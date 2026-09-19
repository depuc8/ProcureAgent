import json

import pytest
from fastapi.testclient import TestClient
from google.adk.apps import App

from api.backends import LocalBackend
from api.main import create_app
from observability.plugin import TelemetryPlugin
from observability.sink import ListSink
from tests.scripted import scripted_factory
from agents.planner import build_planner

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DEMO = "We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks"


@pytest.fixture
def sink():
    return ListSink()


@pytest.fixture
def client(monkeypatch, sink):
    monkeypatch.setenv("PROCURE_API_TOKEN", TOKEN)
    app = App(name="procure_api_test", root_agent=build_planner(scripted_factory()),
              plugins=[TelemetryPlugin(sink=sink)])
    with TestClient(create_app(LocalBackend(app))) as c:
        yield c


def test_health_is_open(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_requests_without_valid_token_are_rejected(client, headers):
    assert client.post("/v1/chat", json={"message": "hi"}, headers=headers).status_code == 401
    assert client.post("/v1/sessions", headers=headers).status_code == 401


def test_fails_closed_when_no_token_configured(monkeypatch):
    monkeypatch.delenv("PROCURE_API_TOKEN", raising=False)
    monkeypatch.delenv("PROCURE_API_ALLOW_UNAUTHENTICATED", raising=False)
    app = App(name="x", root_agent=build_planner(scripted_factory()))
    with TestClient(create_app(LocalBackend(app))) as c:
        assert c.post("/v1/chat", json={"message": "hi"}).status_code == 503


def test_explicit_dev_override_allows_unauthenticated(monkeypatch):
    monkeypatch.delenv("PROCURE_API_TOKEN", raising=False)
    monkeypatch.setenv("PROCURE_API_ALLOW_UNAUTHENTICATED", "true")
    app = App(name="x", root_agent=build_planner(scripted_factory()))
    with TestClient(create_app(LocalBackend(app))) as c:
        assert c.post("/v1/sessions").status_code == 200


def test_chat_returns_recommendation_with_usage(client):
    r = client.post("/v1/chat", json={"message": DEMO}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "Finance Director" in body["answer"] and body["session_id"] and body["request_id"]
    assert body["latency_s"] > 0 and body["ttft_s"] > 0
    u = body["usage"]
    assert u["request_id"] == body["request_id"] and u["total_tokens"] > 0 and u["cost_usd"] > 0


def test_follow_up_in_same_session_and_state_endpoint(client):
    first = client.post("/v1/chat", json={"message": DEMO}, headers=AUTH).json()
    sid = first["session_id"]
    before = client.get(f"/v1/sessions/{sid}", headers=AUTH).json()["state"]
    assert before["request"]["quantity"] == 50 and len(before["shortlist"]) == 3
    second = client.post("/v1/chat", json={"message": "swap vendor 2 for a cheaper one", "session_id": sid}, headers=AUTH)
    assert second.status_code == 200 and second.json()["session_id"] == sid
    after = client.get(f"/v1/sessions/{sid}", headers=AUTH).json()["state"]
    assert after["request"] == before["request"]
    assert after["shortlist"][1]["unit_price_inr"] < before["shortlist"][1]["unit_price_inr"]


def test_sessions_are_scoped_to_user(client):
    sid = client.post("/v1/chat", json={"message": DEMO, "user_id": "alice"}, headers=AUTH).json()["session_id"]
    assert client.get(f"/v1/sessions/{sid}?user_id=alice", headers=AUTH).status_code == 200
    assert client.get(f"/v1/sessions/{sid}?user_id=mallory", headers=AUTH).status_code == 404
    r = client.post("/v1/chat", json={"message": "swap vendor 2 for a cheaper one", "session_id": sid, "user_id": "mallory"}, headers=AUTH)
    assert r.status_code == 404


def test_unknown_session_is_404(client):
    assert client.get("/v1/sessions/nope", headers=AUTH).status_code == 404
    assert client.post("/v1/chat", json={"message": "hi", "session_id": "nope"}, headers=AUTH).status_code == 404


@pytest.mark.parametrize("body", [
    {"message": ""}, {"message": "x" * 4001}, {"message": "ok", "user_id": "bad user!"},
    {"message": "ok", "session_id": "../etc"}, {},
])
def test_input_validation(client, body):
    assert client.post("/v1/chat", json=body, headers=AUTH).status_code == 422


def test_backend_errors_do_not_leak_internals(client, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret internal detail: db password=hunter2")

    monkeypatch.setattr(client.app.state.backend, "chat", boom)
    r = client.post("/v1/chat", json={"message": "hi"}, headers=AUTH)
    assert r.status_code == 502 and "hunter2" not in r.text and "request_id=" in r.json()["detail"]


def test_stream_emits_session_final_and_done(client):
    with client.stream("POST", "/v1/chat/stream", json={"message": DEMO}, headers=AUTH) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    types_ = [e["type"] for e in events]
    assert types_[0] == "session" and types_[-1] == "done" and "final" in types_
    final = next(e for e in events if e["type"] == "final")
    assert "Finance Director" in final["text"]
    assert events[-1]["usage"]["total_tokens"] > 0 and events[-1]["ttft_s"] > 0


def test_stream_requires_auth(client):
    assert client.post("/v1/chat/stream", json={"message": "hi"}).status_code == 401
