"""Agent backends for the public API.

* AgentEngineBackend - production: proxies to the deployed Agent Engine instance, which owns
  sessions, execution, tracing and the authoritative per-request metrics (-> BigQuery).
* LocalBackend       - runs the same ADK app in-process (dev, tests, single-instance deployments).
  Sessions are in memory, so use AgentEngineBackend when Cloud Run scales past one instance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from observability.metrics import request_scope

STATE_KEYS = ("request", "shortlist", "contract_findings", "approval_decisions",
              "recommended_vendor_id", "review_status")


class SessionNotFound(Exception):
    pass


@dataclass
class ChatResult:
    session_id: str
    answer: str
    request_id: str
    latency_s: float
    ttft_s: float | None
    usage: dict | None = None            # authoritative token/cost numbers (LocalBackend only)
    events: list = field(default_factory=list)


class AgentBackend(Protocol):
    async def create_session(self, user_id: str) -> str: ...
    async def chat(self, user_id: str, session_id: str, message: str, request_id: str) -> ChatResult: ...
    def stream(self, user_id: str, session_id: str, message: str, request_id: str) -> AsyncIterator[dict]: ...
    async def get_state(self, user_id: str, session_id: str) -> dict: ...


def _text_of(parts: list[dict] | None) -> str:
    return "".join(p.get("text", "") or "" for p in parts or [] if not p.get("thought"))


class LocalBackend:
    def __init__(self, app: App | None = None, session_service=None):
        if app is None:
            from agents.agent import app as default_app

            app = default_app
        self.app = app
        self.sessions = session_service or InMemorySessionService()
        self.runner = Runner(app=app, session_service=self.sessions)
        self.root = app.root_agent.name

    async def create_session(self, user_id: str) -> str:
        return (await self.sessions.create_session(app_name=self.app.name, user_id=user_id)).id

    async def _require(self, user_id: str, session_id: str):
        s = await self.sessions.get_session(app_name=self.app.name, user_id=user_id, session_id=session_id)
        if s is None:
            raise SessionNotFound(session_id)
        return s

    async def _run(self, user_id, session_id, message, streaming) -> AsyncIterator:
        await self._require(user_id, session_id)
        cfg = RunConfig(streaming_mode=StreamingMode.SSE if streaming else StreamingMode.NONE)
        async for ev in self.runner.run_async(
                user_id=user_id, session_id=session_id, run_config=cfg,
                new_message=types.Content(role="user", parts=[types.Part(text=message)])):
            yield ev

    async def chat(self, user_id, session_id, message, request_id) -> ChatResult:
        start, first, answer = time.perf_counter(), None, ""
        with request_scope(request_id) as metrics:
            async for ev in self._run(user_id, session_id, message, streaming=False):
                if first is None and ev.content and ev.content.parts:
                    first = time.perf_counter() - start
                if ev.author == self.root and ev.is_final_response() and ev.content and ev.content.parts:
                    answer = "".join(p.text or "" for p in ev.content.parts)
        usage = metrics.to_dict() if metrics.finalized else None
        return ChatResult(session_id, answer, request_id, time.perf_counter() - start, first, usage)

    async def stream(self, user_id, session_id, message, request_id) -> AsyncIterator[dict]:
        start, first = time.perf_counter(), None
        with request_scope(request_id) as metrics:
            async for ev in self._run(user_id, session_id, message, streaming=True):
                if not (ev.content and ev.content.parts):
                    continue
                if first is None:
                    first = time.perf_counter() - start
                if ev.author != self.root:
                    continue
                text = "".join(p.text or "" for p in ev.content.parts)
                if ev.partial and text:
                    yield {"type": "delta", "text": text}
                elif ev.is_final_response() and text:
                    yield {"type": "final", "text": text}
        yield {"type": "done", "request_id": request_id, "latency_s": round(time.perf_counter() - start, 4),
               "ttft_s": None if first is None else round(first, 4),
               "usage": metrics.to_dict() if metrics.finalized else None}

    async def get_state(self, user_id, session_id) -> dict:
        s = await self._require(user_id, session_id)
        return {k: s.state.get(k) for k in STATE_KEYS}


class AgentEngineBackend:
    def __init__(self, project: str, location: str, resource_name: str, client=None):
        if client is None:
            import agentplatform

            client = agentplatform.Client(project=project, location=location)
        self._remote = client.runtimes.get(name=resource_name)

    async def create_session(self, user_id: str) -> str:
        return (await self._remote.async_create_session(user_id=user_id))["id"]

    async def _events(self, user_id, session_id, message) -> AsyncIterator[dict]:
        try:
            await self._remote.async_get_session(user_id=user_id, session_id=session_id)
        except Exception as e:                       # remote raises on unknown / foreign sessions
            raise SessionNotFound(session_id) from e
        async for ev in self._remote.async_stream_query(user_id=user_id, session_id=session_id, message=message):
            yield ev

    async def chat(self, user_id, session_id, message, request_id) -> ChatResult:
        start, first, answer = time.perf_counter(), None, ""
        async for ev in self._events(user_id, session_id, message):
            parts = (ev.get("content") or {}).get("parts")
            if first is None and parts:
                first = time.perf_counter() - start
            if parts and not ev.get("partial") and not any(p.get("function_call") for p in parts) and _text_of(parts):
                answer = _text_of(parts)             # the last non-partial text event is the final answer
        return ChatResult(session_id, answer, request_id, time.perf_counter() - start, first, usage=None)

    async def stream(self, user_id, session_id, message, request_id) -> AsyncIterator[dict]:
        start, first, last = time.perf_counter(), None, ""
        async for ev in self._events(user_id, session_id, message):
            parts = (ev.get("content") or {}).get("parts")
            if not parts:
                continue
            first = first if first is not None else time.perf_counter() - start
            text = _text_of(parts)
            if ev.get("partial") and text:
                yield {"type": "delta", "text": text}
            elif text and not any(p.get("function_call") for p in parts):
                last = text
        yield {"type": "final", "text": last}
        yield {"type": "done", "request_id": request_id, "latency_s": round(time.perf_counter() - start, 4),
               "ttft_s": None if first is None else round(first, 4), "usage": None}

    async def get_state(self, user_id, session_id) -> dict:
        try:
            s: Any = await self._remote.async_get_session(user_id=user_id, session_id=session_id)
        except Exception as e:
            raise SessionNotFound(session_id) from e
        state = s.get("state", {}) if isinstance(s, dict) else getattr(s, "state", {})
        return {k: state.get(k) for k in STATE_KEYS}


def backend_from_env() -> AgentBackend:
    import os

    from settings import get_settings

    s = get_settings()
    kind = os.getenv("AGENT_BACKEND", "local").lower()
    if kind == "agent_engine":
        name = os.environ["AGENT_ENGINE_RESOURCE_NAME"]
        return AgentEngineBackend(s.project, s.location, name)
    return LocalBackend()
