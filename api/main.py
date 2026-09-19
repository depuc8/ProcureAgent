"""Public HTTP API for ProcureAgent (Cloud Run).

    POST /v1/sessions                  -> {session_id}
    POST /v1/chat                      -> {session_id, answer, request_id, latency_s, ttft_s, usage}
    POST /v1/chat/stream               -> Server-Sent Events: delta* final done
    GET  /v1/sessions/{id}?user_id=..  -> persisted session state (request, shortlist, findings, decisions)
    GET  /healthz

Auth: `Authorization: Bearer <PROCURE_API_TOKEN>`, resolved from the environment or Secret
Manager (see settings.get_secret). It fails closed: with no token configured every /v1 call is
refused (503) unless PROCURE_API_ALLOW_UNAUTHENTICATED=true is set for local development.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.backends import AgentBackend, SessionNotFound, backend_from_env
from settings import get_secret

log = logging.getLogger("procure.api")
USER_ID_PATTERN = r"^[A-Za-z0-9_.@-]{1,64}$"
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    user_id: str = Field(default="api-user", pattern=USER_ID_PATTERN)


class SessionRequest(BaseModel):
    user_id: str = Field(default="api-user", pattern=USER_ID_PATTERN)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    request_id: str
    latency_s: float
    ttft_s: float | None
    usage: dict | None = None


def require_auth(authorization: str | None = Header(default=None)) -> None:
    expected = get_secret("PROCURE_API_TOKEN")
    if not expected:
        if os.getenv("PROCURE_API_ALLOW_UNAUTHENTICATED", "").lower() == "true":
            return
        raise HTTPException(503, "API token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip().encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})


def create_app(backend: AgentBackend | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend or backend_from_env()
        yield

    app = FastAPI(title="ProcureAgent API", version="0.1.0", lifespan=lifespan)

    def be() -> AgentBackend:
        return app.state.backend

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/sessions", dependencies=[Depends(require_auth)])
    async def create_session(body: SessionRequest | None = None) -> dict:
        body = body or SessionRequest()
        return {"session_id": await be().create_session(body.user_id), "user_id": body.user_id}

    async def _session(req: ChatRequest) -> str:
        return req.session_id or await be().create_session(req.user_id)

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
    async def chat(req: ChatRequest) -> ChatResponse:
        session_id = await _session(req)
        request_id = uuid.uuid4().hex
        try:
            r = await be().chat(req.user_id, session_id, req.message, request_id)
        except SessionNotFound:
            raise HTTPException(404, "session not found")
        except Exception:
            log.exception("chat failed request_id=%s", request_id)
            raise HTTPException(502, f"agent call failed (request_id={request_id})")
        return ChatResponse(session_id=r.session_id, answer=r.answer, request_id=r.request_id,
                            latency_s=round(r.latency_s, 4), ttft_s=None if r.ttft_s is None else round(r.ttft_s, 4),
                            usage=r.usage)

    @app.post("/v1/chat/stream", dependencies=[Depends(require_auth)])
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        session_id = await _session(req)
        request_id = uuid.uuid4().hex

        async def sse():
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'request_id': request_id})}\n\n"
            try:
                async for item in be().stream(req.user_id, session_id, req.message, request_id):
                    yield f"data: {json.dumps(item, default=str)}\n\n"
            except SessionNotFound:
                yield f"data: {json.dumps({'type': 'error', 'error': 'session not found'})}\n\n"
            except Exception:
                log.exception("stream failed request_id=%s", request_id)
                yield f"data: {json.dumps({'type': 'error', 'error': 'agent call failed', 'request_id': request_id})}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def get_session(session_id: str, user_id: str = Query(default="api-user", pattern=USER_ID_PATTERN)) -> dict:
        try:
            return {"session_id": session_id, "state": await be().get_state(user_id, session_id)}
        except SessionNotFound:
            raise HTTPException(404, "session not found")

    return app


app = create_app()
