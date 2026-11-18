"""ADK entrypoint. `adk web .` (run from the project root) discovers `agents.agent.app`;
Agent Engine deployment imports `root_agent`/`app` from here too."""

from __future__ import annotations

from google.adk.apps import App

from agents.planner import build_planner
from agents.ratelimit import RateLimitPlugin
from observability.plugin import TelemetryPlugin
from observability.tracing import setup_telemetry

setup_telemetry()

root_agent = build_planner()

app = App(name="agents", root_agent=root_agent, plugins=[RateLimitPlugin(), TelemetryPlugin()])
