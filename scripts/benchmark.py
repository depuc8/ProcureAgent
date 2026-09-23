"""Measure latency, tokens and cost.

    python -m scripts.benchmark                      # offline: orchestration overhead + tool latency
    python -m scripts.benchmark --live --runs 3      # real Gemini; needs credentials + PROCURE_RUN_LIVE=1

OFFLINE mode swaps Gemini for scripted models with a configurable simulated model latency. It
measures what this repo controls (ADK orchestration, tools, telemetry plugin, policy engine):
its token counts and cost are *simulated* and are reported only as a plumbing check, never as
model cost. LIVE mode runs the real agents against Gemini and is the only source of real model
latency, TTFT, tokens/sec and cost; it also appends each request to METRICS_JSONL_PATH so the
dashboard picks the runs up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

PROMPTS = [
    "We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks",
    "We need 50 laptops, budget ₹10L, delivery in 3 weeks",
    "We need 2 laptops, budget ₹2L, delivery in 4 weeks",
    "We need 400 laptops for a data-centre refresh, budget ₹3 crore, delivery in 4 weeks",
    "We need 20 laptops with ISO 27001 certification, budget ₹30L, delivery in 4 weeks",
]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def table(rows: list[tuple[str, str]]) -> str:
    w = max(len(a) for a, _ in rows)
    return "\n".join(f"| {a.ljust(w)} | {b} |" for a, b in rows)


def tool_microbench(n: int = 300) -> list[tuple[str, str]]:
    from types import SimpleNamespace

    from tools.contract_tools import scan_contract_clauses
    from tools.policy_tools import check_policy
    from tools.state_tools import save_request
    from tools.vendor_tools import search_vendors

    ctx = SimpleNamespace(state={}, actions=SimpleNamespace(escalate=False, skip_summarization=False))
    save_request("Laptops", 50, 4_000_000, 21, ctx)
    search_vendors(ctx, require_contract_on_file=True)
    jobs = {
        "search_vendors (local backend)": lambda: search_vendors(ctx, require_contract_on_file=True),
        "scan_contract_clauses": lambda: scan_contract_clauses("V0003"),
        "check_policy": lambda: check_policy("V0002", ctx),
    }
    out = []
    for name, fn in jobs.items():
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        out.append((name, f"p50 {pct(times, .5):.3f} ms, p95 {pct(times, .95):.3f} ms ({n} calls)"))
    return out


async def offline(runs: int, model_latency: float) -> list[tuple[str, str]]:
    from tests.scripted import Harness

    lat, calls = [], []
    for i in range(runs):
        h = Harness(latency_s=model_latency)
        t = time.perf_counter()
        await h.ask(PROMPTS[i % len(PROMPTS)])
        lat.append(time.perf_counter() - t)
        calls.append(h.sink.records[-1]["llm_calls"] + h.sink.records[-1]["tool_calls"])
    return [
        (f"Requests (scripted models, {model_latency * 1000:.0f} ms simulated per model call)", str(runs)),
        ("End-to-end latency p50", f"{pct(lat, .5) * 1000:.0f} ms"),
        ("End-to-end latency p95", f"{pct(lat, .95) * 1000:.0f} ms"),
        ("Model + tool calls per request (median)", f"{statistics.median(calls):.0f}"),
    ]


async def live(runs: int) -> list[tuple[str, str]]:
    if os.getenv("PROCURE_RUN_LIVE") != "1":
        raise SystemExit("Live mode calls Gemini and costs money: set PROCURE_RUN_LIVE=1 to confirm.")
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from agents.agent import app

    runner = InMemoryRunner(app=app)
    recs: list[dict] = []
    from observability.metrics import request_scope

    for i in range(runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        sid = (await runner.session_service.create_session(app_name=app.name, user_id="bench")).id
        with request_scope() as m:
            async for _ in runner.run_async(user_id="bench", session_id=sid, run_config=RunConfig(streaming_mode=StreamingMode.SSE),
                                            new_message=types.Content(role="user", parts=[types.Part(text=prompt)])):
                pass
        recs.append(m.to_dict())
        print(f"  [{i + 1}/{runs}] {recs[-1]['latency_s']:.1f}s ${recs[-1]['cost_usd']:.4f} {prompt[:50]}", file=sys.stderr)
    lat = [r["latency_s"] for r in recs]
    f = lambda k: [r[k] for r in recs if r[k] is not None]  # noqa: E731
    return [
        (f"Requests ({recs[0]['model']})", str(len(recs))),
        ("End-to-end latency p50 / p95", f"{pct(lat, .5):.1f} s / {pct(lat, .95):.1f} s"),
        ("Time to first token (mean)", f"{statistics.mean(f('ttft_s')):.2f} s"),
        ("Tokens/sec (mean)", f"{statistics.mean(f('tokens_per_sec')):.0f}"),
        ("Tokens per request (mean, in + out)", f"{statistics.mean(f('total_tokens')):.0f}"),
        ("Cost per request (mean)", f"${statistics.mean(f('cost_usd')):.4f}"),
        ("Pricing rate card", recs[0]["pricing_effective_from"] + (" (FALLBACK rate: model not in pricing.json)" if recs[0]["pricing_fallback"] else "")),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--model-latency-ms", type=float, default=0.0, help="simulated per-call latency (offline)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    a = ap.parse_args()
    rows = asyncio.run(live(a.runs)) if a.live else asyncio.run(offline(a.runs, a.model_latency_ms / 1000)) 
    if not a.live:
        rows += tool_microbench()
    print(("LIVE" if a.live else "OFFLINE (scripted models; orchestration + tools only)") + "\n")
    print(table(rows))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows), f, indent=2)


if __name__ == "__main__":
    main()
