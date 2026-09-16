"""Scripted stand-ins for Gemini so the whole agent graph can run offline.

Each agent gets a ScriptedLlm whose `brain` plays the part a competent model would:
it reads the conversation, decides the next tool call (or the final text) and nothing
else. Everything *around* the model - the ADK runtime, AgentTool delegation, the
LoopAgent, session state, the real tools, the policy engine and the telemetry plugin -
is the production code. What this does NOT test is Gemini's own judgement; the
`live` tests cover that when credentials are available.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.planner import build_planner
from observability.plugin import TelemetryPlugin
from observability.sink import ListSink
from tools.money import format_inr

Call = tuple[str, dict]                       # (tool name, args)
Reply = list[Call] | str


# ------------------------------------------------------------------ conversation view
@dataclass
class Turn:
    """What the brain sees: the latest user text and the tool results since then."""
    user_text: str
    results: list[tuple[str, Any]] = field(default_factory=list)   # (tool name, response)
    last: tuple[str, Any] | None = None                            # last function response in the request

    def result(self, name: str, nth: int = -1) -> Any:
        matches = [r for n, r in self.results if n == name]
        return matches[nth] if matches else None

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.results if n == name)


def _response_payload(fr: types.FunctionResponse) -> Any:
    data = fr.response or {}
    if set(data) == {"result"}:
        data = data["result"]
    if isinstance(data, str):
        try:
            return json.loads(data)
        except ValueError:
            return data
    return data


def view(req: LlmRequest) -> Turn:
    results: list[tuple[str, Any]] = []
    user_text = ""
    for content in reversed(req.contents or []):
        frs = [p.function_response for p in content.parts or [] if p.function_response]
        if frs:
            results = [(fr.name, _response_payload(fr)) for fr in frs] + results
            continue
        texts = [p.text for p in content.parts or [] if p.text]
        if content.role == "user" and texts:
            user_text = " ".join(texts)
            break
    last = None
    if req.contents:
        frs = [p.function_response for p in req.contents[-1].parts or [] if p.function_response]
        if frs:
            last = (frs[-1].name, _response_payload(frs[-1]))
    return Turn(user_text=user_text, results=results, last=last)


class ScriptedLlm(BaseLlm):
    model: str = "gemini-3.8-flash"          # priced model name so cost maths is exercised
    brain: Any = None
    latency_s: float = 0.003
    stream_text: bool = True

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        await asyncio.sleep(self.latency_s)
        reply = self.brain(view(llm_request), llm_request)
        prompt_chars = len(str(llm_request.contents)) + len(str(llm_request.config.system_instruction if llm_request.config else ""))
        if isinstance(reply, str):
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_chars // 4, candidates_token_count=max(1, len(reply) // 4), thoughts_token_count=7)
            if stream and self.stream_text and len(reply) > 40:
                step = max(1, len(reply) // 3)
                for i in range(0, len(reply), step):
                    await asyncio.sleep(self.latency_s / 3)
                    yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=reply[i:i + step])]), partial=True)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=reply)]), usage_metadata=usage)
        else:
            parts = [types.Part.from_function_call(name=n, args=a) for n, a in reply]
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_chars // 4, candidates_token_count=max(1, len(str(reply)) // 4), thoughts_token_count=7)
            yield LlmResponse(content=types.Content(role="model", parts=parts), usage_metadata=usage)


# ------------------------------------------------------------------ the brains
VENDOR_ID = re.compile(r"\bV\d{4}\b")


def vendor_brain(t: Turn, _req) -> Reply:
    swap = re.search(r"swap shortlist position (\d)", t.user_text, re.IGNORECASE)
    if swap and t.last is None:
        return [("find_cheaper_alternative", {"position": int(swap.group(1))})]
    if swap and t.last:
        out = t.last[1]
        if out["status"] == "ok":
            return (f"Swapped position {out['position']}: {out['replaced']['vendor_id']} -> NEW {out['replacement']['vendor_id']} "
                    f"({out['replacement']['name']}), saving {out['total_saving']}.")
        return f"No cheaper vendor available: {out.get('message', out['status'])}"
    if t.last is None:
        return [("search_vendors", {"require_contract_on_file": True})]
    name, out = t.last
    if name == "search_vendors":
        if out["status"] == "ok" or t.count("search_vendors") >= 4:
            sl = out.get("shortlist", [])
            if not sl:
                return "No vendors qualify even after relaxing constraints."
            return "Shortlist: " + "; ".join(
                f"#{v['rank']} {v['vendor_id']} {v['name']} {format_inr(v['unit_price_inr'])}/unit, "
                f"total {format_inr(v['total_price_inr'])}, rating {v['rating']}, {v['delivery_days']}d" for v in sl)
        d, f = out.get("diagnostics", {}), out["filters"]
        # relax one constraint at a time: contract -> delivery -> price (never the rating floor)
        if f["contract_on_file_only"] and d.get("without_contract_requirement", 0) > out["candidates_considered"]:
            return [("search_vendors", {"require_contract_on_file": False})]
        if d.get("without_delivery_limit", 0) > out["candidates_considered"]:
            return [("search_vendors", {"require_contract_on_file": False, "max_delivery_days": 90})]
        if d.get("without_price_ceiling", 0) > out["candidates_considered"]:
            return [("search_vendors", {"require_contract_on_file": False, "max_delivery_days": 90,
                                        "max_unit_price_inr": 10_000_000.0})]
        return "No vendors qualify; no single constraint relaxation helps."
    return "Done."


def contract_brain(t: Turn, _req) -> Reply:
    ids = list(dict.fromkeys(VENDOR_ID.findall(t.user_text)))
    done = t.count("record_contract_findings")
    if done >= len(ids):
        lines = []
        for _, r in [x for x in t.results if x[0] == "record_contract_findings"]:
            f = r["findings"]
            lines.append(f"{f['vendor_id']}: " + ("no contract on file" if f["risk_score"] is None else f"risk {f['risk_score']}/5") +
                         f" - {f['summary']}")
        return "\n".join(lines)
    vid = ids[done]
    if t.last is None or t.last[0] == "record_contract_findings":
        return [("scan_contract_clauses", {"vendor_id": vid})]
    scan = t.last[1]
    if scan["status"] == "not_found":
        return [("record_contract_findings", {"vendor_id": vid, "summary": "No contract on file."})]
    risky = [f["found"] for f in scan["flags"] if f["severity"] >= 3]
    return [("record_contract_findings", {"vendor_id": vid, "risk_score": scan["baseline_score"],
                                          "summary": f"Baseline scanner score {scan['baseline_score']}/5.",
                                          "risky_clauses": risky, "recommendation": "See flagged clauses."})]


def approval_brain(t: Turn, _req) -> Reply:
    ids = list(dict.fromkeys(VENDOR_ID.findall(t.user_text)))
    done = t.count("check_policy")
    if done < len(ids):
        return [("check_policy", {"vendor_id": ids[done]})]
    return "\n".join(f"{r['vendor_id']} {r['vendor_name']}: {r['label']} ({r['total_price']})"
                     for n, r in t.results if n == "check_policy")


def critic_brain(t: Turn, _req) -> Reply:
    if t.last is None:
        return [("get_review_context", {})]
    name, out = t.last
    if name == "get_review_context":
        return [("check_draft_facts", {})]
    if name == "check_draft_facts":
        return [("submit_review", {"issues": out["issues"]})]
    return "Review submitted."


def make_reviser_brain(fixes: bool = True):
    def brain(t: Turn, _req) -> Reply:
        if t.last is None:
            return [("get_review_context", {})]
        name, ctx = t.last
        if name == "get_review_context":
            draft = build_draft(ctx) if fixes else ctx["draft_recommendation"] + "\n(no change)"
            return [("save_revised_draft", {"draft_markdown": draft})]
        return "Revised."
    return brain


# ---- planner ---------------------------------------------------------------
_CATEGORY_WORDS = {"laptop": "Laptops", "mobile": "Mobile Devices", "phone": "Mobile Devices",
                   "desktop": "Desktops & Workstations", "chair": "Office Furniture", "furniture": "Office Furniture"}


def parse_need(text: str) -> dict:
    low = text.lower()
    category = next((c for w, c in _CATEGORY_WORDS.items() if w in low), "Laptops")
    qty = int(re.search(r"(\d[\d,]*)", text).group(1).replace(",", ""))
    weeks, days = re.search(r"(\d+)\s*weeks?", low), re.search(r"(\d+)\s*days?", low)
    delivery = int(weeks.group(1)) * 7 if weeks else int(days.group(1)) if days else 30
    certs = ["ISO 27001"] if "iso 27001" in low else []
    return {"category": category, "quantity": qty, "max_delivery_days": delivery, "required_certifications": certs}


def build_draft(ctx: dict, flaw: str | None = None) -> str:
    """The recommendation a careful planner would write from the verified session facts."""
    req, short = ctx["request"], ctx["shortlist"] or []
    findings, decisions = ctx["contract_findings"] or {}, ctx["approval_decisions"] or {}
    rec_id = ctx.get("recommended_vendor_id") or ""
    lines = [f"# Procurement recommendation",
             f"**Request:** {req['quantity']} x {req['category']}, budget {format_inr(req['budget_inr'])}, "
             f"delivery within {req['max_delivery_days']} days."]
    rec = next((v for v in short if v["vendor_id"] == rec_id), None)
    if rec:
        d = decisions[rec_id]
        approvers = ", ".join(d["required_approvers"]) or "none"
        if flaw == "omit_approver_and_fake_figure":
            approvers = "to be confirmed"
        lines.append(f"**Summary:** Recommend {rec['name']} ({rec_id}) at {format_inr(rec['total_price_inr'])}. "
                     f"Policy outcome: {d['decision']} - approvers: {approvers}.")
    else:
        lines.append("**Summary:** No vendor is recommended - every option was rejected by policy or none qualified. "
                     "Decision: reject.")
    lines.append("## Shortlist comparison")
    for v in short:
        f, d = findings.get(v["vendor_id"]), decisions.get(v["vendor_id"])
        risk = "not reviewed" if f is None else ("no contract on file" if f["risk_score"] is None else f"contract risk {f['risk_score']}/5")
        lines.append(f"- {v['name']} ({v['vendor_id']}): {format_inr(v['unit_price_inr'])}/unit, total {format_inr(v['total_price_inr'])}, "
                     f"rating {v['rating']}, {v['delivery_days']} days, {risk}, policy: {(d['label'] if flaw is None else d['decision']) if d else 'not checked'}"
                     + ("" if not d or d["decision"] != "reject" else " - " + "; ".join(d["reasons"])))
    if rec:
        d = decisions[rec_id]
        lines.append("## Policy and approvals")
        lines.append(f"Decision: {d['label'] if flaw is None else d['decision']}.")
        lines += [f"- Condition: {c}" for c in d["conditions"]]
    if flaw == "omit_approver_and_fake_figure":
        lines.append("Estimated saving versus market rate: ₹7,77,777.")
    return "\n".join(lines)


def make_planner_brain(flaw: str | None = None, flaw_rounds: int = 1):
    def brain(t: Turn, _req) -> Reply:
        low = t.user_text.lower()
        n = len(t.results)
        swap = re.search(r"swap vendor (\d)", low)
        if swap:
            return swap_plan(t, int(swap.group(1)), flaw)
        return new_request_plan(t, n, flaw)
    return brain


def new_request_plan(t: Turn, n: int, flaw) -> Reply:
    need = parse_need(t.user_text)
    steps: list[Callable[[], Reply]] = [
        lambda: [("parse_inr_amount", {"text": t.user_text})],
        lambda: [("save_request", {**need, "budget_inr": t.result("parse_inr_amount")["amount_inr"]})],
        lambda: [("vendor_search_agent", {"request": "Find a shortlist for the saved request."})],
        lambda: [("get_session_summary", {})],
        lambda: [("contract_review_agent", {"request": "Review contracts for " + " ".join(v["vendor_id"] for v in t.result("get_session_summary")["shortlist"])})],
        lambda: [("approval_agent", {"request": "Check policy for " + " ".join(v["vendor_id"] for v in t.result("get_session_summary")["shortlist"])})],
        lambda: [("get_session_summary", {})],
    ]
    if n < 4:
        return steps[n]()
    if not t.result("get_session_summary")["shortlist"]:       # nothing qualified: skip review/policy, explain in the draft
        return finish_plan(t, n - 4, flaw)
    if n < len(steps):
        return steps[n]()
    return finish_plan(t, n - len(steps), flaw)


def choose(ctx: dict) -> str | None:
    ok = [(ctx["contract_findings"][v["vendor_id"]]["risk_score"] or 3, len(d["required_approvers"]), v["total_price_inr"], v["vendor_id"])
          for v in ctx["shortlist"] for d in [ctx["approval_decisions"].get(v["vendor_id"])]
          if d and d["decision"] != "reject"]
    return min(ok)[3] if ok else None


def finish_plan(t: Turn, k: int, flaw) -> Reply:
    ctx = t.result("get_session_summary")
    rec = choose(ctx)
    steps: list[Callable[[], Reply]] = []
    if rec:
        steps.append(lambda: [("set_recommended_vendor", {"vendor_id": rec})])
    else:
        steps.append(lambda: [("get_session_summary", {})])        # nothing to recommend; keep the step count uniform
    steps += [
        lambda: [("save_draft", {"draft_markdown": build_draft({**ctx, "recommended_vendor_id": rec or ""}, flaw)})],
        lambda: [("draft_review_loop", {"request": "Review the saved draft."})],
        lambda: [("get_final_draft", {})],
    ]
    if k < len(steps):
        return steps[k]()
    final = t.result("get_final_draft")
    text = final["draft_recommendation"]
    if final["review_status"] != "approved":
        text += f"\n\n_Review ended with open items ({final['review_status']})._"
    return text


def swap_plan(t: Turn, pos: int, flaw) -> Reply:
    n = len(t.results)
    steps: list[Callable[[], Reply]] = [
        lambda: [("vendor_search_agent", {"request": f"Swap shortlist position {pos} for a cheaper vendor."})],
        lambda: [("get_session_summary", {})],
    ]
    if n < 2:
        return steps[n]()
    swapped = t.result("vendor_search_agent")
    if "NEW" not in swapped:
        return swapped
    new_id = re.search(r"NEW (V\d{4})", swapped).group(1)
    more: list[Callable[[], Reply]] = [
        lambda: [("contract_review_agent", {"request": f"Review contracts for {new_id}"})],
        lambda: [("approval_agent", {"request": f"Check policy for {new_id}"})],
        lambda: [("get_session_summary", {})],
    ]
    if n - 2 < len(more):
        return more[n - 2]()
    return finish_plan(t, n - 5, flaw)


# ------------------------------------------------------------------ assembly + runner
def scripted_factory(planner_flaw: str | None = None, reviser_fixes: bool = True, latency_s: float = 0.003):
    brains = {
        "procure_planner": make_planner_brain(planner_flaw),
        "vendor_search_agent": vendor_brain,
        "contract_review_agent": contract_brain,
        "approval_agent": approval_brain,
        "draft_critic": critic_brain,
        "draft_reviser": make_reviser_brain(reviser_fixes),
    }
    return lambda name: ScriptedLlm(brain=brains[name], latency_s=latency_s)


class Harness:
    """One ADK Runner + session over the scripted agent graph."""

    def __init__(self, planner_flaw=None, reviser_fixes=True, stream=False, latency_s=0.003):
        self.sink = ListSink()
        self.planner = build_planner(scripted_factory(planner_flaw, reviser_fixes, latency_s))
        app = App(name="procure_test", root_agent=self.planner, plugins=[TelemetryPlugin(sink=self.sink)])
        self.runner = InMemoryRunner(app=app)
        self.stream = stream
        self.session_id: str | None = None
        self.events: list = []

    async def ask(self, text: str) -> str:
        if self.session_id is None:
            self.session_id = (await self.runner.session_service.create_session(app_name="procure_test", user_id="u1")).id
        run_config = RunConfig(streaming_mode=StreamingMode.SSE if self.stream else StreamingMode.NONE)
        final = ""
        async for ev in self.runner.run_async(
                user_id="u1", session_id=self.session_id, run_config=run_config,
                new_message=types.Content(role="user", parts=[types.Part(text=text)])):
            self.events.append(ev)
            if ev.author == "procure_planner" and ev.is_final_response() and ev.content and ev.content.parts:
                final = "".join(p.text or "" for p in ev.content.parts)
        return final

    async def state(self) -> dict:
        s = await self.runner.session_service.get_session(app_name="procure_test", user_id="u1", session_id=self.session_id)
        return dict(s.state)
