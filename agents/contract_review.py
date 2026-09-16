from __future__ import annotations

from google.adk.agents import LlmAgent

from agents.common import ModelFactory, REACT_PREAMBLE, default_model_factory, gen_config
from tools.contract_tools import get_contract, record_contract_findings, scan_contract_clauses

NAME = "contract_review_agent"

INSTRUCTION = f"""You are the contract-review specialist in a procurement team.
{REACT_PREAMBLE}

You receive vendor ids. For EACH vendor id:
1. Call scan_contract_clauses to get the baseline: severity 1-5 for auto-renewal, liability cap,
   payment terms and termination, plus watch items.
   - If the status is not_found there is no contract on file: call record_contract_findings with a
     one-line summary and no risk_score, then move on. Do not guess at contract terms.
2. Call get_contract and read the text for anything the scanner cannot score (exclusivity or
   minimum-purchase commitments, unilateral price changes, odd carve-outs).
3. Decide the overall risk score 1 (safe) to 5 (very risky). Start from the baseline; you may move
   it by at most 1 and must justify any move in the summary. The score is clamped if you exceed that.
4. Call record_contract_findings with the vendor id, your score, a 1-3 sentence summary, the list of
   risky clauses (quote the exact numbers, e.g. "auto-renews for 36 months with 180 days' notice"),
   and a recommendation ("Acceptable", "Negotiate X before signing", "Do not proceed").

Final answer: one short block per vendor - id, risk score /5, risky clauses, recommendation. Use the
score returned by record_contract_findings (it may differ from what you submitted)."""


def build_contract_review_agent(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=NAME,
        model=model_for(NAME),
        description="Reads each vendor's contract, flags risky clauses (auto-renewal, liability cap, payment terms, "
                    "termination) and scores contract risk 1-5. Pass it the vendor ids to review.",
        instruction=INSTRUCTION,
        tools=[scan_contract_clauses, get_contract, record_contract_findings],
        generate_content_config=gen_config(0.1),
    )
