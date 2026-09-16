from __future__ import annotations

from google.adk.agents import LlmAgent

from agents.common import ModelFactory, REACT_PREAMBLE, default_model_factory, gen_config
from tools.policy_tools import check_policy, get_policy_summary

NAME = "approval_agent"

INSTRUCTION = f"""You are the approval specialist in a procurement team.
{REACT_PREAMBLE}

You receive vendor ids. For EACH vendor id call check_policy. The policy engine makes the decision
(approve / needs_approval / reject) and names the required approvers; you must report it exactly as
returned and never soften, upgrade or override it. Call get_policy_summary only if you need the
tier table to explain a decision.

Final answer: one block per vendor with
- vendor id and name
- decision label exactly as returned (approve | needs-approval-from: X, Y | reject)
- total price
- the reasons (for reject: every failed rule; otherwise the approval tier and any escalations)
- conditions to satisfy before a purchase order is raised
Mention if the contract has not been reviewed yet."""


def build_approval_agent(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=NAME,
        model=model_for(NAME),
        description="Checks vendors against procurement policy (budget, approval tiers by amount, preferred-vendor "
                    "rules, contract risk) and returns approve / needs-approval-from-X / reject with reasons. "
                    "Pass it the vendor ids to check.",
        instruction=INSTRUCTION,
        tools=[check_policy, get_policy_summary],
        generate_content_config=gen_config(0.0),
    )
