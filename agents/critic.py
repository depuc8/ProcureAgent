from __future__ import annotations

from google.adk.agents import LlmAgent, LoopAgent

from agents.common import ModelFactory, REACT_PREAMBLE, default_model_factory, gen_config
from tools.review_tools import check_draft_facts, get_review_context, save_revised_draft, submit_review

CRITIC = "draft_critic"
REVISER = "draft_reviser"
LOOP = "draft_review_loop"
MAX_ITERATIONS = 3

CRITIC_INSTRUCTION = f"""You are the critic: a skeptical reviewer of a procurement recommendation draft.
{REACT_PREAMBLE}

1. Call get_review_context for the verified facts, then check_draft_facts for deterministic findings.
2. Judge the draft against the facts for:
   - MISSING INFORMATION: budget, quantity, delivery requirement, comparison of all shortlisted
     vendors, recommended vendor with unit and total price, contract risk and the risky clauses, the
     policy decision with named approvers, conditions, next steps.
   - POLICY VIOLATIONS: recommending a vendor the policy engine rejected, naming the wrong or
     missing approvers, ignoring an unmet condition, hiding a budget overrun.
   - UNSUPPORTED CLAIMS: any figure, rating, date or assertion that is not in the session data.
3. Call submit_review exactly once:
   - nothing material wrong -> submit_review with an empty issues list (this approves the draft);
   - otherwise -> submit_review with specific, actionable issues (include every finding from
     check_draft_facts that is real). Do not nitpick style.
Previous-round critique (if any) is in get_review_context: confirm each point was actually fixed."""

REVISER_INSTRUCTION = f"""You are the reviser. The critic found problems with the draft recommendation.
{REACT_PREAMBLE}

1. Call get_review_context: it has the verified facts, the current draft and the critique.
2. Rewrite the COMPLETE draft so every critique point is fixed, using only facts from the context.
   Never invent numbers. If the critique says something is missing and the facts do not contain it,
   say plainly that the information is unavailable instead of making it up.
3. Call save_revised_draft with the full revised markdown. Then stop."""


def build_critic_agent(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=CRITIC, model=model_for(CRITIC),
        description="Reviews the draft recommendation for missing information, policy violations and unsupported claims.",
        instruction=CRITIC_INSTRUCTION,
        tools=[get_review_context, check_draft_facts, submit_review],
        generate_content_config=gen_config(0.0),
    )


def build_reviser_agent(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=REVISER, model=model_for(REVISER),
        description="Revises the draft recommendation to address the critic's issues.",
        instruction=REVISER_INSTRUCTION,
        tools=[get_review_context, save_revised_draft],
        generate_content_config=gen_config(0.2),
    )


def build_review_loop(model_for: ModelFactory = default_model_factory) -> LoopAgent:
    """critic -> reviser, repeated until the critic approves or MAX_ITERATIONS rounds have run."""
    return LoopAgent(
        name=LOOP,
        description="Self-reflection loop: a critic reviews the saved draft and sends it back to a reviser until it "
                    "is approved (max 3 rounds). Call after save_draft. Takes no meaningful input.",
        sub_agents=[build_critic_agent(model_for), build_reviser_agent(model_for)],
        max_iterations=MAX_ITERATIONS,
    )
