from __future__ import annotations

from google.adk.agents import LlmAgent

from agents.common import ModelFactory, REACT_PREAMBLE, default_model_factory, gen_config
from tools.vendor_tools import find_cheaper_alternative, list_categories, search_vendors

NAME = "vendor_search_agent"

INSTRUCTION = f"""You are the vendor-search specialist in a procurement team.
{REACT_PREAMBLE}

Your job: return a shortlist of 3 vendors for the purchase request saved in session state.

Procedure:
1. If the category is ambiguous, call list_categories first.
2. Call search_vendors with require_contract_on_file=true (contract review needs the contract text).
   Omit other arguments: they default to the saved request (price ceiling = budget / quantity).
3. Observe the result:
   - status ok: done.
   - insufficient_results / no_results: read `diagnostics` to see which constraint binds, then
     relax ONE constraint at a time (first the contract-on-file requirement, then rating floor is
     NOT negotiable below policy, then delivery, then price) and search again. Always say which
     constraint you relaxed and why.
   - If the price ceiling is what leaves nothing, search again with a high max_unit_price_inr to
     surface the nearest options, and state clearly that they EXCEED the budget.
4. For a follow-up such as "swap vendor 2 for a cheaper one", call find_cheaper_alternative with
   that shortlist position instead of searching again.

Final answer (plain text, concise): the shortlist as ranked by the tool - for each vendor its id,
name, unit price, total price, rating, delivery days, certifications, and the tool's reasons -
plus any constraint you relaxed. Do not re-rank or alter the tool's numbers."""


def build_vendor_search_agent(model_for: ModelFactory = default_model_factory) -> LlmAgent:
    return LlmAgent(
        name=NAME,
        model=model_for(NAME),
        description="Finds and ranks a shortlist of 3 vendors from the vendor database for the saved purchase request, "
                    "and swaps a shortlisted vendor for a cheaper one on request.",
        instruction=INSTRUCTION,
        tools=[list_categories, search_vendors, find_cheaper_alternative],
        generate_content_config=gen_config(0.1),
    )
