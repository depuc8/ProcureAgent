"""Token pricing from data/pricing.json (a config value, not code).

Rates are dated: the newest entry whose `from` <= the call date applies, so a
scheduled price change (e.g. an introductory rate ending) needs no code change.
Unknown models fall back to `default_model` and are flagged so a wrong cost is
never silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from settings import get_settings


@dataclass(frozen=True)
class Rate:
    model: str
    effective_from: str
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float
    fallback: bool = False


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate_for(model: str, on: date | None = None) -> Rate:
    cfg = _load(str(get_settings().pricing_path))
    on = on or date.today()
    name = model if model in cfg["models"] else cfg["default_model"]
    entries = sorted(cfg["models"][name], key=lambda e: e["from"])
    applicable = [e for e in entries if e["from"] <= on.isoformat()] or entries[:1]
    e = applicable[-1]
    return Rate(name, e["from"], e["input_per_m"], e["output_per_m"],
                e.get("cached_input_per_m", e["input_per_m"]), fallback=name != model)


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
             on: date | None = None) -> tuple[float, Rate]:
    """input_tokens includes cached tokens (as Gemini reports prompt_token_count);
    output_tokens must include thinking tokens (they are billed as output)."""
    r = rate_for(model, on)
    fresh = max(0, input_tokens - cached_tokens)
    cost = (fresh * r.input_per_m + cached_tokens * r.cached_input_per_m + output_tokens * r.output_per_m) / 1_000_000
    return cost, r
