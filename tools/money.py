"""INR amount parsing/formatting. Kept deterministic so the LLM never does
lakh/crore arithmetic itself."""

from __future__ import annotations

import re

_UNITS = {
    "k": 1_000, "thousand": 1_000,
    "l": 100_000, "lac": 100_000, "lacs": 100_000, "lakh": 100_000, "lakhs": 100_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
}
_PATTERN = re.compile(
    r"(₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(thousand|lakhs?|lacs?|crores?|million|cr|mn|k|l|m)?\b",
    re.IGNORECASE,
)


def parse_inr(text: str) -> float | None:
    """'₹40L' -> 4_000_000; '1.5 crore' -> 15_000_000; '40,00,000' -> 4_000_000. None if no amount.

    In free text ("50 laptops, budget ₹40L") a number with a currency marker or a
    lakh/crore-style unit wins over a bare number.
    """
    matches = list(_PATTERN.finditer(text or ""))
    if not matches:
        return None
    m = next((x for x in matches if x.group(1) or x.group(3)), matches[0])
    number = float(m.group(2).replace(",", ""))
    unit = (m.group(3) or "").lower()
    return number * _UNITS.get(unit, 1)


def format_inr(amount: float) -> str:
    """4_000_000 -> '₹40,00,000 (₹40.00 L)'; Indian digit grouping."""
    n = int(round(amount))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    sign = "-" if n < 0 else ""
    if abs(n) >= 10_000_000:
        short = f" (₹{abs(n) / 10_000_000:.2f} Cr)"
    elif abs(n) >= 100_000:
        short = f" (₹{abs(n) / 100_000:.2f} L)"
    else:
        short = ""
    return f"{sign}₹{s}{short}"
