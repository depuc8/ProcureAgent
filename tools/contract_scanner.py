"""Deterministic contract clause scanner.

Gives the contract-review agent a reproducible *baseline*: each of the four
clause families gets a severity 1 (safe) .. 5 (very risky) from a documented
rubric, and the overall score blends the worst clause with the average. The
LLM still reads the full text and may adjust the final score by at most +/-1
(enforced in contract_tools.record_contract_findings) - it is there to catch
what a regex cannot (exclusivity, unilateral price changes, odd carve-outs).

Rubric
------
auto_renewal  none=1; present=2, +1 if renewal term >12 months, +1 if >=36 months,
              +1 if notice >60 days, +1 if notice >120 days (cap 5)
liability_cap supplier cap as % of 12-month fees: >=150%=1, 100-149%=2, 50-99%=3,
              25-49%=4, <25%=5; uncapped=1; no cap clause found=3
payment_terms advance 100%=5, >=50%=4, >0%=3; otherwise Net>=30 =1, Net 15-29=3,
              Net<15=4; +1 if late interest >=2%/month (cap 5)
termination   convenience exit: notice <=60d=1, <=90d=2, >90d=3; no convenience
              exit (cause only)=4; +1 if lock-in >12 months; +1 if early
              termination charge >0%, +2 if >=25% (cap 5)

overall = floor(0.5 * worst + 0.5 * mean + 0.5)
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

_NUM_WORDS = {"twelve": 12, "twenty-four": 24, "thirty-six": 36}


@dataclass
class ClauseFlag:
    clause: str
    severity: int
    found: str
    rationale: str
    excerpt: str


@dataclass
class ScanResult:
    flags: list[ClauseFlag]
    watch_items: list[dict]
    baseline_score: int

    def to_dict(self) -> dict:
        return {"flags": [asdict(f) for f in self.flags], "watch_items": self.watch_items,
                "baseline_score": self.baseline_score}


def _excerpt(text: str, match: re.Match | None, width: int = 220) -> str:
    if not match:
        return ""
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    line = text[start: end if end != -1 else len(text)].strip()
    return line if len(line) <= width else line[: width - 1] + "…"


def _cap(n: int) -> int:
    return max(1, min(5, n))


def _scan_renewal(text: str) -> ClauseFlag:
    m = re.search(
        r"automatically\s+renew.*?terms?\s+of\s+(\d+)\s+months?.*?at\s+least\s+(\d+)\s+days", text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        loose = re.search(r"auto(?:matic(?:ally)?)?[-\s]?renew", text, re.IGNORECASE)
        if loose and not re.search(r"not\s+renew\s+automatically|does\s+not\s+renew", text, re.IGNORECASE):
            return ClauseFlag("auto_renewal", 3, "auto-renewal (terms not parseable)",
                              "Auto-renewal language found but term/notice could not be parsed; verify manually.",
                              _excerpt(text, loose))
        return ClauseFlag("auto_renewal", 1, "no auto-renewal", "Agreement lapses at end of term; extension needs written agreement.",
                          _excerpt(text, re.search(r"does\s+not\s+renew|not\s+renew\s+automatically", text, re.IGNORECASE)))
    months, notice = int(m.group(1)), int(m.group(2))
    sev = _cap(2 + (months > 12) + (months >= 36) + (notice > 60) + (notice > 120))
    return ClauseFlag("auto_renewal", sev, f"auto-renews for {months} months; {notice} days' notice to opt out",
                      "Long renewal terms and long notice windows lock the buyer in." if sev >= 4 else
                      "Auto-renewal present; calendar the notice deadline.", _excerpt(text, m))


def _scan_liability(text: str) -> ClauseFlag:
    m = re.search(r"liability[^.]*?limited\s+to\s+(\d+)%\s+of\s+the\s+fees", text, re.IGNORECASE)
    pct: float | None = float(m.group(1)) if m else None
    if not m:
        m = re.search(r"liability[^.]*?limited\s+to\s+the\s+fees\s+paid\s+in\s+the\s+(\d+)\s+months", text, re.IGNORECASE)
        if m:
            pct = int(m.group(1)) / 12 * 100
    if pct is None:
        unlimited = re.search(r"liability[^.]*?not\s+capped|unlimited\s+liability", text, re.IGNORECASE)
        if unlimited:
            return ClauseFlag("liability_cap", 1, "supplier liability uncapped", "Full recourse for direct damages.", _excerpt(text, unlimited))
        return ClauseFlag("liability_cap", 3, "no liability clause found", "Silent on liability; confirm the default position with Legal.", "")
    sev = 1 if pct >= 150 else 2 if pct >= 100 else 3 if pct >= 50 else 4 if pct >= 25 else 5
    return ClauseFlag("liability_cap", sev, f"supplier liability capped at {pct:.0f}% of 12-month fees",
                      "Cap is low relative to the fees at risk." if sev >= 4 else "Cap is in a conventional range."
                      if sev <= 2 else "Cap is below fees at risk; negotiate upward.", _excerpt(text, m))


def _scan_payment(text: str) -> ClauseFlag:
    late = re.search(r"interest\s+at\s+(\d+(?:\.\d+)?)%\s+per\s+month", text, re.IGNORECASE)
    late_pct = float(late.group(1)) if late else 0.0
    adv = re.search(r"pay\s+(\d+)%\s+of\s+the\s+order\s+value\s+in\s+advance", text, re.IGNORECASE)
    net = re.search(r"\(Net\s+(\d+)\)|within\s+(\d+)\s+days\s+of\s+receipt", text, re.IGNORECASE)
    if adv:
        pct = int(adv.group(1))
        sev, found = (5 if pct >= 100 else 4 if pct >= 50 else 3), f"{pct}% advance payment"
        why = "Advance payment exposes the buyer to non-delivery risk."
        m = adv
    elif net:
        days = int(net.group(1) or net.group(2))
        sev, found = (1 if days >= 30 else 3 if days >= 15 else 4), f"Net {days} payment terms"
        why = "Short credit period." if sev >= 3 else "Standard or better credit period."
        m = net
    else:
        return ClauseFlag("payment_terms", 3, "no payment terms found", "Payment terms not stated; confirm.", "")
    if late_pct >= 2:
        sev, found = _cap(sev + 1), found + f"; late interest {late_pct:g}%/month"
        why += " Punitive late-payment interest."
    return ClauseFlag("payment_terms", sev, found, why, _excerpt(text, m))


def _scan_termination(text: str) -> ClauseFlag:
    conv = re.search(r"terminate\s+this\s+Agreement\s+for\s+convenience\s+by\s+giving\s+(\d+)\s+days", text, re.IGNORECASE)
    if not conv:
        cause = re.search(r"may\s+terminate\s+this\s+Agreement\s+for\s+convenience|Neither\s+party\s+may\s+terminate[^.]*convenience", text, re.IGNORECASE)
        return ClauseFlag("termination", 4, "no termination for convenience (cause only)",
                          "Buyer cannot exit without proving breach.", _excerpt(text, cause))
    notice = int(conv.group(1))
    sev = 1 if notice <= 60 else 2 if notice <= 90 else 3
    found = [f"buyer may exit on {notice} days' notice"]
    lock = re.search(r"not\s+before\s+the\s+end\s+of\s+the\s+first\s+(\d+)\s+months", text, re.IGNORECASE)
    if lock and int(lock.group(1)) > 12:
        sev += 1
        found.append(f"{lock.group(1)}-month minimum commitment")
    fee = re.search(r"early\s+termination\s+charge\s+equal\s+to\s+(\d+)%", text, re.IGNORECASE)
    if fee and int(fee.group(1)) > 0:
        sev += 2 if int(fee.group(1)) >= 25 else 1
        found.append(f"{fee.group(1)}% early-termination charge")
    why = "Exit is costly or constrained." if sev >= 3 else "Buyer retains a workable exit."
    return ClauseFlag("termination", _cap(sev), "; ".join(found), why, _excerpt(text, conv))


_WATCH_PATTERNS = [
    ("exclusivity", r"purchase\s+not\s+less\s+than\s+(\d+)%|minimum\s+purchase", "Minimum-purchase / exclusivity commitment"),
    ("unilateral_price_change", r"may\s+revise\s+(?:unit\s+)?prices", "Supplier can change prices unilaterally"),
    ("assignment", r"assign(?:ment)?\s+without\s+consent", "Assignment without consent"),
]


def scan_contract(text: str) -> ScanResult:
    flags = [_scan_renewal(text), _scan_liability(text), _scan_payment(text), _scan_termination(text)]
    sevs = [f.severity for f in flags]
    baseline = math.floor(0.5 * max(sevs) + 0.5 * (sum(sevs) / len(sevs)) + 0.5)
    watch = []
    for name, pattern, desc in _WATCH_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            watch.append({"item": name, "description": desc, "excerpt": _excerpt(text, m)})
    return ScanResult(flags, watch, max(1, min(5, baseline)))
