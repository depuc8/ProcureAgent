"""Deterministic vendor ranking + human-readable reasons.

The LLM explains the shortlist; it does not compute it. Weights are
explicit so the ordering is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.synthetic import Vendor

W_PRICE, W_RATING, W_DELIVERY = 0.40, 0.30, 0.20
BONUS_PREFERRED, BONUS_CONTRACT = 0.05, 0.05


@dataclass(frozen=True)
class Ranked:
    vendor: Vendor
    score: float
    reasons: list[str]


def rank_vendors(candidates: list[Vendor], max_delivery_days: int | None, top_n: int = 3) -> list[Ranked]:
    if not candidates:
        return []
    prices = [v.unit_price_inr for v in candidates]
    p_min, p_max = min(prices), max(prices)
    cheapest = {v.vendor_id for v in candidates if v.unit_price_inr == p_min}
    top_rating = max(v.rating for v in candidates)

    ranked: list[Ranked] = []
    for v in candidates:
        price_s = 1.0 if p_max == p_min else (p_max - v.unit_price_inr) / (p_max - p_min)
        rating_s = min(1.0, max(0.0, (v.rating - 3.0) / 2.0))
        horizon = max_delivery_days or max(c.delivery_days for c in candidates) or 1
        delivery_s = min(1.0, max(0.0, 1 - v.delivery_days / horizon))
        score = (W_PRICE * price_s + W_RATING * rating_s + W_DELIVERY * delivery_s
                 + (BONUS_PREFERRED if v.is_preferred else 0)
                 + (BONUS_CONTRACT if v.contract_on_file else 0))

        reasons: list[str] = []
        if v.vendor_id in cheapest and len(candidates) > 1:
            reasons.append(f"Lowest unit price among {len(candidates)} qualifying vendors")
        elif price_s >= 0.66:
            reasons.append(f"Price in the lowest third of {len(candidates)} qualifying vendors")
        if v.rating == top_rating:
            reasons.append(f"Highest rating among qualifying vendors ({v.rating}/5)")
        elif v.rating >= 4.2:
            reasons.append(f"Strong rating ({v.rating}/5)")
        if max_delivery_days:
            reasons.append(f"Delivers in {v.delivery_days} days (requirement: {max_delivery_days})")
        else:
            reasons.append(f"Delivers in {v.delivery_days} days")
        if v.is_preferred:
            reasons.append("Preferred vendor")
        if v.contract_on_file:
            reasons.append("Contract on file for review")
        if v.certifications:
            reasons.append("Certifications: " + ", ".join(v.certifications))
        ranked.append(Ranked(v, round(score, 4), reasons))

    ranked.sort(key=lambda r: (-r.score, r.vendor.unit_price_inr, r.vendor.vendor_id))
    return ranked[:top_n]
