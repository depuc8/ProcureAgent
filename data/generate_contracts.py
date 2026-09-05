"""Generates the 10 sample supply contracts in data/contracts/ (one per anchor vendor).

The generated .txt files are committed; re-run this script only to change the
clause mix. Each contract is plain English with numbered sections, mixing safe
and risky terms for auto-renewal, liability caps, payment terms and termination.
`risk_note` documents the intended (hand-assigned) risk so tests can pin the
scanner's rubric against design intent rather than against its own output.
"""

from __future__ import annotations

from pathlib import Path

from data.synthetic import ANCHORS

OUT_DIR = Path(__file__).parent / "contracts"

# vendor_id -> clause parameters + intended risk score (1 safe .. 5 very risky)
SPECS: dict[str, dict] = {
    "V0001": dict(payment=("net", 30, 1), renewal=None, liability=("pct", 200), term=("t4c", 30, 0, 0),
                  intended=1, risk_note="Balanced, buyer-friendly terms."),
    "V0002": dict(payment=("net", 45, 1), renewal=(12, 60), liability=("pct", 100), term=("t4c", 30, 0, 0),
                  intended=2, risk_note="Short auto-renewal with workable notice."),
    "V0003": dict(payment=("net", 15, 3), renewal=(36, 180), liability=("pct", 10), term=("cause", 0, 0, 0),
                  intended=5, risk_note="36-month auto-renewal, 10% liability cap, no exit for convenience."),
    "V0004": dict(payment=("net", 30, 1), renewal=None, liability=("months", 3), term=("t4c", 30, 0, 0),
                  intended=3, risk_note="Liability capped at 3 months of fees."),
    "V0005": dict(payment=("advance", 100, 0), renewal=None, liability=("pct", 150), term=("t4c", 30, 0, 0),
                  intended=4, risk_note="100% advance payment exposes the buyer to non-delivery."),
    "V0006": dict(payment=("net", 60, 1), renewal=None, liability=("pct", 150), term=("t4c", 60, 0, 0),
                  intended=1, risk_note="Best-in-set terms."),
    "V0007": dict(payment=("net", 30, 1), renewal=(36, 120), liability=("pct", 100), term=("t4c", 30, 0, 40),
                  extra="exclusivity", intended=4,
                  risk_note="36-month auto-renewal, 40% early-termination charge, 80% exclusivity."),
    "V0008": dict(payment=("net", 30, 1), renewal=None, liability=("pct", 40), term=("t4c", 45, 24, 0),
                  extra="price_revision", intended=3,
                  risk_note="40% liability cap, 24-month lock-in, unilateral price revision."),
    "V0009": dict(payment=("net", 30, 1), renewal=(12, 90), liability=("pct", 100), term=("t4c", 30, 0, 0),
                  intended=2, risk_note="12-month auto-renewal with 90-day notice."),
    "V0010": dict(payment=("net", 30, 1), renewal=(12, 90), liability=("pct", 50), term=("t4c", 90, 0, 0),
                  intended=3, risk_note="50% liability cap, 90-day exit notice, auto-renewal."),
}

EXTRAS = {
    "exclusivity": (
        "7.2 Exclusivity. During the Term the Buyer shall purchase not less than 80% of its laptop "
        "requirement from the Supplier."
    ),
    "price_revision": (
        "7.2 Price Revision. The Supplier may revise unit prices on 30 days' written notice, and revised "
        "prices apply to all purchase orders issued after the notice date."
    ),
}


def _payment_clause(kind: str, value: int, late_pct: int) -> str:
    if kind == "net":
        text = (f"The Buyer shall pay each undisputed invoice within {value} days of receipt of the "
                f"invoice (Net {value}).")
    elif value >= 100:
        text = ("The Buyer shall pay 100% of the order value in advance upon placing a purchase order. "
                "No credit period applies.")
    else:
        text = (f"The Buyer shall pay {value}% of the order value in advance upon placing a purchase "
                f"order, with the balance payable within 30 days of delivery (Net 30).")
    if late_pct:
        text += f" Overdue amounts attract interest at {late_pct}% per month."
    return text


def _renewal_clause(renewal: tuple[int, int] | None) -> str:
    if renewal is None:
        return ("This Agreement expires at the end of the Initial Term and does not renew automatically. "
                "Any extension must be agreed in writing by both parties.")
    months, notice = renewal
    return (f"This Agreement shall automatically renew for successive renewal terms of {months} months each "
            f"unless either party gives written notice of non-renewal at least {notice} days before the end "
            f"of the then-current term.")


def _liability_clause(kind: str, value: int) -> str:
    if kind == "pct":
        return (f"The Supplier's aggregate liability under this Agreement is limited to {value}% of the fees "
                f"paid or payable in the twelve (12) months preceding the claim.")
    if kind == "months":
        return (f"The Supplier's aggregate liability under this Agreement is limited to the fees paid in the "
                f"{value} months preceding the claim.")
    return "The Supplier's liability for direct damages under this Agreement is not capped."


def _termination_clause(kind: str, notice: int, lock_in: int, fee_pct: int) -> str:
    if kind == "cause":
        return ("Neither party may terminate this Agreement for convenience. Either party may terminate only "
                "for material breach that remains uncured 30 days after written notice.")
    text = f"The Buyer may terminate this Agreement for convenience by giving {notice} days' written notice"
    if lock_in:
        text += f", but not before the end of the first {lock_in} months (the minimum commitment period)"
    text += "."
    if fee_pct:
        text += (f" On termination for convenience the Buyer shall pay an early termination charge equal to "
                 f"{fee_pct}% of the fees projected for the remainder of the then-current term.")
    return text


def render(vendor_id: str, name: str, price: float, spec: dict) -> str:
    pay_kind, pay_val, late = spec["payment"]
    t_kind, t_notice, t_lock, t_fee = spec["term"]
    annual_value = int(price * 600)  # nominal 600 units a year, for the contract header only
    extra = EXTRAS.get(spec.get("extra", ""), "7.2 Notices. Notices must be in writing and sent to the addresses on page 1.")
    return f"""MASTER SUPPLY AGREEMENT
Contract Ref: MSA-{vendor_id}
Supplier: {name} ("Supplier")
Buyer: Buyer Enterprise Ltd ("Buyer")
Category: Laptops
Estimated Annual Contract Value: INR {annual_value:,}

1. SCOPE
1.1 The Supplier shall supply laptops, accessories and related on-site support to the Buyer against
purchase orders issued under this Agreement. Specifications, quantities and delivery dates are set out in
each purchase order.
1.2 The Supplier warrants that all goods are new, genuine and free from defects for 12 months from delivery.

2. PRICE AND PAYMENT
2.1 Payment Terms. {_payment_clause(pay_kind, pay_val, late)}
2.2 All prices are in Indian Rupees and exclusive of GST.

3. TERM AND RENEWAL
3.1 Initial Term. This Agreement commences on the Effective Date and continues for 12 months (the "Initial Term").
3.2 Renewal. {_renewal_clause(spec["renewal"])}

4. LIABILITY
4.1 Liability Cap. {_liability_clause(*spec["liability"])}
4.2 Neither party is liable for indirect or consequential loss, except for fraud or wilful misconduct.

5. TERMINATION
5.1 Termination for Convenience. {_termination_clause(t_kind, t_notice, t_lock, t_fee)}
5.2 Either party may terminate immediately on written notice if the other party becomes insolvent.

6. CONFIDENTIALITY AND DATA
6.1 Each party shall keep the other's confidential information confidential for three years after the Term.
6.2 The Supplier shall securely wipe or destroy any storage media returned under warranty.

7. GENERAL
7.1 This Agreement is governed by the laws of India and the courts of Bengaluru have exclusive jurisdiction.
{extra}
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vid, name, price, *_ in ANCHORS:
        (OUT_DIR / f"{vid}.txt").write_text(render(vid, name, price, SPECS[vid]), encoding="utf-8")
    print(f"wrote {len(ANCHORS)} contracts to {OUT_DIR}")


if __name__ == "__main__":
    main()
