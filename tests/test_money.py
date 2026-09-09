import pytest

from tools.money import format_inr, parse_inr
from tools.state_tools import parse_inr_amount


@pytest.mark.parametrize("text,expected", [
    ("₹40L", 4_000_000), ("budget 40 lakh", 4_000_000), ("Rs 1.5 Cr", 15_000_000),
    ("40,00,000", 4_000_000), ("INR 2.5k", 2_500), ("2 crore", 20_000_000),
    ("We need 50 laptops, budget ₹40L, delivery in 3 weeks", 4_000_000),
])
def test_parse_inr(text, expected):
    assert parse_inr(text) == expected


def test_parse_inr_no_amount():
    assert parse_inr("no numbers here") is None
    assert parse_inr_amount("nothing")["status"] == "error"


def test_parse_inr_amount_tool():
    out = parse_inr_amount("₹40L")
    assert out["status"] == "ok" and out["amount_inr"] == 4_000_000


@pytest.mark.parametrize("amount,expected", [
    (74_500, "₹74,500"), (200_000, "₹2,00,000 (₹2.00 L)"),
    (4_000_000, "₹40,00,000 (₹40.00 L)"), (123_456_789, "₹12,34,56,789 (₹12.35 Cr)"),
])
def test_format_inr(amount, expected):
    assert format_inr(amount) == expected
