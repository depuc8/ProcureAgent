"""Deterministic synthetic vendor data (~200 vendors).

V0001-V0010 are hand-defined "anchor" laptop vendors, each with a sample
contract in data/contracts/. The remaining vendors are generated from a seeded
RNG so the local backend, the BigQuery seed script and the tests all see the
same universe.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SEED = 20260401


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    category: str
    unit_price_inr: float
    rating: float
    delivery_days: int
    certifications: tuple[str, ...]
    min_order_qty: int
    is_preferred: bool
    contract_on_file: bool
    region: str

    def to_row(self) -> dict:
        row = asdict(self)
        row["certifications"] = list(self.certifications)
        return row


# (category, count, price_low, price_high, delivery_low, delivery_high, cert pool)
CATEGORIES: list[tuple[str, int, float, float, int, int, list[str]]] = [
    ("Laptops", 24, 45_000, 220_000, 7, 35, ["ISO 9001", "ISO 27001", "BIS", "Energy Star", "RoHS"]),
    ("Desktops & Workstations", 18, 40_000, 250_000, 7, 35, ["ISO 9001", "BIS", "Energy Star", "RoHS"]),
    ("Networking Equipment", 22, 8_000, 800_000, 10, 45, ["ISO 9001", "ISO 27001", "BIS", "CE"]),
    ("Office Furniture", 24, 4_000, 45_000, 10, 50, ["ISO 9001", "ISO 14001", "BIFMA", "FSC"]),
    ("Software Licenses", 22, 2_000, 60_000, 1, 10, ["ISO 27001", "SOC 2", "ISO 9001"]),
    ("Printers & Imaging", 16, 12_000, 300_000, 7, 30, ["ISO 9001", "Energy Star", "BIS", "RoHS"]),
    ("Mobile Devices", 16, 10_000, 150_000, 5, 25, ["BIS", "ISO 9001", "RoHS", "CE"]),
    ("Stationery & Supplies", 22, 100, 5_000, 2, 15, ["ISO 9001", "ISO 14001", "FSC"]),
    ("Security Services", 16, 15_000, 120_000, 5, 30, ["ISO 9001", "ISO 27001", "PSARA"]),
    ("Cloud & Hosting Services", 20, 3_000, 90_000, 1, 10, ["ISO 27001", "SOC 2", "ISO 9001", "MeitY Empanelled"]),
]

REGIONS = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]

_PREFIX = [
    "Apex", "Brightwave", "Cobalt", "Dynamo", "Elevate", "Fulcrum", "Granite", "Horizon", "Ironwood",
    "Juniper", "Keystone", "Lumen", "Monarch", "Northstar", "Onyx", "Pinnacle", "Quanta", "Redwood",
    "Summit", "Tessera", "Umbra", "Vanguard", "Willow", "Xenon", "Yellowstone", "Zephyr", "Aurora",
    "Beacon", "Catalyst", "Delta", "Ember", "Falcon", "Harbor", "Indus", "Kaveri", "Lotus", "Mandala",
    "Narmada", "Orbit", "Prism", "Rivulet", "Sahara", "Trident", "Utkarsh", "Vertex", "Zenara",
]
_SUFFIX = {
    "Laptops": ["Computing", "Devices", "Infotech", "Digital", "Systems"],
    "Desktops & Workstations": ["Workstations", "Computing", "Systems", "Infotech"],
    "Networking Equipment": ["Networks", "Telecom", "Connect", "Infrastructure"],
    "Office Furniture": ["Furnishings", "Interiors", "Workspaces", "Furniture Co."],
    "Software Licenses": ["Software", "Cloudware", "Licensing", "Technologies"],
    "Printers & Imaging": ["Imaging", "Print Solutions", "Document Systems", "Office Tech"],
    "Mobile Devices": ["Mobility", "Wireless", "Communications", "Handsets"],
    "Stationery & Supplies": ["Supplies", "Office Mart", "Paper & Co.", "Traders"],
    "Security Services": ["Security", "Guarding", "Protection", "Risk Services"],
    "Cloud & Hosting Services": ["Cloud", "Hosting", "Datacentres", "Managed Services"],
}
_LEGAL = ["Pvt Ltd", "Ltd", "LLP", "India Pvt Ltd"]

# vendor_id, name, unit_price_inr, rating, delivery_days, certs, min_qty, preferred
ANCHORS: list[tuple] = [
    ("V0001", "Veritas Computing Pvt Ltd", 74_500, 4.6, 14, ("ISO 9001", "ISO 27001", "BIS", "Energy Star"), 5, True),
    ("V0002", "Bluepeak Systems India Pvt Ltd", 68_900, 4.4, 18, ("ISO 9001", "BIS"), 10, True),
    ("V0003", "Nimbus Devices Ltd", 59_900, 4.1, 12, ("ISO 9001", "BIS"), 1, False),
    ("V0004", "Kestrel Technologies Pvt Ltd", 71_200, 4.3, 21, ("ISO 9001", "ISO 27001", "BIS"), 5, False),
    ("V0005", "Orion Digital Supply LLP", 63_500, 4.0, 10, ("BIS",), 1, False),
    ("V0006", "Zenith Infotech Ltd", 82_000, 4.7, 16, ("ISO 9001", "ISO 27001", "BIS", "Energy Star", "RoHS"), 10, True),
    ("V0007", "Atlas Hardware Solutions Pvt Ltd", 66_400, 4.2, 25, ("ISO 9001", "BIS"), 5, False),
    ("V0008", "Meridian Compute Pvt Ltd", 77_800, 4.5, 20, ("ISO 9001", "ISO 27001", "BIS"), 5, False),
    ("V0009", "Sapphire Tech Distributors LLP", 69_900, 4.3, 15, ("ISO 9001", "BIS", "Energy Star"), 1, True),
    ("V0010", "Crimson Networks & Devices Pvt Ltd", 72_600, 4.4, 30, ("ISO 9001", "BIS"), 5, False),
]


def _round_price(p: float) -> float:
    if p >= 10_000:
        return float(round(p / 100) * 100 - 100)  # e.g. 74,900 style price points
    if p >= 1_000:
        return float(round(p / 10) * 10 - 1)
    return float(round(p, 0))


def generate_vendors(seed: int = SEED) -> list[Vendor]:
    rng = random.Random(seed)
    vendors: list[Vendor] = [
        Vendor(vid, name, "Laptops", float(price), rating, days, certs, minq, pref, True,
               REGIONS[i % len(REGIONS)])
        for i, (vid, name, price, rating, days, certs, minq, pref) in enumerate(ANCHORS)
    ]
    used_names = {v.name for v in vendors}
    next_id = len(vendors) + 1

    pending: list[Vendor] = []
    for category, count, p_lo, p_hi, d_lo, d_hi, cert_pool in CATEGORIES:
        generated = count - (len(ANCHORS) if category == "Laptops" else 0)
        for _ in range(generated):
            while True:
                name = f"{rng.choice(_PREFIX)} {rng.choice(_SUFFIX[category])} {rng.choice(_LEGAL)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            # log-uniform price so cheap vendors are as common as premium ones
            price = _round_price(p_lo * (p_hi / p_lo) ** rng.random())
            rating = round(min(5.0, max(2.6, rng.gauss(4.0, 0.5))), 1)
            n_certs = rng.randint(0, min(3, len(cert_pool)))
            pending.append(Vendor(
                vendor_id="",
                name=name,
                category=category,
                unit_price_inr=price,
                rating=rating,
                delivery_days=rng.randint(d_lo, d_hi),
                certifications=tuple(sorted(rng.sample(cert_pool, n_certs))),
                min_order_qty=rng.choice([1, 1, 1, 5, 10, 25]),
                is_preferred=rng.random() < 0.18,
                contract_on_file=False,
                region=rng.choice(REGIONS),
            ))
    rng.shuffle(pending)
    for v in pending:
        vendors.append(Vendor(**{**asdict(v), "vendor_id": f"V{next_id:04d}",
                                 "certifications": v.certifications}))
        next_id += 1
    return vendors


def vendor_rows(seed: int = SEED) -> list[dict]:
    return [v.to_row() for v in generate_vendors(seed)]
