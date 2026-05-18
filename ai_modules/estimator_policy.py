"""
Config-driven estimator policy: feasibility caps, floor cost multipliers.

JSON lives under `config/` at project root (same cwd as FastAPI).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _config_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "config"


def _read_json(name: str) -> Any:
    p = _config_dir() / name
    if not p.exists():
        p = Path(os.getcwd()) / "config" / name
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class FeasibilityBand:
    max_plot_sqft: int
    max_bed: int
    max_bath: int
    max_kitchen: int


def load_feasibility_bands() -> List[FeasibilityBand]:
    raw = _read_json("estimator_feasibility_bands.json")
    bands = raw.get("bands") or []
    out: List[FeasibilityBand] = []
    for b in bands:
        out.append(
            FeasibilityBand(
                max_plot_sqft=int(b.get("max_plot_sqft", 0) or 0),
                max_bed=int(b.get("max_bed", 0) or 0),
                max_bath=int(b.get("max_bath", 0) or 0),
                max_kitchen=int(b.get("max_kitchen", 0) or 0),
            )
        )
    return sorted(out, key=lambda x: x.max_plot_sqft)


def feasibility_caps(plot_sqft: int, bands: Optional[List[FeasibilityBand]] = None) -> Tuple[int, int, int]:
    """Return (max_bed, max_bath, max_kitchen) for plot footprint sqft."""
    bs = bands or load_feasibility_bands()
    if not bs:
        return 20, 20, 5
    ps = max(int(plot_sqft or 0), 0)
    for b in bs:
        if ps <= b.max_plot_sqft:
            return b.max_bed, b.max_bath, b.max_kitchen
    last = bs[-1]
    return last.max_bed, last.max_bath, last.max_kitchen


def clamp_layout_inputs(
    plot_sqft: int,
    bedrooms: int,
    washrooms: int,
    kitchens: int,
    bands: Optional[List[FeasibilityBand]] = None,
) -> Tuple[int, int, int, bool, str]:
    """
    Clamp bedrooms / washrooms / kitchens to feasible max for plot size.
    Returns (beds, baths, kits, clamped, message).
    """
    mx_b, mx_bt, mx_k = feasibility_caps(plot_sqft, bands)
    b = max(0, min(int(bedrooms or 0), mx_b))
    w = max(0, min(int(washrooms or 0), mx_bt))
    k = max(0, min(int(kitchens or 0), mx_k))
    raw_b, raw_w, raw_k = int(bedrooms or 0), int(washrooms or 0), int(kitchens or 0)
    clamped = (raw_b != b) or (raw_w != w) or (raw_k != k)
    msg = ""
    if clamped:
        msg = (
            f"Adjusted bedrooms/bathrooms/kitchens to the maximum typical for ~{int(plot_sqft)} sqft plot "
            f"({b} bed, {w} bath, {k} kitchen)."
        )
    return b, w, k, clamped, msg


def load_floor_policy() -> Dict[str, Any]:
    return _read_json("estimator_floor_multipliers.json") or {
        "floor_index_multipliers": [1.0],
        "upper_floor_default_multiplier": 1.0,
        "foundation_phases": ["Site Preparation", "Excavation & Foundation"],
        "structural_per_floor_phases": ["Grey Structure", "Masonry & Walls"],
    }


def floor_multiplier_series(floors: int, policy: Optional[Dict[str, Any]] = None) -> List[float]:
    """Return one multiplier per floor index 0..floors-1."""
    p = policy or load_floor_policy()
    m = [float(x) for x in (p.get("floor_index_multipliers") or [1.0])]
    default_upper = float(p.get("upper_floor_default_multiplier") or 0.92)
    n = max(int(floors or 1), 1)
    out: List[float] = []
    for i in range(n):
        if i < len(m):
            out.append(m[i])
        else:
            out.append(default_upper)
    return out


def floor_cost_factors(floors: int, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mults = floor_multiplier_series(floors, policy)
    structural_blend = float(sum(mults)) / max(len(mults), 1)
    foundation_mult = float(mults[0]) if mults else 1.0
    return {
        "floors": int(floors),
        "per_floor_multipliers": mults,
        "foundation_mult": foundation_mult,
        "structural_blend_mult": structural_blend,
    }


def public_feasibility_bands_raw() -> List[dict]:
    """Bands as stored in JSON (for API/UI without importing dataclasses)."""
    raw = _read_json("estimator_feasibility_bands.json")
    return list(raw.get("bands") or [])


# ── BHK → default residential layout (Pakistan typical marketing / planning) ──
# Used when the client sends `bhk` instead of separate bedroom/bathroom counts.
# washrooms = wet stacks (attached/shared baths); kitchens = working kitchens.
BHK_LAYOUT_DEFAULTS: Dict[int, Dict[str, Any]] = {
    1: {"bedrooms": 1, "washrooms": 1, "kitchens": 1, "note": "1 bed, 1 wet stack, 1 kitchen"},
    2: {"bedrooms": 2, "washrooms": 2, "kitchens": 1, "note": "2 bed, 2 baths, 1 kitchen"},
    3: {"bedrooms": 3, "washrooms": 3, "kitchens": 1, "note": "3 bed, 3 baths (suite-style), 1 kitchen"},
    4: {"bedrooms": 4, "washrooms": 3, "kitchens": 1, "note": "4 bed, 3 baths (shared), 1 kitchen"},
    5: {"bedrooms": 5, "washrooms": 4, "kitchens": 1, "note": "5 bed, 4 baths, 1 kitchen"},
    6: {"bedrooms": 6, "washrooms": 4, "kitchens": 2, "note": "6 bed, 4 baths, 2 kitchens (main + dirty)"},
}


def resolve_layout_from_bhk(bhk: int) -> Tuple[int, int, int, str]:
    """
    Map BHK (1..6) to bedrooms / washrooms / kitchens and a short assumption string.
    Values outside range clamp to nearest supported BHK.
    """
    k = int(bhk or 3)
    k = max(1, min(6, k))
    row = BHK_LAYOUT_DEFAULTS[k]
    note = str(row.get("note", ""))
    return int(row["bedrooms"]), int(row["washrooms"]), int(row["kitchens"]), f"{k} BHK — {note}"
