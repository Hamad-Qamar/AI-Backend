"""
Canonical finishing tier definitions — loaded only from `config/finishingTiers.json`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _config_path() -> Path:
    p = Path(__file__).resolve().parent.parent / "config" / "finishingTiers.json"
    if p.exists():
        return p
    return Path(os.getcwd()) / "config" / "finishingTiers.json"


def finishing_catalog_meta() -> Dict[str, Any]:
    """`_meta` block from `finishingTiers.json` (pricing model knobs for the cost engine)."""
    raw = load_finishing_catalog()
    return dict(raw.get("_meta") or {})


@lru_cache(maxsize=1)
def load_finishing_catalog() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tier_keys() -> List[str]:
    raw = load_finishing_catalog()
    return list(raw.get("tier_order") or ("economy", "standard", "premium", "luxury"))


def normalize_finishing_tier(grade: str) -> str:
    g = (grade or "standard").strip().lower()
    if g not in tier_keys():
        return "standard"
    return g


def tier_index(tier: str) -> int:
    keys = tier_keys()
    t = normalize_finishing_tier(tier)
    return keys.index(t) if t in keys else keys.index("standard")


def finishing_tier_config(tier: str) -> Dict[str, Any]:
    raw = load_finishing_catalog()
    tiers = raw.get("tiers") or {}
    t = normalize_finishing_tier(tier)
    return dict(tiers.get(t) or tiers.get("standard") or {})


def package_phases() -> List[str]:
    raw = load_finishing_catalog()
    return list(raw.get("package_phases") or [])


def calculate_finishing_package_cost(total_sqft: float, tier: str) -> Tuple[float, Dict[str, Any]]:
    """Return (pkr_amount, tier_block) for a finishing allowance covering package_phases."""
    cfg = finishing_tier_config(tier)
    rate = float(cfg.get("price_per_sqft") or 0)
    amt = max(0.0, float(total_sqft) * rate)
    return amt, cfg


def explain_tier(tier: str) -> str:
    cfg = finishing_tier_config(tier)
    return str(cfg.get("summary") or "")


def feature_display_rows(tier: str) -> List[Dict[str, Any]]:
    """Rows for UI: key, label, value (formatted)."""
    raw = load_finishing_catalog()
    labels = raw.get("feature_labels") or {}
    cfg = finishing_tier_config(tier)
    feats = cfg.get("features") or {}
    rows: List[Dict[str, Any]] = []
    for k, label in labels.items():
        v = feats.get(k)
        if v is None:
            disp = "—"
        elif isinstance(v, bool):
            disp = "Yes" if v else "No"
        elif isinstance(v, (int, float)):
            disp = str(int(v)) if k.endswith("_l") or "tank" in k else str(v)
        else:
            disp = str(v)
        rows.append({"key": k, "label": label, "value": disp})
    return rows


def upsell_hints(plot_sqft: int, tier: str) -> List[Dict[str, Any]]:
    raw = load_finishing_catalog()
    out: List[Dict[str, Any]] = []
    for rule in raw.get("upsells") or []:
        if rule.get("when") == "plot_sqft_gte" and int(plot_sqft) >= int(rule.get("value") or 0):
            allowed = [normalize_finishing_tier(x) for x in (rule.get("if_tier_in") or [])]
            if normalize_finishing_tier(tier) in allowed:
                out.append(rule)
    return out


def item_eligible_for_finishing_tier(item_min_tier: str, user_tier: str) -> bool:
    """Product is shown if user's finishing tier is at least the product minimum."""
    return tier_index(user_tier) >= tier_index(item_min_tier or "economy")
