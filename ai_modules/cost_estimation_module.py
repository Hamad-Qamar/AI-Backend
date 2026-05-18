"""
CostEstimationModule — Module 4 (Extended)
Estimates construction costs from structured OR text-based inputs.
Loads pricing from prices.json — never hardcoded.
"""

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .estimator_policy import (
    clamp_layout_inputs,
    feasibility_caps,
    floor_cost_factors,
    load_feasibility_bands,
    load_floor_policy,
    public_feasibility_bands_raw,
    resolve_layout_from_bhk,
)
from . import finishing_catalog
from .llm_helper import LLMHelper
from .phase2_repository import Phase2Paths, get_phase2_repository

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — defined once, never scattered inline
# ─────────────────────────────────────────────────────────────────────────────

# Pakistan standard area conversions → sqft (fallbacks only).
# City-aware marla conversions come from `city_area_standards.csv`.
AREA_CONVERSIONS: Dict[str, float] = {
    # Standardize marla to an integer to avoid subtle float mismatches across
    # backend/frontend, CSVs, and tests.
    "marla":  272.0,
    "kanal":  5445.0,
    "sqft":   1.0,
    "sft":    1.0,
    "sqm":    10.764,
    "sqyard": 9.0,
    "gaj":    9.0,     
}

# Minimum viable covered area (sqft) to prevent absurd estimates (e.g., 1 sqft).
MIN_VIABLE_SQFT = 120
# Minimum plot / covered input accepted by the estimator (product policy).
MIN_INPUT_MARLA_EQUIV = 2.0


def _safe_catalog_text(val: Any) -> str:
    """Normalize Supabase/CSV text cells (NaN/None → empty) for BOQ metadata."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    sl = s.lower()
    if sl in ("nan", "none", "<na>", "null", "nil", "-", "--", "n/a"):
        return ""
    return " ".join(s.split())


def _catalog_id_str(val: Any) -> str:
    """Stable string for ``material_id`` / keys (preserve IDs; skip NaN/None)."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ""
    if isinstance(val, float) and float(val).is_integer():
        s = str(int(val))
    else:
        s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    return s


def normalize_city_key(city: str) -> str:
    """Strip, lowercase, collapse internal whitespace — join pricing_data with city_area_standards + API city inputs."""
    return " ".join(str(city or "").strip().lower().split())


# Construction types (strict separation)
CONSTRUCTION_TYPES = ("grey_structure", "full_construction", "renovation")


def ui_bucket_for_phase_or_labour_label(label: str) -> str:
    """
    Map a materials_master `phase` or a synthetic labour line label to a high-level UI bucket.
    Keeps labour tops-ups aligned with the same buckets as BOQ rows (avoids Grey labour in Finishing).
    """
    p = (label or "").strip().lower()
    if "grey package labour" in p or ("grey" in p and "package" in p and "labour" in p):
        return "Grey Structure"
    if "electrical" in p and "labour realism" in p:
        return "Electrical"
    if "plumbing" in p and "sanitary" in p and "labour realism" in p:
        return "Plumbing"
    if "finishing trades" in p:
        return "Finishing"
    if any(
        k in p
        for k in (
            "excavation",
            "foundation",
            "site preparation",
            "grey structure",
            "masonry",
        )
    ):
        return "Grey Structure"
    if "plumbing" in p or "sanitary" in p:
        return "Plumbing"
    if "electrical" in p:
        return "Electrical"
    if any(
        k in p
        for k in (
            "floor",
            "tiling",
            "paint",
            "finishing",
            "carpentry",
            "kitchen",
            "wardrobe",
            "aluminum",
            "glass",
            "plaster",
        )
    ):
        return "Finishing"
    return "Misc"


def allowed_ui_buckets_for_construction(construction_type: str) -> Optional[frozenset]:
    """
    Which UI buckets may appear in totals / charts for this job type.
    None means all buckets (full construction). Excluded buckets are stripped from the response.
    """
    ct = (construction_type or "full_construction").lower().replace(" ", "_")
    if ct == "grey_structure":
        return frozenset({"Grey Structure", "Plumbing", "Electrical", "Misc"})
    if ct == "renovation":
        return frozenset({"Finishing", "Plumbing", "Electrical", "Misc"})
    return None  # full — no bucket excluded


FULL_SCOPE_UI_BUCKETS = frozenset({"Grey Structure", "Plumbing", "Electrical", "Finishing", "Misc"})

# Short definitions for BOQ UI tooltips (frontend may also mirror this list).
BOQ_TOOLTIP_GLOSSARY: Dict[str, str] = {
    "PCC": "Plain cement concrete — a lean mix used for blinding, levelling, or mass fill before structural RCC.",
    "RCC": "Reinforced cement concrete — structural concrete cast with steel reinforcement (beams, slabs, columns).",
    "rebar": "Steel reinforcing bars (sarya/deformed bars) embedded in concrete to resist tension and shear.",
    "sarya": "Local term for steel reinforcement bars (rebar) used in RCC members.",
    "bajri": "Crushed stone aggregate mixed in concrete or used as hardcore under floors.",
    "cft": "Cubic feet — common PK unit for bulk sand, crush, and concrete volume on site.",
    "marla": "Traditional plot area unit; city-specific conversion to sqft is applied from rate-card standards.",
}

# Phase groupings for BOQ generation from `materials_master.csv`
GREY_STRUCTURE_PHASES = {
    "Site Preparation",
    "Excavation & Foundation",
    "Grey Structure",
    "Masonry & Walls",
}
FULL_CONSTRUCTION_PHASES = None  # means: include all phases
RENOVATION_PHASES = {
    "Plastering & Screeding",
    "Flooring & Tiling",
    "Electrical (Rough + Final)",
    "Plumbing (Rough + Final)",
    "Paint & Finishing",
    "Carpentry & Woodwork",
    "Kitchen & Wardrobes",
    "Sanitary & Bathroom Fittings",
    "Aluminum & Glass Work",
    "Roofing & Waterproofing",
    "External Works",
}

# Renovation must be scope-based (not one linear multiplier).
# Each phase has its own typical coverage percentage for renovation work.
RENOVATION_PHASE_COVERAGE: Dict[str, float] = {
    "Paint & Finishing": 1.00,                 # often full repaint
    "Flooring & Tiling": 0.60,                 # usually partial replacement
    "Plastering & Screeding": 0.35,
    "Electrical (Rough + Final)": 0.30,
    "Plumbing (Rough + Final)": 0.30,
    "Carpentry & Woodwork": 0.35,
    "Kitchen & Wardrobes": 0.50,
    "Sanitary & Bathroom Fittings": 0.50,
    "Aluminum & Glass Work": 0.25,
    "Roofing & Waterproofing": 0.25,
    "External Works": 0.25,
}

# Contingency by tier (materials+labour)
CONTINGENCY_BY_GRADE: Dict[str, float] = {
    "economy": 0.05,
    "standard": 0.07,
    "premium": 0.10,
}

# Quality tier multipliers (applied to categories that vary heavily by spec).
# Grey structure is mostly quantity-driven; finishes/fixtures/labour move most.
QUALITY_TIER_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "economy": {"Finishing": 0.85, "Electrical": 0.90, "Plumbing": 0.90, "Misc": 0.95, "Grey Structure": 1.00, "labour": 0.92},
    "standard": {"Finishing": 1.00, "Electrical": 1.00, "Plumbing": 1.00, "Misc": 1.00, "Grey Structure": 1.00, "labour": 1.00},
    "premium": {"Finishing": 1.25, "Electrical": 1.18, "Plumbing": 1.20, "Misc": 1.10, "Grey Structure": 1.02, "labour": 1.08}
}

# Benchmark enforcement bounds (BOQ total vs benchmark total)
BENCHMARK_LOW = 0.70
BENCHMARK_HIGH = 1.30

# Benchmark quality factors (used only when applying a benchmark floor).
# Turnkey benchmarks vary a lot by spec; grey-structure varies less.
BENCHMARK_QUALITY_FACTOR_TURNKEY: Dict[str, float] = {
    "economy": 0.85,
    "standard": 1.00,
    "premium": 1.25,
}
BENCHMARK_QUALITY_FACTOR_GREY: Dict[str, float] = {
    "economy": 0.92,
    "standard": 1.00,
    "premium": 1.10,
}

# Thumb-rule quantity multipliers vs AREA Covered / built-up area.
# The user-provided reference row is authoritative here:
#   1000 sqft => cement 400 bags, sand 81.6 ton, aggregate 60.8 ton,
#                steel 4000 kg, paint 180 L, bricks 8000 pcs, flooring 1300 sqft.
# The prompt's tabular sand/aggregate values were off by one decimal place
# (0.816 and 0.608 would yield 816/608 tons for 1000 sqft), so we use the
# explicit cross-check row: 81.6 and 60.8 tons per 1000 sqft.
THUMB_CEMENT_BAGS_PER_SQFT = 0.40
THUMB_SAND_TON_PER_SQFT = 0.0816
THUMB_AGGREGATE_TON_PER_SQFT = 0.0608
THUMB_STEEL_KG_PER_SQFT = 4.0
THUMB_PAINT_L_PER_SQFT = 0.18
THUMB_BRICKS_PER_SQFT = 8.0  # legacy reference row only (1000 sqft → 8000 pcs); BOQ brick target uses wall geometry below
THUMB_FLOORING_SQFT_PER_SQFT = 1.30
THUMB_FINISHERS_SHARE_OF_TOTAL = 0.165
THUMB_FITTINGS_SHARE_OF_TOTAL = 0.228
TON_TO_CFT_AT_1600_KG_PER_M3 = (1000.0 / 1600.0) * 35.3147

GREY_BOQ_MIN_STEEL_KG_PER_SQFT = THUMB_STEEL_KG_PER_SQFT
GREY_BOQ_MIN_CEMENT_BAGS_PER_SQFT = THUMB_CEMENT_BAGS_PER_SQFT
GREY_BOQ_MIN_CEMENT_BAGS_ABS = 40.0
GREY_BOQ_MIN_CRUSH_CFT_PER_SQFT = THUMB_AGGREGATE_TON_PER_SQFT * TON_TO_CFT_AT_1600_KG_PER_M3
GREY_BOQ_MIN_SAND_CFT_PER_SQFT = THUMB_SAND_TON_PER_SQFT * TON_TO_CFT_AT_1600_KG_PER_M3
GREY_LABOUR_PKR_PER_SQFT_FLOOR = 580.0

GREY_BOQ_PHASE_SET = frozenset(GREY_STRUCTURE_PHASES)

# Brick envelope (PK common modular brick exposed face 9" × 3" = 27 sq in).
BRICK_FACE_SQIN = 27.0
BRICK_STOREY_WALL_HEIGHT_FT = 10.0


def _structural_steel_kg_mask(df: pd.DataFrame) -> pd.Series:
    """Catalog-agnostic mask for structural reinforcement rows (kg) to merge in BOQ.

    Includes rebar/sarya lines even when ``category`` is RCC/concrete, not ``steel``.
    Excludes mesh and binding wire so small accessory lines are not lumped in.
    """
    unit_lower = df["unit"].astype(str).str.lower()
    name_lower = df["name"].astype(str).str.lower()
    cat_lower = df["category"].astype(str).str.lower()

    kg_ok = unit_lower.eq("kg")
    structural_hint = (
        cat_lower.eq("steel")
        | name_lower.str.contains("rebar", regex=False)
        | name_lower.str.contains("sarya", regex=False)
        | name_lower.str.contains("sariya", regex=False)
        | name_lower.str.contains("deformed", regex=False)
        | name_lower.str.contains("stirrup", regex=False)
        | (
            name_lower.str.contains("steel", regex=False)
            & (
                name_lower.str.contains("rcc", regex=False)
                | name_lower.str.contains("column", regex=False)
                | name_lower.str.contains("beam", regex=False)
                | name_lower.str.contains("slab", regex=False)
                | name_lower.str.contains("foundation", regex=False)
                | name_lower.str.contains("structure", regex=False)
            )
        )
    )
    exclude = (
        name_lower.str.contains("mesh", regex=False)
        | name_lower.str.contains("weldmesh", regex=False)
        | name_lower.str.contains("binding wire", regex=False)
        | name_lower.str.contains("fence", regex=False)
    )
    return kg_ok & structural_hint & ~exclude


def _brick_target_from_wall_geometry(footprint_sqft: float, floors: int) -> float:
    """Rough brick count from square footprint: perimeter 4√A × (storey height × floors) / brick face.

    Uses per-floor covered area for √A (not total built-up across floors). Wall height stacks
    as BRICK_STOREY_WALL_HEIGHT_FT per storey (default 10 ft). Openings are not deducted.
    """
    a = max(float(footprint_sqft), 1.0)
    side_ft = math.sqrt(a)
    perim_ft = 4.0 * side_ft
    fl = max(1, int(floors))
    height_ft = BRICK_STOREY_WALL_HEIGHT_FT * float(fl)
    wall_sqft = perim_ft * height_ft
    wall_sqin = wall_sqft * 144.0
    return wall_sqin / BRICK_FACE_SQIN


def _norm_boq_item_key(name: str) -> str:
    """Normalize display name for stable lookup against materials_master.name."""
    return " ".join(str(name or "").lower().split())


def _coarse_construction_phase(phase_raw: str) -> str:
    """Map detailed Phase-2 phase strings to a short badge for UI (Foundation / Structure / …)."""
    p = str(phase_raw or "").lower()
    # Foundation / earliest works
    if any(
        x in p
        for x in (
            "excavat",
            "excavation",
            "foundation",
            "site preparation",
            "pile",
            "earthwork",
            "grading",
            "demolition",
            "survey",
            "soil",
            "boundary",
            "substructure",
        )
    ):
        return "Foundation"
    if any(
        x in p
        for x in (
            "grey structure",
            "roofing & waterproof",
            "waterproof",
            "roofing",
            "structural",
            "rcc",
            "precast",
            "slab",
            "steelwork",
            "formwork",
            "shuttering",
            "steel & metals",
            "roofing materials",
            "bulk materials",
            "supply chain",
        )
    ):
        return "Structure"
    if any(
        x in p
        for x in (
            "brick",
            "blockwork",
            "block work",
            "aac",
            "mason",
            "masonry",
            "binder",
            "pointing",
            "stone cladding",
            "kota",
            "chips",
            "compound wall",
        )
    ):
        return "Masonry"
    if any(
        x in p
        for x in (
            "plaster",
            "skim",
            "putty",
            "primer",
            "plumbing",
            "sanitary",
            "bathroom",
            "fixture",
            "electrical",
            "lighting",
            "switchgear",
            "dbe",
            "paint",
            "tile",
            "tiling",
            "granite",
            "marble",
            "flooring",
            "parquet",
            "epoxy",
            "kitchen",
            "wardrobe",
            "aluminum",
            "aluminium",
            "glass",
            "carpentry",
            "kitchen & wardrobes",
            "fixtures",
            "watertank",
            "water tank",
            "gypsum",
            "false ceiling",
            "drywall",
            "partition",
            "ceil",
            "acoustic",
            "hvac",
            "climate",
            "mechanical",
            "insulation",
            "cladding",
            "woodwork",
            "stair",
            "railing",
            "steel door",
            "wood door",
        )
    ):
        return "Finishing"
    if "external" in p or "driveway" in p or " pavement" in p:
        return "General"
    if "chemical" in p or "admixture" in p:
        return "Structure"
    return "General"


def _apply_grey_structure_boq_minimums(
    mm_q: pd.DataFrame,
    total_sqft: float,
    construction_type: str,
    footprint_sqft: float,
    floors: int = 1,
) -> Tuple[pd.DataFrame, bool, bool]:
    """Raise thin catalogue-derived grey BOQ toward typical structural completeness.

    Applies to both grey_structure and full_construction. Cement / steel / sand /
    crush follow built-up-area thumb rules vs total_sqft. Brick totals follow
    perimeter × wall-height geometry (square footprint, nominal 9\"×3\" face).

    Steel consolidation: multiple catalogue rows (foundation rebar, column
    rebar, stirrup wire) can each survive group_key dedup because they have
    distinct names/subcategories. To prevent inflation, the minimum check
    works on the total, then scales only the primary (highest-qty) row
    and zeros any secondary steel rows so the UI shows one clean line.

    Returns:
        (updated_df, any_minimum_touched, steel_rebar_catalogue_merged)
    """
    if construction_type not in ("grey_structure", "full_construction") or total_sqft <= 0:
        return mm_q, False, False
    sq = float(total_sqft)
    df = mm_q.copy()
    touched = False
    steel_rebar_catalogue_merged = False

    cat_lower = df["category"].astype(str).str.lower()
    name_lower = df["name"].astype(str).str.lower()
    phase_str = df["phase"].astype(str)
    unit_lower = df["unit"].astype(str).str.lower()

    # Steel: same built-up-area thumb rule for grey_structure and full_construction.
    # Consolidate all structural rebar kg rows (including RCC-labelled catalogue lines)
    # into the primary row (highest quantity_raw) to eliminate duplicate sarya lines.
    steel_mask = _structural_steel_kg_mask(df)
    steel_indices = df.index[steel_mask].tolist()
    cur_steel = float(df.loc[steel_mask, "quantity_raw"].sum())
    tgt_steel = GREY_BOQ_MIN_STEEL_KG_PER_SQFT * sq
    if cur_steel > 0 and len(steel_indices) > 0:
        # Pick the row with the highest existing quantity as the canonical steel line.
        primary_idx = df.loc[steel_mask, "quantity_raw"].idxmax()
        secondary_indices = [i for i in steel_indices if i != primary_idx]
        # Zero out secondary rows — their quantities are already captured in the target.
        if secondary_indices:
            df.loc[secondary_indices, "quantity_raw"] = 0.0
            touched = True
            steel_rebar_catalogue_merged = True
        # Set primary row exactly to the thumb-rule target, not just a lower bound.
        if abs(float(df.loc[primary_idx, "quantity_raw"]) - tgt_steel) > 1e-9:
            df.loc[primary_idx, "quantity_raw"] = tgt_steel
            touched = True

    # Brick minimum — applies to both types (same envelope rule for grey vs full).
    brick_mask = unit_lower.isin(["pcs", "piece", "pieces"]) & name_lower.str.contains(
        "brick", regex=False
    )
    cur_b = float(df.loc[brick_mask, "quantity_raw"].sum())
    tgt_b = _brick_target_from_wall_geometry(float(footprint_sqft), int(floors))
    if cur_b > 0 and abs(tgt_b - cur_b) > 1e-9:
        df.loc[brick_mask, "quantity_raw"] *= tgt_b / cur_b
        touched = True

    # Cement minimum (grey-phase rows only) — applies to both types.
    # Built-up-area rule: cement = AREA Covered × 0.4 bags.
    phase_ok = phase_str.isin(GREY_BOQ_PHASE_SET)
    cement_mask = (
        phase_ok
        & unit_lower.str.contains("bag", regex=False)
        & name_lower.str.contains("cement", regex=False)
        & ~name_lower.str.contains("solvent", regex=False)
    )
    cur_cem = float(df.loc[cement_mask, "quantity_raw"].sum())
    min_cem_sqft = GREY_BOQ_MIN_CEMENT_BAGS_PER_SQFT
    tgt_cem = max(min_cem_sqft * sq, GREY_BOQ_MIN_CEMENT_BAGS_ABS)
    if cur_cem > 0 and abs(tgt_cem - cur_cem) > 1e-9:
        df.loc[cement_mask, "quantity_raw"] *= tgt_cem / cur_cem
        touched = True

    crush_mask = phase_ok & (
        name_lower.str.contains("crush", regex=False)
        | name_lower.str.contains("bajri", regex=False)
    )
    cur_cr = float(df.loc[crush_mask, "quantity_raw"].sum())
    tgt_cr = GREY_BOQ_MIN_CRUSH_CFT_PER_SQFT * sq
    if cur_cr > 0 and abs(tgt_cr - cur_cr) > 1e-9:
        df.loc[crush_mask, "quantity_raw"] *= tgt_cr / cur_cr
        touched = True

    sand_mask = (
        phase_ok
        & name_lower.str.contains("sand", regex=False)
        & ~name_lower.str.contains("solvent", regex=False)
    )
    cur_sa = float(df.loc[sand_mask, "quantity_raw"].sum())
    tgt_sa = GREY_BOQ_MIN_SAND_CFT_PER_SQFT * sq
    if cur_sa > 0 and abs(tgt_sa - cur_sa) > 1e-9:
        df.loc[sand_mask, "quantity_raw"] *= tgt_sa / cur_sa
        touched = True

    return df, touched, steel_rebar_catalogue_merged


# Turnkey MEP + exposed finishing BOQ floors vs covered area (PK practice; scales past “1 marla reference”).
ELECTRICAL_LABOUR_PKR_PER_SQFT_FLOOR = 52.0
PLUMBING_LABOUR_PKR_PER_SQFT_FLOOR = 58.0
FINISHING_TRADES_LABOUR_PKR_PER_SQFT_FLOOR = 265.0


def _apply_turnkey_mep_finishing_boq_minimums(
    mm_q: pd.DataFrame,
    total_sqft: float,
    construction_type: str,
    baths: int,
    rooms: int,
    kitchens: int,
) -> Tuple[pd.DataFrame, bool]:
    """Raise thin MEP/finishing catalogue quantities toward credible whole-home scope."""
    if construction_type not in ("full_construction", "renovation") or total_sqft <= 0:
        return mm_q, False
    sq = float(total_sqft)
    df = mm_q.copy()
    touched = False
    name_lower = df["name"].astype(str).str.lower()
    unit_lower = df["unit"].astype(str).str.lower()
    phase_str = df["phase"].astype(str)

    baths_i = max(int(baths), 0)
    rooms_i = max(int(rooms), 0)
    kits_i = max(int(kitchens), 0)
    baths_eff = max(baths_i, 1) if construction_type == "full_construction" else max(baths_i, 0)
    rooms_e = max(rooms_i, 1)

    # ── Sanitary / shower: whole sets vs wet rooms ──
    # Enforce a two-sided constraint: raise to baths_eff if under-stated, AND
    # cap to baths_eff if over-stated.  Without the upper cap, a catalogue ratio
    # of 0.0045/sqft × 225 sqft = 1.01 sets (after phase multiplier) for a 1-bath
    # house can silently exceed 1 when phase_multiplier > 1.0, producing 2 sets
    # for a house that can only physically fit 1 bathroom.
    if baths_eff >= 1:
        for substr, uneedle in (("sanitary ware", "sets"), ("shower set", "set")):
            mask = name_lower.str.contains(substr, regex=False) & unit_lower.str.contains(
                uneedle, regex=False
            )
            if not mask.any():
                continue
            for idx in df.index[mask]:
                v = float(df.at[idx, "quantity_raw"])
                if v + 1e-9 < float(baths_eff):
                    df.at[idx, "quantity_raw"] = float(baths_eff)
                    touched = True
                elif v > float(baths_eff) + 1e-9:
                    # Cap: e.g. 0.0045×225×phase_mult > 1 should not create 2 sets
                    df.at[idx, "quantity_raw"] = float(baths_eff)
                    touched = True

    # ── Solvent cement: retail whole cans ──
    m_sol = unit_lower.eq("can") & name_lower.str.contains("solvent", regex=False)
    if m_sol.any():
        for idx in df.index[m_sol]:
            v = float(df.at[idx, "quantity_raw"])
            nv = max(1.0, float(math.ceil(v))) if v > 1e-9 else 1.0
            if abs(nv - v) > 1e-6:
                df.at[idx, "quantity_raw"] = nv
                touched = True

    # ── Ball valves: at least two functional stops on small houses ──
    m_bv = phase_str.str.contains("Plumbing", regex=False) & name_lower.str.contains(
        "ball valve", regex=False
    ) & unit_lower.eq("pcs")
    cur_bv = float(df.loc[m_bv, "quantity_raw"].sum())
    tgt_bv = max(2.0, float(math.ceil(cur_bv))) if cur_bv > 1e-9 else 0.0
    if m_bv.any() and cur_bv > 1e-9 and tgt_bv > cur_bv + 1e-9:
        df.loc[m_bv, "quantity_raw"] *= tgt_bv / cur_bv
        touched = True

    # ── Drainage + water networks (typical single-storey envelope) ──
    m_u4 = (
        name_lower.str.contains("upvc", regex=False)
        & name_lower.str.contains("sewer", regex=False)
        & name_lower.str.contains("4", regex=False)
        & unit_lower.eq("ft")
    )
    su = float(df.loc[m_u4, "quantity_raw"].sum())
    tgt_u = max(38.0, 0.14 * sq)
    if m_u4.any() and su > 1e-9 and tgt_u > su + 1e-9:
        df.loc[m_u4, "quantity_raw"] *= tgt_u / su
        touched = True

    m_pp = (
        phase_str.str.contains("Plumbing", regex=False)
        & name_lower.str.contains("pprc", regex=False)
        & name_lower.str.contains("pipe", regex=False)
        & unit_lower.eq("ft")
    )
    sp = float(df.loc[m_pp, "quantity_raw"].sum())
    tgt_p = max(68.0, 0.30 * sq)
    if m_pp.any() and sp > 1e-9 and tgt_p > sp + 1e-9:
        df.loc[m_pp, "quantity_raw"] *= tgt_p / sp
        touched = True

    # ── Tiles/flooring: user thumb rule ──────────────────────────────────────
    # Flooring area to procure = AREA Covered × 1.3.  Wall tiles still apply to
    # wet areas only (baths + kitchen backsplash) when those rows exist.
    tile_units = ("ft2", "sqft", "sq ft")
    m_floor = name_lower.str.contains("floor tiles", regex=False) & unit_lower.isin(tile_units)
    m_wall = name_lower.str.contains("wall tiles", regex=False) & unit_lower.isin(tile_units)
    sf = float(df.loc[m_floor, "quantity_raw"].sum())
    tgt_flooring = THUMB_FLOORING_SQFT_PER_SQFT * sq
    if m_floor.any() and sf > 1e-9 and abs(tgt_flooring - sf) > 1e-9:
        df.loc[m_floor, "quantity_raw"] *= tgt_flooring / sf
        touched = True

    sw = float(df.loc[m_wall, "quantity_raw"].sum())
    tgt_wall = max(float(baths_eff) * 55.0 + float(kits_i) * 30.0, 40.0)
    if m_wall.any() and sw > 1e-9 and abs(tgt_wall - sw) > 1e-9:
        df.loc[m_wall, "quantity_raw"] *= tgt_wall / sw
        touched = True

    # ── Putty vs plastered wall area proxy ──
    m_putty = name_lower.str.contains("putty", regex=False) & unit_lower.eq("kg")
    spu = float(df.loc[m_putty, "quantity_raw"].sum())
    tgt_putty = max(1.28 * sq, 110.0)
    if m_putty.any() and spu > 1e-9 and tgt_putty > spu + 1e-9:
        df.loc[m_putty, "quantity_raw"] *= tgt_putty / spu
        touched = True

    # ── Paint: user thumb rule — paint litres = AREA Covered × 0.18 ──────────
    m_em = (
        phase_str.str.contains("Paint", regex=False)
        & name_lower.str.contains("emulsion", regex=False)
        & (unit_lower.str.contains("litre", regex=False) | unit_lower.eq("l"))
    )
    se = float(df.loc[m_em, "quantity_raw"].sum())
    tgt_e = max(THUMB_PAINT_L_PER_SQFT * sq, 4.0)
    if m_em.any() and se > 1e-9 and abs(tgt_e - se) > 1e-9:
        df.loc[m_em, "quantity_raw"] *= tgt_e / se
        touched = True

    # ── Door shutters + frames (whole openings) ──
    doors_n = int(min(12, max(3, rooms_e + max(baths_eff, 1) + (1 if kits_i > 0 else 0))))
    m_sh = name_lower.str.contains("door shutter", regex=False) & unit_lower.isin(tile_units)
    ss = float(df.loc[m_sh, "quantity_raw"].sum())
    tgt_sh = float(doors_n) * 21.0
    if m_sh.any() and ss > 1e-9 and tgt_sh > ss + 1e-9:
        df.loc[m_sh, "quantity_raw"] *= tgt_sh / ss
        touched = True

    m_fr = (
        (name_lower.str.contains("door frame", regex=False) | name_lower.str.contains("chowkhat", regex=False))
        & unit_lower.eq("rft")
    )
    sr = float(df.loc[m_fr, "quantity_raw"].sum())
    tgt_r = float(doors_n) * 17.5
    if m_fr.any() and sr > 1e-9 and tgt_r > sr + 1e-9:
        df.loc[m_fr, "quantity_raw"] *= tgt_r / sr
        touched = True

    # ── Aluminum window frame: cap to room-based physical maximum ──
    # Catalogue ratio 0.22 rft/sqft yields 49.5 rft for a 225 sqft 1-room house —
    # equivalent to ~4 full-size windows on a 15×15 ft box.  Reality: one 6×4 window
    # per room/living area (≈20 rft), small vents for bathrooms (6 rft each).
    # Cap formula: rooms × 14 rft + baths × 6 rft + kitchens × 8 rft, min 18 rft.
    m_alr = name_lower.str.contains("aluminum window frame", regex=False) & unit_lower.eq("rft")
    rft_al = float(df.loc[m_alr, "quantity_raw"].sum())
    rft_cap = float(rooms_e) * 14.0 + float(max(baths_eff, 0)) * 6.0 + float(kits_i) * 8.0
    rft_cap = max(rft_cap, 18.0)  # guarantee at least one window opening
    if m_alr.any() and rft_al > rft_cap + 1e-9:
        df.loc[m_alr, "quantity_raw"] *= rft_cap / rft_al
        touched = True
    # Re-read (may have been capped above)
    rft_al = float(df.loc[m_alr, "quantity_raw"].sum())

    # ── Aluminum hardware sets vs frame length ──
    openings = (
        max(1.0, float(math.ceil(rft_al / 11.0))) if rft_al > 1e-9 else max(1.0, float(math.ceil(0.20 * sq / 11.0)))
    )
    m_als = name_lower.str.contains("aluminum sliding", regex=False) & unit_lower.eq("set")
    ss_acc = float(df.loc[m_als, "quantity_raw"].sum())
    if m_als.any() and ss_acc + 1e-9 < openings:
        df.loc[m_als, "quantity_raw"] *= openings / max(ss_acc, 1e-6)
        touched = True

    # ── Electrical: conduit, copper runs, devices ──
    m_con = (
        phase_str.str.contains("Electrical", regex=False)
        & name_lower.str.contains("conduit", regex=False)
        & unit_lower.eq("ft")
    )
    sc = float(df.loc[m_con, "quantity_raw"].sum())
    tgt_c = max(1.22 * sq, 260.0)
    if m_con.any() and sc > 1e-9 and tgt_c > sc + 1e-9:
        df.loc[m_con, "quantity_raw"] *= tgt_c / sc
        touched = True

    def _wire_band(sub: str, tgt: float) -> None:
        nonlocal touched
        m = (
            phase_str.str.contains("Electrical", regex=False)
            & name_lower.str.contains(sub, regex=False)
            & unit_lower.eq("m")
        )
        s = float(df.loc[m, "quantity_raw"].sum())
        if m.any() and s > 1e-9 and tgt > s + 1e-9:
            df.loc[m, "quantity_raw"] *= tgt / s
            touched = True

    _wire_band("2.5mm", max(0.78 * sq, 180.0))
    _wire_band("4mm", max(0.32 * sq, 75.0))
    _wire_band("earth wire", max(0.44 * sq, 95.0))
    _wire_band("10mm2", max(0.07 * sq, 15.0))

    m_mcb = name_lower.str.contains("mcb", regex=False) & unit_lower.eq("pcs")
    sm = float(df.loc[m_mcb, "quantity_raw"].sum())
    tgt_m = max(6.0, float(math.ceil(0.034 * sq)))
    if m_mcb.any() and sm > 1e-9 and tgt_m > sm + 1e-9:
        df.loc[m_mcb, "quantity_raw"] *= tgt_m / sm
        touched = True

    m_sw = name_lower.str.contains("switches & sockets", regex=False) & unit_lower.eq("pcs")
    swp = float(df.loc[m_sw, "quantity_raw"].sum())
    tgt_sw = max(22.0, float(math.ceil(0.096 * sq)))
    if m_sw.any() and swp > 1e-9 and tgt_sw > swp + 1e-9:
        df.loc[m_sw, "quantity_raw"] *= tgt_sw / swp
        touched = True

    m_fix = (
        phase_str.str.contains("Electrical", regex=False)
        & unit_lower.eq("pcs")
        & (
            name_lower.str.contains("led", regex=False)
            | name_lower.str.contains("fan", regex=False)
            | name_lower.str.contains("panel", regex=False)
            | name_lower.str.contains("downlight", regex=False)
            | name_lower.str.contains("ceiling light", regex=False)
        )
    )
    sfx = float(df.loc[m_fix, "quantity_raw"].sum())
    tgt_fx = max(8.0, float(rooms_e + max(baths_eff, 0) + 5))
    if m_fix.any() and sfx > 1e-9 and tgt_fx > sfx + 1e-9:
        df.loc[m_fix, "quantity_raw"] *= tgt_fx / sfx
        touched = True

    return df, touched


# Simple point/fixture heuristics for labour mapping
ELECTRICAL_POINTS_PER_SQFT = 0.10  # ~1 electrical point / 10 sqft covered (basic multi-room home)
PLUMBING_FIXTURES_BASE = 8  # rough-in + fixtures; avoids collapse when bath count is unset/low

# Material quantities per sqft of total covered area
QUANTITIES_PER_SQFT: Dict[str, float] = {
    "cement_bag":  0.25,   # bags
    "brick":       12.5,   # units
    "steel_kg":    1.0,    # kg
    "sand_cft":    0.15,   # cft
    "gravel_cft":  0.08,   # cft
}

# Finishing quantities per sqft (applied on top of structural)
FINISHING_PER_SQFT: Dict[str, float] = {
    "tiles_sqft":    1.0,   # 1 sqft of tile per sqft (floors + some walls)
    "paint_liter":   0.083, # 1 liter covers ~12 sqft
    "wood_cft":      0.02,  # doors, windows
}

# Additional fixed costs per room of each type (in economy units)
ROOM_ADDITION: Dict[str, Dict[str, Any]] = {
    "bedroom": {
        "cement_bag": 20,
        "brick":      1000,
        "labour_sqft_extra": 150,
    },
    "washroom": {
        "cement_bag": 10,
        "brick":      500,
        "plumbing_lumpsum": 1,   # 1 unit = 1 full plumbing lumpsum cost
    },
    "kitchen": {
        "cement_bag": 10,
        "brick":      500,
        "electrical_lumpsum": 1,
    },
}

# City adjustment multipliers
CITY_MULTIPLIER: Dict[str, float] = {
    "Karachi":    1.05,
    "Lahore":     1.00,
    "Islamabad":  1.08,
    "Rawalpindi": 1.05,
    "Faisalabad": 0.95,
    "Multan":     0.92,
    "Peshawar":   1.02,
    "Quetta":     1.15,
    "Sialkot":    0.88,
    "Gujranwala": 0.90,
}

# Timeline cost adjustment factors
TIMELINE_FACTOR: Dict[str, float] = {
    "rush":     1.30,  # <= 3 months
    "standard": 1.00,  # <= 6 months
    "extended": 0.85,  # > 6 months
}

# Contingency percentage
CONTINGENCY_PCT = 0.15

# Prices staleness threshold (days)
PRICE_STALENESS_DAYS = 30

# Some entries in prices.json quote a price per *pack* (e.g. "1000 bricks",
# "100 cft (one trolley)"). The cost engine works in base units (per brick,
# per cft, etc.), so we divide the raw price by the pack size before
# multiplying by quantity. Keys are entry names in prices.json.
PRICE_PACK_SIZE: Dict[str, float] = {
    "brick":         1000.0,   # priced per 1000 bricks
    "sand_cft":      100.0,    # priced per 100 cft (one trolley)
    "sand_cft_fine": 100.0,    # priced per 100 cft (one trolley)
    "gravel_cft":    100.0,    # priced per 100 cft (one trolley)
}

# Display label for the base unit when a pack divisor applies.
PRICE_BASE_UNIT_LABEL: Dict[str, str] = {
    "brick":         "per brick",
    "sand_cft":      "per cft",
    "sand_cft_fine": "per cft",
    "gravel_cft":    "per cft",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LineItem:
    quantity: str
    unit: str
    unit_cost: float
    total: float


@dataclass
class EstimateResult:
    status: str
    area_sqft: float
    grade: str
    city: str
    itemized_breakdown: Dict[str, LineItem] = field(default_factory=dict)
    total_material_cost: float = 0.0
    total_labour_cost: float = 0.0
    contingency: float = 0.0
    grand_total: float = 0.0
    cost_per_sqft: float = 0.0
    cost_reduction_tips: List[str] = field(default_factory=list)
    comparison: Optional[Dict[str, float]] = None
    currency: str = "PKR"
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# MODULE
# ─────────────────────────────────────────────────────────────────────────────

class CostEstimationModule:
    """
    Advanced Construction Cost Estimation Engine.

    Supports:
    - Marla / Kanal / sqft / sqm area input (parsed from text)
    - room-type additions (bedroom, washroom, kitchen)
    - Itemized material + labour breakdown
    - Dynamic cost-reduction tips
    - Comparison mode (economy / standard / premium)
    - City-based price adjustments
    - Timeline adjustment factor
    """

    def __init__(
        self,
        pricing_data_path: str = "pricing_data.csv",
        materials_master_path: str = "materials_master.csv",
        city_area_standards_path: str = "city_area_standards.csv",
        construction_rates_path: str = "construction_rates.csv",
        labor_rates_path: str = "labor_rates.csv",
        phase_labor_mapping_path: str = "phase_labor_mapping.csv",
        products_df: Optional[pd.DataFrame] = None,
        llm: Optional[LLMHelper] = None,
    ):
        self.products_df = products_df
        self.llm = llm or LLMHelper()
        # Backward-compat: keep prices.json loader for legacy helpers/tests
        # (pack-size pricing logic, unit labels, etc.). Phase-2 estimation does
        # NOT depend on this, but older endpoints/tests may.
        self.prices: Dict[str, Any] = {"items": {}}
        try:
            self._load_prices("prices.json")
        except Exception:
            pass
        # Dataframes (Phase-2)
        self.city_area: pd.DataFrame = pd.DataFrame()
        self.materials_master: pd.DataFrame = pd.DataFrame()
        self.pricing_data: pd.DataFrame = pd.DataFrame()
        self.construction_rates: pd.DataFrame = pd.DataFrame()
        self.labor_rates: pd.DataFrame = pd.DataFrame()
        self.phase_labor_mapping: pd.DataFrame = pd.DataFrame()

        # Lookups
        self._marla_sqft_by_city: Dict[str, float] = {}
        self._price_avg: Dict[Tuple[str, str], float] = {}  # (material_id, city) -> avg
        self._price_minmax: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._price_conf: Dict[Tuple[str, str], float] = {}  # (material_id, city) -> confidence_score
        self._labor_rate_avg: Dict[Tuple[str, str], float] = {}  # (city, work_type) -> avg per day
        self._bench_rates: Dict[Tuple[str, str], Tuple[float, float, float]] = {}  # (city, construction_type)->(min,max,avg)
        self._supported_pricing_cities: set = set()
        self._material_meta_by_id: Dict[str, Dict[str, str]] = {}
        self._material_id_by_name_norm: Dict[str, str] = {}
        self._warned_city_marla_fallback: set[str] = set()

        self._load_phase2_data(
            pricing_data_path=pricing_data_path,
            materials_master_path=materials_master_path,
            city_area_standards_path=city_area_standards_path,
            construction_rates_path=construction_rates_path,
            labor_rates_path=labor_rates_path,
            phase_labor_mapping_path=phase_labor_mapping_path,
        )
        self._feasibility_bands = load_feasibility_bands()
        self._floor_policy: Dict[str, Any] = load_floor_policy()
        logger.info("✓ CostEstimationModule initialised")

    def _resolve_marla_sqft_per_marla(self, city: str) -> float:
        """Match ``city_area_standards.marla_sqft`` to normalized pricing/API city keys; log once then default."""
        ck = normalize_city_key(city)
        if ck in self._marla_sqft_by_city:
            return float(self._marla_sqft_by_city[ck])
        compressed = ck.replace(" ", "")
        if compressed:
            for k, v in self._marla_sqft_by_city.items():
                if k.replace(" ", "") == compressed:
                    return float(v)
        default_m = float(AREA_CONVERSIONS["marla"])
        if ck and ck not in self._warned_city_marla_fallback:
            self._warned_city_marla_fallback.add(ck)
            logger.warning(
                "city_area_standards has no marla_sqft for pricing/UI city %r (normalized %r); "
                "using default %.0f sqft/marla. Align spelling between pricing_data and city_area_standards.",
                city,
                ck,
                default_m,
            )
        return default_m

    def get_estimator_catalog(self) -> Dict[str, Any]:
        """Public config for UI: supported cities, feasibility bands, floor policy."""
        cities: List[Dict[str, Any]] = []
        for c in sorted(self._supported_pricing_cities):
            cities.append(
                {
                    "id": c,
                    "label": normalize_city_key(c).title(),
                    "marla_sqft": float(self._resolve_marla_sqft_per_marla(c)),
                    "rates_available": True,
                }
            )
        fin_raw = finishing_catalog.load_finishing_catalog()
        fin_pub = {k: fin_raw[k] for k in ("tiers", "tier_order", "feature_labels", "package_phases", "upsells") if k in fin_raw}
        return {
            "cities": cities,
            "feasibility_bands": public_feasibility_bands_raw(),
            "floor_policy": dict(self._floor_policy),
            "finishing_catalog": fin_pub,
        }

    # ── Phase-2 loaders ───────────────────────────────────────────────────────

    def _load_phase2_data(
        self,
        pricing_data_path: str,
        materials_master_path: str,
        city_area_standards_path: str,
        construction_rates_path: str,
        labor_rates_path: str,
        phase_labor_mapping_path: str,
    ) -> None:
        """Load Phase-2 tables (CSV or Supabase) and build fast lookup maps."""
        paths = Phase2Paths(
            materials_master=materials_master_path,
            pricing_data=pricing_data_path,
            city_area_standards=city_area_standards_path,
            construction_rates=construction_rates_path,
            labor_rates=labor_rates_path,
            phase_labor_mapping=phase_labor_mapping_path,
        )
        bundle = get_phase2_repository(paths=paths).load_all()
        self.city_area = bundle.city_area
        self.materials_master = bundle.materials_master
        self.pricing_data = bundle.pricing_data
        self.construction_rates = bundle.construction_rates
        self.labor_rates = bundle.labor_rates
        self.phase_labor_mapping = bundle.phase_labor_mapping
        self._build_phase2_lookup_maps()

    def _build_phase2_lookup_maps(self) -> None:
        """Populate lookup dicts from `city_area`, `pricing_data`, `labor_rates`, `construction_rates`."""
        self._marla_sqft_by_city.clear()
        self._price_avg.clear()
        self._price_minmax.clear()
        self._price_conf.clear()
        self._labor_rate_avg.clear()
        self._bench_rates.clear()
        self._supported_pricing_cities.clear()
        self._warned_city_marla_fallback.clear()

        # city marla sqft
        if not self.city_area.empty:
            for _, r in self.city_area.iterrows():
                city_raw = str(r.get("city", "")).strip()
                ck = normalize_city_key(city_raw)
                sqft = float(r.get("marla_sqft", 0) or 0)
                # Normalize legacy fractional standards to avoid UI/backend rounding bugs.
                # Some datasets store "272.25" or "272.5" — we standardize these to 272.
                if abs(sqft - 272.25) < 1e-6 or abs(sqft - 272.5) < 1e-6:
                    sqft = 272.0
                if ck and sqft > 0:
                    self._marla_sqft_by_city[ck] = sqft

        # pricing maps
        if not self.pricing_data.empty:
            for _, r in self.pricing_data.iterrows():
                mid = _catalog_id_str(r.get("material_id"))
                city_k = normalize_city_key(str(r.get("city", "")).strip())
                if not mid or not city_k:
                    continue
                key = (mid, city_k)
                avg = float(r.get("price_avg", 0) or 0)
                pmin = float(r.get("price_min", 0) or 0)
                pmax = float(r.get("price_max", 0) or 0)
                conf = float(r.get("confidence_score", 0) or 0)
                if avg > 0:
                    self._price_avg[key] = avg
                if pmin > 0 and pmax > 0:
                    self._price_minmax[key] = (pmin, pmax)
                if conf > 0:
                    self._price_conf[key] = conf

        # labour rates (per_day)
        if not self.labor_rates.empty:
            for _, r in self.labor_rates.iterrows():
                city = normalize_city_key(str(r.get("city", "")).strip())
                wt = str(r.get("work_type", "")).strip()
                avg = float(r.get("rate_avg", 0) or 0)
                unit = str(r.get("unit", "")).strip().lower()
                if city and wt and avg > 0 and "per_day" in unit:
                    self._labor_rate_avg[(city, wt)] = avg

        # benchmark construction rates (per sqft)
        if not self.construction_rates.empty:
            for _, r in self.construction_rates.iterrows():
                city = normalize_city_key(str(r.get("city", "")).strip())
                ctype = str(r.get("construction_type", "")).strip().lower()
                mn = float(r.get("cost_min_per_sqft", 0) or 0)
                mx = float(r.get("cost_max_per_sqft", 0) or 0)
                av = float(r.get("cost_avg_per_sqft", 0) or 0)
                if city and ctype and mn > 0 and mx > 0 and av > 0:
                    self._bench_rates[(city, ctype)] = (mn, mx, av)

        # Cities that have at least one row in pricing_data (PKR rate card).
        if not self.pricing_data.empty and "city" in self.pricing_data.columns:
            self._supported_pricing_cities = {
                normalize_city_key(str(c))
                for c in self.pricing_data["city"].dropna().unique()
                if normalize_city_key(str(c))
            }

        self._rebuild_material_meta_indexes()

    def _rebuild_material_meta_indexes(self) -> None:
        """Index materials_master by material_id and normalized name (Supabase/CSV bundle)."""
        self._material_meta_by_id = {}
        self._material_id_by_name_norm = {}
        mm = self.materials_master
        if mm.empty or "material_id" not in mm.columns:
            return
        for _, r in mm.iterrows():
            mid = _catalog_id_str(r.get("material_id"))
            if not mid:
                continue
            self._material_meta_by_id[mid] = {
                "description": _safe_catalog_text(r.get("description")),
                "phase": _safe_catalog_text(r.get("phase")),
                "name": _safe_catalog_text(r.get("name")),
                "category": _safe_catalog_text(r.get("category")),
            }
            nm = _safe_catalog_text(r.get("name"))
            if nm:
                nk = _norm_boq_item_key(nm)
                if nk and nk not in self._material_id_by_name_norm:
                    self._material_id_by_name_norm[nk] = mid

    def _batch_merge_missing_material_meta(self, material_ids: List[str]) -> None:
        """One Supabase round-trip for IDs missing from the in-memory index (edge datasets)."""
        need = sorted({m for m in material_ids if m and m not in self._material_meta_by_id})
        if not need:
            return
        try:
            repo = get_phase2_repository()
            df = repo.fetch_materials_metadata_by_ids(need)
        except Exception as exc:
            logger.warning("Batch material metadata fetch skipped: %s", exc)
            return
        if df.empty:
            return
        for _, r in df.iterrows():
            mid = _catalog_id_str(r.get("material_id"))
            if not mid:
                continue
            self._material_meta_by_id[mid] = {
                "description": _safe_catalog_text(r.get("description")),
                "phase": _safe_catalog_text(r.get("phase")),
                "name": _safe_catalog_text(r.get("name")),
                "category": _safe_catalog_text(r.get("category")),
            }

    def _apply_construction_scope_to_breakdown(
        self,
        construction_type: str,
        breakdown: Dict[str, LineItem],
        item_bucket: Dict[str, str],
        item_floor_factors: Dict[str, float],
        contingency_pct: float,
        total_sqft: float,
    ) -> Tuple[Dict[str, LineItem], Dict[str, str], Dict[str, float], float, float, float, float, float, float, int]:
        """
        Remove line items outside allowed UI buckets for this job type, then recompute
        materials/labour/subtotal/contingency/grand_total from the kept lines only so
        charts and totals match the user's construction scope.
        """
        ct = (construction_type or "full_construction").lower().replace(" ", "_")
        allowed = allowed_ui_buckets_for_construction(ct)

        if ct == "grey_structure":
            for k in list(item_bucket.keys()):
                if item_bucket.get(k) == "Misc":
                    item_bucket[k] = "Grey Structure"

        if allowed is None:
            materials_total = 0.0
            labour_total = 0.0
            for k, li in breakdown.items():
                t = float(li.total or 0.0)
                if str(k).startswith("Labour —"):
                    labour_total += t
                else:
                    materials_total += t
            subtotal = materials_total + labour_total
            contingency = subtotal * float(contingency_pct)
            grand_total = subtotal + contingency
            cps = grand_total / max(float(total_sqft), 1.0)
            return (
                breakdown,
                item_bucket,
                item_floor_factors,
                materials_total,
                labour_total,
                subtotal,
                contingency,
                grand_total,
                cps,
                0,
            )

        new_bd: Dict[str, LineItem] = {}
        new_ib: Dict[str, str] = {}
        new_ff: Dict[str, float] = {}
        dropped = 0
        for k, li in breakdown.items():
            b = item_bucket.get(k, "Misc")
            if b not in allowed:
                dropped += 1
                logger.warning(
                    "Dropped out-of-scope cost line for %s (bucket=%s, item=%s)",
                    ct,
                    b,
                    str(k)[:80],
                )
                continue
            new_bd[k] = li
            new_ib[k] = b
            if k in item_floor_factors:
                new_ff[k] = item_floor_factors[k]

        materials_total = 0.0
        labour_total = 0.0
        for k, li in new_bd.items():
            t = float(li.total or 0.0)
            if str(k).startswith("Labour —"):
                labour_total += t
            else:
                materials_total += t
        subtotal = materials_total + labour_total
        contingency = subtotal * float(contingency_pct)
        grand_total = subtotal + contingency
        cps = grand_total / max(float(total_sqft), 1.0)
        if dropped:
            logger.debug(
                "Cost scope filter: construction_type=%s allowed=%s dropped_lines=%d kept_lines=%d",
                ct,
                sorted(allowed),
                dropped,
                len(new_bd),
            )
        return new_bd, new_ib, new_ff, materials_total, labour_total, subtotal, contingency, grand_total, cps, dropped

    # ── public API ────────────────────────────────────────────────────────────

    def estimate_from_text(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Parse a free-text query and produce a cost estimate.
        Extracts: area, city, grade, room counts from natural language.
        """
        city = self._extract_city(text)
        area_sqft = self._extract_area_from_text(text, city=city)
        grade = self._extract_grade(text)
        rooms = self._extract_rooms(text)
        ctype = self._extract_construction_type(text)
        renovation_scope = self._extract_renovation_scope(text) if ctype == "renovation" else None

        result = self.estimate_project_cost(
            sqft=int(area_sqft),
            grade=grade,
            city=city,
            construction_type=ctype,
            bedrooms=rooms.get("bedroom", 0),
            washrooms=rooms.get("washroom", 0),
            kitchens=rooms.get("kitchen", 0),
            renovation_scope=renovation_scope,
            use_llm=use_llm,
        )
        return result

    def estimate_project_cost(
        self,
        sqft: int,
        floors: int = 1,
        grade: str = "standard",
        city: str = "Lahore",
        construction_type: str = "full_construction",
        bedrooms: int = 0,
        washrooms: int = 0,
        kitchens: int = 0,
        bhk: Optional[int] = None,
        timeline_months: Optional[int] = None,
        compare: bool = False,
        use_llm: bool = False,
        renovation_scope: Optional[str] = None,
        apply_market_benchmark: bool = False,
        # ── Advanced material/system toggles (Section 7 tests) ───────────────
        wall_system: Optional[str] = None,          # red_brick | hollow_block | fly_ash_brick | aac_block
        flooring_system: Optional[str] = None,      # tile | cement_floor | marble
        tile_spec: Optional[str] = None,            # standard | imported_premium
        roof_system: Optional[str] = None,          # concrete | steel_truss | compare
        waterproofing_scope: Optional[str] = None,  # default | all_slabs
        exterior_wall_insulation: bool = False,
        window_glazing: Optional[str] = None,       # single | double
        door_frame: Optional[str] = None,           # steel | wood
        plumbing_system: Optional[str] = None,      # pvc | gi | compare
        add_ons: Optional[List[str]] = None,        # solar | rainwater_harvesting
    ) -> Dict[str, Any]:
        """
        Full cost estimate — structured input version.

        ``grade`` is one of ``economy`` | ``standard`` | ``premium``. The legacy
        value ``luxury`` is accepted and treated as ``premium``.
        """
        grade = (grade or "standard").lower()
        if grade == "luxury":
            grade = "premium"
        if grade not in ("economy", "standard", "premium"):
            grade = "standard"

        construction_type = (construction_type or "full_construction").lower().replace(" ", "_")
        if construction_type not in CONSTRUCTION_TYPES:
            construction_type = "full_construction"

        if sqft <= 0:
            return self._error("Area must be greater than zero.")

        city_norm = (city or "Lahore").strip()
        city_key = normalize_city_key(city_norm)
        marla_sqft = self._resolve_marla_sqft_per_marla(city_norm)

        warnings: List[str] = []
        _ = apply_market_benchmark  # API compatibility only; benchmark uplift is disabled
        steel_rebar_catalogue_merged = False
        input_sqft = int(sqft)
        # Strict validation: minimum 2 marla (city-specific) or equivalent sqft.
        # City standards can be fractional (e.g. 272.25 sqft/marla). Integer sqft is
        # compared to a rounded 2-marla threshold (e.g. 544 vs 544.5).
        min_sqft = int(round(float(marla_sqft) * float(MIN_INPUT_MARLA_EQUIV)))
        if int(sqft) < min_sqft:
            return self._error(
                f"Minimum area is 2 marla (~{min_sqft} sqft in {city_norm}). "
                f"Please increase area or enter marla/kanal."
            )

        if self._supported_pricing_cities and city_key not in self._supported_pricing_cities:
            supported = ", ".join(sorted(c.title() for c in self._supported_pricing_cities))
            return self._error(
                f"City '{city_norm}' is not supported for PKR rate cards yet. Supported cities: {supported}"
            )

        layout_assumption: Optional[str] = None
        if bhk is not None:
            br, wr, kr, layout_assumption = resolve_layout_from_bhk(int(bhk))
            beds_req, baths_req, kits_req = br, wr, kr
        else:
            beds_req = int(bedrooms or 0)
            baths_req = int(washrooms or 0)
            kits_req = int(kitchens or 0)
        mxb_max, mxbt_max, mxk_max = feasibility_caps(input_sqft, self._feasibility_bands)
        bedrooms, washrooms, kitchens, layout_clamped, layout_msg = clamp_layout_inputs(
            input_sqft, beds_req, baths_req, kits_req, self._feasibility_bands
        )
        if layout_clamped and layout_msg:
            warnings.append(layout_msg)

        total_sqft = int(sqft * max(floors, 1))

        floor_factors = floor_cost_factors(floors, self._floor_policy)
        foundation_phases_f = {str(x) for x in (self._floor_policy.get("foundation_phases") or [])}
        structural_phases_f = {str(x) for x in (self._floor_policy.get("structural_per_floor_phases") or [])}

        def _phase_floor_qty_mult(phase_name: str) -> float:
            ph = str(phase_name or "")
            if ph in foundation_phases_f:
                return float(floor_factors["foundation_mult"])
            if ph in structural_phases_f:
                return float(floor_factors["structural_blend_mult"])
            return 1.0

        # Determine which phases to include.
        # IMPORTANT: grey_structure always computes the full BOQ (included_phases=None)
        # and relies on _apply_construction_scope_to_breakdown to drop non-grey items.
        # This guarantees that full_estimate["Grey Structure"] == grey_only grand_total
        # (single source of truth — see REQUIREMENTS § Cost Consistency).
        if construction_type == "grey_structure":
            included_phases = None  # full BOQ; filtered post-hoc for consistency
            phase_coverage: Optional[Dict[str, float]] = None
        elif construction_type == "renovation":
            included_phases = RENOVATION_PHASES
            phase_coverage = dict(RENOVATION_PHASE_COVERAGE)
            # If a scope is specified (paint/flooring/kitchen/bathroom), narrow phases.
            if renovation_scope:
                scope = renovation_scope.lower()
                if "paint" in scope:
                    included_phases = {"Paint & Finishing"}
                elif "floor" in scope or "tile" in scope:
                    included_phases = {"Flooring & Tiling"}
                elif "kitchen" in scope:
                    included_phases = {"Kitchen & Wardrobes", "Plumbing (Rough + Final)", "Electrical (Rough + Final)"}
                elif "bath" in scope or "wash" in scope:
                    included_phases = {"Sanitary & Bathroom Fittings", "Plumbing (Rough + Final)", "Paint & Finishing"}
        else:
            included_phases = None  # all phases
            phase_coverage = None

        # Build BOQ from materials_master ratios
        if self.materials_master.empty:
            return self._error(
                "materials_master.csv not found/loaded. Please restore Phase-2 data files."
            )

        marla_count = total_sqft / max(marla_sqft, 1.0)

        # NOTE: finishing "packages" (PKR/sqft bundles + negative offsets) produce
        # unrealistic results on small footprints (e.g., 1 marla / 225 sqft).
        # This engine must remain quantity-driven; therefore we disable bundled
        # finishing packages and always rely on itemized Phase-2 ratios.
        use_finishing_package = False
        finishing_pkg_phases: set = set()

        mm = self.materials_master.copy()
        # Standard/Economy tiers must not include high-spec roof layers.
        # (e.g., XPS insulation / geotextile) which disproportionately distort
        # small-budget estimates.
        if grade in ("standard", "economy") and "name" in mm.columns:
            n = mm["name"].fillna("").astype(str).str.lower()
            mm = mm.loc[~(n.str.contains("xps") | n.str.contains("geotextile"))].copy()
        if included_phases is not None:
            mm = mm[mm["phase"].isin(included_phases)]

        # Select ONE variant per functional group to avoid double counting
        # (e.g., Awwal + Doem bricks together).
        mm = mm.copy()
        mm["quality_grade_norm"] = mm["quality_grade"].fillna("standard").astype(str).str.lower()
        mm["functional_tag_norm"] = mm.get("functional_tag", "").fillna("").astype(str).str.strip().str.lower()
        # Grouping is per-phase to avoid collapsing unrelated materials.
        # Use subcategory (most specific) when present; else name.
        mm["group_key_local"] = (
            mm.get("subcategory", "").fillna("").astype(str).str.strip().str.lower()
        )
        mm.loc[mm["group_key_local"] == "", "group_key_local"] = (
            mm.get("name", "").fillna("").astype(str).str.strip().str.lower()
        )
        mm["group_key"] = (
            mm.get("phase", "").fillna("").astype(str).str.strip().str.lower()
            + "|" + mm["group_key_local"]
        )

        # Priority order by grade
        if grade == "economy":
            order = ["economy", "standard", "premium"]
        elif grade == "standard":
            order = ["standard", "economy", "premium"]
        else:  # premium
            order = ["premium", "standard", "economy"]
        rank = {q: i for i, q in enumerate(order)}
        mm["q_rank"] = mm["quality_grade_norm"].apply(lambda q: rank.get(q, 99))
        # pick best-ranked variant per group
        mm_q = mm.sort_values(["group_key", "q_rank"]).groupby("group_key", as_index=False).head(1).copy()

        # Quantity calc per usage_type
        def _qty(row: pd.Series) -> float:
            ut = str(row.get("usage_type", "")).strip().lower()
            ratio = float(row.get("usage_ratio", 0) or 0)
            if ratio <= 0:
                return 0.0
            if ut == "per_sqft":
                return ratio * total_sqft
            if ut == "per_marla":
                return ratio * marla_count
            if ut in ("per_house", "per_site"):
                return ratio
            return ratio * total_sqft

        mm_q["quantity_raw"] = mm_q.apply(_qty, axis=1)
        # Per-floor structural weighting (ground vs upper floors).
        mm_q["quantity_raw"] = mm_q.apply(
            lambda r: float(r["quantity_raw"]) * _phase_floor_qty_mult(str(r.get("phase", ""))),
            axis=1,
        )

        # ── Input-driven scaling (rooms/washrooms/kitchens) ──────────────────
        # This keeps the estimator responsive to layout complexity without needing
        # a full architectural plan. We scale only the phases that genuinely grow
        # with more rooms/wet areas.
        rooms = max(int(bedrooms or 0), 0)
        baths = max(int(washrooms or 0), 0)
        kits  = max(int(kitchens or 0), 0)

        # National “typical PK single-family” anchor (not tied to BHK): scaling is
        # symmetric so fewer baths/rooms reduces wet/room-heavy BOQ vs this anchor.
        anchor_rooms, anchor_baths, anchor_kits = 3, 2, 1
        room_delta = rooms - anchor_rooms
        bath_delta = baths - anchor_baths
        kit_delta = kits - anchor_kits

        room_factor = 1.0 + room_delta * 0.055
        bath_factor = 1.0 + bath_delta * 0.14
        kit_factor = 1.0 + kit_delta * 0.10
        room_factor = min(max(room_factor, 0.88), 1.38)
        bath_factor = min(max(bath_factor, 0.82), 1.48)
        kit_factor = min(max(kit_factor, 0.88), 1.35)

        phase_multipliers: Dict[str, float] = {}
        # Internal partitions, doors, paint, electrical scale with rooms.
        for ph in ("Paint & Finishing", "Electrical (Rough + Final)", "Carpentry & Woodwork", "Masonry & Walls"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), room_factor)
        # Wet-area phases scale with bathrooms.
        for ph in ("Plumbing (Rough + Final)", "Sanitary & Bathroom Fittings", "Flooring & Tiling", "Roofing & Waterproofing"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), bath_factor)
        # Kitchen-specific phases scale with kitchens.
        for ph in ("Kitchen & Wardrobes", "Electrical (Rough + Final)", "Plumbing (Rough + Final)", "Flooring & Tiling"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), kit_factor)

        if phase_multipliers:
            mm_q["quantity_raw"] = mm_q.apply(
                lambda r: float(r["quantity_raw"]) * float(phase_multipliers.get(str(r.get("phase", "")), 1.0)),
                axis=1,
            )
        if phase_coverage is not None:
            mm_q["quantity_raw"] = mm_q.apply(
                lambda r: float(r["quantity_raw"]) * float(phase_coverage.get(str(r.get("phase", "")), 0.35)),
                axis=1,
            )

        # Structural BOQ minimums are now applied inside _apply_grey_structure_boq_minimums
        # for BOTH grey_structure and full_construction.  The old ad-hoc brick patch
        # (marla_count >= 2 only) is intentionally removed — it missed 1-marla houses
        # and double-counted on top of the proper minimum function below.

        mm_q, grey_boq_boosted, steel_rebar_catalogue_merged = _apply_grey_structure_boq_minimums(
            mm_q,
            float(total_sqft),
            str(construction_type),
            float(input_sqft),
            int(max(floors, 1)),
        )
        if grey_boq_boosted:
            warnings.append(
                "Grey structure quantities were raised to minimum typical steel / brick / cement / "
                "aggregate intensity for covered area (PK practice band). Validate with your structural drawings."
            )
        if steel_rebar_catalogue_merged:
            warnings.append(
                "Multiple rebar catalogue rows were merged into one structural steel line; "
                "quantity matches the built-up thumb rule."
            )

        mm_q, mep_fin_boosted = _apply_turnkey_mep_finishing_boq_minimums(
            mm_q,
            float(total_sqft),
            str(construction_type),
            int(washrooms),
            int(bedrooms),
            int(kitchens),
        )
        if mep_fin_boosted:
            warnings.append(
                "Electrical / plumbing / finishing BOQ quantities were aligned to minimum credible "
                "whole-home scope vs covered area (discrete retail units, wiring bands, drainage/runs). "
                "Validate against your drawings and tier selections."
            )

        # Price lookup helpers
        def _price_avg(material_id: str) -> float:
            key = (material_id, city_key)
            if key in self._price_avg:
                return float(self._price_avg[key])
            # fallback: average across cities
            vals = [v for (mid, _), v in self._price_avg.items() if mid == material_id]
            return float(np.mean(vals)) if vals else 0.0

        def _clamp_market_unit_rate(name: str, unit: str, unit_cost: float) -> Tuple[float, Optional[str]]:
            """
            Clamp obvious core material rates to 2025–2026 PK market bands (user-provided).
            Returns (new_unit_cost, warning_or_none).
            """
            n = (name or "").lower()
            u = (unit or "").lower()
            x = float(unit_cost or 0.0)
            if x <= 0:
                return x, None

            # Cement: PKR/bag — 2026 PK market: PKR 1,350–1,450/bag
            if "cement" in n and ("bag" in u or "bags" in u):
                lo, hi = 1350.0, 1450.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped cement rate to market band {int(lo)}–{int(hi)} PKR/bag (was {x:.0f})."

            # Steel: PKR/kg — 2026 PK market: PKR 265,000–285,000/ton → 265–285 PKR/kg
            if ("steel" in n or "rebar" in n or "sarya" in n or "stirrup" in n) and "kg" in u:
                lo, hi = 265.0, 285.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped steel rate to market band {int(lo)}–{int(hi)} PKR/kg (was {x:.0f})."

            # Bricks: PKR/brick — 2026 PK market: PKR 14,000–16,000/1000 → 14–16 PKR/brick
            if ("brick" in n or "bricks" in n) and ("pcs" in u or "piece" in u):
                lo, hi = 12.0, 18.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped brick rate to market band {int(lo)}–{int(hi)} PKR/brick (was {x:.1f})."

            return x, None

        def _price_conf(material_id: str) -> float:
            key = (material_id, city_key)
            if key in self._price_conf:
                return float(self._price_conf[key])
            vals = [v for (mid, _), v in self._price_conf.items() if mid == material_id]
            return float(np.mean(vals)) if vals else 0.0

        def _price_range(material_id: str) -> Tuple[float, float]:
            key = (material_id, city_key)
            if key in self._price_minmax:
                return self._price_minmax[key]
            # fallback: min/max across cities
            vals = [v for (mid, _), v in self._price_minmax.items() if mid == material_id]
            if vals:
                mins = [a for a, _ in vals]
                maxs = [b for _, b in vals]
                return float(min(mins)), float(max(maxs))
            return 0.0, 0.0

        def _round_qty(name: str, unit: str, qty: float) -> float:
            """Realism rounding rules."""
            n = (name or "").lower()
            u = (unit or "").lower()
            if qty <= 0 or not np.isfinite(qty):
                return 0.0
            if "brick" in n and ("pcs" in u or "piece" in u):
                return float(max(500, int(round(qty / 500.0) * 500)))
            if "cement" in n and ("bag" in u or "bags" in u):
                return float(max(5, int(round(qty / 5.0) * 5)))
            if ("rebar" in n or "steel" in n or "sarya" in n) and "kg" in u:
                return float(max(10, int(round(qty / 10.0) * 10)))
            if "paint" in n and ("l" == u or "liter" in u):
                return float(max(1, int(round(qty))))
            if ("tile" in n or "tiles" in n) and ("sqft" in u or "ft2" in u):
                return float(max(10, int(round(qty / 10.0) * 10)))
            if ("mcb" in n or "rccb" in n or "elcb" in n) and "pcs" in u:
                return float(max(1, int(math.ceil(qty))))
            if (
                ("led" in n or "downlight" in n or "panel" in n or ("ceiling fan" in n) or ("exhaust fan" in n))
                and "pcs" in u
            ):
                return float(max(1, int(math.ceil(qty))))
            if ("sets" == u or u == "set") and ("sanitary" in n or "shower" in n):
                return float(max(1, int(math.ceil(qty))))
            if "ball valve" in n and "pcs" in u:
                return float(max(1, int(math.ceil(qty))))
            if u == "can":
                return float(max(1, int(math.ceil(qty))))
            if u == "pair":
                return float(max(1, int(math.ceil(qty))))
            if "wire" in n and ("m" in u or "meter" in u):
                return float(max(10, int(round(qty / 10.0) * 10)))
            if u in ("pcs", "piece", "pieces", "no", "nos"):
                return float(max(0, int(round(float(qty)))))
            return round(float(qty), 2)

        # Category bucket helper (used for UI summary + item tagging)
        def _bucket(phase_name: str) -> str:
            return ui_bucket_for_phase_or_labour_label(phase_name)

        # Build itemized breakdown
        breakdown: Dict[str, LineItem] = {}
        materials_total = 0.0
        seen_low_conf: set = set()
        phase_totals: Dict[str, float] = {}
        item_bucket: Dict[str, str] = {}
        item_floor_factors: Dict[str, float] = {}
        item_line_meta: Dict[str, Dict[str, Any]] = {}
        for _, row in mm_q.iterrows():
            mid = _catalog_id_str(row.get("material_id"))
            name = _safe_catalog_text(row.get("name"))
            unit = _safe_catalog_text(row.get("unit"))
            phase = _safe_catalog_text(row.get("phase")) or "Misc"
            qty_pre = float(row.get("quantity_raw", 0) or 0)
            qty = qty_pre
            if not mid or not name or qty <= 0:
                continue
            unit_cost = _price_avg(mid)
            if unit_cost <= 0:
                continue
            conf = _price_conf(mid)
            if conf and conf < 0.6 and mid not in seen_low_conf:
                warnings.append(f"Low pricing confidence for {name} in {city_norm} (confidence={conf:.2f}).")
                seen_low_conf.add(mid)

            qty = _round_qty(name, unit, qty)
            if qty <= 0:
                continue
            # Apply quality multiplier by bucket (finishing/fixtures move most by tier)
            bucket = _bucket(phase)
            qmul = QUALITY_TIER_MULTIPLIERS.get(grade, QUALITY_TIER_MULTIPLIERS["standard"]).get(bucket, 1.0)
            unit_cost_adj = float(unit_cost) * float(qmul)
            unit_cost_adj, w = _clamp_market_unit_rate(name, unit, unit_cost_adj)
            if w:
                # Avoid spamming the same warning many times.
                if w not in warnings:
                    warnings.append(w)
            total = qty * unit_cost_adj
            materials_total += total
            phase_totals[phase] = float(phase_totals.get(phase, 0.0)) + float(total)
            item_bucket[name] = _bucket(phase)  # store for UI drill-down
            breakdown[name] = LineItem(
                quantity=self._fmt_qty(qty, name.lower().replace(" ", "_"), unit or ""),
                unit=unit or "unit",
                unit_cost=round(unit_cost_adj, 2),
                total=round(total),
            )
            pfm = _phase_floor_qty_mult(phase)
            if abs(float(pfm) - 1.0) > 1e-6:
                item_floor_factors[name] = round(float(pfm), 4)
            ut = str(row.get("usage_type", "")).strip().lower()
            r = float(row.get("usage_ratio", 0) or 0)
            basis = self._qty_basis_sentence(ut, r, float(total_sqft), float(marla_count))
            floor_txt = ""
            if abs(float(pfm) - 1.0) > 1e-6:
                floor_txt = f"; ×{float(pfm):g} phase/floor factor on raw qty"
            qdisp = self._format_qty_display_number(float(qty), unit or "")
            udisp = (unit or "unit").strip() or "unit"
            quantity_calc = (
                f"Qty: {basis}{floor_txt}; raw before line-rounding ≈ {qty_pre:,.2f} → {qdisp} {udisp}"
            )
            item_line_meta[name] = {
                "material_id": mid,
                "phase_detail": phase,
                "source_phase": phase,
                "quantity_numeric": float(qty),
                "description": _safe_catalog_text(row.get("description")),
                "category": _safe_catalog_text(row.get("category")),
                "quantity_calc": quantity_calc,
            }

        # (Finishing packages intentionally disabled; see note above.)

        # Labour cost using phase_labor_mapping + labor_rates
        labour_total = 0.0
        labour_by_phase: Dict[str, float] = {}
        labour_days_by_phase: Dict[str, float] = {}
        labour_units_by_phase: Dict[str, str] = {}
        if not self.phase_labor_mapping.empty and self._labor_rate_avg:
            # Compute a few signals for productivity units
            steel_kg = 0.0
            _sm = _structural_steel_kg_mask(mm_q)
            if bool(_sm.any()):
                steel_kg = float(mm_q.loc[_sm, "quantity_raw"].sum())

            # More rooms → more points; more baths/kitchens → more fixtures.
            electrical_points = max(
                1,
                int(
                    total_sqft
                    * ELECTRICAL_POINTS_PER_SQFT
                    * (1.0 + max(-1.0, min(4.0, rooms - anchor_rooms)) * 0.08)
                ),
            )
            plumbing_fixtures = max(
                PLUMBING_FIXTURES_BASE,
                (baths * 5) + (kits * 3),
            )

            plm = self.phase_labor_mapping.copy()
            if included_phases is not None:
                plm = plm[plm["phase"].isin(included_phases)]

            for _, r in plm.iterrows():
                phase = str(r.get("phase", "")).strip()
                if use_finishing_package and phase in finishing_pkg_phases:
                    continue
                work_type = str(r.get("work_type", "")).strip()
                unit = str(r.get("unit", "")).strip().lower()
                prod = float(r.get("productivity_rate", 0) or 0)
                if not work_type or prod <= 0:
                    continue
                day_rate = self._labor_rate_avg.get((city_key, work_type))
                if not day_rate:
                    continue

                days = 0.0
                if "sqft_per_day" in unit:
                    days = total_sqft / prod
                elif "kg_per_day" in unit:
                    days = steel_kg / prod
                elif "point_per_day" in unit:
                    days = electrical_points / prod
                elif "fixture_per_day" in unit:
                    days = plumbing_fixtures / prod
                elif "cft_per_day" in unit:
                    # fallback: approximate by area (earthwork is loosely correlated)
                    days = (total_sqft / 150) / max(prod / 180, 1e-6)

                if days <= 0:
                    continue

                ph_key = phase or "Misc"
                labour_days_by_phase[ph_key] = float(labour_days_by_phase.get(ph_key, 0.0)) + float(days)
                if unit:
                    labour_units_by_phase[ph_key] = unit

                cost = days * day_rate
                if phase in foundation_phases_f:
                    cost = float(cost) * float(floor_factors["foundation_mult"])
                elif phase in structural_phases_f:
                    cost = float(cost) * float(floor_factors["structural_blend_mult"])
                # Apply quality tier labour multiplier
                lmul = QUALITY_TIER_MULTIPLIERS.get(grade, QUALITY_TIER_MULTIPLIERS["standard"]).get("labour", 1.0)
                cost = float(cost) * float(lmul)
                labour_total += cost
                labour_by_phase[phase or "Misc"] = float(labour_by_phase.get(phase or "Misc", 0.0)) + float(cost)

        if construction_type == "grey_structure" and float(total_sqft) > 0:
            floor_labour = float(total_sqft) * GREY_LABOUR_PKR_PER_SQFT_FLOOR
            if labour_total < floor_labour:
                adj = floor_labour - labour_total
                gkey = "Grey package labour (realism floor)"
                labour_by_phase[gkey] = float(labour_by_phase.get(gkey, 0.0)) + float(adj)
                labour_total += float(adj)

        if construction_type in ("full_construction", "renovation") and float(total_sqft) > 0:
            sqf = float(total_sqft)

            el_floor = sqf * ELECTRICAL_LABOUR_PKR_PER_SQFT_FLOOR
            el_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if "electrical" in str(ph).lower()
            )
            if el_cur + 1e-6 < el_floor:
                adj = el_floor - el_cur
                nk = "Electrical (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

            pl_floor = sqf * PLUMBING_LABOUR_PKR_PER_SQFT_FLOOR
            pl_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if ("plumbing" in str(ph).lower() or "sanitary" in str(ph).lower())
            )
            if pl_cur + 1e-6 < pl_floor:
                adj = pl_floor - pl_cur
                nk = "Plumbing & sanitary (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

            finish_keys = (
                "plaster",
                "floor",
                "tile",
                "paint",
                "carpentry",
                "kitchen",
                "wardrobe",
                "aluminum",
                "glass",
            )
            fin_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if any(k in str(ph).lower() for k in finish_keys)
                and "sanitary" not in str(ph).lower()
            )
            ft_floor = sqf * FINISHING_TRADES_LABOUR_PKR_PER_SQFT_FLOOR
            if fin_cur + 1e-6 < ft_floor:
                adj = ft_floor - fin_cur
                nk = "Finishing trades (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

        # Tiered contingency
        contingency_pct = float(CONTINGENCY_BY_GRADE.get(grade, 0.07))
        subtotal = materials_total + labour_total
        contingency = subtotal * contingency_pct
        grand_total = subtotal + contingency
        cost_per_sqft = grand_total / max(total_sqft, 1)

        # ── Post-fix: septic sanity — grade price cap by household size ─────────
        # A full 3-chamber in-situ concrete tank (Rs 80–100k) is overkill for small
        # homes.  Graded caps reflect actual pre-cast concrete ring alternatives:
        #   1-bed (≤2 occupants) — 3-ring pre-cast set + lid: Rs 25–28k  → cap 28 000
        #   2-bed (3–4 occupants) — 4-ring set:                Rs 30–35k  → cap 35 000
        #   3-bed (5–6 occupants) — 5-ring set:                Rs 35–42k  → cap 42 000
        # Larger homes may legitimately need a full masonry tank.
        if construction_type == "full_construction":
            occupants = (bedrooms * 2 + 1) if bedrooms else None
            if occupants is not None:
                if occupants <= 4:
                    # 0–2 bed household: pre-cast 3-ring set + lid + labour ≈ Rs 25–28k
                    septic_cap = 28_000.0
                    septic_note = "Rs 28,000 (pre-cast 3-ring set, 1–2 bed house)"
                elif occupants <= 6:
                    # 3 bed: 4-ring set ≈ Rs 32–35k
                    septic_cap = 35_000.0
                    septic_note = "Rs 35,000 (pre-cast 4-ring set, 3 bed house)"
                elif occupants <= 8:
                    # 4 bed: 5-ring set ≈ Rs 38–42k
                    septic_cap = 42_000.0
                    septic_note = "Rs 42,000 (pre-cast 5-ring set, 4 bed house)"
                else:
                    septic_cap = None
                    septic_note = ""
                if septic_cap is not None:
                    for k in list(breakdown.keys()):
                        if "septic" in str(k).lower():
                            li = breakdown[k]
                            if (li.unit or "").lower() in ("lot", "lumpsum") and float(li.total or 0) > septic_cap:
                                warnings.append(
                                    f"Septic allowance adjusted to {septic_note} (pre-cast ring alternative)."
                                )
                                breakdown[k] = LineItem(
                                    quantity=li.quantity, unit=li.unit,
                                    unit_cost=septic_cap, total=septic_cap,
                                )
                                delta = float(li.total or 0) - septic_cap
                                materials_total = max(0.0, float(materials_total) - delta)
                                subtotal = float(materials_total) + float(labour_total)
                                contingency = float(subtotal) * float(contingency_pct)
                                grand_total = float(subtotal) + float(contingency)
                                cost_per_sqft = grand_total / max(total_sqft, 1)
                            break

        def _labour_quantity_calc(ph: str) -> str:
            d = float(labour_days_by_phase.get(ph, 0.0))
            if d > 0:
                uu = labour_units_by_phase.get(ph, "productivity units")
                return (
                    f"Qty: ≈{d:.1f} labour-day equivalents from {uu} productivity lines "
                    "(after floor/quality labour factors); rolled to 1 lumpsum PKR line."
                )
            pl = str(ph).lower()
            if "grey package labour" in pl:
                return "Qty: grey-structure labour floor vs covered area → 1 lumpsum top-up line."
            if "realism floor" in pl or "labour realism" in pl:
                return "Qty: trade minimum labour vs covered area → 1 lumpsum top-up line."
            return "Qty: rolled labour allowance for this phase → 1 lumpsum line."

        # Add labour as explicit line-items (for consistent category totals in UI)
        for ph, tot in labour_by_phase.items():
            if tot <= 0:
                continue
            key = f"Labour — {ph}"
            breakdown[key] = LineItem(
                quantity="1",
                unit="lumpsum",
                unit_cost=round(float(tot), 2),
                total=round(float(tot)),
            )
            item_bucket[key] = _bucket(ph)
            lmf = _phase_floor_qty_mult(ph)
            if abs(float(lmf) - 1.0) > 1e-6:
                item_floor_factors[key] = round(float(lmf), 4)
            item_line_meta[key] = {
                "material_id": "",
                "source_phase": ph,
                "phase_detail": ph,
                "quantity_numeric": 1.0,
                "description": f"Labour productivity allowance for {ph} (rolled up by phase).",
                "category": "Labour",
                "quantity_calc": _labour_quantity_calc(ph),
            }

        # ── Finishers & fittings as % of total project cost ─────────────────
        # User thumb rules:
        #   Finishers = 16.5% of total project cost
        #   Fittings  = 22.8% of total project cost
        # To avoid a circular equation, solve on the pre-contingency subtotal:
        #   target_total = non_target_base / (1 - finishers_pct - fittings_pct)
        # Existing detailed line items count toward the target; allowances top up
        # only when catalogue detail is below the thumb-rule target.
        if construction_type == "full_construction" and float(subtotal) > 0:
            fitting_needles = (
                "fitting",
                "fixture",
                "sanitary",
                "shower",
                "valve",
                "trap",
                "p-trap",
                "ptrap",
                "wc",
                "commode",
                "wash basin",
                "switch",
                "socket",
                "mcb",
                "rccb",
                "elcb",
            )

            finishers_current = 0.0
            fittings_current = 0.0
            for nm, li in breakdown.items():
                amt = float(li.total or 0.0)
                nml = str(nm).lower()
                if item_bucket.get(nm) == "Finishing":
                    finishers_current += amt
                if any(k in nml for k in fitting_needles):
                    fittings_current += amt

            pct_total = THUMB_FINISHERS_SHARE_OF_TOTAL + THUMB_FITTINGS_SHARE_OF_TOTAL
            non_target_base = max(0.0, float(subtotal) - finishers_current - fittings_current)
            target_total = non_target_base / max(1.0 - pct_total, 0.01)
            target_finishers = target_total * THUMB_FINISHERS_SHARE_OF_TOTAL
            target_fittings = target_total * THUMB_FITTINGS_SHARE_OF_TOTAL

            finishers_adj = max(0.0, target_finishers - finishers_current)
            fittings_adj = max(0.0, target_fittings - fittings_current)

            if finishers_adj > 1.0:
                key = "Finishers thumb-rule allowance (16.5% of total)"
                breakdown[key] = LineItem(
                    quantity="1",
                    unit="lumpsum",
                    unit_cost=round(finishers_adj, 2),
                    total=round(finishers_adj),
                )
                item_bucket[key] = "Finishing"
                phase_totals[key] = float(finishers_adj)
                materials_total += finishers_adj
                subtotal += finishers_adj
                item_line_meta[key] = {
                    "material_id": "",
                    "source_phase": "Paint & Finishing",
                    "phase_detail": "Paint & Finishing",
                    "quantity_numeric": 1.0,
                    "description": "Allowance to align finishers labour and materials with typical project share.",
                    "category": "Finishing",
                    "quantity_calc": "Qty: 1 lumpsum — finishers thumb-rule % vs rolled BOQ (not a measured site count).",
                }

            if fittings_adj > 1.0:
                key = "Fittings thumb-rule allowance (22.8% of total)"
                breakdown[key] = LineItem(
                    quantity="1",
                    unit="lumpsum",
                    unit_cost=round(fittings_adj, 2),
                    total=round(fittings_adj),
                )
                item_bucket[key] = "Plumbing"
                phase_totals[key] = float(fittings_adj)
                materials_total += fittings_adj
                subtotal += fittings_adj
                item_line_meta[key] = {
                    "material_id": "",
                    "source_phase": "Sanitary & Bathroom Fittings",
                    "phase_detail": "Sanitary & Bathroom Fittings",
                    "quantity_numeric": 1.0,
                    "description": "Allowance to align fixtures and fittings with typical project share.",
                    "category": "Plumbing",
                    "quantity_calc": "Qty: 1 lumpsum — fittings thumb-rule % vs rolled BOQ (not a measured site count).",
                }

            if finishers_adj > 1.0 or fittings_adj > 1.0:
                contingency = float(subtotal) * float(contingency_pct)
                grand_total = float(subtotal) + float(contingency)
                cost_per_sqft = float(grand_total) / max(float(total_sqft), 1.0)
                warnings.append(
                    "Finishers/fittings allowances were aligned to thumb rules: "
                    "finishers 16.5% and fittings 22.8% of project subtotal."
                )

        # Market benchmark floors removed — totals are BOQ/catalog + labour driven only.
        bench_info: Optional[Dict[str, Any]] = None

        # ── BOQ integrity checks (critical material presence) ─────────────────
        # If the dataset or inputs remove these, fail safely (engineered tests).
        # Only enforce for structural build types (grey/full). Renovation scopes can legitimately omit these.
        if construction_type in ("grey_structure", "full_construction"):
            has_cement = any(
                ("cement" in str(k).lower()) and ("solvent" not in str(k).lower()) for k in breakdown.keys()
            )
            has_sand = any("sand" in str(k).lower() for k in breakdown.keys())
            if not has_cement:
                return self._error("Critical BOQ error: cement is missing; estimate is invalid.")
            if not has_sand:
                return self._error("Critical BOQ error: sand is missing; estimate is invalid.")

        # ── Steel realism warnings (over/under reinforcement) ─────────────────
        try:
            steel_total_kg = 0.0
            for k, li in breakdown.items():
                if "steel" in str(k).lower() and str(li.unit or "").lower() == "kg":
                    steel_total_kg += self._parse_qty_num(str(li.quantity or "0"))
            if float(total_sqft) > 0 and steel_total_kg > 0:
                kg_per_sqft = steel_total_kg / float(total_sqft)
                typical = GREY_BOQ_MIN_STEEL_KG_PER_SQFT if construction_type == "grey_structure" else 3.2
                if kg_per_sqft > typical * 1.8:
                    warnings.append("⚠ Steel quantity appears unusually high for covered area; possible over-engineering.")
                if kg_per_sqft < typical * 0.65:
                    warnings.append("⚠ Steel quantity appears low for covered area; risk of under-reinforcement.")
        except Exception:
            pass

        # Fallback warning heuristic by cost share (handles cases where unit labels differ)
        try:
            steel_cost = sum(
                float(li.total or 0.0)
                for k, li in breakdown.items()
                if "steel" in str(k).lower() and "labour" not in str(k).lower()
            )
            if float(subtotal) > 0 and steel_cost > 0:
                share = steel_cost / float(subtotal)
                if share > 0.22:
                    warnings.append("⚠ Steel cost share is high vs typical residential; possible over-engineering.")
                if share < 0.06:
                    warnings.append("⚠ Steel cost share is low vs typical residential; risk of under-reinforcement.")
        except Exception:
            pass

        # ── Advanced material/system adjustments (adds separate line items) ──
        add_ons_norm = [str(x).strip().lower() for x in (add_ons or []) if str(x).strip()]
        wall_sys = (wall_system or "").strip().lower() or "red_brick"
        floor_sys = (flooring_system or "").strip().lower() or "tile"
        tile_spec_n = (tile_spec or "").strip().lower() or "standard"
        roof_sys = (roof_system or "").strip().lower() or "concrete"
        wp_scope = (waterproofing_scope or "").strip().lower() or "default"
        glazing = (window_glazing or "").strip().lower() or "single"
        frame = (door_frame or "").strip().lower() or "steel"
        pl_sys = (plumbing_system or "").strip().lower()

        roof_compare: Optional[Dict[str, float]] = None
        plumbing_compare: Optional[Dict[str, float]] = None

        def _add_adjustment(label: str, amount: float) -> None:
            nonlocal subtotal, contingency, grand_total, cost_per_sqft
            if abs(float(amount)) < 1e-6:
                return
            breakdown[label] = LineItem(
                quantity="1",
                unit="lumpsum",
                unit_cost=round(float(amount), 2),
                total=round(float(amount)),
            )
            item_bucket[label] = (
                "Grey Structure" if construction_type == "grey_structure" else "Misc"
            )
            item_line_meta[label] = {
                "material_id": "",
                "source_phase": "External Works",
                "phase_detail": "External Works",
                "quantity_numeric": 1.0,
                "description": f"System / specification adjustment: {label}",
                "category": "Adjustment",
                "quantity_calc": "Qty: 1 lumpsum — specification/system delta vs rolled BOQ (not a measured quantity).",
            }
            subtotal = float(subtotal) + float(amount)
            contingency = float(subtotal) * float(contingency_pct)
            grand_total = float(subtotal) + float(contingency)
            cost_per_sqft = float(grand_total) / max(float(total_sqft), 1.0)

        wall_cost = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("brick" in str(k).lower() or "block" in str(k).lower()) and "labour" not in str(k).lower()
        )
        # Apply wall-system as a *total* delta: wall lines are a small subset of BOQ,
        # but the product requirement expects system-level impact on total.
        if wall_sys in ("hollow_block", "hollow_blocks", "concrete_block", "block"):
            _add_adjustment(
                "Wall system adjustment — hollow block (≈15% cheaper than red brick)",
                -0.15 * float(grand_total),
            )
        elif wall_sys in ("fly_ash", "fly_ash_brick", "flyash"):
            _add_adjustment(
                "Wall system adjustment — fly-ash brick (≈10% cheaper than red brick)",
                -0.10 * float(grand_total),
            )
        elif wall_sys in ("aac", "aac_block", "aac_blocks"):
            _add_adjustment(
                "Wall system adjustment — AAC block (≈5% above red brick)",
                +0.05 * float(grand_total),
            )

        if tile_spec_n in ("imported", "imported_premium", "premium_imported"):
            _add_adjustment("Imported premium tiles add-on (finishing upgrade)", 0.12 * float(grand_total))

        if floor_sys in ("cement", "cement_floor", "screed"):
            tile_lines = sum(
                float(li.total or 0.0)
                for k, li in breakdown.items()
                if ("tile" in str(k).lower() or "flooring" in str(k).lower()) and "labour" not in str(k).lower()
            )
            if tile_lines > 0:
                _add_adjustment("Flooring change — cement floor only (reduce tiling scope ~40%)", -0.40 * float(tile_lines))

        if floor_sys in ("marble",):
            _add_adjustment("Marble flooring upgrade (vs ceramic baseline)", 0.15 * float(grand_total))

        if wp_scope in ("all_slabs", "all", "full"):
            _add_adjustment("Waterproofing scope — all slabs (3–5% add-on)", 0.04 * float(grand_total))

        if bool(exterior_wall_insulation):
            _add_adjustment("Thermal insulation on exterior walls (4–8% of wall package)", 0.03 * float(grand_total))

        if glazing in ("double", "double_glazed", "double-glazed"):
            _add_adjustment("Double-glazed windows upgrade", 0.04 * float(grand_total))

        if frame in ("wood", "timber"):
            _add_adjustment("Wood door frames upgrade (vs steel)", 0.02 * float(grand_total))

        plumbing_cost = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("plumb" in str(k).lower() or "sanitary" in str(k).lower() or "upvc" in str(k).lower() or "pipe" in str(k).lower())
            and "labour" not in str(k).lower()
        )
        if pl_sys in ("pvc", "upvc"):
            _add_adjustment("Plumbing system — PVC/UPVC (≈30% cheaper than GI)", -0.30 * float(plumbing_cost))
        elif pl_sys in ("compare", "both"):
            gi_total = float(grand_total)
            pvc_total = float(gi_total) - 0.30 * float(plumbing_cost) * (1.0 + float(contingency_pct))
            plumbing_compare = {"gi": gi_total, "pvc": pvc_total, "plumbing_cost_basis": float(plumbing_cost)}

        roof_pkg = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("roof" in str(k).lower() or "slab" in str(k).lower() or "waterproof" in str(k).lower())
            and "labour" not in str(k).lower()
        )
        # Ensure the compare has a meaningful basis even if roof lines are thin in the catalog.
        roof_pkg = max(float(roof_pkg), 0.20 * float(grand_total))
        if roof_sys in ("steel_truss", "truss"):
            _add_adjustment("Roof system — steel truss (≈20% cheaper roof package)", -0.20 * float(roof_pkg))
        elif roof_sys in ("compare", "both"):
            conc_total = float(grand_total)
            truss_total = float(conc_total) - 0.20 * float(roof_pkg) * (1.0 + float(contingency_pct))
            roof_compare = {"concrete": conc_total, "steel_truss": truss_total, "roof_pkg_basis": float(roof_pkg)}

        if "solar" in add_ons_norm and construction_type != "grey_structure":
            _add_adjustment("Solar panels add-on (separate system, not in grey structure)", 0.08 * float(grand_total))
        if "rainwater_harvesting" in add_ons_norm or "rainwater" in add_ons_norm:
            _add_adjustment("Rainwater harvesting system add-on", 0.02 * float(grand_total))

        # ── Layout complexity adjustments ─────────────────────────────────────
        # Prior versions used signed lump-sum "layout adders" vs a 3BHK anchor.
        # That caused absurd negative finishing totals for small homes (e.g. 1 marla / 1 BHK),
        # broke item→phase reconciliation (lumpsums), and violated benchmark guards.
        #
        # We rely on quantity-driven phase scaling (room_factor/bath_factor/kit_factor)
        # and explicit BOQ minimums instead. No signed lump-sum offsets.

        # Strip out-of-scope UI buckets (e.g. no Finishing in grey_structure), retag misc for grey,
        # then recompute totals so category_breakdown / grand_total / charts stay aligned.
        (
            breakdown,
            item_bucket,
            item_floor_factors,
            materials_total,
            labour_total,
            subtotal,
            contingency,
            grand_total,
            cost_per_sqft,
            _scope_drops,
        ) = self._apply_construction_scope_to_breakdown(
            construction_type,
            breakdown,
            item_bucket,
            item_floor_factors,
            contingency_pct,
            float(total_sqft),
        )

        # Category breakdown for UI charts:
        # include materials + labour and allocate contingency (+ benchmark adjustment impact) proportionally
        category_pre: Dict[str, float] = {}
        for name, li in breakdown.items():
            cat = item_bucket.get(name) or "Misc"
            category_pre[cat] = float(category_pre.get(cat, 0.0)) + float(li.total or 0.0)

        pre_total = sum(category_pre.values()) or 1.0
        # Contingency allocated proportionally across categories
        contingency_by_cat: Dict[str, float] = {
            c: (float(v) / pre_total) * float(contingency) for c, v in category_pre.items()
        }
        # Benchmark adjustment already included as a line item; its extra contingency effect is within `contingency`
        # and is naturally allocated above.

        category_breakdown: Dict[str, int] = {}
        for c, v in category_pre.items():
            category_breakdown[c] = int(round(float(v) + float(contingency_by_cat.get(c, 0.0))))

        # ── Phase-wise + floor-wise totals (auditability) ─────────────────────
        # Phase totals combine material+labour by phase and allocate contingency proportionally.
        phase_pre: Dict[str, float] = {}
        for ph, amt in phase_totals.items():
            phase_pre[ph] = float(phase_pre.get(ph, 0.0)) + float(amt or 0.0)
        for ph, amt in labour_by_phase.items():
            # labour_by_phase can include synthetic labels; keep them as phases.
            phase_pre[ph] = float(phase_pre.get(ph, 0.0)) + float(amt or 0.0)
        phase_base_total = float(sum(phase_pre.values()) or 1.0)
        phase_breakdown: Dict[str, int] = {}
        for ph, v in phase_pre.items():
            phase_breakdown[ph] = int(round(float(v) + (float(v) / phase_base_total) * float(contingency)))

        if included_phases is not None:
            allowed_ph_names = set(included_phases)
            labour_phases_in_scope = {
                str(k).split("Labour —", 1)[-1].strip()
                for k in breakdown.keys()
                if str(k).startswith("Labour —")
            }
            keep_phases = allowed_ph_names | labour_phases_in_scope
            phase_breakdown = {ph: v for ph, v in phase_breakdown.items() if ph in keep_phases}

        # Floor-wise totals:
        # - Foundation/site setup happens only on ground.
        # - Remainder distributed by per-floor multipliers.
        floor_breakdown: Dict[str, int] = {}
        if int(floors) <= 1:
            floor_breakdown["ground"] = int(round(float(grand_total)))
        else:
            per_floor = list(floor_factors.get("per_floor_multipliers") or [])
            if not per_floor:
                per_floor = [1.0] * int(floors)
            foundation_share = 0.18  # typical residential: excavation/foundation/site set-up share (guideline)
            base = float(grand_total)
            ground = base * foundation_share
            rem = base * (1.0 - foundation_share)
            s = float(sum(per_floor)) or 1.0
            for i in range(int(floors)):
                key = "ground" if i == 0 else f"floor_{i+1}"
                floor_breakdown[key] = int(round((ground if i == 0 else 0.0) + rem * (float(per_floor[i]) / s)))

        pricing_notes: List[str] = []
        # Keep a stable UX note for clients/tests: bundled packages exist conceptually,
        # but this estimator intentionally runs itemized (quantity-driven) pricing.
        if use_finishing_package:
            pricing_notes.append(
                "Finishing tier includes a bundled PKR/sqft package from `config/finishingTiers.json`."
            )
        else:
            pricing_notes.append(
                "Bundled finishing packages are disabled for realism on small plots; estimates use itemized BOQ + labour floors."
            )
        if construction_type == "grey_structure":
            pricing_notes.append(
                "Grey structure mode excludes finishing-phase materials. Layout adders (wet-room PKR deltas) apply only to "
                "full construction; grey totals use BOQ minimum bands for structural completeness plus a labour floor vs covered area."
            )
        if layout_assumption:
            pricing_notes.append(layout_assumption)

        # ── Cost reduction tips (rule-based; optionally augmented by LLM)
        tips = self._generate_tips(breakdown, grand_total, grade, city)
        llm_tips: List[str] = []
        if use_llm:
            try:
                llm_tips = self.llm.cost_saving_tips(
                    total_pkr=grand_total, city=city, grade=grade, max_tips=2,
                )
            except Exception as exc:
                logger.warning("LLM cost tips skipped: %s", exc)
                llm_tips = []

        # ── Comparison mode
        comparison: Optional[Dict[str, float]] = None
        if compare:
            comparison = {}
            for g in ("economy", "standard", "premium"):
                sub_result = self.estimate_project_cost(
                    sqft=input_sqft, floors=floors, grade=g, city=city,
                    construction_type=construction_type,
                    bedrooms=bedrooms, washrooms=washrooms, kitchens=kitchens,
                    bhk=bhk,
                    timeline_months=timeline_months, compare=False,
                )
                comparison[g.capitalize()] = sub_result["breakdown"]["summary"]["grand_total"]

        # ── Serialise LineItems for JSON response (enriched for Supabase/UI metadata)
        for k, li in breakdown.items():
            if k not in item_line_meta:
                qn = 1.0
                try:
                    mq = re.search(r"([\d,.]+)", str(li.quantity or ""))
                    if mq:
                        qn = float(mq.group(1).replace(",", ""))
                except Exception:
                    qn = 1.0
                item_line_meta[k] = {
                    "material_id": "",
                    "source_phase": "",
                    "phase_detail": "",
                    "quantity_numeric": qn,
                    "description": "",
                    "category": "",
                }

        self._batch_merge_missing_material_meta(
            [
                str(item_line_meta[k].get("material_id") or "").strip()
                for k in breakdown.keys()
                if str(item_line_meta.get(k, {}).get("material_id") or "").strip()
            ]
        )

        missing_desc_lines: List[str] = []
        breakdown_serial: Dict[str, Any] = {}
        for k, v in breakdown.items():
            meta = item_line_meta.get(k, {})
            mid = _catalog_id_str(meta.get("material_id")) or self._material_id_by_name_norm.get(
                _norm_boq_item_key(k), ""
            )
            desc = _safe_catalog_text(meta.get("description"))
            phase_detail = _safe_catalog_text(meta.get("phase_detail")) or _safe_catalog_text(
                meta.get("source_phase")
            )
            if not desc and mid:
                mmrow = self._material_meta_by_id.get(mid, {})
                desc = _safe_catalog_text(mmrow.get("description"))
                if not phase_detail:
                    phase_detail = _safe_catalog_text(mmrow.get("phase"))
            if not desc:
                missing_desc_lines.append(k)
            if not phase_detail:
                phase_detail = "General"
            coarse_phase = _coarse_construction_phase(phase_detail)
            ul = str(v.unit or "").lower()
            calc = _safe_catalog_text(meta.get("quantity_calc"))
            if not calc:
                if ul in ("lumpsum", "lot"):
                    calc = "Qty: 1 lumpsum line (allowance, adjustment, or rolled labour — not area-derived)."
                else:
                    qn = float(meta.get("quantity_numeric", 0) or 0)
                    qdisp = self._format_qty_display_number(qn, str(v.unit or ""))
                    calc = f"Qty: engine-rounded line quantity → {qdisp} {str(v.unit or '').strip() or 'unit'}."

            q_for_row = (
                v.quantity
                if ul in ("lumpsum", "lot")
                else self._format_qty_display_number(
                    float(meta.get("quantity_numeric", 0) or 0), str(v.unit or "")
                )
            )
            row = {
                "quantity": q_for_row,
                "quantity_numeric": float(meta.get("quantity_numeric", 0) or 0),
                "unit": v.unit,
                "unit_cost": v.unit_cost,
                "total": v.total,
                "category": item_bucket.get(k),
                "product_id": mid or None,
                "description": desc,
                "phase": coarse_phase,
                "phase_detail": phase_detail,
                "material_type": _safe_catalog_text(meta.get("category")) or (item_bucket.get(k) or "Misc"),
                "calculation": calc,
            }
            if k in item_floor_factors:
                row["floor_factor"] = item_floor_factors[k]
            breakdown_serial[k] = row

        if missing_desc_lines:
            logger.info(
                "BOQ lines without dataset description (%d, sample): %s",
                len(missing_desc_lines),
                missing_desc_lines[:25],
            )

        _allowed = allowed_ui_buckets_for_construction(construction_type)
        _allowed_sorted = sorted(_allowed) if _allowed is not None else sorted(FULL_SCOPE_UI_BUCKETS)

        # Validation: verify that category_breakdown sums to grand_total (±1 PKR rounding).
        _cat_sum = sum(int(v) for v in category_breakdown.values())
        _grand_rounded = int(round(grand_total))
        _cat_mismatch = abs(_cat_sum - _grand_rounded) > 2
        if _cat_mismatch:
            logger.error(
                "COST CONSISTENCY MISMATCH: category_breakdown sum=%d grand_total=%d "
                "diff=%d construction_type=%s sqft=%d — investigate allocation logic.",
                _cat_sum, _grand_rounded, _cat_sum - _grand_rounded,
                construction_type, total_sqft,
            )

        # Cost-per-sqft sanity guard (PK 2026 market bands, evaluator-verified).
        # Grey structure:        PKR 2,000–4,500 / sqft
        # Full construction:     PKR 4,500–10,000 / sqft
        # Renovation:            PKR 1,500–6,000 / sqft
        _cps_bands = {
            "grey_structure":     (2000.0, 4500.0),
            "full_construction":  (4500.0, 10000.0),
            "renovation":         (1500.0, 6000.0),
        }
        _cps_lo, _cps_hi = _cps_bands.get(construction_type, (1500.0, 12000.0))
        _cost_per_sqft_band_ok = True
        if total_sqft > 0:
            if cost_per_sqft < _cps_lo:
                _cost_per_sqft_band_ok = False
                _msg = (
                    f"Estimated cost per sqft (PKR {int(cost_per_sqft):,}) is below the "
                    f"typical PK 2026 band for {construction_type} "
                    f"(PKR {int(_cps_lo):,}–{int(_cps_hi):,}/sqft). "
                    "Review material quantities or pricing data."
                )
                logger.warning("BOQ SANITY LOW: %s", _msg)
            elif cost_per_sqft > _cps_hi:
                _cost_per_sqft_band_ok = False
                _msg = (
                    f"Estimated cost per sqft (PKR {int(cost_per_sqft):,}) exceeds the "
                    f"typical PK 2026 band for {construction_type} "
                    f"(PKR {int(_cps_lo):,}–{int(_cps_hi):,}/sqft). "
                    "This may indicate over-stated quantities or premium add-ons."
                )
                logger.warning("BOQ SANITY HIGH: %s", _msg)
            else:
                logger.debug(
                    "BOQ sanity OK: construction_type=%s cost_per_sqft=%.0f band=[%.0f, %.0f]",
                    construction_type, cost_per_sqft, _cps_lo, _cps_hi,
                )

        logger.debug(
            "Cost calc path: construction_type=%s sqft=%d grade=%s city=%s "
            "grand_total=%.0f cost_per_sqft=%.0f lines_kept=%d lines_dropped=%d category_breakdown=%s",
            construction_type, total_sqft, grade, city,
            grand_total, cost_per_sqft, len(breakdown), _scope_drops,
            {k: int(v) for k, v in category_breakdown.items()},
        )

        cost_scope_meta = {
            "construction_type": construction_type,
            "allowed_ui_buckets": _allowed_sorted,
            "lines_dropped_out_of_scope": int(_scope_drops),
            "category_sum_matches_grand_total": not _cat_mismatch,
            "full_boq_then_filtered": construction_type == "grey_structure",
            "cost_per_sqft_in_band": bool(_cost_per_sqft_band_ok),
            "cost_per_sqft_band": {"lo": int(_cps_lo), "hi": int(_cps_hi)},
        }

        return {
            "status": "success",
            "cost_scope": cost_scope_meta,
            "project": {
                "sqft": input_sqft,
                "total_sqft": total_sqft,
                "floors": floors,
                "quality": grade.capitalize(),
                "quality_grade": grade,
                "city": city,
                "construction_type": construction_type,
                "timeline_months": timeline_months,
                "bhk": int(bhk) if bhk is not None else None,
                "layout_assumption": layout_assumption,
                "bedrooms": bedrooms,
                "washrooms": washrooms,
                "kitchens": kitchens,
                "bedrooms_requested": beds_req,
                "washrooms_requested": baths_req,
                "kitchens_requested": kits_req,
            },
            "pricing_notes": pricing_notes or None,
            "advanced_materials": {
                "wall_system": wall_sys,
                "flooring_system": floor_sys,
                "tile_spec": tile_spec_n,
                "roof_system": roof_sys,
                "waterproofing_scope": wp_scope,
                "exterior_wall_insulation": bool(exterior_wall_insulation),
                "window_glazing": glazing,
                "door_frame": frame,
                "plumbing_system": pl_sys or None,
                "add_ons": add_ons_norm,
                "roof_compare": roof_compare,
                "plumbing_compare": plumbing_compare,
            },
            "feasibility": {
                "plot_sqft": int(input_sqft),
                "max_bedrooms": int(mxb_max),
                "max_washrooms": int(mxbt_max),
                "max_kitchens": int(mxk_max),
                "clamped": bool(layout_clamped),
                "message": layout_msg or None,
            },
            "floor_factors": {
                **floor_factors,
                "foundation_phases": sorted(foundation_phases_f),
                "structural_per_floor_phases": sorted(structural_phases_f),
            },
            "breakdown": {
                "summary": {
                    "total_sqft": int(total_sqft),
                    "total_material": round(materials_total),
                    "total_labour": round(labour_total),
                    "subtotal": round(subtotal),
                    "contingency_pct": contingency_pct,
                    "contingency": round(contingency),
                    "grand_total": round(grand_total),
                }
            },
            "itemized_breakdown": breakdown_serial,
            "boq_tooltip_glossary": dict(BOQ_TOOLTIP_GLOSSARY),
            "category_breakdown": category_breakdown,
            "phase_breakdown": phase_breakdown,
            "floor_breakdown": floor_breakdown,
            "cost_per_sqft": round(cost_per_sqft),
            "cost_reduction_tips": tips,
            "llm_cost_tips": llm_tips or None,
            "comparison": comparison,
            "currency": "PKR",
            "warnings": warnings or None,
            "validation_warnings": None,
            "benchmark": bench_info,
            "rate_breakdown": {
                "quality_grade": grade,
            },
            "llm": {
                "enabled": bool(use_llm),
                "available": (use_llm and self.llm.is_available) if use_llm else None,
                "model": self.llm.model_name if use_llm else None,
            },
        }

    def estimate_phase_cost(
        self,
        phase_name: str,
        sqft: int,
        quality: str = "Standard",
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Estimate cost for a single construction phase."""
        PHASE_COSTS = {
            "foundation":           0.08,
            "grey_structure":       0.35,
            "roofing":              0.12,
            "plumbing":             0.10,
            "electrical":           0.08,
            "flooring":             0.15,
            "finishing":            0.12,
        }
        phase_name = phase_name.lower()
        if phase_name not in PHASE_COSTS:
            return {"status": "error", "message": f"Unknown phase: {phase_name}. Choose from: {list(PHASE_COSTS)}"}

        full = self.estimate_project_cost(sqft, 1, quality.lower(), city)
        grand = full["breakdown"]["summary"]["grand_total"]
        pct = PHASE_COSTS[phase_name]
        phase_cost = grand * pct

        return {
            "status": "success",
            "phase": phase_name,
            "sqft": sqft,
            "quality": quality,
            "city": city,
            "estimated_cost": round(phase_cost),
            "percentage_of_total": f"{int(pct * 100)}%",
            "cost_per_sqft": round(phase_cost / max(sqft, 1)),
        }

    # ── v2 extraction helpers (construction type + renovation scope) ──────────

    @staticmethod
    def _extract_construction_type(text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ("grey structure", "gray structure", "grey_structure", "structure only")):
            return "grey_structure"
        if any(k in t for k in ("renovation", "renovate", "remodel", "upgrade", "finishing only")):
            return "renovation"
        return "full_construction"

    @staticmethod
    def _extract_renovation_scope(text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ("paint", "repaint", "whitewash")):
            return "paint"
        if any(k in t for k in ("floor", "tile", "tiles", "marble")):
            return "flooring"
        if "kitchen" in t:
            return "kitchen"
        if any(k in t for k in ("bath", "bathroom", "washroom", "toilet")):
            return "bathroom"
        return None

    def estimate_material_cost(
        self,
        materials: List[Dict],
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Estimate cost for a list of specific materials."""
        city_mult = CITY_MULTIPLIER.get(city, 1.0)
        total = 0
        detailed = []

        for mat in materials:
            name = mat.get("name", "").lower().replace(" ", "_")
            qty = float(mat.get("quantity", 0))
            grade = (mat.get("grade") or "standard")
            if isinstance(grade, str):
                grade = grade.strip().lower()
                if grade == "luxury":
                    grade = "premium"
                if grade not in ("economy", "standard", "premium"):
                    grade = "standard"
            else:
                grade = "standard"
            unit_cost = self._unit_price_per_base(name, grade) * city_mult
            item_total = unit_cost * qty
            total += item_total
            detailed.append({
                "material": name,
                "quantity": qty,
                "unit": mat.get("unit", self._unit(name)),
                "unit_cost_pkr": round(unit_cost),
                "total_cost_pkr": round(item_total),
            })

        return {
            "status": "success",
            "city": city,
            "city_multiplier": city_mult,
            "materials": detailed,
            "total_estimated_cost": round(total),
            "currency": "PKR",
        }

    def compare_quality_tiers(
        self,
        sqft: int,
        floors: int = 1,
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Compare costs across Economy / Standard / Premium tiers."""
        comparison: Dict[str, Any] = {}
        for grade in ("economy", "standard", "premium"):
            r = self.estimate_project_cost(sqft, floors, grade, city)
            grand = r["breakdown"]["summary"]["grand_total"]
            comparison[grade.capitalize()] = {
                "total_cost": round(grand),
                "cost_per_sqft": r["cost_per_sqft"],
            }

        base = comparison["Economy"]["total_cost"]
        for g in comparison:
            diff = ((comparison[g]["total_cost"] - base) / max(base, 1)) * 100
            comparison[g]["percentage_diff_vs_economy"] = round(diff, 1)

        return {
            "status": "success",
            "project": {"sqft": sqft, "total_sqft": sqft * floors, "floors": floors, "city": city},
            "quality_comparison": comparison,
            "cost_variation": {
                "min": comparison["Economy"]["total_cost"],
                "max": comparison["Premium"]["total_cost"],
                "range": comparison["Premium"]["total_cost"] - comparison["Economy"]["total_cost"],
            },
        }

    # ── Universal (all building types) estimator ─────────────────────────────

    def _universal_rate_lookup(self, city_lower: str, key: str) -> Optional[float]:
        """
        Best-effort material rate lookup from Phase-2 pricing CSVs.
        Returns PKR per unit for logical keys:
          - cement_bag, steel_kg, sand_cft, crush_cft, brick
        """
        c = (city_lower or "").strip().lower()
        k = (key or "").strip().lower()

        # If we have materials_master/pricing_data, try semantic matching by name.
        if self.materials_master.empty or self.pricing_data.empty:
            return None

        name_col = "material_name" if "material_name" in self.materials_master.columns else None
        id_col = "material_id" if "material_id" in self.materials_master.columns else None
        if not name_col or not id_col:
            return None

        # Candidate name patterns (broad, but safe).
        patterns: List[str]
        if k == "cement_bag":
            patterns = ["cement", "opc", "portland"]
        elif k == "steel_kg":
            patterns = ["steel", "rebar", "sariya"]
        elif k == "sand_cft":
            patterns = ["sand", "ravi", "chenab"]
        elif k == "crush_cft":
            patterns = ["crush", "aggregate", "bajri", "gravel"]
        elif k == "brick":
            patterns = ["brick", "bricks", "eent"]
        else:
            return None

        # Find material_ids whose names match.
        mm = self.materials_master
        names = mm[name_col].astype(str).str.lower()
        mask = False
        for p in patterns:
            mask = mask | names.str.contains(p, na=False)
        mids = mm.loc[mask, id_col].astype(str).str.strip().tolist()
        if not mids:
            return None

        # Find the first mid that has a city rate avg.
        for mid in mids[:25]:
            avg = self._price_avg.get((mid, c))
            if avg and float(avg) > 0:
                return float(avg)

        return None

    def estimate_universal_boq(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal estimator: derive quantities from first principles and return
        structured JSON + validation warnings (no lump-sum packages).
        """
        from .universal_cost_engine import _RateLookup, estimate_universal

        rl = _RateLookup(get_rate_pkr=self._universal_rate_lookup)
        return estimate_universal(
            building_type=str(payload.get("building_type") or ""),
            city=str(payload.get("city") or "Lahore"),
            area=dict(payload.get("area") or {}),
            floors=payload.get("floors"),
            bhk=payload.get("bhk"),
            finishing_tier=payload.get("finishing_tier"),
            capacity=dict(payload.get("capacity") or {}),
            wall_height_ft=payload.get("wall_height_ft"),
            span_m=payload.get("span_m"),
            industrial_heavy=bool(payload.get("industrial_heavy") or False),
            rate_lookup=rl,
        )

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "prices_loaded": bool(self.prices),
            "price_items": len(self.prices.get("items", {})),
            "last_updated": self.prices.get("last_updated", "unknown"),
            "features": [
                "project_cost_estimation",
                "text_based_estimation",
                "room_based_additions",
                "itemized_breakdown",
                "cost_reduction_tips",
                "comparison_mode",
                "phase_cost_breakdown",
                "material_cost_calculator",
                "quality_tier_comparison",
                "city_based_pricing",
                "timeline_optimization",
                "marla_kanal_area_parsing",
            ],
        }

    # ── text parsers ─────────────────────────────────────────────────────────

    def parse_area(self, raw: str, city: Optional[str] = None) -> float:
        """
        Parse area from a string like '5 marla', '1 kanal', '2000 sqft', '185 sqm'.
        Returns area in sqft.
        Raises ValueError with user-readable message if unparseable.
        """
        raw = raw.strip().lower()
        # Try "number unit" or "number unit" with optional spaces
        for unit, factor in sorted(AREA_CONVERSIONS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}"
            m = re.search(pattern, raw)
            if m:
                value = float(m.group(1))
                if unit == "marla" and city:
                    marla_sqft = self._resolve_marla_sqft_per_marla(city)
                    return value * float(marla_sqft)
                return value * factor
        raise ValueError(f"Could not parse area from: '{raw}'. Use formats like '5 marla', '2000 sqft', '1 kanal'.")

    def _extract_area_from_text(self, text: str, city: Optional[str] = None) -> float:
        """Extract area from free text; default to 1000 sqft if not found."""
        text_lower = text.lower()
        for unit, factor in sorted(AREA_CONVERSIONS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}"
            m = re.search(pattern, text_lower)
            if m:
                value = float(m.group(1))
                if unit == "marla" and city:
                    marla_sqft = self._resolve_marla_sqft_per_marla(city)
                    return value * float(marla_sqft)
                return value * factor
        # Try standalone number (assume sqft)
        m = re.search(r"\b(\d{3,5})\b", text_lower)
        if m:
            return float(m.group(1))
        return 1000.0  # Sensible default

    def _extract_city(self, text: str) -> str:
        text_lower = text.lower()
        for city in CITY_MULTIPLIER:
            if city.lower() in text_lower:
                return city
        return "Lahore"

    def _extract_grade(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ("luxury", "luxurious", "premium", "high quality", "best quality")):
            return "premium"
        if any(w in text_lower for w in ("economy", "cheap", "low cost", "budget", "sasta")):
            return "economy"
        return "standard"

    def _extract_rooms(self, text: str) -> Dict[str, int]:
        rooms: Dict[str, int] = {"bedroom": 0, "washroom": 0, "kitchen": 0}
        patterns: Dict[str, List[str]] = {
            "bedroom":  [r"(\d+)\s*bed", r"(\d+)\s*room", r"(\d+)\s*kamr"],
            "washroom": [r"(\d+)\s*wash", r"(\d+)\s*bath", r"(\d+)\s*toilet", r"(\d+)\s*wc"],
            "kitchen":  [r"(\d+)\s*kitchen", r"(\d+)\s*rasoi"],
        }
        text_lower = text.lower()
        for room_type, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, text_lower)
                if m:
                    rooms[room_type] = int(m.group(1))
                    break
        return rooms

    # ── pricing helpers ───────────────────────────────────────────────────────

    def _price(self, item: str, grade: str) -> float:
        """
        Raw price from prices.json (per pack, exactly as recorded). Use
        `_unit_price_per_base()` when computing line totals — that respects
        the pack size declared in prices.json (e.g. "1000 bricks").
        """
        items = self.prices.get("items", {})
        entry = items.get(item, {})
        return float(entry.get(grade, entry.get("standard", 100)))

    def _pack_size(self, item: str) -> float:
        """
        Pack size for a price entry (default 1 → already per base unit).

        Only honours the PRICE_PACK_SIZE entry when the item actually exists
        in prices.json. If it doesn't, the price came from the fallback
        (100), which is already per base unit — applying a divisor there
        would silently shrink the total.
        """
        items = self.prices.get("items", {})
        if item not in items:
            return 1.0
        return float(PRICE_PACK_SIZE.get(item, 1.0))

    def _unit_price_per_base(self, item: str, grade: str) -> float:
        """
        Per-base-unit price (PKR per brick, per cft, etc.).
        This is the price you should multiply by raw quantity.
        """
        return self._price(item, grade) / self._pack_size(item)

    def _unit(self, item: str) -> str:
        """
        Display label for the *base* unit. For pack-priced items (bricks,
        sand-trolley etc.) we override the raw "1000 bricks" label so the
        line item reads naturally: quantity × per-base-unit price = total.
        """
        if item in PRICE_BASE_UNIT_LABEL:
            return PRICE_BASE_UNIT_LABEL[item]
        items = self.prices.get("items", {})
        return items.get(item, {}).get("unit", "unit")

    @staticmethod
    def _qty_basis_sentence(
        usage_type: Any,
        usage_ratio: float,
        total_sqft: float,
        marla_count: float,
    ) -> str:
        ut = str(usage_type or "").strip().lower()
        r = float(usage_ratio or 0.0)
        if r <= 0:
            return "from catalogue minimums / structural calibration (usage ratio not set)"
        if ut == "per_sqft":
            return f"usage_ratio {r:g} × {total_sqft:,.0f} sqft covered (per_sqft)"
        if ut == "per_marla":
            return f"usage_ratio {r:g} × {marla_count:.2f} marla-equivalent (per_marla)"
        if ut in ("per_house", "per_site"):
            return f"fixed allowance ratio {r:g} ({ut.replace('_', ' ')})"
        return f"usage_ratio {r:g} × {total_sqft:,.0f} sqft (scaled)"

    @staticmethod
    def _format_qty_display_number(qty: float, unit: str) -> str:
        """Quantity column for API/UI (unit is a separate field)."""
        ul = (unit or "").strip().lower()
        if ul in ("pcs", "piece", "pieces", "no", "nos"):
            return f"{max(0, int(round(float(qty)))):,}"
        if "kg" in ul:
            return f"{float(qty):,.0f}"
        if "bag" in ul:
            return f"{float(qty):,.0f}"
        if "cft" in ul:
            return f"{float(qty):,.1f}"
        if "sqft" in ul or "ft2" in ul:
            return f"{float(qty):,.0f}"
        if ul == "l" or "liter" in ul:
            return f"{float(qty):,.1f}"
        if ul == "m" or "meter" in ul:
            return f"{float(qty):,.0f}"
        if "ton" in ul:
            return f"{float(qty):,.2f}"
        return f"{float(qty):,.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _fmt_qty(qty: float, mat: str, unit: str = "") -> str:
        ul = (unit or "").strip().lower()
        if ul in ("pcs", "piece", "pieces", "no", "nos"):
            return CostEstimationModule._format_qty_display_number(qty, ul)
        if mat == "brick":
            return f"{qty:,.0f} units"
        if "_kg" in mat:
            return f"{qty:,.0f} kg"
        if "_bag" in mat:
            return f"{qty:,.0f} bags"
        if "_cft" in mat:
            return f"{qty:,.1f} cft"
        if "_sqft" in mat:
            return f"{qty:,.0f} sqft"
        if "_liter" in mat:
            return f"{qty:,.1f} liters"
        return CostEstimationModule._format_qty_display_number(qty, ul)

    @staticmethod
    def _parse_qty_num(qty_str: str) -> float:
        """Extract the numeric part from a formatted quantity string."""
        m = re.search(r"[\d,]+(?:\.\d+)?", qty_str.replace(",", ""))
        return float(m.group()) if m else 0.0

    @staticmethod
    def _timeline_factor(months: Optional[int]) -> float:
        if months is None:
            return 1.0
        if months <= 3:
            return TIMELINE_FACTOR["rush"]
        if months <= 6:
            return TIMELINE_FACTOR["standard"]
        return TIMELINE_FACTOR["extended"]

    # ── cost reduction tips ───────────────────────────────────────────────────

    def _generate_tips(
        self,
        breakdown: Dict[str, LineItem],
        grand_total: float,
        grade: str,
        city: str,
    ) -> List[str]:
        """
        Generate short, actionable tips.

        v2: the cost engine is driven by `materials_master.csv` + `pricing_data.csv`,
        so breakdown keys are human names, not `prices.json` keys. Keep tips generic
        and market-aligned (Pakistan), never dependent on price keys existing.
        """
        tips: List[str] = []
        grade = (grade or "standard").lower()
        if grade == "luxury":
            grade = "premium"

        # Highlight top cost drivers (by total) for relevance.
        top = sorted(
            [(k, v.total) for k, v in breakdown.items() if getattr(v, "total", 0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        if top:
            drivers = ", ".join(k for k, _ in top)
            tips.append(f"Your biggest cost drivers are: {drivers}. Compare suppliers for these first.")

        # Grade guidance
        if grade == "premium":
            tips.append("Use Premium only for wet areas/exterior; Standard is usually enough for internal walls to save cost.")
        if grade == "economy":
            tips.append("Economy grade is cost-efficient. Prioritise quality checks (fresh cement date, brick strength) to avoid rework.")

        # Market actions
        tips.append(f"In {city}, get at least 3 quotes before bulk purchasing cement/steel/tiles.")
        tips.append("Keep a 10% contingency buffer for price fluctuations and site surprises.")
        return tips[:4]

    def _load_prices(self, prices_path: str) -> None:
        """Load prices.json; log warning if file is stale."""
        checked_paths = [prices_path, "prices.json"]
        for path in checked_paths:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.prices = json.load(f)
                    # Staleness check
                    last_updated_str = self.prices.get("last_updated", "")
                    if last_updated_str:
                        try:
                            last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
                            age = (date.today() - last_updated).days
                            if age > PRICE_STALENESS_DAYS:
                                logger.warning(
                                    "prices.json is %d days old (last updated %s). "
                                    "Consider updating prices for accuracy.",
                                    age, last_updated_str,
                                )
                        except ValueError:
                            pass
                    logger.info("Prices loaded from %s (%d items)", path, len(self.prices.get("items", {})))
                    return
                except Exception as exc:
                    logger.warning("Failed to load prices from %s: %s", path, exc)

        logger.error("prices.json not found — using fallback prices of 100 PKR per unit")
        self.prices = {"items": {}}

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {"status": "error", "message": msg}
