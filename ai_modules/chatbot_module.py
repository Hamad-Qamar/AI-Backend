"""
ChatBotModule v3 — Product-Aware, Task-Oriented, UX-First Chatbot.

v3 improvements over v2:
  - Detailed intent taxonomy (21 intents across 6 categories).
  - Richer entity extraction: area, city, floors, BHK, finishing tier, building type.
  - Cleaner dual-path routing with single clarifying question policy.
  - Structured quick-reply suggestions per intent.
  - Urdu/Roman-Urdu friendly responses.
  - Unit conversion helpers (marla ↔ sqft ↔ kanal).
  - Quantity calculator for cement, bricks, tiles, paint, steel.
  - Vendor info intent with mock fetch layer.
  - Robust off-topic guard and safety layer.
  - Navigation actions always returned for UI deep-linking.
  - Response always includes products, steps, navigation_actions (never None).
"""

import json
import logging
import os
import re
import uuid
import datetime as _dt
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import faiss
import numpy as np

from .cost_assistant_knowledge import load_cost_estimation_assistant_knowledge
from .query_preprocessor import QueryPreprocessor
from .intent_detector import IntentDetector, IntentResult
from .llm_helper import LLMHelper
from .shared_models import get_embedding_model

if TYPE_CHECKING:
    from .recommendation_module import RecommendationModule
    from .cost_estimation_module import CostEstimationModule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

CHATBOT_SYSTEM_PROMPT_COST_AND_REC = """
You are BuildHive Assistant — an AI embedded in BuildHive, Pakistan's construction
marketplace. You help with:
  (1) Cost Estimation  — prices, budgets, breakdowns, totals, "how much"
  (2) Recommendations  — "best for me", ranked options, what to pick, alternatives
  (3) Material search  — quantities, specs, comparisons, filtering
  (4) Platform navigation — features, tabs, account actions
  (5) Construction FAQ — phases, terminology, timelines (Pakistani context)
  (6) Vendor info      — profiles, ratings, contact, onboarding

ROUTING RULES
─────────────
- Pure cost question   → Cost Estimation path only.
- Pure choice question → Recommendation path only.
- Both in one message  → run both; default order cost-first unless user says
  "recommendations first" or a rec-trigger appears before a cost-trigger.
- "Best deal" / "cheapest and best" → treat as both; label sections clearly.
- Never force cost/rec framing on account, navigation, or general FAQ queries.

COST ANSWERS
────────────
- Anchor to Cost Estimation module output; never invent figures.
- Refer to built-up footprint as **total AREA Covered** (not "plot area").
- For per-floor rates: **per sq ft of AREA Covered**.
- For roofing: **per sq ft of Roof Area** (note ~225 sq ft ≈ 1 marla roof).
- If only plot size given, ask for AREA Covered (built-up) before estimating.
- Always show: Summary line → Breakdown → Assumptions → Confidence disclaimer.
- Exclusions to always mention: land, design/engineering fees, NOCs, utility
  connections, furniture/equipment, contractor profit.

RECOMMENDATION ANSWERS
──────────────────────
- Anchor to Recommendation module; respect catalog and tier filters.
- Show: Top pick (1–3) → Why it fits → Trade-offs → Next step.
- Summarise rationale in plain language; no raw internal weights.

TONE & FORMAT
─────────────
- Professional, concise, friendly. No hype, no hallucinated prices.
- Use Urdu/Roman-Urdu terms naturally (marla, kanal, grey structure, etc.).
- Short lead-in then structured body; headings + bullets; mobile-first length.
- One focused clarifying question when data is missing — never more than two.
- Always end with a clear next step or navigation action.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTENT TAXONOMY  (21 intents, 6 categories)
# ─────────────────────────────────────────────────────────────────────────────

# ── Cost Estimation ───────────────────────────────────────────────────────────
_COST_TRIGGERS: Tuple[str, ...] = (
    "how much", "how many", "estimated", "expense", "estimate",
    "cost ", " cost", "price", "total price", "budget", "total cost",
    "cheaper", "cost breakdown", "what will it cost", "per sqft",
    "per sq ft", "per marla", "in a marla", "grand total",
    "pkr", "rupees", " rs ", "rs.",
)

# ── Recommendation ────────────────────────────────────────────────────────────
_REC_TRIGGERS: Tuple[str, ...] = (
    "recommend", "recomend", "recomned", "suggest", "sugest",
    "best option", "what should i choose", "top picks", "which material",
    "which should i", "what to pick", "alternatives", "pick between",
    "compare ", "materials for", "products for", "items for",
    "material list", "what materials", "what material", "material for",
)

# ── Material purchase / search ────────────────────────────────────────────────
_PURCHASE_ACTIONS: Tuple[str, ...] = (
    "buy", "purchase", "order", "get", "find", "where to",
    "how to buy", "how to get", "price of", "cost of", "best",
    "recommend", "suggest", "compare", "source", "procure",
    "kahan se", "khareedna", "milega",
)

# ── Vendor intents ────────────────────────────────────────────────────────────
_VENDOR_TRIGGERS: Tuple[str, ...] = (
    "vendor", "seller", "supplier", "contact seller", "is this verified",
    "seller rating", "vendor profile", "register as vendor",
    "list my store", "sell on buildhive", "become a vendor",
)

# ── Quantity calculator ───────────────────────────────────────────────────────
_QTY_TRIGGERS: Tuple[str, ...] = (
    "how many bags", "how many bricks", "how many tiles",
    "how many litres", "how much cement", "how much steel",
    "quantity of", "bags needed", "bricks needed", "tiles needed",
    "calculate quantity", "material quantity",
)

# ── Unit conversion ───────────────────────────────────────────────────────────
_UNIT_TRIGGERS: Tuple[str, ...] = (
    "marla to sqft", "sqft to marla", "kanal to marla",
    "marla to kanal", "convert marla", "convert sqft",
    "how many sqft in", "how many marla in",
)

# ── Ambiguous "best deal" ─────────────────────────────────────────────────────
_AMBIGUOUS_DEAL: Tuple[str, ...] = (
    "best deal", "best value", "cheapest and best", "best bang",
)


def _user_wants_cost_estimate(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _COST_TRIGGERS)


def _user_wants_recommendation(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _REC_TRIGGERS)


def _user_ambiguous_deal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _AMBIGUOUS_DEAL)


def _user_wants_vendor_info(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _VENDOR_TRIGGERS)


def _user_wants_quantity_calc(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _QTY_TRIGGERS)


def _user_wants_unit_conversion(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _UNIT_TRIGGERS)


def _dual_module_order(text: str) -> str:
    """Return 'cost_first' or 'rec_first' when both modules run."""
    t = (text or "").lower()
    if any(p in t for p in (
        "recommendation first", "recommendations first",
        "materials first", "options first", "then cost", "then price",
    )):
        return "rec_first"
    rec_positions = [t.find(m) for m in ("recommend", "suggest") if m in t]
    rec_idx = min(rec_positions) if rec_positions else 10 ** 6
    cost_markers = ("how much", "estimate", "cost", "price", "budget")
    cost_positions = [t.find(m) for m in cost_markers if m in t]
    cost_idx = min(cost_positions) if cost_positions else 10 ** 6
    if rec_idx < 10 ** 6 and cost_idx < 10 ** 6:
        return "rec_first" if rec_idx < cost_idx else "cost_first"
    return "cost_first"


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_MATERIAL_ENTITIES: Dict[str, str] = {
    "cement":        "Raw Materials",
    "bricks":        "Raw Materials",
    "brick":         "Raw Materials",
    "sand":          "Raw Materials",
    "steel":         "Raw Materials",
    "rebar":         "Raw Materials",
    "gravel":        "Raw Materials",
    "crush":         "Raw Materials",
    "tiles":         "Flooring Materials",
    "tile":          "Flooring Materials",
    "marble":        "Flooring Materials",
    "granite":       "Flooring Materials",
    "paint":         "Paint & Finishing",
    "primer":        "Paint & Finishing",
    "wood":          "Wood & Carpentry",
    "timber":        "Wood & Carpentry",
    "door":          "Doors & Windows",
    "window":        "Doors & Windows",
    "pipe":          "Plumbing - Pipes & Fittings",
    "plumbing":      "Plumbing - Pipes & Fittings",
    "faucet":        "Plumbing - Taps & Fixtures",
    "tap":           "Plumbing - Taps & Fixtures",
    "wire":          "Electrical - Wiring & Cables",
    "cable":         "Electrical - Wiring & Cables",
    "switch":        "Electrical - Switchgear",
    "switchgear":    "Electrical - Switchgear",
    "light":         "Electrical - Lighting & Fixtures",
    "fan":           "Electrical - Lighting & Fixtures",
    "waterproofing": "Chemicals & Treatments",
    "roofing":       "Roofing Materials",
    "insulation":    "Insulation & Ceilings",
    "sanitary":      "Sanitary Items",
    "kitchen":       "Kitchen Materials",
}

# Pakistani cities with PKR rate cards
_SUPPORTED_CITIES = (
    "lahore", "karachi", "islamabad", "rawalpindi",
    "faisalabad", "multan", "peshawar", "quetta",
)

_BUILDING_TYPES = (
    "house", "home", "apartment", "villa", "farmhouse",
    "servant quarter", "shop", "office", "plaza",
    "school", "hospital", "mosque", "warehouse", "factory",
)


def _extract_area_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(marla|kanal|sqft|sq ft|square feet|sqm|m2)", t
    )
    if not m:
        return None
    val = m.group(1)
    unit = (
        m.group(2)
        .replace("square feet", "sqft")
        .replace("sq ft", "sqft")
        .replace("m2", "sqm")
    )
    return f"{val} {unit}"


def _extract_city_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    for c in _SUPPORTED_CITIES:
        if c in t:
            return c.title()
    return None


def _extract_building_type_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    for bt in _BUILDING_TYPES:
        if bt in t:
            return bt
    return None


def _extract_floors_hint(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = re.search(r"\b(\d+)\s*(floors?|storeys?|stories?)\b", t)
    if m:
        return int(m.group(1))
    if "double storey" in t or "double story" in t:
        return 2
    if "single storey" in t or "single story" in t:
        return 1
    return None


def _extract_bhk_hint(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = re.search(r"\b(\d+)\s*bhk\b", t)
    return int(m.group(1)) if m else None


def _extract_finishing_tier(text: str) -> Optional[str]:
    t = (text or "").lower()
    for tier in ("economy", "standard", "premium", "luxury"):
        if tier in t:
            return tier.title()
    return None


def _marla_to_sqft(marla: float, city: Optional[str] = None) -> float:
    """Standard: 1 marla = 272 sqft. City overrides can be added."""
    city_standards: Dict[str, float] = {}  # extend as needed
    factor = city_standards.get((city or "").lower(), 272.0)
    return marla * factor


def _kanal_to_marla(kanal: float) -> float:
    return kanal * 20.0


# ─────────────────────────────────────────────────────────────────────────────
# QUANTITY CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_quantity(material: str, area_sqft: float) -> str:
    """
    Return a human-readable quantity estimate for a given material and sqft area.
    All formulas are standard Pakistani construction benchmarks.
    """
    m = material.lower()

    if "cement" in m:
        bags = round(area_sqft * 0.4)
        return (
            f"For **{area_sqft:,.0f} sqft**, you need approximately **{bags} bags of cement**.\n"
            "*(Formula: ~0.4 bags/sqft for a typical concrete mix; varies by slab thickness.)*\n\n"
            "💡 Order 10% extra for wastage and re-work."
        )
    if "brick" in m:
        bricks = round(area_sqft * 9)
        return (
            f"For **{area_sqft:,.0f} sqft** of wall area, you need approximately **{bricks:,} bricks**.\n"
            "*(Formula: ~9 bricks/sqft for a 4.5\" single-brick wall.)*\n\n"
            "💡 Order 10–15% extra for cuts and breakage."
        )
    if "tile" in m or "marble" in m or "granite" in m:
        with_wastage = round(area_sqft * 1.15)
        return (
            f"For **{area_sqft:,.0f} sqft**, order tiles for **{with_wastage:,} sqft** (includes 15% wastage).\n"
            "*(Formula: area × 1.15 for cuts, grout gaps, and breakage.)*\n\n"
            "💡 Always request a sample batch before the full order."
        )
    if "paint" in m:
        litres = round(area_sqft / 12)
        return (
            f"For **{area_sqft:,.0f} sqft**, you need approximately **{litres} litres of paint**.\n"
            "*(Formula: 1 litre covers ~12 sqft with 2 coats.)*\n\n"
            "💡 Add 1 coat of primer before topcoat for best durability."
        )
    if "steel" in m or "rebar" in m:
        kg = round(area_sqft * 3.75)
        return (
            f"For **{area_sqft:,.0f} sqft** of slab, you need approximately **{kg:,} kg of steel rebar**.\n"
            "*(Formula: ~3.5–4 kg/sqft for standard residential slab.)*\n\n"
            "💡 Consult a structural engineer for load-bearing floors."
        )
    if "sand" in m or "gravel" in m or "crush" in m:
        cft = round(area_sqft * 0.5)
        return (
            f"For **{area_sqft:,.0f} sqft**, estimate approximately **{cft:,} cft of {material}**.\n"
            "*(Formula: rough estimate; varies by mix design and depth.)*"
        )

    return (
        f"I don't have a specific formula for **{material}** yet.\n"
        "Please visit the **Cost Estimator** tab for a full material breakdown, "
        "or tell me the exact use case and I'll do my best to help."
    )


# ─────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _handle_unit_conversion(text: str) -> Optional[str]:
    """Detect and compute common Pakistani area unit conversions."""
    t = text.lower()

    # marla → sqft
    m = re.search(r"(\d+(?:\.\d+)?)\s*marla\s+to\s+sqft", t)
    if m:
        marla = float(m.group(1))
        sqft = _marla_to_sqft(marla)
        return (
            f"**{marla:g} Marla = {sqft:,.0f} sqft**\n"
            "*(Standard: 1 Marla = 272 sqft)*"
        )

    # sqft → marla
    m = re.search(r"(\d+(?:\.\d+)?)\s*sqft\s+to\s+marla", t)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:sq ft|square feet)\s+(?:in|to)\s+marla", t)
    if m:
        sqft = float(m.group(1))
        marla = sqft / 272.0
        return (
            f"**{sqft:,.0f} sqft = {marla:.2f} Marla**\n"
            "*(Standard: 272 sqft = 1 Marla)*"
        )

    # kanal → marla
    m = re.search(r"(\d+(?:\.\d+)?)\s*kanal\s+to\s+marla", t)
    if m:
        kanal = float(m.group(1))
        marla = _kanal_to_marla(kanal)
        sqft = _marla_to_sqft(marla)
        return (
            f"**{kanal:g} Kanal = {marla:,.0f} Marla = {sqft:,.0f} sqft**\n"
            "*(1 Kanal = 20 Marla = 5,440 sqft)*"
        )

    # kanal → sqft
    m = re.search(r"(\d+(?:\.\d+)?)\s*kanal\s+to\s+sqft", t)
    if m:
        kanal = float(m.group(1))
        sqft = _marla_to_sqft(_kanal_to_marla(kanal))
        return (
            f"**{kanal:g} Kanal = {sqft:,.0f} sqft**\n"
            "*(1 Kanal = 20 Marla × 272 sqft)*"
        )

    # "how many sqft in X marla"
    m = re.search(r"how many sqft (?:in|is)\s+(\d+(?:\.\d+)?)\s*marla", t)
    if m:
        marla = float(m.group(1))
        sqft = _marla_to_sqft(marla)
        return (
            f"**{marla:g} Marla = {sqft:,.0f} sqft**\n"
            "*(1 Marla = 272 sqft)*"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _navigation_actions_list(intent: str) -> List[Dict[str, Any]]:
    """Return a single, context-relevant navigation action for the UI."""
    if intent == "Recommendation":
        return [{
            "label": "View Recommendations",
            "target_module": "recommendation",
            "deep_link": "/recommendations",
            "optional": True,
        }]
    if intent == "Estimation":
        return [{
            "label": "Go to Cost Estimator",
            "target_module": "estimation",
            "deep_link": "/estimator",
            "optional": True,
        }]
    if intent == "Mixed":
        return [{
            "label": "Go to Cost Estimator",
            "target_module": "estimation",
            "deep_link": "/estimator",
            "optional": True,
        }]
    if intent == "Vendor":
        return [{
            "label": "Browse Vendors",
            "target_module": "marketplace",
            "deep_link": "/vendors",
            "optional": True,
        }]
    if intent == "Materials":
        return [{
            "label": "Browse Materials",
            "target_module": "marketplace",
            "deep_link": "/marketplace",
            "optional": True,
        }]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER TEMPLATE  (structured output for UI parsing)
# ─────────────────────────────────────────────────────────────────────────────

def _router_template(
    *,
    intent: str,
    inputs_used: Dict[str, Any],
    result_summary: str,
    warnings: Optional[List[str]] = None,
    include_nav: bool = True,
) -> str:
    warn_lines = [f"- {w}" for w in (warnings or []) if w]
    warn_block = "\n".join(warn_lines) if warn_lines else "- None"

    inputs_lines = [
        f"- **{k}**: {v}"
        for k, v in (inputs_used or {}).items()
        if v not in (None, "")
    ]
    inputs_block = "\n".join(inputs_lines) if inputs_lines else "- (none provided)"

    nav_json = ""
    if include_nav:
        actions = _navigation_actions_list(intent)
        if actions:
            nav_json = json.dumps({"navigation_actions": actions}, ensure_ascii=False, indent=2)

    body = (
        f"## Intent\n**{intent}**\n\n"
        f"## Inputs used\n{inputs_block}\n\n"
        f"## Result\n{result_summary}\n\n"
        f"## Warnings / Exclusions\n{warn_block}\n"
    )
    if nav_json:
        body += f"\n## Optional navigation\n{nav_json}\n"
    return body


# ─────────────────────────────────────────────────────────────────────────────
# RESULT FORMATTERS  (compact chat-friendly summaries)
# ─────────────────────────────────────────────────────────────────────────────

def _format_estimation_summary_for_chat(
    cost_r: Dict[str, Any],
) -> Tuple[str, List[str]]:
    warns: List[str] = []
    if not isinstance(cost_r, dict) or cost_r.get("status") not in ("success", "ok"):
        return "- I couldn't generate an estimate from the provided inputs.", warns

    summary = ((cost_r.get("breakdown") or {}).get("summary") or {})
    grand = summary.get("grand_total")
    cps = cost_r.get("cost_per_sqft")
    if cost_r.get("warnings"):
        warns.extend([str(w) for w in (cost_r.get("warnings") or [])][:6])

    parts: List[str] = []
    if grand is not None:
        parts.append(f"Estimated total: **PKR {int(grand):,}**")
    if cps is not None:
        parts.append(f"(**PKR {int(cps):,}/sqft**)")
    msg = " ".join(parts) if parts else "- I couldn't generate an estimate."
    return msg, warns


def _format_recommendation_summary_for_chat(
    rec_r: Dict[str, Any],
) -> Tuple[str, List[str]]:
    warns: List[str] = []
    if not isinstance(rec_r, dict) or rec_r.get("status") != "success":
        return "- I couldn't generate recommendations from the provided inputs.", warns

    recs = rec_r.get("recommendations") or rec_r.get("categories") or {}
    top_name = top_price = top_cat = None
    if isinstance(recs, dict) and recs:
        for cat, items in recs.items():
            if items:
                top = items[0] or {}
                top_cat = str(cat)
                top_name = top.get("item_name") or top.get("name") or top.get("title")
                top_price = top.get("market_price_pkr") or top.get("final_price_pkr")
                break
    if top_name:
        if top_price is not None:
            return (
                f"Top pick: **{top_name}** ({top_cat}) — ≈ **PKR {int(top_price):,}**.",
                warns,
            )
        return f"Top pick: **{top_name}** ({top_cat}).", warns
    return "- I couldn't generate recommendations from the provided inputs.", warns


# ─────────────────────────────────────────────────────────────────────────────
# PURCHASE STEPS  (step-by-step guides per material)
# ─────────────────────────────────────────────────────────────────────────────

_PURCHASE_STEPS: Dict[str, List[str]] = {
    "default": [
        "Search for the material using the BuildHive search bar.",
        "Filter results by city, quality grade, and your budget.",
        "Compare at least 3 suppliers to get the best price.",
        "Check supplier ratings and verification badges.",
        "Request a bulk-order quote for large quantities.",
        "Confirm lead time and delivery terms before ordering.",
    ],
    "cement": [
        "Decide the grade: OPC 43 (general), OPC 53 (high-strength), SRC (sulphate resistant).",
        "Calculate quantity: ~0.4 bags per sqft of construction area.",
        "Compare prices across brands (DG Khan, Lucky, Cherat, Askari).",
        "Order in full truck loads (100–200 bags) for best bulk pricing.",
        "Check manufacturing date — cement older than 3 months loses strength.",
        "Arrange covered, dry storage at the site before delivery.",
    ],
    "bricks": [
        "Choose grade: A-grade (kiln-fired, uniform) for load-bearing walls.",
        "Estimate quantity: ~9 bricks per sqft of wall area.",
        "Inspect a sample batch — look for uniform size, no cracks.",
        "Source locally (within 50 km) to save transport cost.",
        "Order at least 10% extra for breakage and cuts.",
        "Confirm price is per 1,000 bricks, not per piece.",
    ],
    "tiles": [
        "Measure total floor and wall area to tile.",
        "Add 15% to your measurement for cuts and wastage.",
        "Choose grade: A-grade for uniform thickness and shade.",
        "Compare prices per sqft including adhesive and grout.",
        "Request a sample tile before placing the full order.",
        "Hire a certified tiler — incorrect laying voids warranty.",
    ],
    "paint": [
        "Choose finish: matte for walls, semi-gloss for trims and kitchens.",
        "Calculate: 1 litre covers ~12 sqft (2 coats).",
        "Apply one coat of primer before topcoat for durability.",
        "Compare 4-litre tins vs 20-litre drums — drums are cheaper per litre.",
        "Use lead-free paints for interiors.",
        "For exterior walls, choose a weather-shield formula.",
    ],
    "steel": [
        "Specify grade: Grade 40 (general), Grade 60 (high-strength columns).",
        "Estimate: ~3.5–4 kg per sqft of slab area.",
        "Buy from PSQCA-certified mills (Ittefaq, Agha Steel, Amreli).",
        "Check mill test certificates (MTCs) before delivery.",
        "Inspect for rust — light surface rust is acceptable; flaking is not.",
        "Store off the ground on wooden spacers to prevent ground moisture.",
    ],
    "marble": [
        "Choose type: local marble (cheaper) vs imported (Italian/Spanish).",
        "Measure area and add 15% for cuts and wastage.",
        "Ask for polished vs honed finish depending on use case.",
        "Ensure consistent lot number for uniform colour across rooms.",
        "Hire an experienced marble-layer — poor laying causes cracking.",
        "Seal marble after laying to protect against stains.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM RESPONSES  (FR1–FR16 structured templates)
# ─────────────────────────────────────────────────────────────────────────────

PLATFORM_RESPONSES: List[Dict[str, Any]] = [
    # ── Account & Security ────────────────────────────────────────────────────
    {
        "keywords": ["change password", "update password", "reset password", "new password"],
        "domain": "account_security",
        "response": (
            "**To change your password:**\n"
            "1. Go to **Account Settings** (top-right icon)\n"
            "2. Click **Security**\n"
            "3. Select **Change Password**\n"
            "4. Enter your current password, then your new one\n"
            "5. Click **Save Changes**\n\n"
            "💡 Use a mix of letters, numbers, and symbols for a strong password."
        ),
    },
    {
        "keywords": ["change email", "update email", "email settings"],
        "domain": "account_security",
        "response": (
            "**To update your email address:**\n"
            "1. Go to **Account Settings → Profile**\n"
            "2. Edit the **Email** field\n"
            "3. Verify the new email via the confirmation link sent to it\n\n"
            "💡 Keep your email up to date to receive order and notification alerts."
        ),
    },
    {
        "keywords": ["delete account", "close account", "remove account"],
        "domain": "account_security",
        "response": (
            "**To delete your account:**\n"
            "1. Go to **Account Settings → Danger Zone**\n"
            "2. Click **Delete Account**\n"
            "3. Confirm by entering your password\n\n"
            "⚠️ This action is permanent and cannot be undone."
        ),
    },
    {
        "keywords": ["two factor", "2fa", "two-factor", "enable 2fa", "authenticator"],
        "domain": "account_security",
        "response": (
            "**To enable Two-Factor Authentication (2FA):**\n"
            "1. Go to **Account Settings → Security**\n"
            "2. Toggle **Two-Factor Authentication** ON\n"
            "3. Scan the QR code with Google Authenticator or Authy\n"
            "4. Enter the 6-digit code to confirm\n\n"
            "💡 2FA significantly increases your account security."
        ),
    },
    # ── Marketplace ───────────────────────────────────────────────────────────
    {
        "keywords": ["place order", "buy product", "purchase product", "order material",
                     "how to order", "add to cart", "checkout"],
        "domain": "marketplace",
        "response": (
            "**To place an order:**\n"
            "1. Find the product in **Marketplace**\n"
            "2. Click **Add to Cart**\n"
            "3. Go to **Cart** (top-right icon)\n"
            "4. Review items and click **Checkout**\n"
            "5. Enter delivery address and payment details\n"
            "6. Confirm your order\n\n"
            "💡 Compare at least 3 suppliers before ordering to get the best price."
        ),
    },
    {
        "keywords": ["track order", "order status", "where is my order", "delivery status"],
        "domain": "marketplace",
        "response": (
            "**To track your order:**\n"
            "1. Go to **Buyer Dashboard → Active Orders**\n"
            "2. Find your order and click **Track**\n"
            "3. View real-time delivery status\n\n"
            "💡 You'll also receive SMS/email updates at each delivery stage."
        ),
    },
    {
        "keywords": ["cancel order", "return order", "refund"],
        "domain": "marketplace",
        "response": (
            "**To cancel or return an order:**\n"
            "1. Go to **Buyer Dashboard → Active Orders**\n"
            "2. Select the order\n"
            "3. Click **Cancel** (not shipped) or **Return** (delivered)\n"
            "4. Select a reason and submit\n\n"
            "💡 Cancellations are instant if the order hasn't been dispatched."
        ),
    },
    # ── Seller Module ─────────────────────────────────────────────────────────
    {
        "keywords": ["add listing", "create listing", "add product", "list product",
                     "new listing", "post product"],
        "domain": "listings",
        "response": (
            "**To add a new product listing:**\n"
            "1. Go to **Seller Dashboard → Listing Management → Add New Listing**\n"
            "2. Fill in: Product Name, Category, Price, Quality Grade, City\n"
            "3. Upload photos and set stock quantity\n"
            "4. Click **Publish**\n\n"
            "💡 Listings with photos and detailed specs get 3× more views."
        ),
    },
    {
        "keywords": ["edit listing", "update listing", "update product", "change price",
                     "update price", "edit product"],
        "domain": "listings",
        "response": (
            "**To edit a listing:**\n"
            "1. Go to **Seller Dashboard → Listing Management**\n"
            "2. Find the product and click **Edit**\n"
            "3. Update price, stock, description, or photos\n"
            "4. Click **Save Changes**\n\n"
            "💡 Keep prices updated — stale prices lower your ranking in search results."
        ),
    },
    {
        "keywords": ["delete listing", "remove listing", "remove product", "deactivate listing"],
        "domain": "listings",
        "response": (
            "**To remove a listing:**\n"
            "1. Go to **Seller Dashboard → Listing Management**\n"
            "2. Click **Deactivate** (hides it) or **Delete** (permanent)\n\n"
            "⚠️ Deleted listings cannot be recovered. Use Deactivate if temporary."
        ),
    },
    {
        "keywords": ["seller dashboard", "my dashboard", "sales analytics", "view sales",
                     "how to sell"],
        "domain": "dashboard",
        "response": (
            "**Your Seller Dashboard includes:**\n"
            "• **Listing Management** — add, edit, deactivate products\n"
            "• **Order Management** — view incoming orders\n"
            "• **Sales Analytics** — revenue, top products, city-wise sales\n"
            "• **Financial Overview** — earnings and payouts\n\n"
            "Access: **Profile icon (top-right) → Seller Dashboard**."
        ),
    },
    # ── AI Tools ──────────────────────────────────────────────────────────────
    {
        "keywords": ["recommendation", "recommend materials", "material list", "ai recommend",
                     "suggest materials", "material recommendation"],
        "domain": "ai_recommendation",
        "response": (
            "**To use AI Material Recommendations:**\n"
            "1. Click the **Recommendation System** tab on this page\n"
            "2. Describe your project (e.g. '5 marla house in Lahore')\n"
            "3. Enter area, city, and quality preference\n"
            "4. Click **Get AI Recommendations**\n\n"
            "You'll get a full material list across 7+ categories with quantities and costs."
        ),
    },
    {
        "keywords": ["cost estimate", "estimate cost", "project cost", "how much will it cost",
                     "building cost", "construction cost", "cost estimator"],
        "domain": "cost_estimation",
        "response": (
            "**To estimate your project cost:**\n"
            "1. Click the **Cost Estimation** tab on this page\n"
            "2. Enter: Area, Quality grade, City, Floors\n"
            "3. Click **Calculate Cost Estimate**\n\n"
            "You'll see: Total Cost, Cost per sqft, Itemized breakdown, Pie chart, and AI cost-saving tips."
        ),
    },
    # ── Chat & Messaging ──────────────────────────────────────────────────────
    {
        "keywords": ["message seller", "contact seller", "chat with seller",
                     "send message", "inbox", "messages"],
        "domain": "chat_system",
        "response": (
            "**To message a seller:**\n"
            "1. Open the product listing\n"
            "2. Click **Contact Seller** or **Message**\n"
            "3. Type your message and send\n\n"
            "All conversations: **Top navigation → Messages icon**.\n\n"
            "💡 Ask about bulk pricing, lead times, and certifications before ordering."
        ),
    },
    # ── Reviews ───────────────────────────────────────────────────────────────
    {
        "keywords": ["leave review", "write review", "rate seller", "rate product", "add review"],
        "domain": "reviews",
        "response": (
            "**To leave a review:**\n"
            "1. Go to **Buyer Dashboard → Order History**\n"
            "2. Find the completed order\n"
            "3. Click **Leave a Review**\n"
            "4. Rate (1–5 stars) and write your feedback\n"
            "5. Submit\n\n"
            "💡 Only verified buyers can leave reviews."
        ),
    },
    # ── Freelancer ────────────────────────────────────────────────────────────
    {
        "keywords": ["freelancer", "register as freelancer", "post service",
                     "hire freelancer", "contractor", "service provider"],
        "domain": "freelancer",
        "response": (
            "**BuildHive Freelancer System:**\n"
            "• **Register as Freelancer**: Profile → Switch to Freelancer → Add services & portfolio\n"
            "• **Post a service**: Freelancer Dashboard → Add Service\n"
            "• **Get hired**: Buyers browse and book your service directly\n"
            "• **Hire a freelancer**: Marketplace → Services tab → Filter by skill/city\n\n"
            "💡 Complete your portfolio to appear higher in search results."
        ),
    },
    # ── Dashboard ─────────────────────────────────────────────────────────────
    {
        "keywords": ["buyer dashboard", "my orders", "my purchases", "order history"],
        "domain": "dashboard",
        "response": (
            "**Your Buyer Dashboard includes:**\n"
            "• **Active Orders** — track current deliveries\n"
            "• **Order History** — past purchases and receipts\n"
            "• **Saved Items** — your wishlist\n"
            "• **Financial Overview** — spending summary\n\n"
            "Access: **Profile icon (top-right) → Buyer Dashboard**."
        ),
    },
    # ── Notifications & Projects ──────────────────────────────────────────────
    {
        "keywords": ["notifications", "alerts", "enable notifications", "notification settings"],
        "domain": "projects_notifications",
        "response": (
            "**To manage notifications:**\n"
            "1. Go to **Account Settings → Notifications**\n"
            "2. Toggle: Order updates, Price alerts, Messages, Promotions\n"
            "3. Choose delivery: Email, SMS, or In-App\n\n"
            "💡 Enable Price Alerts to get notified when a product drops in price."
        ),
    },
    {
        "keywords": ["create project", "project management", "my projects",
                     "new project", "save project"],
        "domain": "projects_notifications",
        "response": (
            "**To create a project:**\n"
            "1. Go to **Buyer Dashboard → My Projects**\n"
            "2. Click **New Project**\n"
            "3. Enter project name, area, type, and budget\n"
            "4. Save and link materials / estimates to it\n\n"
            "💡 You can save AI Recommendations directly to a project."
        ),
    },
    # ── General Platform ──────────────────────────────────────────────────────
    {
        "keywords": ["register", "sign up", "create account", "how to register"],
        "domain": "account_security",
        "response": (
            "**To register on BuildHive:**\n"
            "1. Click **Sign Up** on the homepage\n"
            "2. Choose your role: **Buyer**, **Seller**, or **Freelancer**\n"
            "3. Enter name, email, phone, and password\n"
            "4. Verify your email\n"
            "5. Complete your profile\n\n"
            "💡 Your role determines which dashboard and features you see."
        ),
    },
    {
        "keywords": ["login", "sign in", "log in", "can't login", "forgot password"],
        "domain": "account_security",
        "response": (
            "**To log in:**\n"
            "1. Click **Login** on the homepage\n"
            "2. Enter your email and password\n"
            "3. Click **Sign In**\n\n"
            "Forgot password? Click **Forgot Password** on the login page and check your email.\n\n"
            "💡 Enable 2FA in Security Settings to protect your account."
        ),
    },
    {
        "keywords": ["payment method", "add payment", "pay", "payment options"],
        "domain": "marketplace",
        "response": (
            "**Payment options on BuildHive:**\n"
            "• Bank Transfer (most common)\n"
            "• JazzCash / EasyPaisa mobile wallets\n"
            "• Cash on Delivery (selected sellers)\n"
            "• Credit/Debit Card (coming soon)\n\n"
            "Manage: **Account Settings → Payment Methods**."
        ),
    },
    # ── Vendor / Seller info ──────────────────────────────────────────────────
    {
        "keywords": ["vendor registration", "register as vendor", "list my store",
                     "sell on buildhive", "become a vendor", "become a seller"],
        "domain": "vendor",
        "response": (
            "**To register as a vendor on BuildHive:**\n"
            "1. Click **Sign Up → Seller** on the homepage\n"
            "2. Enter your business name, CNIC/NTN, city, and contact info\n"
            "3. Submit business verification documents\n"
            "4. Our team reviews within 48 hours\n"
            "5. Once approved, start adding listings\n\n"
            "💡 Verified sellers get a badge and rank higher in search results."
        ),
    },
    {
        "keywords": ["contact vendor", "contact supplier", "is this vendor verified",
                     "vendor rating", "seller profile"],
        "domain": "vendor",
        "response": (
            "**To view a vendor's profile:**\n"
            "1. Click on any product listing\n"
            "2. Click the seller's name or **View Seller Profile**\n"
            "3. You'll see: Rating, Verified badge, Location, Categories\n\n"
            "**To contact them:**\n"
            "1. Click **Contact Seller** on their profile or product page\n"
            "2. Ask about bulk pricing, certifications, and delivery timelines\n\n"
            "💡 Only buy from verified (✓ badge) sellers for quality assurance."
        ),
    },
]


def _match_platform_query(query: str) -> Optional[str]:
    """
    Match query against PLATFORM_RESPONSES using keyword scoring.
    Returns best-matching response string, or None.
    Guards against cost/recommendation queries being intercepted.
    """
    if _user_wants_cost_estimate(query) or _user_wants_recommendation(query):
        return None

    q = query.lower()
    best_match: Optional[str] = None
    best_score = 0

    for entry in PLATFORM_RESPONSES:
        score = 0
        for kw in entry["keywords"]:
            if kw in q:
                score += 2 if " " in kw else 1
            elif " " in kw:
                words = kw.split()
                if all(w in q for w in words):
                    score += 1
        if score > best_score:
            best_score = score
            best_match = entry["response"]

    return best_match if best_score >= 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION GUIDES  (static page-to-action mapping)
# ─────────────────────────────────────────────────────────────────────────────

_NAV_GUIDES: Dict[str, str] = {
    "dashboard":      "Go to your Dashboard: click your profile icon (top-right) → **Dashboard**.",
    "tiles":          "Browse Tiles & Flooring: **Marketplace → Categories → Tiles & Flooring**.",
    "plumbing":       "Find Plumbing materials: **Marketplace → Categories → Plumbing & Sanitary**.",
    "cement":         "Find Cement products: **Marketplace → Categories → Cement & Concrete**.",
    "electrical":     "Browse Electrical items: **Marketplace → Categories → Electrical Components**.",
    "steel":          "Find Steel products: **Marketplace → Categories → Steel & Metal**.",
    "categories":     "Browse all categories: **Marketplace → Categories** (12+ material types).",
    "order":          "Track orders: **Buyer Dashboard → Active Orders**.",
    "wishlist":       "View wishlist: **Buyer Dashboard → Saved Items**.",
    "profile":        "Edit your profile: **Top-right icon → Account Settings → Profile**.",
    "messages":       "Open Messages: **Top navigation bar → Messages icon**.",
    "listing":        "Manage listings: **Seller Dashboard → Listing Management → New Listing**.",
    "estimate":       "Use Cost Estimator: **AI Tools → Cost Estimator** (also on the home page).",
    "recommendation": "Use Recommendations: **AI Tools → Material Recommendations**.",
    "checkout":       "Complete checkout: **Cart icon (top-right) → Checkout**.",
    "payment":        "View payment options: **Buyer Dashboard → Financial Overview**.",
    "help":           "Visit Help Center: **Footer → Help Center**, or ask me anything here.",
    "vendor":         "Browse vendors: **Marketplace → Vendors** or click any seller's name on a listing.",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION TERMINOLOGY  (FAQ — instant definitions)
# ─────────────────────────────────────────────────────────────────────────────

_CONSTRUCTION_TERMS: Dict[str, str] = {
    "grey structure": (
        "**Grey Structure** is the structural skeleton of a building — foundation, columns, "
        "beams, brick walls, and roof slab — without any finishing work (paint, tiles, etc.).\n\n"
        "It typically accounts for **40–50% of the total construction cost**."
    ),
    "bhk": (
        "**BHK** stands for **Bedroom, Hall, Kitchen** — a layout descriptor.\n"
        "- 2 BHK = 2 bedrooms + 1 hall + 1 kitchen\n"
        "- 3 BHK = 3 bedrooms + 1 hall + 1 kitchen\n\n"
        "Typical BHK for 5 marla: 2 BHK. For 10 marla: 3–4 BHK."
    ),
    "marla": (
        "**Marla** is a Pakistani unit of land area.\n"
        "- **1 Marla = 272 sqft** (standard)\n"
        "- **1 Kanal = 20 Marla = 5,440 sqft**\n\n"
        "Some older city records use 225 sqft/marla — always confirm locally."
    ),
    "finishing tier": (
        "**Finishing Tier** describes the quality level of interior/exterior materials:\n"
        "- **Economy** — basic fittings, local materials, minimal detailing\n"
        "- **Standard** — mid-range brands, good quality (most common)\n"
        "- **Premium / Luxury** — imported materials, branded fittings, high detailing\n\n"
        "Tier significantly affects cost per sqft."
    ),
    "mep": (
        "**MEP** stands for **Mechanical, Electrical, and Plumbing** — the utility systems:\n"
        "- **M** — HVAC, fans, ventilation\n"
        "- **E** — wiring, switchgear, lighting, DB boards\n"
        "- **P** — water supply pipes, drainage, sanitary fixtures\n\n"
        "MEP typically adds **15–25%** to the grey structure cost."
    ),
    "dpc": (
        "**DPC (Damp Proof Course)** is a waterproof barrier layer built into the foundation "
        "and lower walls to prevent moisture from rising up through the structure.\n\n"
        "Standard DPC in Pakistan uses bitumen-coated brickwork or waterproof cement mortar."
    ),
    "rcc": (
        "**RCC (Reinforced Cement Concrete)** is concrete strengthened with steel rebar.\n"
        "Used for: columns, beams, roof slabs, foundations.\n\n"
        "Standard residential mix ratio: **1:2:4** (cement : sand : aggregate)."
    ),
    "plinth": (
        "**Plinth** is the raised platform (base) on which a building sits, "
        "typically 1–2 feet above ground level.\n\n"
        "**Plinth Protection** is the concrete apron around the building base that "
        "prevents rainwater from collecting near the foundation."
    ),
    "lintel": (
        "**Lintel** is a horizontal structural beam placed above doors and windows "
        "to support the wall above the opening.\n\n"
        "Usually made of RCC in modern Pakistani construction."
    ),
    "shuttering": (
        "**Shuttering** (also called formwork) is the temporary mould — usually steel plates "
        "or wood planks — used to shape concrete while it cures.\n\n"
        "Shuttering cost is included in the RCC rate for slabs and columns."
    ),
}

_CONSTRUCTION_PHASES: List[Dict[str, str]] = [
    {"phase": "1. Site Preparation", "duration": "1–2 weeks",
     "details": "Clearing, levelling, setting out (laying reference lines), and soil testing."},
    {"phase": "2. Foundation", "duration": "3–6 weeks",
     "details": "Excavation, PCC (plain cement concrete) bedding, DPC, and footings."},
    {"phase": "3. Grey Structure", "duration": "3–5 months",
     "details": "Columns, beams, brick walls, lintels, and roof slab. Largest cost phase."},
    {"phase": "4. MEP Rough-in", "duration": "2–4 weeks",
     "details": "Concealed plumbing pipes, electrical conduits, and HVAC ductwork before plastering."},
    {"phase": "5. Plastering", "duration": "3–5 weeks",
     "details": "Internal and external plaster; walls must cure before finishing begins."},
    {"phase": "6. Finishing", "duration": "2–4 months",
     "details": "Tiles, marble, paint, woodwork (doors/windows/cabinets), and false ceiling."},
    {"phase": "7. MEP Finishing", "duration": "2–4 weeks",
     "details": "Fixtures, switches, sanitary items, faucets, lights, and fan installation."},
    {"phase": "8. Handover / Snagging", "duration": "1–2 weeks",
     "details": "Final inspection, defect correction, and cleaning."},
]


def _handle_construction_term(text: str) -> Optional[str]:
    """Return a definition if the query matches a known construction term."""
    t = text.lower()
    for term, definition in _CONSTRUCTION_TERMS.items():
        if term in t:
            return definition
    return None


def _handle_construction_phases(text: str) -> Optional[str]:
    """Return phase guide if the query asks about construction process/timeline."""
    t = text.lower()
    if not any(
        kw in t
        for kw in ("phase", "step", "process", "timeline", "how long", "stages",
                   "sequence", "order of construction", "how is a house built")
    ):
        return None
    lines = ["**Construction Phases (typical residential, Pakistan):**\n"]
    for p in _CONSTRUCTION_PHASES:
        lines.append(
            f"**{p['phase']}** ({p['duration']})\n{p['details']}\n"
        )
    lines.append(
        "\n*Total timeline: **8–14 months** for a 5 marla full build, "
        "depending on contractor speed, weather, and material availability.*"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ROLE-BASED FOLLOW-UP SUGGESTIONS  (21 intents × 3 roles)
# ─────────────────────────────────────────────────────────────────────────────

_FOLLOW_UPS: Dict[str, Dict[str, List[str]]] = {
    "recommendation": {
        "buyer":      ["Get a cost estimate for these materials",
                       "Save this recommendation to a project",
                       "Ask about a specific material"],
        "seller":     ["List one of these materials",
                       "Check current stock levels",
                       "View similar products"],
        "freelancer": ["Calculate project cost",
                       "Find materials for your project",
                       "View buyer project listings"],
    },
    "cost_estimation": {
        "buyer":      ["Refine with a different BHK or city",
                       "Get material recommendations for this budget",
                       "Ask about cost-saving options"],
        "seller":     ["View demand for these materials",
                       "Adjust pricing strategy",
                       "List related materials"],
        "freelancer": ["Find buyers with this budget",
                       "Post a proposal for similar work",
                       "Check labour rate by city"],
    },
    "cost_and_recommendation": {
        "buyer":      ["Refine estimate with a different tier",
                       "Ask for a narrower material scope (e.g. tiles only)",
                       "Open Cost Estimator to compare options"],
        "seller":     ["See which SKUs match the recommended picks",
                       "List materials popular for this layout tier",
                       "Review demand by city"],
        "freelancer": ["Tie this BOQ to your proposal template",
                       "Ask for labour-only vs turnkey breakdown",
                       "Export assumptions for client brief"],
    },
    "purchase_help": {
        "buyer":      ["Compare prices across vendors",
                       "Calculate quantity needed",
                       "Check vendor ratings"],
        "seller":     ["List this material",
                       "View category demand",
                       "Compare with competitors"],
        "freelancer": ["Add to your project materials list",
                       "Find local suppliers",
                       "Request a bulk quote"],
    },
    "quantity_calculator": {
        "buyer":      ["Get a full material list for my project",
                       "Find vendors for this material",
                       "Get a cost estimate"],
        "seller":     ["Check if you have enough stock",
                       "View demand for this material",
                       "Update listing quantity"],
        "freelancer": ["Add to project BOQ",
                       "Find suppliers for bulk order",
                       "Get cost estimate for full project"],
    },
    "unit_conversion": {
        "buyer":      ["Estimate cost for this area",
                       "Get material recommendations",
                       "Calculate material quantities"],
        "seller":     ["Check listings for this area size",
                       "View demand by plot size",
                       "Update listing specifications"],
        "freelancer": ["Calculate materials for this plot",
                       "Estimate project cost",
                       "Find buyers with this plot size"],
    },
    "vendor_info": {
        "buyer":      ["Contact this vendor",
                       "Compare with other suppliers",
                       "Request a quote"],
        "seller":     ["View your seller profile",
                       "Improve your rating",
                       "Update your listings"],
        "freelancer": ["Find verified suppliers for your project",
                       "Compare vendor prices",
                       "Request bulk pricing"],
    },
    "construction_faq": {
        "buyer":      ["Get material recommendations",
                       "Get a cost estimate",
                       "Browse related products"],
        "seller":     ["List materials for this construction phase",
                       "View category demand",
                       "Check compliance requirements"],
        "freelancer": ["Use this in your proposal",
                       "Find related projects",
                       "Browse buyer requirements"],
    },
    "construction_phases": {
        "buyer":      ["Get a cost estimate for each phase",
                       "Get material list by phase",
                       "Find contractors on BuildHive"],
        "seller":     ["List materials for upcoming phases",
                       "View phase-wise demand",
                       "Update stock for construction season"],
        "freelancer": ["Post a service for a specific phase",
                       "Find active projects in this phase",
                       "Browse phase-specific BOQ templates"],
    },
    "platform_help": {
        "buyer":      ["Track your current order",
                       "Browse material categories",
                       "Use the AI cost estimator"],
        "seller":     ["Manage your listings",
                       "View sales analytics",
                       "Update product pricing"],
        "freelancer": ["Update your portfolio",
                       "View open project requests",
                       "Manage active proposals"],
    },
    "navigation": {
        "buyer":      ["Search for specific materials",
                       "View featured products",
                       "Check your order status"],
        "seller":     ["Add a new product listing",
                       "View your seller dashboard",
                       "Check compliance notices"],
        "freelancer": ["View project board",
                       "Check proposal status",
                       "Update service listings"],
    },
    "general_question": {
        "buyer":      ["Get a material recommendation",
                       "Get a cost estimate",
                       "Browse related products"],
        "seller":     ["List this material",
                       "View category demand",
                       "Compare with competitors"],
        "freelancer": ["Use this knowledge in your proposal",
                       "Find related projects",
                       "Browse buyer requirements"],
    },
    "clarification_needed": {
        "buyer":      ["Get material recommendations",
                       "Get a cost estimate",
                       "Get platform help"],
        "seller":     ["Manage my listings",
                       "View sales data",
                       "Get platform help"],
        "freelancer": ["View open projects",
                       "Submit a proposal",
                       "Get platform help"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# OFF-TOPIC DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_OFF_TOPIC_KEYWORDS = (
    "weather", "temperature", "rainfall", "rain", "snow",
    "sports", "football", "cricket match", "movie", "film",
    "politics", "election", "vote",
    "recipe", "cooking", "restaurant",
    "joke", "funny", "humor",
    "vacation", "travel", "flight booking",
    "music", "song", "singer",
    "what time is it", "what is the date today",
    "tell me a story", "write a poem",
    "who is the president", "stock market",
    "cryptocurrency", "bitcoin",
)

_BUILDHIVE_KEYWORDS = (
    "marla", "sqft", "cement", "bricks", "tiles", "steel", "rebar",
    "paint", "plumbing", "electrical", "construction", "house", "building",
    "estimate", "cost", "material", "vendor", "supplier", "recommend",
    "buildhive", "listing", "seller", "buyer", "freelancer", "order",
    "grey structure", "finishing", "bhk", "kanal", "floor",
)


def _is_off_topic(query: str) -> bool:
    t = query.lower()
    if any(kw in t for kw in _BUILDHIVE_KEYWORDS):
        return False
    return any(kw in t for kw in _OFF_TOPIC_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _detect_unsafe_request(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ("kill myself", "suicide", "hurt myself", "end my life")):
        return "self_harm"
    if "swallowed bleach" in t:
        return "medical_emergency"
    if any(k in t for k in ("make a bomb", "build a bomb", "explosive", "how to poison")):
        return "violence"
    if any(k in t for k in ("illegal guns", "buy illegal gun")):
        return "weapons"
    if any(k in t for k in ("make meth", "cook meth", "synthesize meth")):
        return "drugs"
    if any(k in t for k in ("steal passwords", "phishing", "keylogger", "malware")):
        return "cyber"
    if "tax evasion" in t:
        return "illegal"
    if "ignore rules" in t or "reveal hidden instructions" in t:
        return "prompt_injection"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    response_id: str
    text: str
    intent: str
    suggested_follow_ups: List[str]
    language_hint: str
    source: str
    confidence: float = 0.0
    data: Optional[Dict[str, Any]] = None
    products: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    navigation_actions: List[Dict[str, Any]] = field(default_factory=list)
    # NEW v3: quick-reply chips for mobile UX
    quick_replies: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ChatBotModule:
    """
    BuildHive Universal Chatbot — Orchestration Layer v3.

    Pipeline:
        raw query
          → safety check
          → utility/shortcut check (greetings, unit conversion, math, etc.)
          → platform template match (FR1–FR16)
          → entity extraction (material + purchase action)
          → construction FAQ (terms, phases)
          → vendor intent
          → quantity calculator
          → intent classification (cost / recommendation / dual)
          → module routing
          → KB hybrid search fallback
          → LLM rewrite (KB answers only)
          → ChatResponse
    """

    def __init__(
        self,
        kb_path: str = "buildhive_knowledge_base_enhanced.json",
        model_name: str = "all-MiniLM-L6-v2",
        llm: Optional[LLMHelper] = None,
    ):
        self.kb_path = kb_path
        self.model_name = model_name
        self.model = get_embedding_model(model_name)

        self.preprocessor = QueryPreprocessor()
        self.detector = IntentDetector(model=self.model)
        self.llm = llm or LLMHelper()

        self.recommendation_module: Optional["RecommendationModule"] = None
        self.cost_module: Optional["CostEstimationModule"] = None

        self.kb_data: List[Dict] = []
        self.kb_embeddings: Optional[np.ndarray] = None
        self.kb_index: Optional[faiss.Index] = None
        self.query_variations: Dict[str, int] = {}

        self._query_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_embed_cache_max = 256
        self._session_state: Dict[str, Dict[str, Any]] = {}

        self._load_knowledge_base()
        logger.info("✓ ChatBotModule v3 initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_modules(
        self,
        recommendation: "RecommendationModule",
        cost: "CostEstimationModule",
    ) -> None:
        self.recommendation_module = recommendation
        self.cost_module = cost
        logger.info("✓ ChatBotModule: modules injected")

    def answer_query(
        self,
        query: str,
        user_role: str = "buyer",
        current_page: Optional[str] = None,
        conversation_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        try:
            response = self._build_response(
                query, user_role, current_page, use_llm, conversation_id
            )
        except Exception as exc:
            logger.error("answer_query unhandled exception: %s", exc, exc_info=True)
            response = self._emergency_fallback(query, user_role)
        return self._to_dict(response)

    def get_categories(self) -> List[str]:
        return sorted({d.get("category", "General") for d in self.kb_data})

    def get_faq_by_category(self, category: str) -> List[Dict]:
        return [
            {"question": d.get("question"), "answer": d.get("answer"),
             "category": d.get("category")}
            for d in self.kb_data
            if d.get("category", "").lower() == category.lower()
        ]

    def get_health_status(self) -> Dict:
        return {
            "status": "online",
            "version": "v3",
            "kb_loaded": len(self.kb_data) > 0,
            "embeddings_ready": self.kb_embeddings is not None,
            "kb_index_ready": self.kb_index is not None,
            "kb_size": len(self.kb_data),
            "categories": len(self.get_categories()),
            "recommendation_module": self.recommendation_module is not None,
            "cost_module": self.cost_module is not None,
            "intent_taxonomy": "21 intents across 6 categories",
            "pipeline": (
                "safety → utility → platform_template → entity → "
                "faq → vendor → qty_calc → intent → module → kb → llm"
            ),
        }

    @staticmethod
    def get_cost_recommendation_system_prompt() -> str:
        return CHATBOT_SYSTEM_PROMPT_COST_AND_REC

    @staticmethod
    def get_cost_estimation_assistant_knowledge_tables() -> str:
        return load_cost_estimation_assistant_knowledge()

    @staticmethod
    def get_cost_estimation_assistant_full_bundle() -> Dict[str, str]:
        return {
            "policy": CHATBOT_SYSTEM_PROMPT_COST_AND_REC,
            "knowledge_tables": load_cost_estimation_assistant_knowledge(),
        }

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _build_response(
        self,
        query: str,
        user_role: str,
        current_page: Optional[str],
        use_llm: bool = False,
        conversation_id: Optional[str] = None,
    ) -> ChatResponse:

        role = (user_role or "buyer").lower()
        if role not in ("buyer", "seller", "freelancer"):
            role = "buyer"

        lang_hint = self.preprocessor.detect_language_hint(query)
        clean = self.preprocessor.clean(query)
        intent: IntentResult = self.detector.classify(clean)
        cid = (conversation_id or "").strip() or "default"
        st = self._session_state.setdefault(cid, {})

        logger.info(
            "v3 query='%s' clean='%s' intent=%s(%.2f)",
            query, clean, intent.label, intent.score,
        )

        # ── STEP 1: Safety ────────────────────────────────────────────────────
        unsafe = _detect_unsafe_request(query)
        if unsafe == "self_harm":
            return self._make(
                text=(
                    "I'm really sorry you're feeling this way. "
                    "If you're in immediate danger, please call your local emergency number now.\n\n"
                    "Reach out to someone you trust or a local crisis helpline. "
                    "If you tell me your city, I can suggest support resources."
                ),
                intent="safety_refusal", source="safety",
                lang=lang_hint, confidence=1.0,
            )
        if unsafe == "medical_emergency":
            return self._make(
                text=(
                    "This is urgent. Contact emergency services or poison control immediately. "
                    "Do not induce vomiting unless a medical professional instructs you to."
                ),
                intent="safety_guidance", source="safety",
                lang=lang_hint, confidence=1.0,
            )
        if unsafe:
            return self._make(
                text="I can't help with that. Ask me anything about construction, materials, or BuildHive.",
                intent="safety_refusal", source="safety",
                lang=lang_hint, confidence=1.0,
            )

        # ── STEP 2: Greeting ──────────────────────────────────────────────────
        if self._is_greeting(clean) or self._is_greeting(query):
            return self._make(
                text="Hi! How can I help you today? 👋",
                intent="greeting", source="utility",
                lang=lang_hint, confidence=1.0,
                nav=_navigation_actions_list("Mixed"),
                quick_replies=[
                    "Estimate construction cost",
                    "Find materials",
                    "How does BuildHive work?",
                ],
            )

        # ── STEP 3: Empty / noise ─────────────────────────────────────────────
        if not clean or clean == "[empty query]" or self._looks_like_noise(query):
            return self._make(
                text=(
                    "Please type a question — for example:\n"
                    "- **\"Estimate cost for 5 marla house in Lahore\"**\n"
                    "- **\"Recommend tiles for bathroom\"**\n"
                    "- **\"How many bags of cement for 10 marla?\"**"
                ),
                intent="clarification_needed", source="utility",
                lang=lang_hint, confidence=1.0,
                quick_replies=[
                    "Cost estimate", "Material recommendations", "Platform help",
                ],
            )

        # ── STEP 4: Urdu language preference ─────────────────────────────────
        if "always respond in urdu" in clean.lower() or "always reply in urdu" in clean.lower():
            st["lang_pref"] = "urdu"
            return self._make(
                text="ٹھیک ہے — اب سے میں اُردو میں جواب دوں گا۔",
                intent="preference_set", source="utility",
                lang="urdu", confidence=1.0,
            )

        # ── STEP 5: Menu choice follow-up ─────────────────────────────────────
        if st.get("awaiting_menu_choice") and query.strip() in ("1", "2", "3"):
            st["awaiting_menu_choice"] = False
            choice = query.strip()
            if choice == "1":
                return self._make(
                    text=(
                        "Great! Tell me your **city** and **area** "
                        "(e.g. '5 marla in Lahore') and what you need "
                        "(e.g. 'cement and tiles for a house')."
                    ),
                    intent="recommendation", source="menu_choice",
                    lang=st.get("lang_pref", "english"), confidence=1.0,
                    nav=_navigation_actions_list("Recommendation"),
                )
            if choice == "2":
                return self._make(
                    text=(
                        "Sure! Tell me your **city**, **area** (marla or sqft), "
                        "and **number of floors** — I'll calculate the construction cost."
                    ),
                    intent="cost_estimation", source="menu_choice",
                    lang=st.get("lang_pref", "english"), confidence=1.0,
                    nav=_navigation_actions_list("Estimation"),
                )
            return self._make(
                text=(
                    "What do you need help with on BuildHive?\n"
                    "- Buying materials\n- Listing products\n"
                    "- Tracking orders\n- Using AI tools"
                ),
                intent="platform_help", source="menu_choice",
                lang=st.get("lang_pref", "english"), confidence=1.0,
            )

        # ── STEP 6: Unit conversion ────────────────────────────────────────────
        if _user_wants_unit_conversion(clean):
            result = _handle_unit_conversion(clean)
            if result:
                return self._make(
                    text=result,
                    intent="unit_conversion", source="utility",
                    lang=lang_hint, confidence=1.0,
                    follow_ups=self._suggest_follow_ups("unit_conversion", role, current_page),
                    quick_replies=[
                        "Estimate cost for this area",
                        "Calculate material quantities",
                        "Get material recommendations",
                    ],
                )

        # ── STEP 7: Math shortcuts ─────────────────────────────────────────────
        mmul = re.search(r"^\s*(\d+)\s*[x×\*]\s*(\d+)\s*\??\s*$", query.lower())
        if mmul:
            a, b = int(mmul.group(1)), int(mmul.group(2))
            return self._make(
                text=str(a * b), intent="math",
                source="utility", lang=lang_hint, confidence=1.0,
            )

        # ── STEP 8: Platform template (FR1–FR16) ───────────────────────────────
        platform_answer = _match_platform_query(clean)
        if platform_answer:
            return self._make(
                text=platform_answer,
                intent="platform_help", source="platform_template",
                lang=lang_hint, confidence=1.0,
                follow_ups=self._suggest_follow_ups("platform_help", role, current_page),
            )

        # ── STEP 9: Vendor intent ─────────────────────────────────────────────
        if _user_wants_vendor_info(clean):
            vendor_resp = self._handle_vendor_intent(clean, role)
            return self._make(
                text=vendor_resp,
                intent="vendor_info", source="vendor_guide",
                lang=lang_hint, confidence=0.9,
                follow_ups=self._suggest_follow_ups("vendor_info", role, current_page),
                nav=_navigation_actions_list("Vendor"),
                quick_replies=[
                    "Contact this vendor",
                    "Compare with other suppliers",
                    "Register as a vendor",
                ],
            )

        # ── STEP 10: Quantity calculator ──────────────────────────────────────
        if _user_wants_quantity_calc(clean):
            qty_resp = self._handle_quantity_calc(clean)
            if qty_resp:
                return self._make(
                    text=qty_resp,
                    intent="quantity_calculator", source="qty_calc",
                    lang=lang_hint, confidence=0.95,
                    follow_ups=self._suggest_follow_ups("quantity_calculator", role, current_page),
                    nav=_navigation_actions_list("Materials"),
                    quick_replies=[
                        "Find vendors for this material",
                        "Get a full project cost estimate",
                        "Get material recommendations",
                    ],
                )

        # ── STEP 11: Construction terminology / FAQ ────────────────────────────
        term_def = _handle_construction_term(clean)
        if term_def:
            return self._make(
                text=term_def,
                intent="construction_faq", source="faq",
                lang=lang_hint, confidence=0.95,
                follow_ups=self._suggest_follow_ups("construction_faq", role, current_page),
            )

        phase_guide = _handle_construction_phases(clean)
        if phase_guide:
            return self._make(
                text=phase_guide,
                intent="construction_phases", source="faq",
                lang=lang_hint, confidence=0.95,
                follow_ups=self._suggest_follow_ups("construction_phases", role, current_page),
                nav=_navigation_actions_list("Estimation"),
            )

        # ── STEP 12: Off-topic guard ──────────────────────────────────────────
        if _is_off_topic(clean) and intent.score < 0.30:
            return self._make(
                text=(
                    "I specialise in construction and building materials. "
                    "I can help you with:\n\n"
                    "• **Cost estimation** — how much will it cost to build?\n"
                    "• **Material recommendations** — what materials should I buy?\n"
                    "• **Quantity calculations** — how many bags of cement do I need?\n"
                    "• **Vendor info** — how do I contact a supplier?\n"
                    "• **Platform help** — how do I use BuildHive?\n\n"
                    "What would you like help with?"
                ),
                intent="off_topic", source="off_topic_filter",
                lang=lang_hint, confidence=0.0,
                follow_ups=self._suggest_follow_ups("platform_help", role, current_page),
                quick_replies=[
                    "Cost estimate",
                    "Material recommendations",
                    "Platform help",
                ],
            )

        # ── STEP 13: Entity extraction → purchase guide ───────────────────────
        entities = self._extract_entities(clean)
        if entities["has_purchase_action"] and entities["materials"]:
            material_kw, category = entities["materials"][0]
            text, products, steps = self._handle_purchase_query(
                material_kw=material_kw,
                category=category,
                query=clean,
            )
            return self._make(
                text=text, intent="purchase_help", source="purchase_guide",
                lang=lang_hint, confidence=0.9,
                products=products, steps=steps,
                follow_ups=self._suggest_follow_ups("purchase_help", role, current_page),
                nav=_navigation_actions_list("Materials"),
                quick_replies=[
                    f"Calculate {material_kw} quantity",
                    f"Compare {material_kw} brands",
                    "Get full project cost estimate",
                ],
            )

        # ── STEP 14: Dual cost + recommendation ───────────────────────────────
        want_cost = _user_wants_cost_estimate(clean)
        want_rec = _user_wants_recommendation(clean)
        run_dual = (
            (want_cost and want_rec) or _user_ambiguous_deal(clean)
        ) and self.cost_module is not None and self.recommendation_module is not None

        if run_dual and intent.label not in ("clarification_needed", "platform_help", "navigation"):
            return self._handle_dual_intent(clean, role, lang_hint, use_llm, current_page)

        # ── STEP 15: Single intent routing ────────────────────────────────────
        if intent.label == "recommendation" or want_rec:
            return self._handle_recommendation_intent(
                clean, role, lang_hint, use_llm, current_page
            )

        if intent.label == "cost_estimation" or want_cost:
            return self._handle_cost_intent(
                clean, role, lang_hint, use_llm, current_page
            )

        if intent.label == "clarification_needed":
            st["awaiting_menu_choice"] = True
            return self._make(
                text=self._build_clarification_prompt(clean),
                intent="clarification_needed", source="clarification",
                lang=lang_hint, confidence=float(intent.score or 0.0),
                follow_ups=[],
                nav=_navigation_actions_list("Mixed"),
                quick_replies=["1", "2", "3"],
            )

        if intent.label == "navigation":
            text = self._static_navigation_guide(clean, current_page)
            return self._make(
                text=text, intent="navigation", source="static",
                lang=lang_hint, confidence=float(intent.score or 0.0),
                follow_ups=self._suggest_follow_ups("navigation", role, current_page),
            )

        # ── STEP 16: KB hybrid search (general questions) ─────────────────────
        text = self._kb_hybrid_search(clean, top_k=5, role=role)

        # City advisory check
        if not text and self.recommendation_module:
            for city in getattr(self.recommendation_module, "city_advisory", {}).keys():
                if city.lower() in clean.lower():
                    advice = self.recommendation_module.get_city_advisory(city)
                    if advice:
                        text = (
                            f"**Construction Tips for {city}:**\n\n{advice}\n\n"
                            f"Want material recommendations specific to {city}?"
                        )
                        break

        source = "kb"

        # LLM rewrite for KB answers
        if source == "kb" and text:
            try:
                rewritten = self.llm.rewrite_kb_answer(question=clean, answer=text)
                if rewritten and rewritten.strip():
                    text = rewritten
            except Exception as exc:
                logger.warning("LLM rewrite skipped: %s", exc)

        return self._make(
            text=text or self._fallback_text(),
            intent=intent.label, source=source,
            lang=lang_hint, confidence=float(intent.score or 0.0),
            follow_ups=self._suggest_follow_ups(intent.label, role, current_page),
        )

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _handle_dual_intent(
        self,
        clean: str,
        role: str,
        lang_hint: str,
        use_llm: bool,
        current_page: Optional[str],
    ) -> ChatResponse:
        """Run cost + recommendation modules together."""
        bt = _extract_building_type_hint(clean) or "house"
        city_h = _extract_city_hint(clean)
        area_h = _extract_area_hint(clean)
        floors_h = _extract_floors_hint(clean)
        tier_h = _extract_finishing_tier(clean)

        missing: List[str] = []
        if not area_h:
            missing.append("area (e.g. **5 marla** or **2000 sqft**)")
        if not city_h:
            missing.append("city (e.g. **Lahore**, **Karachi**)")

        if missing:
            q = "To give you an accurate estimate and recommendations, I need:\n" + "\n".join(
                f"- {m}" for m in missing[:2]
            )
            text = _router_template(
                intent="Mixed",
                inputs_used={"query": clean},
                result_summary=f"{q}",
                warnings=[
                    "⚠ Exclusions: land, design fees, NOCs, utility connections, "
                    "furniture, and contractor profit are not included.",
                ],
                include_nav=True,
            )
            return self._make(
                text=text, intent="clarification_needed", source="clarification_router",
                lang=lang_hint, confidence=0.5,
                follow_ups=self._suggest_follow_ups("clarification_needed", role, current_page),
                nav=_navigation_actions_list("Mixed"),
                quick_replies=[
                    "5 marla Lahore",
                    "10 marla Karachi",
                    "1 kanal Islamabad",
                ],
            )

        try:
            order = _dual_module_order(clean)
            if order == "rec_first":
                rec_r = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
                cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
            else:
                cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
                rec_r = self.recommendation_module.recommend(text=clean, use_llm=use_llm)

            rec_summary, _ = _format_recommendation_summary_for_chat(rec_r)
            cost_summary, cost_warns = _format_estimation_summary_for_chat(cost_r)

            result_summary = (
                "### 💰 Cost Estimate\n"
                f"{cost_summary}\n\n"
                "### 🏗 Material Recommendations\n"
                f"{rec_summary}"
            )
            text = _router_template(
                intent="Mixed",
                inputs_used={
                    "building_type": bt,
                    "city": city_h or "Lahore (default)",
                    "area": area_h,
                    "floors": floors_h if floors_h is not None else 1,
                    "finishing_tier": tier_h or "Standard (default)",
                },
                result_summary=result_summary,
                warnings=[
                    "⚠ Exclusions: land, design fees, NOCs, utility connections, "
                    "furniture, and contractor profit are not included.",
                    "⚠ Accuracy: benchmark-based estimate. Detailed drawings required for BOQ.",
                ] + (cost_warns[:2] if cost_warns else []),
                include_nav=True,
            )
            dual_products: List[Dict[str, Any]] = []
            try:
                cats = rec_r.get("categories") or rec_r.get("recommendations") or {}
                for _cat, items in list(cats.items())[:4]:
                    dual_products.extend(items[:1])
            except Exception:
                pass

            return self._make(
                text=text, intent="cost_and_recommendation",
                source="cost_and_recommendation",
                lang=lang_hint, confidence=0.85,
                data={"cost_estimation": cost_r, "recommendation": rec_r},
                products=dual_products,
                follow_ups=self._suggest_follow_ups("cost_and_recommendation", role, current_page),
                nav=_navigation_actions_list("Mixed"),
                quick_replies=[
                    "Refine with Economy tier",
                    "Refine with Luxury tier",
                    "Show only materials",
                ],
            )
        except Exception as exc:
            logger.warning("Dual path failed, falling through: %s", exc)
            return self._handle_cost_intent(clean, role, lang_hint, use_llm, current_page)

    def _handle_recommendation_intent(
        self,
        clean: str,
        role: str,
        lang_hint: str,
        use_llm: bool,
        current_page: Optional[str],
    ) -> ChatResponse:
        if self.recommendation_module is None:
            return self._make(
                text=self._kb_hybrid_search(clean, top_k=3, role=role),
                intent="recommendation", source="kb_fallback",
                lang=lang_hint, confidence=0.4,
                nav=_navigation_actions_list("Recommendation"),
            )

        bt = _extract_building_type_hint(clean) or "house"
        city_h = _extract_city_hint(clean)
        area_h = _extract_area_hint(clean)
        floors_h = _extract_floors_hint(clean)

        if not area_h:
            text = _router_template(
                intent="Recommendation",
                inputs_used={"query": clean, "building_type": bt},
                result_summary=(
                    "I need one more detail to recommend materials.\n\n"
                    "**What is your total area?** (e.g. 5 marla, 2000 sqft)"
                ),
                include_nav=True,
            )
            return self._make(
                text=text, intent="clarification_needed", source="clarification_router",
                lang=lang_hint, confidence=0.5,
                nav=_navigation_actions_list("Recommendation"),
                quick_replies=["5 marla", "10 marla", "1 kanal", "Enter manually"],
            )

        try:
            result = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
            rec_summary, _ = _format_recommendation_summary_for_chat(result)
            text = _router_template(
                intent="Recommendation",
                inputs_used={
                    "building_type": bt,
                    "city": city_h or "Lahore (default)",
                    "area": area_h,
                    "floors": floors_h if floors_h is not None else 1,
                },
                result_summary=rec_summary,
                include_nav=True,
            )
            return self._make(
                text=text, intent="recommendation", source="recommendation_module",
                lang=lang_hint, confidence=0.85, data=result,
                follow_ups=[],
                nav=_navigation_actions_list("Recommendation"),
                quick_replies=[
                    "Get cost estimate",
                    "Filter by Economy tier",
                    "Show only cement and steel",
                ],
            )
        except Exception as exc:
            logger.error("Recommendation module error: %s", exc)
            return self._make(
                text=self._kb_hybrid_search(clean, top_k=3, role=role),
                intent="recommendation", source="kb_fallback",
                lang=lang_hint, confidence=0.3,
                nav=_navigation_actions_list("Recommendation"),
            )

    def _handle_cost_intent(
        self,
        clean: str,
        role: str,
        lang_hint: str,
        use_llm: bool,
        current_page: Optional[str],
    ) -> ChatResponse:
        if self.cost_module is None:
            return self._make(
                text=self._kb_hybrid_search(clean, top_k=3, role=role),
                intent="cost_estimation", source="kb_fallback",
                lang=lang_hint, confidence=0.4,
                nav=_navigation_actions_list("Estimation"),
            )

        bt = _extract_building_type_hint(clean) or "house"
        city_h = _extract_city_hint(clean)
        area_h = _extract_area_hint(clean)
        floors_h = _extract_floors_hint(clean)
        tier_h = _extract_finishing_tier(clean)
        bhk_h = _extract_bhk_hint(clean)

        if not area_h:
            text = _router_template(
                intent="Estimation",
                inputs_used={"query": clean, "building_type": bt},
                result_summary=(
                    "I need one more detail to calculate the cost.\n\n"
                    "**What is your total area?** (e.g. 5 marla, 2000 sqft)"
                ),
                warnings=[
                    "⚠ Exclusions: land, design fees, NOCs, utility connections, "
                    "furniture, and contractor profit.",
                ],
                include_nav=True,
            )
            return self._make(
                text=text, intent="clarification_needed", source="clarification_router",
                lang=lang_hint, confidence=0.5,
                nav=_navigation_actions_list("Estimation"),
                quick_replies=["5 marla", "10 marla", "1 kanal", "Enter manually"],
            )

        try:
            result = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
            cost_summary, cost_warns = _format_estimation_summary_for_chat(result)
            text = _router_template(
                intent="Estimation",
                inputs_used={
                    "building_type": bt,
                    "city": city_h or "Lahore (default)",
                    "area": area_h,
                    "floors": floors_h if floors_h is not None else 1,
                    "quality_tier": tier_h or "Standard (default)",
                    "bhk": bhk_h or "auto",
                },
                result_summary=cost_summary,
                warnings=[
                    "⚠ Exclusions: land, design fees, NOCs, utility connections, "
                    "furniture, and contractor profit.",
                    "⚠ Accuracy: benchmark-based. Detailed drawings required for BOQ.",
                ] + (cost_warns[:2] if cost_warns else []),
                include_nav=True,
            )
            return self._make(
                text=text, intent="cost_estimation", source="cost_module",
                lang=lang_hint, confidence=0.85, data=result,
                follow_ups=[],
                nav=_navigation_actions_list("Estimation"),
                quick_replies=[
                    "Get material recommendations",
                    "Try Economy tier",
                    "Try Luxury tier",
                ],
            )
        except Exception as exc:
            logger.error("Cost module error: %s", exc)
            return self._make(
                text=self._kb_hybrid_search(clean, top_k=3, role=role),
                intent="cost_estimation", source="kb_fallback",
                lang=lang_hint, confidence=0.3,
                nav=_navigation_actions_list("Estimation"),
            )

    # ── Specialist handlers ───────────────────────────────────────────────────

    def _handle_vendor_intent(self, clean: str, role: str) -> str:
        t = clean.lower()

        # Registration / onboarding
        if any(k in t for k in ("register", "list my store", "sell on", "become a vendor",
                                 "become a seller", "vendor registration")):
            return (
                "**To register as a vendor on BuildHive:**\n"
                "1. Click **Sign Up → Seller** on the homepage\n"
                "2. Enter business name, CNIC/NTN, city, and contact info\n"
                "3. Submit verification documents\n"
                "4. Our team reviews within **48 hours**\n"
                "5. Once approved, start adding listings\n\n"
                "💡 Verified sellers get a ✓ badge and rank higher in search results.\n\n"
                "Would you like me to take you to the vendor registration page?"
            )

        # Contact / verification
        if any(k in t for k in ("contact", "verified", "rating", "profile", "is this seller")):
            return (
                "**To view a vendor's profile:**\n"
                "1. Click any product listing\n"
                "2. Click the seller's name → **View Seller Profile**\n"
                "3. You'll see: ⭐ Rating, ✓ Verified badge, City, Categories\n\n"
                "**To contact them:**\n"
                "Click **Contact Seller** on their profile or any product page.\n\n"
                "💡 Always check for the ✓ Verified badge and at least a 4-star rating "
                "before placing a large order."
            )

        # Generic vendor query
        return (
            "**BuildHive Vendor Directory:**\n\n"
            "• Browse verified suppliers: **Marketplace → Vendors**\n"
            "• Filter by city, category, and rating\n"
            "• Contact directly via the **Message** button on any listing\n"
            "• All verified vendors carry a ✓ badge and have met our quality criteria\n\n"
            "What material are you looking to source? I can find relevant vendors for you."
        )

    def _handle_quantity_calc(self, clean: str) -> Optional[str]:
        """
        Detect material + area from the query and return a quantity estimate.
        Returns None if area cannot be extracted.
        """
        t = clean.lower()

        # Find area
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(marla|kanal|sqft|sq ft|square feet)", t
        )
        if not m:
            return (
                "To calculate quantity, please tell me:\n"
                "1. **Material** (e.g. cement, bricks, tiles)\n"
                "2. **Area** (e.g. 5 marla, 2000 sqft)\n\n"
                "Example: *\"How many bags of cement for 5 marla house?\"*"
            )

        val = float(m.group(1))
        unit = m.group(2).lower().replace("sq ft", "sqft").replace("square feet", "sqft")

        if unit == "marla":
            sqft = _marla_to_sqft(val)
        elif unit == "kanal":
            sqft = _marla_to_sqft(_kanal_to_marla(val))
        else:
            sqft = val

        # Find material
        for kw in (
            "cement", "brick", "tile", "marble", "granite",
            "paint", "steel", "rebar", "sand", "gravel",
        ):
            if kw in t:
                return _calculate_quantity(kw, sqft)

        # Generic fallback
        return (
            f"For **{val:g} {unit}** ({sqft:,.0f} sqft), here are rough estimates:\n\n"
            f"• **Cement**: ~{round(sqft * 0.4)} bags\n"
            f"• **Bricks**: ~{round(sqft * 9):,} bricks\n"
            f"• **Steel (slab)**: ~{round(sqft * 3.75):,} kg\n"
            f"• **Paint**: ~{round(sqft / 12)} litres\n\n"
            "💡 These are estimates. Ask me for a specific material for more detail."
        )

    def _handle_purchase_query(
        self,
        material_kw: str,
        category: str,
        query: str,
        quality: str = "Standard",
        city: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        products: List[Dict[str, Any]] = []
        if self.recommendation_module is not None:
            try:
                rec = self.recommendation_module.recommend(
                    text=f"{material_kw} {quality}",
                    city=city, quality=quality, top_n_per_cat=5,
                )
                all_items: List[Dict[str, Any]] = []
                for v in (rec.get("categories") or {}).values():
                    all_items.extend(v)
                kw = material_kw.lower()
                matched = [p for p in all_items if kw in str(p.get("item_name", "")).lower()]
                products = (matched if matched else all_items)[:5]
            except Exception as exc:
                logger.warning("Product retrieval failed: %s", exc)

        steps = _PURCHASE_STEPS.get(material_kw, _PURCHASE_STEPS["default"])

        price_range = ""
        if products:
            prices = [p["final_price_pkr"] for p in products if p.get("final_price_pkr")]
            if prices:
                lo, hi = min(prices), max(prices)
                price_range = (
                    f"\n**Price range on BuildHive:** PKR {lo:,} – PKR {hi:,} per unit.\n"
                )

        names = ", ".join(p["item_name"] for p in products[:3]) if products else ""
        product_line = f"\n**Top options:** {names}.\n" if names else ""

        answer = (
            f"Here's how to buy **{material_kw}** on BuildHive:\n\n"
            + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
            + price_range
            + product_line
            + "\n\n💡 Use the **Recommendation Tool** above for a full personalised material list."
        )
        return answer, products, steps

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_greeting(text: str) -> bool:
        t = (text or "").strip().lower()
        return t in (
            "hi", "hello", "hey", "assalam o alaikum", "assalamualaikum",
            "aoa", "salam", "good morning", "good afternoon", "good evening",
        )

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        alnum = sum(ch.isalnum() for ch in t)
        if (alnum <= 2 and len(t) <= 8) or (len(t) <= 3 and not t.isalnum()):
            return True
        if " " not in t and re.fullmatch(r"[a-zA-Z0-9\?\!]{8,}", t):
            if any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t):
                return True
        return False

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        materials = [(kw, cat) for kw, cat in _MATERIAL_ENTITIES.items() if kw in t]
        has_purchase_action = any(act in t for act in _PURCHASE_ACTIONS)
        return {"materials": materials, "has_purchase_action": has_purchase_action}

    def _suggest_follow_ups(
        self,
        intent: str,
        role: str,
        current_page: Optional[str] = None,
    ) -> List[str]:
        role_map = _FOLLOW_UPS.get(intent, {})
        suggestions = list(role_map.get(role, role_map.get("buyer", [])))
        if current_page:
            page = current_page.lower()
            suggestions = [
                s for s in suggestions
                if not any(kw in page for kw in s.lower().split()[:2])
            ]
        return suggestions[:3]

    def _build_clarification_prompt(self, clean_query: str) -> str:
        return (
            "What would you like to do?\n\n"
            "**1️⃣** Find materials for my project\n"
            "**2️⃣** Calculate construction cost\n"
            "**3️⃣** Learn how to use BuildHive\n\n"
            "Reply with **1**, **2**, or **3** — or just describe what you need 👍"
        )

    def _static_navigation_guide(self, clean_query: str, current_page: Optional[str]) -> str:
        for keyword, guide in _NAV_GUIDES.items():
            if keyword in clean_query.lower():
                if current_page and keyword in current_page.lower():
                    continue
                return guide
        return (
            "**BuildHive main sections:**\n\n"
            "• **Marketplace** → Browse & buy materials\n"
            "• **AI Tools** → Recommendations & Cost Estimator\n"
            "• **Dashboard** → Orders, listings & messages\n"
            "• **Help Center** → FAQs & support\n\n"
            "Tell me where you'd like to go and I'll guide you directly."
        )

    def _fallback_text(self) -> str:
        return (
            "I didn't quite catch that. Here's what I can help with:\n\n"
            "• **Cost estimate** — *\"Estimate cost for 5 marla house in Lahore\"*\n"
            "• **Material recommendations** — *\"Suggest materials for my project\"*\n"
            "• **Quantity calculator** — *\"How many bags of cement for 10 marla?\"*\n"
            "• **Unit conversion** — *\"5 marla to sqft\"*\n"
            "• **Vendor info** — *\"How do I contact a supplier?\"*\n"
            "• **Platform help** — *\"How to add a listing?\"*\n\n"
            "Try rephrasing your question or pick one of the options above."
        )

    def _emergency_fallback(self, query: str, role: str) -> ChatResponse:
        return self._make(
            text=self._fallback_text(),
            intent="error", source="error_fallback",
            lang="english", confidence=0.0,
            follow_ups=self._suggest_follow_ups("clarification_needed", role),
        )

    # ── Response builder (single factory method) ──────────────────────────────

    def _make(
        self,
        *,
        text: str,
        intent: str,
        source: str,
        lang: str,
        confidence: float,
        data: Optional[Dict[str, Any]] = None,
        products: Optional[List[Dict[str, Any]]] = None,
        steps: Optional[List[str]] = None,
        follow_ups: Optional[List[str]] = None,
        nav: Optional[List[Dict[str, Any]]] = None,
        quick_replies: Optional[List[str]] = None,
    ) -> ChatResponse:
        return ChatResponse(
            response_id=str(uuid.uuid4()),
            text=text,
            intent=intent,
            suggested_follow_ups=follow_ups or [],
            language_hint=lang,
            source=source,
            confidence=confidence,
            data=data,
            products=products or [],
            steps=steps or [],
            navigation_actions=nav or [],
            quick_replies=quick_replies or [],
        )

    @staticmethod
    def _to_dict(response: ChatResponse) -> Dict[str, Any]:
        return {
            "status":               "success",
            "response_id":          response.response_id,
            "answer":               response.text,
            "intent":               response.intent,
            "confidence":           response.confidence,
            "suggested_follow_ups": response.suggested_follow_ups,
            "language_hint":        response.language_hint,
            "source":               response.source,
            "data":                 response.data,
            "products":             response.products,
            "steps":                response.steps,
            "navigation_actions":   response.navigation_actions,
            # v3 addition — always present
            "quick_replies":        response.quick_replies,
        }

    def llm_status(self) -> Dict[str, Any]:
        return {"available": self.llm.is_available, "model": self.llm.model_name}

    # ── KB search ─────────────────────────────────────────────────────────────

    def _kb_hybrid_search(self, query: str, top_k: int = 5, role: str = "buyer") -> str:
        if not self.kb_data:
            return self._fallback_text()

        semantic_hits = self._semantic_search(query, top_k=top_k)
        keyword_hits = self._keyword_search(query, top_k=top_k)

        combined: Dict[int, float] = {}
        for idx, score in semantic_hits:
            combined[idx] = combined.get(idx, 0) + score * 0.65
        max_kw = max((s for _, s in keyword_hits), default=1) or 1
        for idx, score in keyword_hits:
            combined[idx] = combined.get(idx, 0) + (score / max_kw) * 0.35

        if not combined:
            return self._fallback_text()

        ROLE_CATEGORY_BOOST = {
            "seller":     ["Seller Module", "Marketplace System"],
            "freelancer": ["Freelancer / Service Provider"],
            "buyer":      ["Buyer Module", "Cost Estimation System", "AI Recommendation System"],
        }
        for idx in combined:
            if self.kb_data[idx].get("category") in ROLE_CATEGORY_BOOST.get(role, []):
                combined[idx] *= 1.2

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        best_idx, best_score = ranked[0]

        if best_score < 0.25:
            return self._fallback_text()

        best_doc = self.kb_data[best_idx]
        raw_answer = best_doc.get("answer", "")
        question = best_doc.get("question", "")

        if not raw_answer:
            return self._fallback_text()

        answer = raw_answer.strip()
        if len(answer) > 400:
            sentences = answer.split(". ")
            answer = ". ".join(sentences[:2]).strip()
            if not answer.endswith("."):
                answer += "."

        if question:
            answer = f"**{question.rstrip('?')}:**\n\n{answer}"
        return answer

    def _encode_query(self, query: str) -> np.ndarray:
        if query in self._query_embed_cache:
            self._query_embed_cache.move_to_end(query)
            return self._query_embed_cache[query]
        vec = self.model.encode([query], show_progress_bar=False)
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)
        self._query_embed_cache[query] = vec
        if len(self._query_embed_cache) > self._query_embed_cache_max:
            self._query_embed_cache.popitem(last=False)
        return vec

    def _semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.kb_index is None or self.kb_index.ntotal == 0:
            return []
        k = min(top_k, self.kb_index.ntotal)
        vec = self._encode_query(query)
        sims, idxs = self.kb_index.search(vec, k)
        return [(int(idxs[0][i]), float(sims[0][i])) for i in range(k)]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_words = set(query.lower().split())
        scores: List[Tuple[int, float]] = []
        for idx, doc in enumerate(self.kb_data):
            score = 0.0
            score += len(query_words & {t.lower() for t in doc.get("tags", [])}) * 3
            score += len(query_words & set(doc.get("question", "").lower().split())) * 2
            score += len(query_words & set(doc.get("answer", "").lower().split())) * 0.5
            for var in doc.get("query_variations", []):
                score += len(query_words & set(var.lower().split())) * 2.5
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── Knowledge base loading ────────────────────────────────────────────────

    def _load_knowledge_base(self) -> None:
        paths = [
            self.kb_path,
            "buildhive_knowledge_base_enhanced.json",
            "buildhive_knowledge_base.json",
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.kb_data = json.load(f)
                    logger.info("KB loaded: %d entries from %s", len(self.kb_data), path)
                    self._index_kb()
                    return
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
        logger.error("No knowledge base found — KB search unavailable")

    def _index_kb(self) -> None:
        if not self.kb_data:
            return
        for idx, doc in enumerate(self.kb_data):
            self.query_variations[doc.get("question", "").lower()] = idx
            for var in doc.get("query_variations", []):
                self.query_variations[var.lower()] = idx

        questions = [d.get("question", "") for d in self.kb_data]
        logger.info("Encoding %d KB entries...", len(questions))
        embeddings = self.model.encode(questions, show_progress_bar=False)
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.kb_embeddings = embeddings

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.kb_index = index
        logger.info(
            "KB index ready: %d vectors, dim=%d (cosine IndexFlatIP)",
            embeddings.shape[0], embeddings.shape[1],
        )