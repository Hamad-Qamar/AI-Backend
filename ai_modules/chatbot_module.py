"""
ChatBotModule v4 FINAL — BuildHive Construction Copilot
========================================================

What's new vs v3:
  ✅ 113-product Construction KB  — "what is cement", "explain pprc pipe", "what is MCB"
  ✅ Developer KB                 — DB schema, API routes, frontend helpers, AI proxy
  ✅ PRODUCT_KNOWLEDGE intent     — completely separate from BUY intent
  ✅ Natural conversational tone  — no robotic bullet walls for simple answers
  ✅ buy cement  ≠  what is cement  (routing fixed)
  ✅ Expanded construction terms  — PCC, RCC, shuttering, curing, lintel, MEP, DPC…
  ✅ 25 platform response templates
  ✅ Backward compatible API      — same answer_query() / inject_modules() / ChatResponse
"""

import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
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

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────
CHATBOT_SYSTEM_PROMPT_COST_AND_REC = """
You are BuildHive Assistant — a helpful, knowledgeable AI built into Pakistan's
leading construction marketplace. You talk like a real person, not a robot.

You help users with:
  1. What a construction product IS — describe it in plain, useful language
  2. HOW TO BUY something — guide them to the marketplace, show steps
  3. Cost estimation — realistic budgets for Pakistani construction
  4. Material recommendations — what to pick and why
  5. Quantity calculations — how much of X do I need
  6. Platform help — how to use BuildHive features
  7. Construction knowledge — phases, terminology, tips
  8. Developer KB — DB schema, API routes, frontend helpers

ROUTING RULES
─────────────
"What is cement?"         → explain the product (Product KB)
"How to buy cement?"      → show purchase steps (Purchase Guide)
"Best cement brand?"      → recommend (Recommendation Module)
"How much cement for 5 marla?" → quantity calculator
"Estimate cost 5 marla Lahore" → Cost Estimation Module

TONE
────
Talk like a knowledgeable friend, not a customer service bot.
Short answers for simple questions. Depth only when needed.
Use Urdu construction terms naturally — sarya, bajri, choona, chowkhat etc.
One clarifying question max if data is missing.
Never start with "Certainly!" or "Great question!".
"""


# ──────────────────────────────────────────────────────────────────────────────
# INTENT TRIGGERS
# ──────────────────────────────────────────────────────────────────────────────

# ── What-is / knowledge triggers (NOT buy/purchase) ───────────────────────────
_KNOWLEDGE_QUESTION_TRIGGERS: Tuple[str, ...] = (
    "what is ", "what are ", "what's a ", "what's the ",
    "define ", "meaning of ", "what does ", "what do you mean by ",
    "explain ", "describe ", "tell me about ",
    "uses of ", "purpose of ", "difference between ",
    "types of ", "how is ", "why use ", "benefits of ",
    "properties of ", "grades of ", "sizes of ", "specifications of ",
)

# ── Buy / purchase triggers (NOT what-is) ─────────────────────────────────────
_PURCHASE_ACTIONS: Tuple[str, ...] = (
    "how to buy", "how do i buy", "where to buy", "where can i buy",
    "how to purchase", "how do i purchase", "where to get",
    "how to order", "where to order", "kahan se", "khareedna", "milega",
    "buy ", "purchase ", "order ", "procure ",
)

# ── Cost / estimation ─────────────────────────────────────────────────────────
_COST_TRIGGERS: Tuple[str, ...] = (
    "how much", "estimated", "expense", "estimate", "cost ", " cost",
    "price", "total price", "budget", "total cost", "cheaper",
    "cost breakdown", "what will it cost", "per sqft", "per sq ft",
    "per marla", "grand total", "pkr", "rupees", " rs ", "rs.",
)

# ── Recommendation ────────────────────────────────────────────────────────────
_REC_TRIGGERS: Tuple[str, ...] = (
    "recommend", "recomend", "suggest", "best option",
    "what should i choose", "top picks", "which material",
    "which should i", "what to pick", "alternatives",
    "compare ", "materials for", "products for",
    "material list", "what materials", "what material",
    "best brand", "best cement", "best tiles", "best rebar",
    "which brand", "which grade",
)

# ── Quantity ──────────────────────────────────────────────────────────────────
_QTY_TRIGGERS: Tuple[str, ...] = (
    "how many bags", "how many bricks", "how many tiles",
    "how many litres", "how much cement", "how much steel",
    "quantity of", "bags needed", "bricks needed", "tiles needed",
    "calculate quantity", "material quantity", "how much material",
)

# ── Unit conversion ───────────────────────────────────────────────────────────
_UNIT_TRIGGERS: Tuple[str, ...] = (
    "marla to sqft", "sqft to marla", "kanal to marla",
    "marla to kanal", "convert marla", "convert sqft",
    "how many sqft in", "how many marla in", "kanal to sqft",
)

# ── Vendor ────────────────────────────────────────────────────────────────────
_VENDOR_TRIGGERS: Tuple[str, ...] = (
    "vendor", "supplier", "contact seller", "is this verified",
    "seller rating", "vendor profile", "register as vendor",
    "list my store", "sell on buildhive", "become a vendor",
    "become a seller",
)

# ── Developer KB ──────────────────────────────────────────────────────────────
_DEV_KB_TRIGGERS: Tuple[str, ...] = (
    "database schema", "db schema", "table schema", "api endpoint",
    "api route", "controller", "backend route", "supabase",
    "express route", "frontend convention", "enum mapper",
    "date formatter", "address formatter", "stripe", "webhook",
    "ai proxy", "ai gateway", "token usage", "ai_tool_usage",
    "buildhive architecture", "directory structure", "users table",
    "products table", "orders table", "businesses table",
    "services table", "proposals table", "projects table",
    "categories table", "order_items table", "disputes table",
    "cost estimator formula", "phase allocation",
    "service package", "packages jsonb", "mapEnum", "formatDateTime",
    "formatAddressBlock", "displayValue",
)

_AMBIGUOUS_DEAL: Tuple[str, ...] = ("best deal", "best value", "cheapest and best")


def _wants_knowledge(text: str) -> bool:
    """User is asking what something IS — not how to buy it."""
    t = text.lower()
    # If they're asking HOW TO BUY, that's purchase, not knowledge
    if any(k in t for k in _PURCHASE_ACTIONS):
        return False
    return any(k in t for k in _KNOWLEDGE_QUESTION_TRIGGERS)


def _wants_purchase(text: str) -> bool:
    """User wants to buy/source something."""
    t = text.lower()
    return any(k in t for k in _PURCHASE_ACTIONS)


def _wants_cost(text: str) -> bool:
    return any(k in text.lower() for k in _COST_TRIGGERS)

def _wants_recommendation(text: str) -> bool:
    return any(k in text.lower() for k in _REC_TRIGGERS)

def _wants_vendor(text: str) -> bool:
    return any(k in text.lower() for k in _VENDOR_TRIGGERS)

def _wants_quantity(text: str) -> bool:
    return any(k in text.lower() for k in _QTY_TRIGGERS)

def _wants_unit_conversion(text: str) -> bool:
    return any(k in text.lower() for k in _UNIT_TRIGGERS)

def _wants_dev_kb(text: str) -> bool:
    return any(k in text.lower() for k in _DEV_KB_TRIGGERS)

def _is_ambiguous_deal(text: str) -> bool:
    return any(k in text.lower() for k in _AMBIGUOUS_DEAL)

def _dual_order(text: str) -> str:
    t = text.lower()
    if any(p in t for p in ("recommendation first", "then cost", "then price")):
        return "rec_first"
    return "cost_first"


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCT KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────
_PRODUCTS_JSON_PATHS = [
    "construction_products.json",
    os.path.join(os.path.dirname(__file__), "construction_products.json"),
    os.path.join(os.path.dirname(__file__), "knowledge", "construction_products.json"),
]


def _load_products() -> List[Dict]:
    for path in _PRODUCTS_JSON_PATHS:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Product KB: %d products from %s", len(data), path)
                return data
            except Exception as exc:
                logger.warning("Could not load products: %s", exc)
    logger.error("construction_products.json not found")
    return []


def _build_product_response(p: Dict) -> str:
    """
    Natural, conversational product answer.
    Short for simple products, more detail when there's something useful to say.
    """
    name        = p.get("name", "this material")
    aliases     = p.get("aliases", [])
    category    = p.get("category", "")
    description = p.get("description", "")
    uses        = p.get("uses", [])
    types_      = p.get("types", [])
    grades      = p.get("grades", [])
    sizes       = p.get("sizes", [])
    thickness   = p.get("thickness", [])
    unit        = p.get("unit", "")
    brands      = p.get("common_brands", [])
    tip         = p.get("tip", "")
    qty_guide   = p.get("quantity_guide", "")
    std_mix     = p.get("standard_mix", "")
    psqca       = p.get("psqca", "")

    local_terms = [a for a in aliases
                   if a.lower() not in (name.lower(), "") and len(a) > 2][:3]

    parts = []

    # Header — name + local term inline
    if local_terms:
        parts.append(f"**{name}** *(also: {', '.join(local_terms)})*\n")
    else:
        parts.append(f"**{name}**\n")

    # Category — only if not obvious
    if category and category.lower() not in name.lower():
        parts.append(f"Category: {category}\n")

    # Core description
    if description:
        parts.append(description)

    # Uses — keep it tight
    if uses:
        use_str = ", ".join(uses[:5])
        parts.append(f"\nUsed for: {use_str}.")

    # Types
    if types_:
        parts.append("\nTypes:")
        for t in types_[:4]:
            parts.append(f"• {t}")

    # Grades
    if grades:
        parts.append("\nGrades:")
        for g in grades:
            parts.append(f"• {g}")

    # Sizes / thickness — inline if short
    if sizes:
        parts.append(f"\nSizes: {', '.join(sizes[:4])}")
    if thickness:
        parts.append(f"\nThickness options: {', '.join(thickness[:4])}")

    # How it's sold
    if unit:
        parts.append(f"\nSold by: {unit}")

    # Standard mix
    if std_mix:
        parts.append(f"\n{std_mix}")

    # PSQCA note
    if psqca:
        parts.append(f"\n📋 {psqca}")

    # Quantity guide
    if qty_guide:
        parts.append(f"\nQuick quantity guide: {qty_guide}")

    # Brands
    if brands:
        parts.append(f"\nPopular brands in Pakistan: {', '.join(brands[:5])}.")

    # Practical tip — most useful part
    if tip:
        parts.append(f"\n💡 {tip}")

    return "\n".join(parts)


class ProductKnowledgeService:
    """
    Two-stage search:
      1. Precise alias match (subject extracted from question)
      2. FAISS semantic fallback
    """

    def __init__(self, model=None):
        self.products: List[Dict] = _load_products()
        self.model = model
        self._faiss_index: Optional[faiss.Index] = None
        self._embed_map: List[int] = []
        self._build_faiss()

    def _build_faiss(self) -> None:
        if not self.products or self.model is None:
            return
        try:
            texts, self._embed_map = [], []
            for i, p in enumerate(self.products):
                doc = (p.get("name", "") + " "
                       + " ".join(p.get("aliases", []))
                       + " " + p.get("description", "")[:80])
                texts.append(doc.lower())
                self._embed_map.append(i)
            vecs = self.model.encode(texts, show_progress_bar=False)
            vecs = np.ascontiguousarray(vecs, dtype="float32")
            faiss.normalize_L2(vecs)
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            self._faiss_index = idx
            logger.info("Product FAISS index: %d entries", len(texts))
        except Exception as exc:
            logger.warning("Product FAISS build failed: %s", exc)

    def search(self, query: str) -> Optional[Dict]:
        result = self._precise(query)
        if result:
            return result
        if self._faiss_index is not None and self.model is not None:
            return self._semantic(query)
        return None

    def _precise(self, query: str) -> Optional[Dict]:
        """
        Strip question prefix to get the subject, then score by longest matching alias.
        Longer alias = more specific = wins.
        """
        q = query.lower().strip()
        subject = q
        prefixes = (
            "what is a ", "what is an ", "what is ", "what are ",
            "explain ", "define ", "meaning of ", "uses of ",
            "purpose of ", "types of ", "tell me about ",
            "describe ", "what does ", "how is ", "why use ",
            "benefits of ", "properties of ", "grades of ",
        )
        for p in prefixes:
            if q.startswith(p):
                subject = q[len(p):].strip("? ")
                break

        best, best_score = None, 0
        for p in self.products:
            all_terms = sorted(
                [p.get("name", "").lower()] + [a.lower() for a in p.get("aliases", [])],
                key=len, reverse=True,
            )
            score = 0
            for term in all_terms:
                if term == subject:
                    score = 100000          # exact subject match — always wins
                elif subject.startswith(term) or term.startswith(subject):
                    score = max(score, len(term) * 5)
                elif term in q:
                    score = max(score, len(term) * 2)
            if score > best_score:
                best_score = score
                best = p
        return best if best_score > 0 else None

    def _semantic(self, query: str, top_k: int = 1) -> Optional[Dict]:
        try:
            vec = self.model.encode([query.lower()], show_progress_bar=False)
            vec = np.ascontiguousarray(vec, dtype="float32")
            faiss.normalize_L2(vec)
            sims, idxs = self._faiss_index.search(vec, top_k)
            if float(sims[0][0]) < 0.45:
                return None
            return self.products[self._embed_map[int(idxs[0][0])]]
        except Exception as exc:
            logger.warning("Product semantic search error: %s", exc)
            return None


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION TERMINOLOGY KB
# ──────────────────────────────────────────────────────────────────────────────
_CONSTRUCTION_TERMS: Dict[str, str] = {
    "grey structure": (
        "Grey structure is basically everything that makes a building stand up — "
        "foundation, columns, beams, brick walls, and roof slab — before any "
        "finishing work like paint or tiles happens.\n\n"
        "It's usually 40–50% of total construction cost. "
        "This phase matters most: bad grey structure can't be fixed after finishing."
    ),
    "difference between pcc and rcc": (
        "PCC (Plain Cement Concrete) and RCC (Reinforced Cement Concrete) are different things:\n\n"
        "\n\n"
        "**PCC** has no steel in it. It is weak lean concrete poured under foundations "
        "\n\n"
        "**RCC** has steel rebar(sarya) inside. Strong structural concrete "
        "used for columns, beams, slabs and everything that actually carries load.\n\n"
        "PCC always goes first, then RCC on top of it."
    ),
    "pcc": (
        "PCC stands for Plain Cement Concrete is a lean, low-strength concrete mix (1:4:8) "
        "that is poured under foundations before the actual structural concrete.\n\n"
        "It gives you a clean, level surface to work on and acts as a damp barrier. "
        "Typically 75–100mm thick. It doesn't carry any structural load it is just a base."
    ),
    "rcc": (
        "RCC is Reinforced Cement Concrete is regular concrete with steel rebar (sarya) "
        "embedded inside to handle tensile forces.\n\n"
        "Standard residential mix in Pakistan: 1 cement : 2 sand : 4 crush.\n"
        "Used everywhere structural strength is needed columns, beams, slabs, lintels, footings.\n\n"
        "Quality depends on four things: correct mix ratio, low water-cement ratio, "
        "proper vibration while pouring, and adequate curing after. Skip any of these "
        "and the slab loses serious strength."
    ),
    "shuttering": (
        "Shuttering (or formwork) is the temporary mould that holds wet concrete in shape until it hardens. Once the concrete sets, the shuttering is removed.\n\n"
        "Usually made from film-faced plywood supported by timber battens and steel props underneath.\n\n"
        "You apply shuttering oil first so the concrete doesn't stick. Columns can be stripped in 24–48 hours, but slab props should stay a minimum 7–14 days. Removing them too early is one of the most common causes of slab collapse on site."
    ),
    "curing": (
        "Curing means keeping fresh concrete moist for several days after pouring so it can hydrate properly and reach full strength.\n\n"
        "so it can hydrate properly and reach full strength.\n\n"
        "Without curing, concrete can lose up to 40% of its design strength which is a big deal for your columns and slabs.\n\n"
        "Minimum 7 days of curing, 14 is better. Methods: ponding water on the slab, covering with wet hessian and plastic, or spraying curing compound. It costs almost nothing but gets skipped constantly on Pakistani sites."
    ),
    "lintel": (
        "A lintel is a horizontal beam placed above a door or window opening. "
        "It carries the wall load above the gap and transfers it to the walls on either side.\n\n"
        "In Pakistan it's almost always RCC. "
        "It needs to bear at least 225mm (9 inches) into the wall on each side.\n\n"
        "Skip the lintel and you'll see diagonal cracks shooting up from the corners "
        "of doors and windows within a year or two."
    ),
    "mep": (
        "MEP stands for Mechanical, Electrical, and Plumbing — the utility systems inside a building.\n\n"
        "M = HVAC, fans, ventilation\n"
        "E = electrical wiring, DB boards, switches, sockets, lighting\n"
        "P = water supply (PPRC pipes), drainage (UPVC), sanitary fittings\n\n"
        "MEP rough-in — running the concealed pipes and conduits — must happen before plastering. "
        "That's why electricians and plumbers need to be on site right after grey structure.\n\n"
        "Budget-wise, MEP typically adds 15–25% on top of grey structure cost."
    ),
    "dpc": (
        "DPC stands for Damp Proof Course — a waterproof barrier built into the lower part "
        "of the foundation walls to stop moisture from rising up through the brickwork.\n\n"
        "In Pakistan it's usually done with bitumen-coated brickwork or waterproof cement mortar "
        "(1:2 ratio with waterproofing additive) at or just above ground level.\n\n"
        "Skip this and you'll have rising damp in your lower walls — a nightmare to fix later."
    ),
    "plinth": (
        "The plinth is the raised base on which a building sits, usually 1 to 2 feet "
        "above finished ground level. It keeps the floor above damp ground and protects "
        "from rainwater flooding.\n\n"
        "Plinth protection is the concrete apron around the base of the building "
        "that stops rainwater from pooling near the footings."
    ),
    "bhk": (
        "BHK = Bedroom, Hall, Kitchen. It's just a quick way to describe a flat or house layout.\n\n"
        "\n\n"
        "2 BHK = 2 bedrooms + 1 hall + 1 kitchen\n"
        "3 BHK = 3 bedrooms + 1 hall + 1 kitchen\n\n"
        "For 5 marla plots, 2–3 BHK is typical. 10 marla gives you 3–4 BHK comfortably."
    ),
    "marla": (
        "Marla is the standard land unit across Pakistan.\n\n"
        "1 Marla = 272 sqft (standard in most cities)\n"
        "1 Kanal = 20 Marla = 5,440 sqft\n\n"
        "Some older properties and rural areas use 225 sqft per marla always confirm locally."
    ),
    "finishing tier": (
        "When people talk about Economy, Standard, or Premium construction, they're describing "
        "the quality of finishing materials such as tiles, paint, fittings, woodwork.\n\n"
        "\n\n"
        "Economy: local brands, basic fittings → ~PKR 800–1,200/sqft\n"
        "Standard: mid-range brands, decent quality (most common) → ~PKR 1,500–2,500/sqft\n"
        "Premium: imported materials, branded fittings → PKR 3,000–5,000+/sqft\n\n"
        "The grey structure cost stays roughly the same — finishing is where the budget difference really shows up."
    ),
    "boq": (
        "BOQ = Bill of Quantities. It's a detailed document listing every material, "
        "labour task, and cost item for a construction project with quantities and unit rates.\n\n"
        "A proper BOQ is prepared by a Quantity Surveyor (QS) from working drawings. "
        "Contractors use it to price their tenders.\n\n"
        "BuildHive's Cost Estimator gives you an automated BOQ-style breakdown for budget planning."
    ),
    "damp proofing": (
        "Damp proofing means treating walls, floors, and foundations to stop moisture getting in.\n\n"
        "Common causes of dampness in Pakistan: rising damp from ground, penetrating damp "
        "through walls, and condensation inside walls.\n\n"
        "Fix it at construction stage , treating dampness retrospectively costs 5–10× more."
    ),
    "honeycombing": (
        "Honeycombing is when concrete has voids or cavities in it after setting — "
        "it looks like a honeycomb when the shuttering is removed.\n\n"
        "Caused by poor compaction (not vibrating the concrete), too little water, "
        "or aggregate segregation.\n\n"
        "It's a serious defect in structural concrete. Small honeycombs can be filled "
        "with non-shrink grout. Large ones may need the member to be broken out and recast."
    ),
}


def _check_construction_term(text: str) -> Optional[str]:
    t = text.lower()
    for term in sorted(_CONSTRUCTION_TERMS.keys(), key=len, reverse=True):
        if term in t:
            return _CONSTRUCTION_TERMS[term]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DEVELOPER KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────
_DEV_KB: Dict[str, str] = {

    "buildhive architecture": (
        "BuildHive runs as a **modular monolith** "
        "\n\n"
        "Three client apps:\n"
        "\n\n"
        "•**Buildhive-market**: Public marketplace (React + Vite + TypeScript)\n"
        "\n\n"
        "•**DashboardBuildhive**: Admin/Supplier panel (React + Vite + TypeScript)\n"
        "\n\n"
        "•**AI microservice**: Python backend deployed at `https://ai-backend-b3yd.onrender.com`\n\n"
        "\n\n"
        "All routes are available at both `/auth/login` and `/api/auth/login`."
    ),

    "directory_structure": """
BuildHive Backend Overview

Location:
buildhive-v-2-new/src/

Core Files
  
• server.js
  Starts the Express server and listens for incoming requests.

• app.js
  Registers middleware, API routes, security settings, and request handlers.

Modules
 
• identity
  Handles user authentication, registration, email verification,
  password recovery, profile management, and saved addresses.

• marketplace
  Manages products, categories, supplier businesses,
  reviews, and customer questions.

• commerce
  Handles carts, checkout, orders, payments,
  refunds, and Stripe integration.

• communication
  Provides chat, notifications, alerts,
  and customer support tickets.

• serviceMarketplace
  Handles contractor services, projects,
  proposals, milestones, and disputes.

• operations
  Provides AI features, search, analytics,
  announcements, and compliance tools.

Database
 -
• database/
  SQL scripts and database utilities.

• supabase/migrations/
  Database schema migration files.
""",

"users_table": """
Users registered on the platform.

Fields:
• id - Unique user ID
• email - Login email address
• full_name - User's full name
• phone - Contact number
• role - buyer, supplier, contractor, or admin
• profile_image - User profile picture
• email_verified - Email verification status
• status - active, inactive, or suspended
• failed_login_attempts - Failed login counter
• locked_until - Account lock timestamp
• last_login - Last login time
• password_hash - Encrypted password
• created_at - Creation timestamp
• updated_at - Last update timestamp
""",

"user_addresses_table": """
Stores user delivery and contact addresses.

Fields:
• id - Address ID
• user_id - Linked user
• label - Home, Office, etc.
• address_line1
• address_line2
• city
• state
• postal_code
• country
• phone
• is_default - Default address flag
""",

"businesses_table": """
Supplier and manufacturer business profiles.

Fields:
• id - Business ID
• user_id - Business owner
• business_name
• business_type - manufacturer, supplier, dealer
• description
• city
• state
• country
• postal_code
• address details
• phone
• email
• website
• logo
• status - pending, approved, rejected
• verified_at - Verification timestamp
""",

"categories_table": """
Product and service categories.

Fields:
• id
• name
• slug
• description
• parent_id - Parent category
• type - product or service
• image
• display_order
• is_active
""",

"products_table": """
Construction materials listed by suppliers.

Fields:
• id
• business_id
• category_id
• name
• slug
• code (SKU)
• description
• price
• compare_at_price
• cost_price
• quantity
• min_order_qty
• weight
• unit
• tags
• track_quantity
• is_active
• featured
• status - pending, approved, rejected
• average_rating
• total_reviews
""",

"orders_table": """
Customer purchase orders.

Fields:
• id
• order_number
• user_id (buyer)
• business_id (supplier)
• subtotal
• tax_amount
• shipping_fee
• discount_amount
• total_amount

Order Status:
• pending
• processing
• shipped
• delivered
• cancelled

Payment Status:
• pending
• paid
• refunded
• failed

Additional Data:
• shipping information
• carrier details
• tracking number
• tracking history
• stripe payment reference
""",

"order_items_table": """
Products included in an order.

Fields:
• id
• order_id
• product_id
• product_name
• quantity
• unit_price
• total_price
""",

"services_table": """
Contractor services available for hire.

Fields:
• id
• contractor_id
• category_id
• title
• description
• starting_price
• price_type (Fixed or Hourly)
• delivery_time
• status
• packages
• rating
• total_reviews
• total_orders

Packages include:
• Basic
• Standard
• Premium
""",

"projects_table": """
Projects between clients and contractors.

Fields:
• id
• contractor_id
• client_id
• service_id
• title
• description
• budget
• status
• progress
• payment_status
""",

"proposals_table": """
Contractor bids submitted for projects.

Fields:
• id
• contractor_id
• client_id
• project_id
• job_title
• job_description
• amount
• delivery_time
• cover_letter
• status
""",

"auth_controller": """
Authentication Controller

register()
• Creates a new user account.
• Generates email verification token.
• Sends confirmation email.

login()
• Validates credentials.
• Tracks failed login attempts.
• Handles account lockout.
• Generates JWT and refresh tokens.
• Updates last login timestamp.

refreshToken()
• Issues a new access token.

verifyEmail()
• Verifies email ownership.

Routes:
POST /auth/register
POST /auth/login
POST /auth/refresh
POST /auth/verify-email
""",

"product_controller": """
Product Controller

getProducts()
• Fetch products with filtering,
  sorting, searching, and pagination.

createProduct()
• Creates a product with
  status = pending.

updateProductStatus()
• Admin approves or rejects products.
• Sends notifications to suppliers.

Routes:
GET /products
POST /products
PATCH /products/:id/status
""",

"order_controller": """
Order Controller

createOrder()
• Creates order records.
• Creates order items.
• Validates stock.
• Updates inventory.
• Clears cart after purchase.

updateOrderStatus()
• Updates order lifecycle.
• Logs status changes to tracking history.

Routes:
POST /orders
PATCH /orders/:id/status
""",

"payment_controller": """
Payment Controller

createStripeCheckoutSession()
• Creates a Stripe checkout session
  for the order amount.

handleStripeWebhook()
• Receives payment confirmation
  from Stripe.
• Updates payment status to paid.

Routes:
POST /payments/stripe/checkout
POST /payments/stripe/webhook

Important:
Stripe webhooks require the raw
request body before JSON parsing.
""",

"operations_controller": """
Operations Controller

proxyAiChat()
• Forwards chatbot requests
  to the AI service.
• Logs usage statistics.

proxyAiRecommend()
• Sends recommendation requests
  to the AI service.
• Records performance metrics.

searchAll()
• Performs global platform search.

getAnnouncementsAdmin()
• Retrieves announcements.

createAnnouncementAdmin()
• Creates and broadcasts announcements
  to selected user roles.
""",

"ai_proxy_flow": """
AI Request Flow

AI Service:
https://ai-backend-b3yd.onrender.com

Flow:
1. User sends a request.
2. JWT token is validated.
3. Request timer starts.
4. Request is forwarded to the AI server.
5. AI returns a response.
6. Usage data is logged.
7. Response is returned to the client.

Supported AI Features:
• Chatbot
• Material Recommendations
• Health Checks
""",

"ai_tool_usage_table": """
Tracks all AI-related activity.

Fields:
• tool_name
  - chatbot
  - material_recommendation
  - ai_health

• action
  - chat
  - recommend
  - health_check

• status
  - completed
  - failed

• duration_ms
  Processing time in milliseconds.

• metadata
  Stores request details,
  routes, and error information.
""",

"cost_estimator": """
Construction Cost Estimator

Area Conversion:
1 Marla = 272 Sq Ft

Inputs:
• City
• Project Type
• Quality Level
• Covered Area
• Floors
• Bedrooms
• Washrooms

Project Types:
• Full Construction
• Grey Structure
• Renovation

Quality Levels:
• Economy
• Standard
• Premium

Cost Breakdown Phases:

1. Site Preparation & Foundation
2. Grey Structure
3. Masonry & Roofing
4. MEP Rough-In
5. Plastering
6. Windows & Doors
7. Flooring & Tiling
8. Paint & Finishes
9. Fixtures & Fittings
10. Testing & Handover
""",

"date_formatter": """
Date Formatting Helper

Purpose:
Convert timestamps into a
clean and readable format.

Example Output:
14 Jun 2026, 9:18 pm

Never display raw ISO dates
directly in the UI.
""",

"enum_mapper": """
Enum Mapping Helper

Converts database values
into user-friendly labels.

Examples:

supplier → Supplier
contractor → Contractor
buyer → Buyer
admin → Admin

pending → Pending
approved → Approved
rejected → Rejected

cash_on_delivery → Cash on Delivery
stripe → Stripe Payment
""",

"address_formatter": """
Address Formatting Helper

Purpose:
Build clean address strings
for display in the UI.

Features:
• Removes empty values
• Removes invalid placeholders
• Prevents messy output

Example:

Input:
Pending, N/A, Lahore, Pakistan

Output:
Lahore, Pakistan

Fallback:
No address on file
""",

"service_packages": """
Service Package Structure

Each service contains
three package levels.

Basic Package
• Price
• Delivery Time
• Revisions
• Features

Standard Package
• Price
• Delivery Time
• Revisions
• Features

Premium Package
• Price
• Delivery Time
• Revisions
• Features

All package information
should always be displayed.
""",

"display_value_rule": """
Display Value Convention

For:
• null
• undefined
• empty strings
• N/A values

Show:
—

Never display raw null
or undefined values in the UI.
""",

"stripe_integration": """
Stripe Payment Flow

1. Create checkout session.
2. Redirect customer to Stripe.
3. Complete payment.
4. Stripe sends webhook event.
5. Payment status becomes Paid.
6. Store payment intent ID.

Stored In:
orders.stripe_payment_intent_id
""",

"order_lifecycle": """
Order Processing Flow

Order Status:

pending
  ↓
processing
  ↓
shipped
  ↓
delivered

Can be cancelled before delivery.

Payment Status:

pending
  ↓
paid

or

pending
  ↓
failed

or

paid
  ↓
refunded

Every status change is
stored in tracking_history.
""",

"product_moderation": """
Product Approval Process

1. Supplier creates a product.
2. Product status = pending.
3. Admin reviews the listing.
4. Product is approved or rejected.
5. Supplier receives notification.
6. Approved products become visible.

Only approved products
can be purchased by buyers.
""",

"user_roles": """
Platform Roles

Buyer
• Browse products
• Purchase materials
• Hire contractors

Supplier
• Manage business profile
• List products
• Process orders

Contractor
• Create services
• Submit proposals
• Manage projects

Admin
• Manage users
• Moderate listings
• Handle announcements
• Monitor platform activity
""",

"global_search": """
Global Search Endpoint

GET /api/operations/search?q=<query>

Searches Across:
• Products
• Services
• Categories
• Businesses

Returns combined results
from all supported modules.
"""
}


def _check_dev_kb(text: str) -> Optional[str]:
    t = text.lower()
    for key in sorted(_DEV_KB.keys(), key=len, reverse=True):
        if key in t:
            return _DEV_KB[key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PURCHASE GUIDES (how to buy — separate from product knowledge)
# ──────────────────────────────────────────────────────────────────────────────
_PURCHASE_GUIDES: Dict[str, str] = {
    "cement": (
        "To buy cement on BuildHive:\n\n"
        "1. Go to **Marketplace → Cement & Concrete** or search 'cement'\n"
        "2. Pick the right grade — OPC 43 for general work, OPC 53 for columns and slabs, "
        "SRC for foundations in clay/sulphate soil\n"
        "3. Compare at least 3 suppliers — price, delivery time, and verified badge\n"
        "4. Order in full truck loads (100–200 bags) for the best bulk price\n"
        "5. Check the manufacturing date on the bag — cement older than 3 months loses strength\n"
        "6. Arrange dry, covered storage at site before delivery\n\n"
        "Popular brands: DG Khan, Lucky, Maple Leaf, Cherat, Askari."
    ),
    "rebar": (
        "To buy steel rebar (sarya) on BuildHive:\n\n"
        "1. Go to **Marketplace → Steel & Metal** or search 'rebar'\n"
        "2. Specify the grade — Grade 40 for general walls, Grade 60 for columns and slabs\n"
        "3. Buy from PSQCA-certified mills only — Ittefaq, Agha Steel, Amreli, Mughal Steel\n"
        "4. Always ask for Mill Test Certificates (MTCs) on delivery\n"
        "5. Light surface rust is fine, flaking rust is not — check before accepting\n"
        "6. Store off the ground on wooden spacers\n\n"
        "Calculate roughly 3.5–4 kg per sqft of slab area."
    ),
    "bricks": (
        "To buy bricks on BuildHive:\n\n"
        "1. Go to **Marketplace → Raw Materials** or search 'bricks'\n"
        "2. Always ask for Awwal (first class) for load-bearing walls\n"
        "3. Request a sample batch — tap the brick, it should ring clearly, no dull thud\n"
        "4. Source locally within 50km to save on transport\n"
        "5. Confirm price is per 1,000 bricks, not per piece\n"
        "6. Order 10–15% extra for cuts and breakage\n\n"
        "Rule of thumb: ~9 bricks per sqft of wall area."
    ),
    "tiles": (
        "To buy tiles on BuildHive:\n\n"
        "1. Go to **Marketplace → Tiles & Flooring** or search 'tiles'\n"
        "2. Add 15% to your measured area for cuts and wastage\n"
        "3. Always request a sample before the full order\n"
        "4. Make sure all tiles in the order share the same batch number — "
        "different batches have slight colour variation\n"
        "5. Compare prices per sqft including adhesive and grout\n\n"
        "Popular brands: Master Tiles, Shabbir Tiles, Ghauri Tiles."
    ),
    "paint": (
        "To buy paint on BuildHive:\n\n"
        "1. Go to **Marketplace → Paint & Finishing** or search 'paint'\n"
        "2. 1 litre covers roughly 12 sqft with 2 coats — calculate accordingly\n"
        "3. Pick the right finish — matte for walls, semi-gloss for kitchens and trims\n"
        "4. 20-litre drums are cheaper per litre than 4-litre tins for large areas\n"
        "5. Use lead-free, low-VOC paint for interiors — especially children's rooms\n\n"
        "Popular brands: ICI Dulux, Berger, Nippon, Boraq, Brighto."
    ),
    "sand": (
        "To buy sand on BuildHive:\n\n"
        "1. Search 'sand' on the Marketplace and filter by your city\n"
        "2. Specify the use: coarse sand for concrete, fine washed sand for plaster\n"
        "3. Check for clay content — rub it between your fingers, it shouldn't feel sticky\n"
        "4. Rough estimate: 0.5 cft per sqft of construction area\n\n"
        "Price is usually per trolley or per cft depending on the supplier."
    ),
    "default": (
        "To find and buy materials on BuildHive:\n\n"
        "1. Use the search bar at the top of the Marketplace\n"
        "2. Filter by city, quality grade, and your budget\n"
        "3. Compare at least 3 suppliers — check ratings and the ✓ Verified badge\n"
        "4. Message the seller directly for bulk pricing or delivery queries\n"
        "5. Add to cart → Checkout → enter delivery address and payment details → confirm"
    ),
}


def _get_purchase_guide(material_kw: str) -> str:
    kw = material_kw.lower()
    for key in _PURCHASE_GUIDES:
        if key in kw or kw in key:
            return _PURCHASE_GUIDES[key]
    return _PURCHASE_GUIDES["default"]


# ──────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
_MATERIAL_ENTITIES: Dict[str, str] = {
    "cement": "Raw Materials", "bricks": "Raw Materials", "brick": "Raw Materials",
    "sand": "Raw Materials", "steel": "Raw Materials", "rebar": "Raw Materials",
    "sarya": "Raw Materials", "gravel": "Raw Materials", "crush": "Raw Materials",
    "tiles": "Flooring Materials", "tile": "Flooring Materials",
    "marble": "Flooring Materials", "granite": "Flooring Materials",
    "paint": "Paint & Finishing", "primer": "Paint & Finishing",
    "wood": "Wood & Carpentry", "door": "Doors & Windows", "window": "Doors & Windows",
    "pipe": "Plumbing", "pprc": "Plumbing", "upvc": "Plumbing",
    "wire": "Electrical", "mcb": "Electrical", "conduit": "Electrical",
    "waterproofing": "Chemicals", "sanitary": "Sanitary", "kitchen": "Kitchen",
}

_SUPPORTED_CITIES = ("lahore", "karachi", "islamabad", "rawalpindi",
                     "faisalabad", "multan", "peshawar", "quetta")
_BUILDING_TYPES   = ("house", "home", "apartment", "villa", "farmhouse",
                     "shop", "office", "plaza", "mosque", "warehouse")


def _extract_area(text: str) -> Optional[str]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(marla|kanal|sqft|sq ft|square feet|sqm)", text.lower())
    if not m: return None
    unit = m.group(2).replace("sq ft", "sqft").replace("square feet", "sqft")
    return f"{m.group(1)} {unit}"

def _extract_city(text: str) -> Optional[str]:
    t = text.lower()
    for c in _SUPPORTED_CITIES:
        if c in t: return c.title()
    return None

def _extract_building_type(text: str) -> Optional[str]:
    t = text.lower()
    for bt in _BUILDING_TYPES:
        if bt in t: return bt
    return None

def _extract_floors(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"\b(\d+)\s*(floors?|storeys?|stories?)\b", t)
    if m: return int(m.group(1))
    if "double storey" in t or "double story" in t: return 2
    if "single storey" in t or "single story" in t: return 1
    return None

def _extract_bhk(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\s*bhk\b", text.lower())
    return int(m.group(1)) if m else None

def _extract_tier(text: str) -> Optional[str]:
    t = text.lower()
    for tier in ("economy", "standard", "premium", "luxury"):
        if tier in t: return tier.title()
    return None

def _marla_to_sqft(v: float) -> float: return v * 272.0
def _kanal_to_sqft(v: float) -> float: return v * 20.0 * 272.0


# ──────────────────────────────────────────────────────────────────────────────
# QUANTITY CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────
def _calc_quantity(material: str, sqft: float) -> str:
    m = material.lower()
    if "cement" in m:
        return (f"For {sqft:,.0f} sqft you'll need roughly **{round(sqft * 0.4)} bags of cement** "
                f"(~0.4 bags/sqft for standard concrete mix). Order 10% extra for wastage.")
    if "brick" in m:
        return (f"For {sqft:,.0f} sqft of wall area you'll need about **{round(sqft * 9):,} bricks** "
                f"(~9 bricks/sqft for a standard 4.5-inch wall). Order 10–15% extra for cuts.")
    if "tile" in m or "marble" in m or "granite" in m:
        return (f"For {sqft:,.0f} sqft, order tiles for **{round(sqft * 1.15):,} sqft** "
                f"— that's your area plus 15% for cuts and breakage.")
    if "paint" in m:
        return (f"For {sqft:,.0f} sqft you'll need about **{round(sqft / 12)} litres of paint** "
                f"(1 litre covers ~12 sqft with 2 coats). Don't forget a primer coat first.")
    if any(k in m for k in ("steel", "rebar", "sarya")):
        return (f"For a {sqft:,.0f} sqft slab, budget roughly **{round(sqft * 3.75):,} kg of steel rebar**. "
                f"That's the standard ~3.5–4 kg/sqft for residential slabs. "
                f"Get a structural engineer to confirm for load-bearing floors.")
    if any(k in m for k in ("sand", "bajri", "crush")):
        return (f"For {sqft:,.0f} sqft, estimate around **{round(sqft * 0.5):,} cft of {material}**. "
                f"This is a rough guide — actual quantity depends on mix design and depth.")
    return (f"I don't have a specific formula for {material} yet. "
            f"Use the **Cost Estimator** tab for a full material breakdown.")


# ──────────────────────────────────────────────────────────────────────────────
# UNIT CONVERSION
# ──────────────────────────────────────────────────────────────────────────────
def _handle_unit_conversion(text: str) -> Optional[str]:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*marla\s+to\s+sqft", t)
    if m:
        v = float(m.group(1))
        return f"{v:g} marla = **{_marla_to_sqft(v):,.0f} sqft** (1 marla = 272 sqft)"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:sqft|sq ft|square feet)\s+(?:in|to)\s*marla", t)
    if m:
        v = float(m.group(1))
        return f"{v:,.0f} sqft = **{v/272:.2f} marla** (272 sqft = 1 marla)"
    m = re.search(r"(\d+(?:\.\d+)?)\s*kanal\s+to\s+marla", t)
    if m:
        v = float(m.group(1))
        return f"{v:g} kanal = **{v*20:,.0f} marla** = **{_kanal_to_sqft(v):,.0f} sqft** (1 kanal = 20 marla)"
    m = re.search(r"(\d+(?:\.\d+)?)\s*kanal\s+to\s+sqft", t)
    if m:
        v = float(m.group(1))
        return f"{v:g} kanal = **{_kanal_to_sqft(v):,.0f} sqft** (1 kanal = 20 marla × 272 sqft)"
    m = re.search(r"how many sqft (?:in|is)\s+(\d+(?:\.\d+)?)\s*marla", t)
    if m:
        v = float(m.group(1))
        return f"{v:g} marla = **{_marla_to_sqft(v):,.0f} sqft**"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PLATFORM RESPONSES
# ──────────────────────────────────────────────────────────────────────────────
PLATFORM_RESPONSES: List[Dict[str, Any]] = [
    {"keywords": ["change password", "update password", "reset password", "new password"],
     "response": "Go to **Account Settings → Security → Change Password**, "
                 "enter your current password then your new one, and save. "
                 "Use a mix of letters, numbers, and symbols for a strong password."},

    {"keywords": ["forgot password", "can't login", "can't log in", "reset my password"],
     "response": "On the login page, click **Forgot Password**, enter your email, "
                 "and check your inbox for a reset link. Also check spam if it doesn't arrive. "
                 "The link expires in a few hours."},

    {"keywords": ["delete account", "close account", "remove account"],
     "response": "Go to **Account Settings → Danger Zone → Delete Account** "
                 "and confirm with your password. Fair warning — this is permanent and can't be undone."},

    {"keywords": ["two factor", "2fa", "two-factor", "enable 2fa", "authenticator"],
     "response": "Go to **Account Settings → Security**, toggle **Two-Factor Authentication ON**, "
                 "scan the QR code with Google Authenticator or Authy, "
                 "then enter the 6-digit code to confirm. Strongly recommended."},

    {"keywords": ["change email", "update email", "email settings"],
     "response": "Go to **Account Settings → Profile**, edit your email, "
                 "then verify the new address via the confirmation link sent to it."},

    {"keywords": ["place order", "how to order", "how do i order", "add to cart", "checkout"],
     "response": "Find the product in the Marketplace → **Add to Cart** → "
                 "open your cart → **Checkout** → enter delivery address and payment → confirm. "
                 "Compare a few suppliers first to get the best price."},

    {"keywords": ["track order", "order status", "where is my order", "delivery status",
                  "track my order", "where is my delivery"],
     "response": "Go to **Buyer Dashboard → Active Orders**, find your order and click **Track**. "
                 "You'll also get SMS and email updates at each delivery stage."},

    {"keywords": ["cancel order", "return order", "refund", "cancel my order"],
     "response": "Go to **Buyer Dashboard → Active Orders**, open the order, "
                 "and click **Cancel** (if not yet dispatched) or **Return** (if already delivered). "
                 "Cancellations are instant if the order hasn't been sent out yet."},

    {"keywords": ["add listing", "create listing", "add product", "list product",
                  "new listing", "post product", "list a product"],
     "response": "Go to **Seller Dashboard → Listing Management → Add New Listing**, "
                 "fill in the product name, category, price, grade, city, and stock, "
                 "upload photos, and hit **Publish**. "
                 "Listings with photos get significantly more views."},

    {"keywords": ["edit listing", "update listing", "update product", "change price",
                  "edit product", "update my listing"],
     "response": "Go to **Seller Dashboard → Listing Management**, find the product, "
                 "click **Edit**, update whatever you need, and save."},

    {"keywords": ["delete listing", "remove listing", "remove product", "deactivate listing"],
     "response": "In **Seller Dashboard → Listing Management**, you can either "
                 "**Deactivate** (hides it temporarily) or **Delete** (permanent). "
                 "Use Deactivate if you plan to relist it later."},

    {"keywords": ["seller dashboard", "my dashboard", "sales analytics", "view sales",
                  "how to sell", "selling on buildhive"],
     "response": "Your **Seller Dashboard** has everything — Listing Management, "
                 "incoming orders, sales analytics with revenue charts, and financial overview. "
                 "Access it from the profile icon in the top-right corner."},

    {"keywords": ["cost estimate", "estimate cost", "project cost", "how much will it cost",
                  "building cost", "construction cost", "cost estimator", "use the estimator"],
     "response": "Open the **Cost Estimation** tab, enter your area, quality grade, city, and floors, "
                 "then click Calculate. You'll get a total cost, cost per sqft, "
                 "full itemized breakdown, and AI cost-saving tips."},

    {"keywords": ["recommendation", "recommend materials", "material list", "ai recommend",
                  "suggest materials", "material recommendation", "get recommendations"],
     "response": "Open the **Recommendation System** tab, describe your project "
                 "(e.g. '5 marla house in Lahore'), enter area, city, and quality preference, "
                 "and hit Get AI Recommendations. "
                 "You'll get a full material list across 7+ categories with quantities and costs."},

    {"keywords": ["message seller", "contact seller", "chat with seller",
                  "send message to seller", "inbox", "messages"],
     "response": "Open the product listing and click **Contact Seller** or **Message**. "
                 "All your conversations are in the **Messages** icon in the top navigation."},

    {"keywords": ["leave review", "write review", "rate seller", "rate product",
                  "add review", "give feedback"],
     "response": "Go to **Buyer Dashboard → Order History**, find the completed order, "
                 "click **Leave a Review**, give a star rating and write your feedback. "
                 "Only verified buyers can leave reviews."},

    {"keywords": ["register", "sign up", "create account", "how to register",
                  "create a buildhive account"],
     "response": "Click **Sign Up** on the homepage, choose your role "
                 "(Buyer, Seller, or Freelancer), fill in your details, and verify your email. "
                 "Your role determines which dashboard and features you get."},

    {"keywords": ["login", "sign in", "log in"],
     "response": "Click **Login**, enter your email and password. "
                 "If you forgot your password, click Forgot Password on the login page."},

    {"keywords": ["vendor registration", "register as vendor", "list my store",
                  "sell on buildhive", "become a vendor", "become a seller", "register as seller"],
     "response": "Click **Sign Up → Seller**, enter your business name, CNIC/NTN, "
                 "city, and contact info, then submit verification documents. "
                 "The team reviews within 48 hours. Once approved you can start adding listings. "
                 "Verified sellers get a ✓ badge and rank higher in search."},

    {"keywords": ["contact vendor", "contact supplier", "is this vendor verified",
                  "vendor rating", "seller profile", "check seller"],
     "response": "Click any product → seller's name → **View Seller Profile**. "
                 "You'll see their rating, verified badge, city, and product categories. "
                 "Hit **Contact Seller** to message them. "
                 "Stick to ✓ Verified sellers for quality assurance."},

    {"keywords": ["freelancer", "register as freelancer", "post service",
                  "hire freelancer", "hire contractor", "service provider"],
     "response": "To register as a freelancer: Profile → Switch to Freelancer → "
                 "add your services and portfolio. "
                 "To hire one: Marketplace → Services tab → filter by skill and city."},

    {"keywords": ["buyer dashboard", "my orders", "my purchases", "order history",
                  "view my orders"],
     "response": "Your **Buyer Dashboard** has Active Orders, Order History, Saved Items, "
                 "and Financial Overview. Access it from the profile icon top-right."},

    {"keywords": ["payment method", "add payment", "payment options", "how to pay"],
     "response": "BuildHive accepts Bank Transfer, JazzCash, EasyPaisa, "
                 "Cash on Delivery (select sellers), and Stripe card payments. "
                 "Manage payment methods in **Account Settings → Payment Methods**."},

    {"keywords": ["notifications", "alerts", "notification settings", "enable notifications"],
     "response": "Go to **Account Settings → Notifications** to control order updates, "
                 "price alerts, messages, and promotions. Choose Email, SMS, or In-App delivery. "
                 "Price Alerts are useful if you're waiting for a product to drop in price."},

    {"keywords": ["wishlist", "saved items", "favorites", "save product"],
     "response": "Click the heart/bookmark icon on any product listing to save it. "
                 "View saved items in **Buyer Dashboard → Saved Items**."},
]


def _match_platform(query: str) -> Optional[str]:
    # Don't intercept cost or rec queries
    if _wants_cost(query) or _wants_recommendation(query):
        return None
    q = query.lower()
    best, best_score = None, 0
    for entry in PLATFORM_RESPONSES:
        score = 0
        for kw in entry["keywords"]:
            if kw in q:
                score += 2 if " " in kw else 1
            elif " " in kw and all(w in q for w in kw.split()):
                score += 1
        if score > best_score:
            best_score = score
            best = entry["response"]
    return best if best_score >= 1 else None


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION PHASES
# ──────────────────────────────────────────────────────────────────────────────
def _check_phases(text: str) -> Optional[str]:
    t = text.lower()
    if not any(k in t for k in ("phase", "step", "process", "timeline", "how long",
                                 "stages", "sequence", "how is a house built",
                                 "order of construction", "how long does construction")):
        return None
    return (
        "Here's how a typical residential build goes in Pakistan:\n\n"
        "**1. Site Preparation** (1–2 weeks) — clearing, levelling, setting out, soil test\n"
        "**2. Foundation** (3–6 weeks) — excavation, PCC, DPC, footings\n"
        "**3. Grey Structure** (3–5 months) — columns, beams, brickwork, lintels, roof slab\n"
        "**4. MEP Rough-in** (2–4 weeks) — concealed PPRC pipes and electrical conduits before plastering\n"
        "**5. Plastering** (3–5 weeks) — internal and external; needs to cure before finishing\n"
        "**6. Finishing** (2–4 months) — tiles, marble, paint, doors, windows, cabinets\n"
        "**7. MEP Finishing** (2–4 weeks) — fixtures, switches, sanitary ware, lights\n"
        "**8. Handover / Snagging** (1–2 weeks) — final punch list, cleanup\n\n"
        "Total for a 5 marla full build: roughly **8–14 months** depending on contractor speed, "
        "materials availability, and weather."
    )


# ──────────────────────────────────────────────────────────────────────────────
# VENDOR HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def _handle_vendor(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("register", "list my store", "sell on", "become a vendor",
                             "become a seller", "vendor registration")):
        return ("To register as a vendor on BuildHive:\n\n"
                "1. Click **Sign Up → Seller** on the homepage\n"
                "2. Enter your business name, CNIC/NTN, city, and contact details\n"
                "3. Submit your verification documents\n"
                "4. The team reviews within 48 hours and notifies you\n"
                "5. Once approved, you can start adding listings\n\n"
                "Verified sellers get a ✓ badge and rank higher in search results.")
    if any(k in t for k in ("contact", "verified", "rating", "profile", "is this seller")):
        return ("Click any product listing → seller's name → **View Seller Profile** "
                "to see their rating, verified badge, city, and product categories.\n\n"
                "To contact them, click **Contact Seller** on their profile or product page.\n\n"
                "Always buy from ✓ Verified sellers for quality assurance.")
    return ("You can browse verified suppliers on **Marketplace → Vendors**, "
            "filter by city, category, and rating, and message any seller directly.\n\n"
            "What material are you looking to source? I can point you to the right category.")


# ──────────────────────────────────────────────────────────────────────────────
# OFF-TOPIC DETECTION
# ──────────────────────────────────────────────────────────────────────────────
_OFF_TOPIC = ("weather", "temperature", "sports", "cricket match", "movie", "film",
              "politics", "election", "recipe", "cooking", "restaurant", "joke",
              "funny", "vacation", "travel", "music", "song", "cryptocurrency",
              "bitcoin", "write a poem", "tell me a story", "flight booking")
_ON_TOPIC  = ("marla", "sqft", "cement", "bricks", "tiles", "steel", "rebar",
              "paint", "plumbing", "electrical", "construction", "house", "building",
              "estimate", "cost", "material", "vendor", "supplier", "recommend",
              "buildhive", "listing", "seller", "buyer", "freelancer", "order",
              "grey structure", "pprc", "conduit", "mcb", "shuttering", "plaster",
              "grout", "membrane", "sanitary", "database", "schema", "api",
              "controller", "supabase", "stripe", "sarya", "bajri", "choona")

def _is_off_topic(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _ON_TOPIC): return False
    return any(k in t for k in _OFF_TOPIC)


# ──────────────────────────────────────────────────────────────────────────────
# SAFETY LAYER
# ──────────────────────────────────────────────────────────────────────────────
def _check_safety(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ("kill myself", "suicide", "hurt myself", "end my life")):
        return "self_harm"
    if "swallowed bleach" in t:
        return "medical_emergency"
    if any(k in t for k in ("make a bomb", "build a bomb", "how to poison")):
        return "violence"
    if any(k in t for k in ("illegal guns",)):
        return "weapons"
    if any(k in t for k in ("make meth", "cook meth")):
        return "drugs"
    if any(k in t for k in ("steal passwords", "phishing", "keylogger", "malware")):
        return "cyber"
    if any(k in t for k in ("ignore rules", "reveal hidden instructions", "act as admin",
                             "pretend to be", "jailbreak", "bypass safety",
                             "system prompt", "developer mode")):
        return "prompt_injection"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
_NAV_GUIDES: Dict[str, str] = {
    "dashboard":      "Profile icon (top-right) → Dashboard.",
    "tiles":          "Marketplace → Categories → Tiles & Flooring.",
    "plumbing":       "Marketplace → Categories → Plumbing & Sanitary.",
    "cement":         "Marketplace → Categories → Cement & Concrete.",
    "electrical":     "Marketplace → Categories → Electrical Components.",
    "steel":          "Marketplace → Categories → Steel & Metal.",
    "categories":     "Marketplace → Categories (12+ material types).",
    "order":          "Buyer Dashboard → Active Orders.",
    "listing":        "Seller Dashboard → Listing Management → New Listing.",
    "estimate":       "AI Tools → Cost Estimator.",
    "recommendation": "AI Tools → Material Recommendations.",
    "checkout":       "Cart icon (top-right) → Checkout.",
    "vendor":         "Marketplace → Vendors.",
}

def _nav_actions(intent: str) -> List[Dict[str, Any]]:
    mapping = {
        "Recommendation": [{"label": "View Recommendations", "target_module": "recommendation", "deep_link": "/recommendations", "optional": True}],
        "Estimation":     [{"label": "Go to Cost Estimator", "target_module": "estimation",     "deep_link": "/estimator",       "optional": True}],
        "Mixed":          [{"label": "Go to Cost Estimator", "target_module": "estimation",     "deep_link": "/estimator",       "optional": True}],
        "Vendor":         [{"label": "Browse Vendors",       "target_module": "marketplace",    "deep_link": "/vendors",         "optional": True}],
        "Materials":      [{"label": "Browse Materials",     "target_module": "marketplace",    "deep_link": "/marketplace",     "optional": True}],
        "ProductKB":      [{"label": "Find on BuildHive",    "target_module": "marketplace",    "deep_link": "/marketplace",     "optional": True}],
    }
    return mapping.get(intent, [])


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER TEMPLATE (structured output for UI parsing)
# ──────────────────────────────────────────────────────────────────────────────
def _router_template(*, intent: str, inputs_used: Dict[str, Any],
                     result_summary: str, warnings: Optional[List[str]] = None,
                     include_nav: bool = True) -> str:
    warn_block    = "\n".join(f"- {w}" for w in (warnings or []) if w) or "- None"
    inputs_block  = "\n".join(f"- **{k}**: {v}" for k, v in (inputs_used or {}).items()
                               if v not in (None, "")) or "- (none provided)"
    nav_json = ""
    if include_nav:
        actions = _nav_actions(intent)
        if actions:
            nav_json = json.dumps({"navigation_actions": actions}, ensure_ascii=False, indent=2)
    body = (f"  Intent\n**{intent}**\n\n  Inputs used\n{inputs_block}\n\n"
            f"  Result\n{result_summary}\n\n  Warnings / Exclusions\n{warn_block}\n")
    if nav_json:
        body += f"\n  Optional navigation\n{nav_json}\n"
    return body
# ──────────────────────────────────────────────────────────────────────────────
# RESULT FORMATTERS
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_estimation(cost_r: Dict[str, Any]) -> Tuple[str, List[str]]:
    warns: List[str] = []
    if not isinstance(cost_r, dict) or cost_r.get("status") not in ("success", "ok"):
        return "Couldn't generate an estimate from those inputs.", warns
    summary = ((cost_r.get("breakdown") or {}).get("summary") or {})
    grand = summary.get("grand_total")
    cps   = cost_r.get("cost_per_sqft")
    if cost_r.get("warnings"):
        warns.extend([str(w) for w in (cost_r.get("warnings") or [])][:6])
    parts = []
    if grand is not None: parts.append(f"Estimated total: **PKR {int(grand):,}**")
    if cps   is not None: parts.append(f"(**PKR {int(cps):,}/sqft**)")
    return " ".join(parts) if parts else "Couldn't generate an estimate.", warns

def _fmt_recommendation(rec_r: Dict[str, Any]) -> Tuple[str, List[str]]:
    warns: List[str] = []
    if not isinstance(rec_r, dict) or rec_r.get("status") != "success":
        return "Couldn't generate recommendations from those inputs.", warns
    recs = rec_r.get("recommendations") or rec_r.get("categories") or {}
    if isinstance(recs, dict):
        for cat, items in recs.items():
            if items:
                top       = items[0] or {}
                top_name  = top.get("item_name") or top.get("name") or top.get("title")
                top_price = top.get("market_price_pkr") or top.get("final_price_pkr")
                if top_name:
                    if top_price: return f"Top pick: **{top_name}** ({cat}) — ≈ **PKR {int(top_price):,}**.", warns
                    return f"Top pick: **{top_name}** ({cat}).", warns
    return "Couldn't generate recommendations from those inputs.", warns


# ──────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP SUGGESTIONS
# ──────────────────────────────────────────────────────────────────────────────
_FOLLOW_UPS: Dict[str, Dict[str, List[str]]] = {
    "product_knowledge": {
        "buyer":      ["Find this on BuildHive", "Get a cost estimate", "Get material recommendations"],
        "seller":     ["List this product", "View category demand", "Check market prices"],
        "freelancer": ["Use this in your proposal", "Find related products", "Get cost estimate"],
    },
    "purchase_help": {
        "buyer":      ["Compare prices across vendors", "Calculate quantity needed", "Check vendor ratings"],
        "seller":     ["List this material", "View category demand", "Compare with competitors"],
        "freelancer": ["Add to your project materials list", "Find local suppliers", "Request a bulk quote"],
    },
    "developer_kb": {
        "buyer":      ["Get a cost estimate", "Get material recommendations", "Platform help"],
        "seller":     ["Manage my listings", "View sales data", "Platform help"],
        "freelancer": ["View open projects", "Submit a proposal", "Platform help"],
    },
    "recommendation": {
        "buyer":      ["Get a cost estimate for these materials", "Save recommendation to a project", "Ask about a specific material"],
        "seller":     ["List one of these materials", "Check current stock levels", "View similar products"],
        "freelancer": ["Calculate project cost", "Find materials for your project", "View buyer project listings"],
    },
    "cost_estimation": {
        "buyer":      ["Refine with a different city or tier", "Get material recommendations", "Ask about cost-saving options"],
        "seller":     ["View demand for these materials", "Adjust pricing strategy", "List related materials"],
        "freelancer": ["Find buyers with this budget", "Post a proposal for similar work", "Check labour rate by city"],
    },
    "cost_and_recommendation": {
        "buyer":      ["Refine with Economy tier", "Refine with Premium tier", "Ask for a narrower scope"],
        "seller":     ["See which SKUs match the picks", "List popular materials for this tier", "Review demand by city"],
        "freelancer": ["Tie this BOQ to your proposal", "Get labour breakdown", "Export assumptions for client"],
    },
    "quantity_calculator": {
        "buyer":      ["Get a full material list for my project", "Find vendors for this material", "Get a cost estimate"],
        "seller":     ["Check if you have enough stock", "View demand for this material", "Update listing quantity"],
        "freelancer": ["Add to project BOQ", "Find suppliers for bulk order", "Get cost estimate"],
    },
    "unit_conversion": {
        "buyer":      ["Estimate cost for this area", "Get material recommendations", "Calculate material quantities"],
        "seller":     ["Check listings for this area size", "View demand by plot size", "Update listing specifications"],
        "freelancer": ["Calculate materials for this plot", "Estimate project cost", "Find buyers with this plot size"],
    },
    "vendor_info": {
        "buyer":      ["Contact this vendor", "Compare with other suppliers", "Request a quote"],
        "seller":     ["View your seller profile", "Improve your rating", "Update your listings"],
        "freelancer": ["Find verified suppliers", "Compare vendor prices", "Request bulk pricing"],
    },
    "construction_faq": {
        "buyer":      ["Get material recommendations", "Get a cost estimate", "Browse related products"],
        "seller":     ["List materials for this phase", "View category demand", "Check compliance requirements"],
        "freelancer": ["Use this in your proposal", "Find related projects", "Browse buyer requirements"],
    },
    "platform_help": {
        "buyer":      ["Track your current order", "Browse material categories", "Use the AI cost estimator"],
        "seller":     ["Manage your listings", "View sales analytics", "Update product pricing"],
        "freelancer": ["Update your portfolio", "View open project requests", "Manage active proposals"],
    },
    "general_question": {
        "buyer":      ["Get a material recommendation", "Get a cost estimate", "Browse related products"],
        "seller":     ["List this material", "View category demand", "Compare with competitors"],
        "freelancer": ["Use this in your proposal", "Find related projects", "Browse buyer requirements"],
    },
    "clarification_needed": {
        "buyer":      ["Get material recommendations", "Get a cost estimate", "Get platform help"],
        "seller":     ["Manage my listings", "View sales data", "Get platform help"],
        "freelancer": ["View open projects", "Submit a proposal", "Get platform help"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA CONTRACT
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ChatResponse:
    response_id:        str
    text:               str
    intent:             str
    suggested_follow_ups: List[str]
    language_hint:      str
    source:             str
    confidence:         float = 0.0
    data:               Optional[Dict[str, Any]] = None
    products:           List[Dict[str, Any]] = field(default_factory=list)
    steps:              List[str] = field(default_factory=list)
    navigation_actions: List[Dict[str, Any]] = field(default_factory=list)
    quick_replies:      List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# CHATBOT MODULE v4 FINAL
# ──────────────────────────────────────────────────────────────────────────────
class ChatBotModule:
    """
    BuildHive Chatbot v4 FINAL.

    Pipeline order:
      1  Safety
      2  Greeting / empty / noise
      3  Unit conversion
      4  Math shortcuts
      5  Platform template (25 templates)
      6  PRODUCT KNOWLEDGE  — "what is X", "explain X"
      7  PURCHASE GUIDE     — "how to buy X", "where to buy X"   ← separated from 6
      8  DEVELOPER KB       — DB schema, API, frontend helpers
      9  Construction terms — PCC, RCC, grey structure, curing…
      10 Construction phases — timeline, stages
      11 Vendor intent
      12 Quantity calculator
      13 Off-topic guard
      14 Entity → purchase guide (fallback for material buy queries)
      15 Dual cost + recommendation
      16 Single intent (cost / recommendation / clarification / navigation)
      17 KB hybrid search (FAISS + keyword)
      18 LLM rewrite fallback
    """

    def __init__(
        self,
        kb_path:    str = "buildhive_knowledge_base_enhanced.json",
        model_name: str = "all-MiniLM-L6-v2",
        llm: Optional["LLMHelper"] = None,
    ):
        self.kb_path    = kb_path
        self.model_name = model_name
        self.model      = get_embedding_model(model_name)

        self.preprocessor = QueryPreprocessor()
        self.detector     = IntentDetector(model=self.model)
        self.llm          = llm or LLMHelper()

        self.recommendation_module: Optional["RecommendationModule"] = None
        self.cost_module:           Optional["CostEstimationModule"] = None

        # Platform KB (FAISS)
        self.kb_data:       List[Dict]            = []
        self.kb_embeddings: Optional[np.ndarray]  = None
        self.kb_index:      Optional[faiss.Index] = None
        self.query_variations: Dict[str, int]     = {}

        # Product KB
        self.product_kb = ProductKnowledgeService(model=self.model)

        # Cache
        self._query_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_embed_cache_max = 256
        self._session_state: Dict[str, Dict[str, Any]] = {}

        self._load_kb()
        logger.info(
            "✓ ChatBotModule v4 FINAL | products: %d | dev KB topics: %d",
            len(self.product_kb.products), len(_DEV_KB),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_modules(self,
                       recommendation: "RecommendationModule",
                       cost: "CostEstimationModule") -> None:
        self.recommendation_module = recommendation
        self.cost_module = cost
        logger.info("✓ Modules injected")

    def answer_query(
        self,
        query:           str,
        user_role:       str  = "buyer",
        current_page:    Optional[str] = None,
        conversation_id: Optional[str] = None,
        use_llm:         bool = True,
    ) -> Dict[str, Any]:
        try:
            resp = self._build_response(query, user_role, current_page, use_llm, conversation_id)
        except Exception as exc:
            logger.error("answer_query error: %s", exc, exc_info=True)
            resp = self._emergency_fallback(query, user_role)
        return self._to_dict(resp)

    def get_categories(self) -> List[str]:
        return sorted({d.get("category", "General") for d in self.kb_data})

    def get_faq_by_category(self, category: str) -> List[Dict]:
        return [{"question": d.get("question"), "answer": d.get("answer"),
                 "category": d.get("category")}
                for d in self.kb_data
                if d.get("category", "").lower() == category.lower()]

    def get_health_status(self) -> Dict:
        return {
            "status":             "online",
            "version":            "v4-final",
            "kb_loaded":          len(self.kb_data) > 0,
            "kb_size":            len(self.kb_data),
            "product_kb_size":    len(self.product_kb.products),
            "product_kb_faiss":   self.product_kb._faiss_index is not None,
            "dev_kb_topics":      len(_DEV_KB),
            "construction_terms": len(_CONSTRUCTION_TERMS),
            "platform_templates": len(PLATFORM_RESPONSES),
            "cost_module":        self.cost_module is not None,
            "rec_module":         self.recommendation_module is not None,
            "pipeline":           (
                "safety → greeting → unit_conversion → platform → "
                "PRODUCT_KB → PURCHASE_GUIDE → DEV_KB → "
                "construction_terms → phases → vendor → qty_calc → "
                "off_topic → entity_purchase → dual_intent → "
                "single_intent → kb_hybrid → llm"
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
        return {"policy":           CHATBOT_SYSTEM_PROMPT_COST_AND_REC,
                "knowledge_tables": load_cost_estimation_assistant_knowledge()}

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _build_response(
        self,
        query:           str,
        user_role:       str,
        current_page:    Optional[str],
        use_llm:         bool,
        conversation_id: Optional[str],
    ) -> ChatResponse:

        role = (user_role or "buyer").lower()
        if role not in ("buyer", "seller", "freelancer"):
            role = "buyer"

        lang_hint = self.preprocessor.detect_language_hint(query)
        clean     = self.preprocessor.clean(query)
        intent: IntentResult = self.detector.classify(clean)
        cid = (conversation_id or "").strip() or "default"
        st  = self._session_state.setdefault(cid, {})

        logger.info("v4-final | '%s' | intent=%s(%.2f)", query, intent.label, intent.score)

        # ── 1. Safety ─────────────────────────────────────────────────────────
        unsafe = _check_safety(query)
        if unsafe == "self_harm":
            return self._make(
                text=("I'm sorry you're going through that. "
                      "Please reach out to someone you trust or call an emergency helpline. "
                      "If you're in immediate danger, call emergency services now."),
                intent="safety_refusal", source="safety", lang=lang_hint, confidence=1.0)
        if unsafe == "medical_emergency":
            return self._make(
                text="That's urgent — contact emergency services or a poison control centre immediately.",
                intent="safety_guidance", source="safety", lang=lang_hint, confidence=1.0)
        if unsafe:
            return self._make(
                text="That's not something I can help with. Ask me anything about construction or BuildHive.",
                intent="safety_refusal", source="safety", lang=lang_hint, confidence=1.0)

        # ── 2. Greeting ───────────────────────────────────────────────────────
        if self._is_greeting(clean) or self._is_greeting(query):
            return self._make(
                text="Hey! What can I help you with? 👋",
                intent="greeting", source="utility", lang=lang_hint, confidence=1.0,
                nav=_nav_actions("Mixed"),
                quick_replies=["What is cement?", "Estimate construction cost",
                               "How to buy tiles", "How does BuildHive work?"])

        # ── 3. Empty / noise ──────────────────────────────────────────────────
        if not clean or clean == "[empty query]" or self._looks_like_noise(query):
            return self._make(
                text=("Not sure what you're asking — try something like:\n\n"
                      "- *\"What is PPRC pipe?\"*\n"
                      "- *\"How to buy cement on BuildHive\"*\n"
                      "- *\"Estimate cost for 5 marla house Lahore\"*"),
                intent="clarification_needed", source="utility", lang=lang_hint, confidence=1.0,
                quick_replies=["What is rebar?", "Cost estimate", "Platform help"])

        # ── 4. Unit conversion ────────────────────────────────────────────────
        if _wants_unit_conversion(clean):
            result = _handle_unit_conversion(clean)
            if result:
                return self._make(
                    text=result, intent="unit_conversion", source="utility",
                    lang=lang_hint, confidence=1.0,
                    follow_ups=self._follow_ups("unit_conversion", role, current_page),
                    quick_replies=["Estimate cost for this area",
                                   "Calculate material quantities",
                                   "Get material recommendations"])

        # ── 5. Math shortcuts ─────────────────────────────────────────────────
        mmul = re.search(r"^\s*(\d+)\s*[x×*]\s*(\d+)\s*\??\s*$", query.lower())
        if mmul:
            a, b = int(mmul.group(1)), int(mmul.group(2))
            return self._make(text=str(a * b), intent="math",
                              source="utility", lang=lang_hint, confidence=1.0)

        # ── 6. Platform template ──────────────────────────────────────────────
        platform_answer = _match_platform(clean)
        if platform_answer:
            return self._make(
                text=platform_answer, intent="platform_help", source="platform_template",
                lang=lang_hint, confidence=1.0,
                follow_ups=self._follow_ups("platform_help", role, current_page))

        # ── 7. PRODUCT KNOWLEDGE — "what is X" ───────────────────────────────
        if _wants_knowledge(clean):
            product = self.product_kb.search(clean)
            if product:
                return self._make(
                    text=_build_product_response(product),
                    intent="product_knowledge", source="product_kb",
                    lang=lang_hint, confidence=0.95,
                    follow_ups=self._follow_ups("product_knowledge", role, current_page),
                    nav=_nav_actions("ProductKB"),
                    quick_replies=[f"How to buy {product.get('name','this')} on BuildHive",
                                   "Get a cost estimate",
                                   "What is the difference between PCC and RCC?"])
            # Try construction terms
            term_def = _check_construction_term(clean)
            if term_def:
                return self._make(
                    text=term_def, intent="construction_faq", source="faq",
                    lang=lang_hint, confidence=0.95,
                    follow_ups=self._follow_ups("construction_faq", role, current_page),
                    quick_replies=["Get material recommendations",
                                   "Get a cost estimate", "Construction phases"])
            # LLM fallback for unknown products
            if use_llm:
                llm_ans = self._llm_product_fallback(clean)
                if llm_ans:
                    return self._make(
                        text=llm_ans, intent="product_knowledge", source="llm_fallback",
                        lang=lang_hint, confidence=0.6,
                        follow_ups=self._follow_ups("general_question", role, current_page))

        # ── 8. PURCHASE GUIDE — "how to buy X" ───────────────────────────────
        if _wants_purchase(clean):
            # Extract the material keyword from the query
            mat_kw = self._extract_material_from_purchase_query(clean)
            guide  = _get_purchase_guide(mat_kw)
            return self._make(
                text=guide, intent="purchase_help", source="purchase_guide",
                lang=lang_hint, confidence=0.9,
                follow_ups=self._follow_ups("purchase_help", role, current_page),
                nav=_nav_actions("Materials"),
                quick_replies=[f"What is {mat_kw}?",
                               f"How many {mat_kw} do I need for 5 marla?",
                               "Get full project cost estimate"])

        # ── 9. DEVELOPER KB ───────────────────────────────────────────────────
        if _wants_dev_kb(clean):
            dev_answer = _check_dev_kb(clean)
            if dev_answer:
                return self._make(
                    text=dev_answer, intent="developer_kb", source="dev_kb",
                    lang=lang_hint, confidence=0.95,
                    follow_ups=self._follow_ups("developer_kb", role, current_page),
                    quick_replies=["Show orders table schema",
                                   "How does the AI proxy work?",
                                   "Frontend date formatter"])

        # ── 10. Construction terminology ──────────────────────────────────────
        term_def = _check_construction_term(clean)
        if term_def:
            return self._make(
                text=term_def, intent="construction_faq", source="faq",
                lang=lang_hint, confidence=0.95,
                follow_ups=self._follow_ups("construction_faq", role, current_page))

        # ── 11. Construction phases ───────────────────────────────────────────
        phase_guide = _check_phases(clean)
        if phase_guide:
            return self._make(
                text=phase_guide, intent="construction_phases", source="faq",
                lang=lang_hint, confidence=0.95,
                follow_ups=self._follow_ups("construction_faq", role, current_page),
                nav=_nav_actions("Estimation"))

        # ── 12. Vendor intent ─────────────────────────────────────────────────
        if _wants_vendor(clean):
            return self._make(
                text=_handle_vendor(clean), intent="vendor_info", source="vendor_guide",
                lang=lang_hint, confidence=0.9,
                follow_ups=self._follow_ups("vendor_info", role, current_page),
                nav=_nav_actions("Vendor"),
                quick_replies=["Contact this vendor",
                               "Compare with other suppliers",
                               "Register as a vendor"])

        # ── 13. Quantity calculator ───────────────────────────────────────────
        if _wants_quantity(clean):
            qty_resp = self._handle_qty(clean)
            if qty_resp:
                return self._make(
                    text=qty_resp, intent="quantity_calculator", source="qty_calc",
                    lang=lang_hint, confidence=0.95,
                    follow_ups=self._follow_ups("quantity_calculator", role, current_page),
                    nav=_nav_actions("Materials"),
                    quick_replies=["Find vendors for this material",
                                   "Get a full project cost estimate",
                                   "Get material recommendations"])

        # ── 14. Off-topic ─────────────────────────────────────────────────────
        if _is_off_topic(clean) and intent.score < 0.30:
            return self._make(
                text=("I'm focused on construction and BuildHive. I can help with:\n\n"
                      "• **What is X?** - any construction product or material\n"
                      "• **How to buy X?** -purchasing guides and steps\n"
                      "• **Cost estimates** - realistic budgets for your project\n"
                      "• **Material recommendations** - what to choose and why\n"
                      "• **Quantity calculations**  how much of X you need\n"
                      "• **Platform help** - how to use BuildHive features\n\n"
                      "What do you need?"),
                intent="off_topic", source="off_topic_filter",
                lang=lang_hint, confidence=0.0,
                follow_ups=self._follow_ups("platform_help", role, current_page),
                quick_replies=["What is rebar?", "Cost estimate", "Material recommendations"])

        # ── 15. Entity → purchase guide (catch-all for material buy queries) ──
        entities = self._extract_entities(clean)
        if entities["has_purchase_action"] and entities["materials"]:
            mat_kw, _ = entities["materials"][0]
            guide     = _get_purchase_guide(mat_kw)
            # Enrich with live product prices if modules available
            extra_products: List[Dict[str, Any]] = []
            if self.recommendation_module:
                try:
                    rec = self.recommendation_module.recommend(
                        text=mat_kw, top_n_per_cat=5)
                    for v in (rec.get("categories") or {}).values():
                        extra_products.extend(v)
                    extra_products = extra_products[:5]
                except Exception:
                    pass
            return self._make(
                text=guide, intent="purchase_help", source="purchase_guide",
                lang=lang_hint, confidence=0.85,
                products=extra_products,
                follow_ups=self._follow_ups("purchase_help", role, current_page),
                nav=_nav_actions("Materials"),
                quick_replies=[f"What is {mat_kw}?",
                               f"Calculate {mat_kw} quantity",
                               "Get full project cost estimate"])

        # ── 16. Dual cost + recommendation ────────────────────────────────────
        want_cost = _wants_cost(clean)
        want_rec  = _wants_recommendation(clean)
        run_dual  = ((want_cost and want_rec) or _is_ambiguous_deal(clean)) \
                    and self.cost_module is not None \
                    and self.recommendation_module is not None

        if run_dual and intent.label not in ("clarification_needed", "platform_help", "navigation"):
            return self._dual_intent(clean, role, lang_hint, use_llm, current_page)

        # ── 17. Single intent ─────────────────────────────────────────────────
        if intent.label == "recommendation" or want_rec:
            return self._rec_intent(clean, role, lang_hint, use_llm, current_page)

        if intent.label == "cost_estimation" or want_cost:
            return self._cost_intent(clean, role, lang_hint, use_llm, current_page)

        if intent.label == "clarification_needed":
            st["awaiting_menu_choice"] = True
            return self._make(
                text=("What would you like help with?\n\n"
                      "\n\n"
                      "**only ask me Questions related to construction and BuildHive**\n"),
                intent="clarification_needed", source="clarification",
                lang=lang_hint, confidence=float(intent.score or 0.0),
                nav=_nav_actions("Mixed"),
                quick_replies=["1", "2", "3"])

        if intent.label == "navigation":
            q  = clean.lower()
            for kw, guide in _NAV_GUIDES.items():
                if kw in q:
                    if current_page and kw in current_page.lower():
                        continue
                    return self._make(text=guide, intent="navigation", source="static",
                                      lang=lang_hint, confidence=float(intent.score or 0.0),
                                      follow_ups=self._follow_ups("platform_help", role, current_page))
            return self._make(
                text=("BuildHive's main sections:\n\n"

                      "• **Marketplace**  It allows you to browse and buy materials\n"
                      "• **AI Tools**  It allows you to get recommendations and cost estimator\n"
                      "• **Dashboard** it allows you to orders, listings, and messages\n\n"
                      "Where do you want to go?"),
                intent="navigation", source="static",
                lang=lang_hint, confidence=float(intent.score or 0.0))

        # ── 18. KB hybrid search ──────────────────────────────────────────────
        text = self._kb_search(clean, top_k=5, role=role)

        if not text and self.recommendation_module:
            for city in getattr(self.recommendation_module, "city_advisory", {}).keys():
                if city.lower() in clean.lower():
                    advice = self.recommendation_module.get_city_advisory(city)
                    if advice:
                        text = (f"Construction tips for **{city}**:\n\n{advice}\n\n"
                                f"Want material recommendations specific to {city}?")
                        break

        if text:
            try:
                rewritten = self.llm.rewrite_kb_answer(question=clean, answer=text)
                if rewritten and rewritten.strip():
                    text = rewritten
            except Exception as exc:
                logger.warning("LLM rewrite skipped: %s", exc)

        return self._make(
            text=text or self._fallback(),
            intent=intent.label, source="kb",
            lang=lang_hint, confidence=float(intent.score or 0.0),
            follow_ups=self._follow_ups(intent.label, role, current_page))

    # ── LLM product fallback ──────────────────────────────────────────────────

    def _llm_product_fallback(self, query: str) -> Optional[str]:
        try:
            prompt = (
                "You're a Pakistani construction expert. Answer this question about a construction "
                "material or product in the Pakistan context. Be concise and practical. "
                "Include local brand names and Urdu terms where relevant. "
                "Don't use bullet walls — write like a knowledgeable person.\n\n"
                f"Question: {query}"
            )
            return self.llm.complete(prompt=prompt)
        except Exception as exc:
            logger.warning("LLM product fallback failed: %s", exc)
            return None

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _dual_intent(self, clean, role, lang_hint, use_llm, current_page):
        bt       = _extract_building_type(clean) or "house"
        city_h   = _extract_city(clean)
        area_h   = _extract_area(clean)
        floors_h = _extract_floors(clean)
        tier_h   = _extract_tier(clean)

        missing = []
        if not area_h: missing.append("area (e.g. **5 marla** or **2000 sqft**)")
        if not city_h: missing.append("city (e.g. **Lahore**, **Karachi**)")

        if missing:
            text = _router_template(
                intent="Mixed", inputs_used={"query": clean},
                result_summary=("To give you an accurate estimate and recommendations, I need:\n"
                                + "\n".join(f"- {m}" for m in missing[:2])),
                warnings=[" Exclusions: land, design fees, NOCs, utility connections, furniture."],
                include_nav=True)
            return self._make(text=text, intent="clarification_needed",
                              source="clarification_router", lang=lang_hint, confidence=0.5,
                              follow_ups=self._follow_ups("clarification_needed", role, current_page),
                              nav=_nav_actions("Mixed"),
                              quick_replies=["5 marla Lahore", "10 marla Karachi", "1 kanal Islamabad"])

        try:
            order = _dual_order(clean)
            if order == "rec_first":
                rec_r  = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
                cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
            else:
                cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
                rec_r  = self.recommendation_module.recommend(text=clean, use_llm=use_llm)

            rec_s,  _          = _fmt_recommendation(rec_r)
            cost_s, cost_warns = _fmt_estimation(cost_r)

            text = _router_template(
                intent="Mixed",
                inputs_used={"building_type": bt,
                             "city":          city_h or "Lahore (default)",
                             "area":          area_h,
                             "floors":        floors_h or 1,
                             "finishing_tier":tier_h or "Standard (default)"},
                result_summary=f"**Cost Estimate**\n{cost_s}\n\n**Material Recommendations**\n{rec_s}",
                warnings=[" Excludes: land, design fees, NOCs, utility connections, furniture.",
                          " Benchmark based estimate > actual costs may vary."]
                         + (cost_warns[:2] if cost_warns else []),
                include_nav=True)

            dual_products: List[Dict[str, Any]] = []
            try:
                for _c, items in list((rec_r.get("categories") or {}).items())[:4]:
                    dual_products.extend(items[:1])
            except Exception:
                pass

            return self._make(
                text=text, intent="cost_and_recommendation",
                source="cost_and_recommendation", lang=lang_hint, confidence=0.85,
                data={"cost_estimation": cost_r, "recommendation": rec_r},
                products=dual_products,
                follow_ups=self._follow_ups("cost_and_recommendation", role, current_page),
                nav=_nav_actions("Mixed"),
                quick_replies=["Refine with Economy tier", "Refine with Luxury tier", "Show only materials"])
        except Exception as exc:
            logger.warning("Dual path failed: %s", exc)
            return self._cost_intent(clean, role, lang_hint, use_llm, current_page)

    def _rec_intent(self, clean, role, lang_hint, use_llm, current_page):
        if self.recommendation_module is None:
            return self._make(text=self._kb_search(clean, top_k=3, role=role),
                              intent="recommendation", source="kb_fallback",
                              lang=lang_hint, confidence=0.4,
                              nav=_nav_actions("Recommendation"))

        bt = _extract_building_type(clean) or "house"
        city_h   = _extract_city(clean)
        area_h   = _extract_area(clean)
        floors_h = _extract_floors(clean)

        if not area_h:
            text = _router_template(
                intent="Recommendation",
                inputs_used={"query": clean, "building_type": bt},
                result_summary="What's the total area? (e.g. 5 marla, 2000 sqft)",
                include_nav=True)
            return self._make(text=text, intent="clarification_needed",
                              source="clarification_router", lang=lang_hint, confidence=0.5,
                              nav=_nav_actions("Recommendation"),
                              quick_replies=["5 marla", "10 marla", "1 kanal", "Enter manually"])

        try:
            result  = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
            rec_s, _ = _fmt_recommendation(result)
            text = _router_template(
                intent="Recommendation",
                inputs_used={"building_type": bt, "city": city_h or "Lahore (default)",
                             "area": area_h, "floors": floors_h or 1},
                result_summary=rec_s, include_nav=True)
            return self._make(text=text, intent="recommendation",
                              source="recommendation_module", lang=lang_hint,
                              confidence=0.85, data=result,
                              nav=_nav_actions("Recommendation"),
                              quick_replies=["Get cost estimate", "Filter by Economy tier",
                                            "Show only cement and steel"])
        except Exception as exc:
            logger.error("Recommendation module error: %s", exc)
            return self._make(text=self._kb_search(clean, top_k=3, role=role),
                              intent="recommendation", source="kb_fallback",
                              lang=lang_hint, confidence=0.3,
                              nav=_nav_actions("Recommendation"))

    def _cost_intent(self, clean, role, lang_hint, use_llm, current_page):
        if self.cost_module is None:
            return self._make(text=self._kb_search(clean, top_k=3, role=role),
                              intent="cost_estimation", source="kb_fallback",
                              lang=lang_hint, confidence=0.4,
                              nav=_nav_actions("Estimation"))

        bt       = _extract_building_type(clean) or "house"
        city_h   = _extract_city(clean)
        area_h   = _extract_area(clean)
        floors_h = _extract_floors(clean)
        tier_h   = _extract_tier(clean)
        bhk_h    = _extract_bhk(clean)

        if not area_h:
            text = "What's the total area? (e.g. 5 marla or 2000 sqft)",
            return self._make(text=text, intent="clarification_needed",
                              source="clarification_router", lang=lang_hint, confidence=0.5,
                              nav=_nav_actions("Estimation"),
                              quick_replies=["5 marla", "10 marla", "1 kanal", "Enter manually"])

        try:
            result   = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
            cost_s, cost_warns = _fmt_estimation(result)
            text = _router_template(
                intent="Estimation",
                inputs_used={"building_type": bt, "city": city_h or "Lahore (default)",
                             "area": area_h, "floors": floors_h or 1,
                             "quality_tier": tier_h or "Standard (default)", "bhk": bhk_h or "auto"},
                result_summary=cost_s,
                warnings=[" Excludes: land, design fees, NOCs, utility connections, furniture.",
                          " Benchmark-based estimate — actual costs vary."]
                         + (cost_warns[:2] if cost_warns else []),
                include_nav=True)
            return self._make(text=text, intent="cost_estimation", source="cost_module",
                              lang=lang_hint, confidence=0.85, data=result,
                              nav=_nav_actions("Estimation"),
                              quick_replies=["Get material recommendations",
                                            "Try Economy tier", "Try Luxury tier"])
        except Exception as exc:
            logger.error("Cost module error: %s", exc)
            return self._make(text=self._kb_search(clean, top_k=3, role=role),
                              intent="cost_estimation", source="kb_fallback",
                              lang=lang_hint, confidence=0.3,
                              nav=_nav_actions("Estimation"))

    # ── Quantity handler ──────────────────────────────────────────────────────

    def _handle_qty(self, clean: str) -> Optional[str]:
        t = clean.lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(marla|kanal|sqft|sq ft|square feet)", t)
        if not m:
            return ("To calculate the quantity, I need:\n"
                    "1. The material (cement, bricks, tiles…)\n"
                    "2. The area (5 marla, 2000 sqft…)\n\n"
                    "Try: *\"How many bags of cement for 5 marla house?\"*")
        val  = float(m.group(1))
        unit = m.group(2).lower().replace("sq ft", "sqft").replace("square feet", "sqft")
        sqft = (_marla_to_sqft(val) if unit == "marla" else
                _kanal_to_sqft(val) if unit == "kanal" else val)

        for kw in ("cement", "brick", "tile", "marble", "granite",
                   "paint", "steel", "rebar", "sarya", "sand", "bajri", "crush"):
            if kw in t:
                return _calc_quantity(kw, sqft)

        return (f"For {val:g} {unit} ({sqft:,.0f} sqft), rough estimates:\n\n"
                f"• Cement: ~{round(sqft * 0.4)} bags\n"
                f"• Bricks (Awwal): ~{round(sqft * 9):,}\n"
                f"• Steel rebar (sarya): ~{round(sqft * 3.75):,} kg\n"
                f"• Paint: ~{round(sqft / 12)} litres\n\n"
                "Ask me about a specific material for a more precise figure.")

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_greeting(text: str) -> bool:
        return (text or "").strip().lower() in (
            "hi", "hello", "hey", "assalam o alaikum", "assalamualaikum",
            "aoa", "salam", "good morning", "good afternoon", "good evening")

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        t = (text or "").strip()
        if not t: return True
        alnum = sum(ch.isalnum() for ch in t)
        return (alnum <= 2 and len(t) <= 8) or (len(t) <= 3 and not t.isalnum())

    @staticmethod
    def _extract_material_from_purchase_query(text: str) -> str:
        """Pull the material name from 'how to buy X' queries."""
        t = text.lower()
        for prefix in _PURCHASE_ACTIONS:
            if prefix.strip() in t:
                idx = t.find(prefix.strip())
                if idx >= 0:
                    after = t[idx + len(prefix.strip()):].strip("? ").strip()
                    if after and len(after) > 1:
                        # Return first 1–2 words
                        return " ".join(after.split()[:2])
        # Fallback: find known material entities in query
        for kw in _MATERIAL_ENTITIES:
            if kw in t:
                return kw
        return "materials"

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        t = text.lower()
        materials = [(kw, cat) for kw, cat in _MATERIAL_ENTITIES.items() if kw in t]
        has_purchase_action = any(act in t for act in _PURCHASE_ACTIONS)
        return {"materials": materials, "has_purchase_action": has_purchase_action}

    def _follow_ups(self, intent: str, role: str,
                    current_page: Optional[str] = None) -> List[str]:
        role_map    = _FOLLOW_UPS.get(intent, _FOLLOW_UPS.get("general_question", {}))
        suggestions = list(role_map.get(role, role_map.get("buyer", [])))
        if current_page:
            page = current_page.lower()
            suggestions = [s for s in suggestions
                           if not any(kw in page for kw in s.lower().split()[:2])]
        return suggestions[:3]

    def _fallback(self) -> str:
        return ("Not sure what you mean. Here's what I can do:\n\n"
                "• **What is X?** — *\"What is cement?\"*, *\"What is MCB?\"*\n"
                "• **How to buy X** — *\"How to buy PPRC pipe on BuildHive\"*\n"
                "• **Cost estimate** — *\"Estimate cost for 5 marla house Lahore\"*\n"
                "• **Recommendations** — *\"Suggest tiles for my bathroom\"*\n"
                "• **Quantities** — *\"How many bags of cement for 10 marla?\"*\n"
                "• **Unit conversion** — *\"5 marla to sqft\"*\n"
                "• **Platform help** — *\"How to add a listing?\"*\n\n"
                "Try rephrasing what you need.")

    def _emergency_fallback(self, query: str, role: str) -> ChatResponse:
        return self._make(text=self._fallback(), intent="error", source="error_fallback",
                          lang="english", confidence=0.0,
                          follow_ups=self._follow_ups("clarification_needed", role))

    # ── Response factory ──────────────────────────────────────────────────────

    def _make(self, *, text: str, intent: str, source: str, lang: str,
              confidence: float,
              data:           Optional[Dict[str, Any]]  = None,
              products:       Optional[List[Dict[str, Any]]] = None,
              steps:          Optional[List[str]] = None,
              follow_ups:     Optional[List[str]] = None,
              nav:            Optional[List[Dict[str, Any]]] = None,
              quick_replies:  Optional[List[str]] = None) -> ChatResponse:
        return ChatResponse(
            response_id          = str(uuid.uuid4()),
            text                 = text,
            intent               = intent,
            suggested_follow_ups = follow_ups or [],
            language_hint        = lang,
            source               = source,
            confidence           = confidence,
            data                 = data,
            products             = products or [],
            steps                = steps or [],
            navigation_actions   = nav or [],
            quick_replies        = quick_replies or [],
        )

    @staticmethod
    def _to_dict(resp: ChatResponse) -> Dict[str, Any]:
        return {
            "status":               "success",
            "response_id":          resp.response_id,
            "answer":               resp.text,
            "intent":               resp.intent,
            "confidence":           resp.confidence,
            "suggested_follow_ups": resp.suggested_follow_ups,
            "language_hint":        resp.language_hint,
            "source":               resp.source,
            "data":                 resp.data,
            "products":             resp.products,
            "steps":                resp.steps,
            "navigation_actions":   resp.navigation_actions,
            "quick_replies":        resp.quick_replies,
        }

    def llm_status(self) -> Dict[str, Any]:
        return {"available": self.llm.is_available, "model": self.llm.model_name}

    # ── KB hybrid search ──────────────────────────────────────────────────────

    def _kb_search(self, query: str, top_k: int = 5, role: str = "buyer") -> str:
        if not self.kb_data:
            return self._fallback()

        sem_hits = self._semantic_search(query, top_k=top_k)
        kw_hits  = self._keyword_search(query, top_k=top_k)

        combined: Dict[int, float] = {}
        for idx, score in sem_hits: combined[idx] = combined.get(idx, 0) + score * 0.65
        max_kw = max((s for _, s in kw_hits), default=1) or 1
        for idx, score in kw_hits:
            combined[idx] = combined.get(idx, 0) + (score / max_kw) * 0.35

        if not combined:
            return self._fallback()

        ROLE_BOOST = {
            "seller":     ["Seller Module", "Marketplace System"],
            "freelancer": ["Freelancer / Service Provider"],
            "buyer":      ["Buyer Module", "Cost Estimation System", "AI Recommendation System"],
        }
        for idx in combined:
            if self.kb_data[idx].get("category") in ROLE_BOOST.get(role, []):
                combined[idx] *= 1.2

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        best_idx, best_score = ranked[0]
        if best_score < 0.25:
            return self._fallback()

        doc        = self.kb_data[best_idx]
        raw_answer = doc.get("answer", "")
        question   = doc.get("question", "")
        if not raw_answer:
            return self._fallback()

        answer = raw_answer.strip()
        if len(answer) > 400:
            sentences = answer.split(". ")
            answer = ". ".join(sentences[:2]).strip()
            if not answer.endswith("."): answer += "."
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
        if self.kb_index is None or self.kb_index.ntotal == 0: return []
        k   = min(top_k, self.kb_index.ntotal)
        vec = self._encode_query(query)
        sims, idxs = self.kb_index.search(vec, k)
        return [(int(idxs[0][i]), float(sims[0][i])) for i in range(k)]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        qw = set(query.lower().split())
        scores: List[Tuple[int, float]] = []
        for idx, doc in enumerate(self.kb_data):
            score = 0.0
            score += len(qw & {t.lower() for t in doc.get("tags", [])}) * 3
            score += len(qw & set(doc.get("question", "").lower().split())) * 2
            score += len(qw & set(doc.get("answer",   "").lower().split())) * 0.5
            for var in doc.get("query_variations", []):
                score += len(qw & set(var.lower().split())) * 2.5
            if score > 0: scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── KB loading ────────────────────────────────────────────────────────────

    def _load_kb(self) -> None:
        paths = [self.kb_path,
                 "buildhive_knowledge_base_enhanced.json",
                 "buildhive_knowledge_base.json"]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.kb_data = json.load(f)
                    logger.info("KB: %d entries from %s", len(self.kb_data), path)
                    self._index_kb()
                    return
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
        logger.error("No platform KB found — KB search unavailable")

    def _index_kb(self) -> None:
        if not self.kb_data: return
        for idx, doc in enumerate(self.kb_data):
            self.query_variations[doc.get("question", "").lower()] = idx
            for var in doc.get("query_variations", []):
                self.query_variations[var.lower()] = idx

        questions  = [d.get("question", "") for d in self.kb_data]
        embeddings = self.model.encode(questions, show_progress_bar=False)
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.kb_embeddings = embeddings
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        idx.add(embeddings)
        self.kb_index = idx
        logger.info("KB FAISS index: %d vectors", embeddings.shape[0])