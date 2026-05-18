"""
Recommendation System Module — Full-coverage, product-aware hybrid AI.

Key design principles (v2 spec):
  - City / budget / relevance: ranking signals only — never exclude products.
  - Finishing tier (from `quality` or `finishing_tier`) vs `finishing_tier_min`
    on each catalog row: hard filter — items below the user tier are omitted.
  - Catalog `room_type` is informational only (not a user filter).
  - For broad project queries ("5 marla house") expand to ALL construction
    categories; each category returns up to top_n_per_cat items.
  - Per-category fallback search: thin categories are topped up via FAISS
    reconstruction, after the same room/tier filters.
  - Scoring (ranking within the filtered pool):
        final_score = 0.65 * cosine + 0.20 * rec_score + 0.10 * quality_match
                    + 0.05 * city_match
  - LLM on for category-level explanations when use_llm=True.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from .finishing_catalog import (
    explain_tier,
    item_eligible_for_finishing_tier,
    normalize_finishing_tier,
)
from .llm_helper import LLMHelper
from .data_store import Phase2DataStore, Phase2Paths
from .shared_models import get_embedding_model

logger = logging.getLogger(__name__)


# Phase-2 product catalog columns (derived from materials_master + pricing_data)
CATALOG_COLUMNS: Tuple[str, ...] = (
    "material_id",
    "item_name",
    "category",
    "phase",
    "subcategory",
    "quality_grade",
    "unit",
    "price_avg_pkr",
    "price_min_pkr",
    "price_max_pkr",
    "confidence_score",
    "usage_type",
    "usage_ratio",
    "functional_tag",
    "synonyms",
    "description",
    "search_text",
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny in-process LRU (Step H — caching)
# ─────────────────────────────────────────────────────────────────────────────

class _LRUCache:
    """Minimal thread-naive LRU for hashable keys → arbitrary values."""

    def __init__(self, max_size: int = 256):
        self._cache: "OrderedDict[Any, Any]" = OrderedDict()
        self._max = max_size

    def get(self, key: Any) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED QUANTITY CONSTANTS
# (kept locally so the recommendation module is self-contained; cost module
#  has its own copy because it derives full BOQ + room-additions on top.)
# ─────────────────────────────────────────────────────────────────────────────

QUANTITIES_PER_SQFT: Dict[str, Dict[str, Any]] = {
    "cement":  {"per_sqft": 0.25, "unit": "bags"},
    "bricks":  {"per_sqft": 12.5, "unit": "units"},
    "steel":   {"per_sqft": 1.0,  "unit": "kg"},
    "sand":    {"per_sqft": 0.15, "unit": "cft"},
    "gravel":  {"per_sqft": 0.08, "unit": "cft"},
}

FINISHING_PER_SQFT: Dict[str, Dict[str, Any]] = {
    "tiles":  {"per_sqft": 1.0,   "unit": "sqft"},
    "paint":  {"per_sqft": 0.083, "unit": "liters"},  # 1 L ≈ 12 sqft
    "wood":   {"per_sqft": 0.02,  "unit": "cft"},
}


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY TIER MAPPING (soft ranking signals — NOT hard filters)
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_TIERS: Dict[str, List[str]] = {
    "Premium":  ["Premium", "Standard", "A-Grade", "Awwal (1st)", "Luxury"],
    "Standard": ["Standard", "Economy", "Premium", "A-Grade", "B-Grade", "Medium", "Fine"],
    "Economy":  ["Economy", "Standard", "B-Grade", "Coarse", "Medium"],
    "Luxury":   ["Luxury", "Premium", "A+++", "Imported"],
}


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT-TYPE DETECTION KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

FULL_PROJECT_KEYWORDS: Tuple[str, ...] = (
    "marla", "kanal", "house", "ghar", "makan", "home",
    "residential", "villa", "building", "floor", "story",
    "construction", "build", "project",
)

# ─────────────────────────────────────────────────────────────────────────────
# ITEM-LEVEL QUANTITY ESTIMATION RULES
# keyword → rule applied when item_name / category contains keyword
# ─────────────────────────────────────────────────────────────────────────────

ITEM_QUANTITY_RULES: List[Dict[str, Any]] = [
    {"keywords": ["cement"],                 "per_sqft": 0.25,  "unit": "bags"},
    {"keywords": ["brick", "bricks"],        "per_sqft": 12.5,  "unit": "units"},
    {"keywords": ["steel", "rebar", "rod"],  "per_sqft": 1.0,   "unit": "kg"},
    {"keywords": ["sand", "baalu"],          "per_sqft": 0.15,  "unit": "cft"},
    {"keywords": ["gravel", "bajri", "crush"], "per_sqft": 0.08, "unit": "cft"},
    {"keywords": ["tile", "tiles"],          "per_sqft": 1.0,   "unit": "sqft"},
    {"keywords": ["paint", "primer"],        "per_sqft": 0.083, "unit": "liters"},
    {"keywords": ["wood", "timber", "plywood"], "per_sqft": 0.02, "unit": "cft"},
    {"keywords": ["pipe", "pipes"],          "per_sqft": 0.05,  "unit": "meters"},
    {"keywords": ["wire", "cable"],          "per_sqft": 0.15,  "unit": "meters"},
]


class RecommendationModule:
    """
    AI-Powered Material Recommendation Engine.

    Pipeline (per plan):
      preprocessed query → FAISS retrieval (cosine) → rule-based filtering
      → plan-scoring → grouped output (+ optional rule-based quantities).
    """

    def __init__(
        self,
        paths: Phase2Paths = Phase2Paths(),
        datastore: Optional[Phase2DataStore] = None,
        index_path: str = "materials.index",
        model: Optional[Any] = None,
        model_name: str = "all-MiniLM-L6-v2",
        llm: Optional[LLMHelper] = None,
        embed_cache_size: int = 256,
        recommend_cache_size: int = 128,
    ):
        self.paths = paths
        self.datastore = datastore or Phase2DataStore(paths=paths)
        self.index_path = index_path
        self.meta_path = index_path + ".meta.json"

        # Shared embedding model (load-once across modules).
        self.model = model or get_embedding_model(model_name)

        # LLM always on — used for category-level explanations.
        self.llm = llm or LLMHelper()

        # Step H — caches for query embeddings + full recommend responses.
        self._embed_cache: _LRUCache = _LRUCache(max_size=embed_cache_size)
        self._recommend_cache: _LRUCache = _LRUCache(max_size=recommend_cache_size)

        # Phase-2 catalog (materials_master enriched with pricing).
        self.products: Optional[pd.DataFrame] = None
        self.index: Optional[faiss.Index] = None

        # Intent → phase categories (derived from materials_master `phase` values).
        # Updated after `_load_data()` once phases are known.
        self.intent_to_categories: Dict[str, List[str]] = {
            "grey_structure": [],
            "finishing": [],
            "full_house": [],
            "foundation": [],
            "roofing": [],
            "plastering": [],
            "flooring": [],
            "bathroom": [],
            "kitchen": [],
            "electrical": [],
            "plumbing": [],
            "cement": [],
            "bricks": [],
            "sand": [],
            "steel": [],
            "tiles": [],
            "paint": [],
            "wood": [],
            "waterproofing": [],
        }

        self.city_advisory: Dict[str, str] = {
            "Karachi": "High coastal humidity: prefer SS-304 fittings. Use waterproof exterior paint. Anti-corrosion treatment for steel recommended.",
            "Lahore": "Hard water area: use scale-resistant faucets. Extreme summers (45C+): choose heat-resistant roofing and thermopore insulation.",
            "Islamabad": "Cold winters: insulate pipes. High rainfall: waterproof exterior walls and use SBS membrane on roof.",
            "Multan": "Extreme heat (50C+): mandatory roof heat-proofing and thermopore insulation. Use UV-resistant paint.",
            "Quetta": "Seismic Zone 3: use TMT Grade-60 steel. Harsh winters: insulate walls and roof.",
        }

        self._load_data()
        logger.info("✓ Recommendation Module initialized (cosine FAISS, plan scoring)")

    # ── finishing tier filter (catalog `finishing_tier_min`) ─────────────────

    @staticmethod
    def _row_matches_finishing_tier(
        row: pd.Series,
        finishing_tier: Optional[str],
        quality: str,
    ) -> bool:
        user_tier = normalize_finishing_tier(finishing_tier or quality)
        mt = str(row.get("finishing_tier_min", "economy") or "economy").strip().lower()
        return item_eligible_for_finishing_tier(mt, user_tier)

    def _filter_finishing_tier(
        self,
        df: pd.DataFrame,
        finishing_tier: Optional[str],
        quality: str,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        mask = df.apply(
            lambda r: self._row_matches_finishing_tier(r, finishing_tier, quality),
            axis=1,
        )
        return df.loc[mask].copy()

    # ── data + index ----------------------------------------------------------

    def _load_data(self) -> None:
        """
        Build Phase-2 catalog:
          materials_master + pricing_data (per city at request-time).

        We index ONLY the material rows (one vector per material_id). Prices are
        joined at request time for the requested city; the semantic index stays
        stable and small.
        """
        self.datastore.load()

        mm = self.datastore.materials_master
        if mm is None or mm.empty:
            raise FileNotFoundError("materials_master.csv not found or empty.")

        # Normalise and keep only necessary columns.
        keep = [
            "material_id", "name", "phase", "category", "subcategory",
            "description", "specifications", "unit", "usage_type", "usage_ratio",
            "quality_grade", "brand", "notes", "functional_tag", "synonyms",
            "room_type", "finishing_tier_min",
        ]
        cols = [c for c in keep if c in mm.columns]
        df = mm[cols].copy()
        df = df.rename(columns={"name": "item_name"})

        if "room_type" not in df.columns:
            df["room_type"] = "general"
        else:
            df["room_type"] = df["room_type"].fillna("general").astype(str)

        if "finishing_tier_min" not in df.columns:
            df["finishing_tier_min"] = "economy"
        else:
            df["finishing_tier_min"] = df["finishing_tier_min"].fillna("economy").astype(str).str.lower()

        # Fill missing text fields for search construction.
        for c in ("phase", "category", "subcategory", "description", "functional_tag", "synonyms"):
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
            else:
                df[c] = ""

        # Category for grouping: use phase primarily; keep subcategory as detail.
        df["category"] = df["phase"].astype(str)

        # Search text: name + phase + category + subcategory + synonyms + tag + description.
        df["search_text"] = (
            df["item_name"].fillna("").astype(str) + " " +
            df["phase"] + " " +
            df["category"] + " " +
            df["subcategory"] + " " +
            df["functional_tag"] + " " +
            df["synonyms"] + " " +
            df["description"]
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        # Keep in memory.
        self.products = df.reset_index(drop=True)

        # Build / load index over search_text.
        self.index = self._load_or_build_cosine_index()

        # Build intent_to_categories based on known phases.
        phases = sorted({p for p in self.products["phase"].dropna().astype(str).tolist() if p.strip()})
        self.intent_to_categories["full_house"] = phases
        # Grey structure phases are a subset.
        self.intent_to_categories["grey_structure"] = [
            p for p in phases if p in (
                "Site Preparation", "Excavation & Foundation", "Grey Structure", "Masonry & Walls"
            )
        ] or phases[: min(6, len(phases))]
        # Finishing and renovation-ish phases.
        self.intent_to_categories["finishing"] = [
            p for p in phases if any(k in p.lower() for k in ("finishing", "tiling", "paint", "carpentry", "kitchen", "sanitary", "aluminum", "glass"))
        ] or phases[-min(6, len(phases)):]

        # Room scopes
        self.intent_to_categories["kitchen"] = [
            p for p in phases if any(k in p.lower() for k in ("kitchen", "wardrobe", "carpentry", "paint", "floor", "tiling", "plumbing", "electrical"))
        ] or self.intent_to_categories.get("finishing", phases[: min(6, len(phases))])
        self.intent_to_categories["bathroom"] = [
            p for p in phases if any(k in p.lower() for k in ("bath", "sanitary", "plumbing", "electrical", "paint", "floor", "tiling"))
        ] or self.intent_to_categories.get("finishing", phases[: min(6, len(phases))])

        # Non-residential scopes: constrain away from home-only categories like kitchen/wardrobes unless requested.
        core_building = [p for p in phases if any(k in p.lower() for k in ("site", "excavation", "foundation", "grey structure", "masonry", "plaster", "electrical", "plumbing", "floor", "tiling", "paint", "finishing", "aluminum", "glass", "external", "roof"))]
        core_building = core_building or phases
        self.intent_to_categories["school"] = core_building
        self.intent_to_categories["hospital"] = core_building
        self.intent_to_categories["plaza"] = core_building
        self.intent_to_categories["marriage_hall"] = core_building
        self.intent_to_categories["mosque"] = core_building

        # Material-specific intents: map to all phases but ranking will pull the right items.
        for k in ("cement", "bricks", "sand", "steel", "tiles", "paint", "wood", "plumbing", "electrical", "roofing", "waterproofing"):
            self.intent_to_categories[k] = phases

        logger.info("Loaded %d Phase-2 materials; phases=%d", len(self.products), len(phases))

    def _load_or_build_cosine_index(self) -> faiss.Index:
        """
        Load cosine (IP) index if available; otherwise migrate an existing L2
        index by reconstructing its vectors (no re-encoding required), or
        as a last resort encode from scratch.
        """
        # Case 1 — meta sidecar says we already have a cosine index.
        meta = self._read_meta()
        if meta.get("metric") == "ip" and os.path.exists(self.index_path):
            try:
                idx = faiss.read_index(self.index_path)
                if idx.metric_type == faiss.METRIC_INNER_PRODUCT:
                    logger.info("FAISS cosine (IP) index loaded from disk")
                    return idx
            except Exception as exc:
                logger.warning("Failed reading cosine index, will rebuild: %s", exc)

        # Case 2 — legacy L2 index exists. Rebuild as IP from existing vectors.
        if os.path.exists(self.index_path):
            try:
                old = faiss.read_index(self.index_path)
                if old.metric_type != faiss.METRIC_INNER_PRODUCT and old.ntotal > 0:
                    logger.warning(
                        "Migrating legacy L2 index to cosine IP (no re-encoding, "
                        "%d vectors)…", old.ntotal,
                    )
                    vecs = np.ascontiguousarray(
                        old.reconstruct_n(0, old.ntotal), dtype="float32",
                    )
                    faiss.normalize_L2(vecs)
                    new_index = faiss.IndexFlatIP(vecs.shape[1])
                    new_index.add(vecs)
                    faiss.write_index(new_index, self.index_path)
                    self._write_meta({"metric": "ip", "dim": int(vecs.shape[1]),
                                      "ntotal": int(new_index.ntotal)})
                    logger.info("Cosine index ready (%d vectors)", new_index.ntotal)
                    return new_index
            except Exception as exc:
                logger.warning("Legacy index migration failed (%s); will re-encode", exc)

        # Case 3 — encode from scratch.
        logger.info("Building cosine FAISS index from scratch…")
        embeddings = self.model.encode(
            self.products["search_text"].tolist(), show_progress_bar=True,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        new_index = faiss.IndexFlatIP(embeddings.shape[1])
        new_index.add(embeddings)
        faiss.write_index(new_index, self.index_path)
        self._write_meta({"metric": "ip", "dim": int(embeddings.shape[1]),
                          "ntotal": int(new_index.ntotal)})
        logger.info("Cosine FAISS index built and saved (%d vectors)", new_index.ntotal)
        return new_index

    def _read_meta(self) -> Dict[str, Any]:
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_meta(self, meta: Dict[str, Any]) -> None:
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            logger.warning("Failed writing FAISS meta sidecar: %s", exc)

    # ── public API ------------------------------------------------------------

    def get_city_advisory(self, city: Optional[str]) -> str:
        if not city:
            return ""
        return self.city_advisory.get(city.capitalize(), "")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode + normalize a query, with an LRU cache (Step H)."""
        cached = self._embed_cache.get(query)
        if cached is not None:
            return cached
        vec = self.model.encode([query])
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)
        self._embed_cache.put(query, vec)
        return vec

    def semantic_search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        """Cosine semantic search via normalized FAISS IP."""
        try:
            vec = self._encode_query(query)
            sims, idxs = self.index.search(vec, top_k)

            results = self.products.iloc[idxs[0]].copy()
            results["semantic_score"] = sims[0]   # cosine in [-1, 1] (typically [0, 1])
            return results
        except Exception as exc:
            logger.error("Search error: %s", exc)
            return pd.DataFrame()

    def estimate_quantities(self, area_sqft: float) -> Dict[str, Any]:
        """
        Rule-based quantity estimation (no LLM, no model calls).
        Returns structured BOQ-style quantities suitable for the
        recommendation response.
        """
        if area_sqft is None or area_sqft <= 0:
            return {}

        structural = {
            mat: {
                "quantity": round(spec["per_sqft"] * area_sqft, 2),
                "unit": spec["unit"],
            }
            for mat, spec in QUANTITIES_PER_SQFT.items()
        }
        finishing = {
            mat: {
                "quantity": round(spec["per_sqft"] * area_sqft, 2),
                "unit": spec["unit"],
            }
            for mat, spec in FINISHING_PER_SQFT.items()
        }

        return {
            "area_sqft": float(area_sqft),
            "structural": structural,
            "finishing": finishing,
            "method": "rule_based",
        }

    # ── project-type detection ─────────────────────────────────────────────────

    def detect_project_type(self, text: str) -> str:
        """
        Classify the query as a specific project type.

        Returns one of: "full_house", "grey_structure", "finishing",
        "foundation", "roofing", "plastering", "flooring", "bathroom",
        "kitchen", "electrical", "plumbing", or a single-material keyword.
        Defaults to "full_house" when the query is broad / ambiguous.
        """
        t = (text or "").lower()

        # Room / scope specific (high priority)
        if any(k in t for k in ("kitchen", "wardrobe", "cabinet", "cabinets")):
            return "kitchen"
        if any(k in t for k in ("bathroom", "washroom", "toilet", "wc")):
            return "bathroom"
        if "electrical" in t or "wiring" in t:
            return "electrical"
        if "plumbing" in t or "pipes" in t:
            return "plumbing"

        # Non-residential project types
        if any(k in t for k in ("school", "college", "university", "classroom")):
            return "school"
        if any(k in t for k in ("hospital", "clinic", "medical", "healthcare")):
            return "hospital"
        if any(k in t for k in ("plaza", "mall", "commercial", "shop", "shops", "market")):
            return "plaza"
        if any(k in t for k in ("marriage hall", "banquet", "hall", "wedding hall")):
            return "marriage_hall"
        if any(k in t for k in ("mosque", "masjid")):
            return "mosque"

        # Broad project indicators → always full-house expansion.
        if any(kw in t for kw in FULL_PROJECT_KEYWORDS):
            # Narrow overrides (e.g. "kitchen renovation", "bathroom tiles")
            for specific in ("grey_structure", "foundation", "roofing",
                             "plastering", "flooring", "bathroom", "kitchen",
                             "electrical", "plumbing", "finishing"):
                if specific in t or specific.replace("_", " ") in t:
                    return specific
            return "full_house"

        # Single-material / specific intents.
        for key in ("cement", "bricks", "sand", "steel", "tiles", "marble",
                    "granite", "paint", "wood", "waterproofing",
                    "plumbing", "electrical", "flooring", "roofing"):
            if key in t:
                return key

        return "full_house"   # safe default

    # ── per-category scoring (fallback when FAISS pool is thin) ──────────────

    def _score_category_products(
        self,
        category: str,
        query_vec: np.ndarray,
        quality: str,
        city: Optional[str],
        top_k: int = 15,
        finishing_tier: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Score ALL products in `category` against `query_vec` using dot-product
        reconstruction from the FAISS index. Used as a fallback when the global
        search pool doesn't surface enough items for a category.
        """
        raw_positions = np.where(self.products["category"].values == category)[0]
        cat_positions = np.array(
            [
                int(i)
                for i in raw_positions
                if self._row_matches_finishing_tier(
                    self.products.iloc[int(i)], finishing_tier, quality
                )
            ],
            dtype=np.int64,
        )
        if len(cat_positions) == 0:
            return pd.DataFrame()

        # Reconstruct normalised vectors from FAISS (IndexFlatIP stores them).
        try:
            vectors = np.array(
                [self.index.reconstruct(int(i)) for i in cat_positions],
                dtype="float32",
            )
        except Exception as exc:
            logger.warning("_score_category_products reconstruct failed: %s", exc)
            return pd.DataFrame()

        cosine_scores = np.dot(query_vec[0], vectors.T)
        cat_df = self.products.iloc[cat_positions].copy().reset_index(drop=True)
        cat_df["semantic_score"] = cosine_scores

        cat_df["quality_match"] = cat_df["quality_grade"].apply(
            lambda g: self._quality_match_score(g, quality)
        )
        # Phase-2: we don't store per-item city; city relevance is a small signal
        # derived from whether city-specific pricing is available.
        if city:
            cat_df["city_match"] = cat_df["material_id"].apply(
                lambda mid: 1.0 if self.datastore.price_avg(str(mid), city) > 0 else 0.2
            )
        else:
            cat_df["city_match"] = 0.5

        if "confidence_score" in cat_df.columns:
            cat_df["rec_score_norm"] = cat_df["confidence_score"].fillna(0.6).astype(float)
        else:
            cat_df["rec_score_norm"] = 0.6
        cat_df["final_score"] = (
            0.65 * cat_df["semantic_score"]
            + 0.20 * cat_df["rec_score_norm"]
            + 0.10 * cat_df["quality_match"]
            + 0.05 * cat_df["city_match"]
        )
        return cat_df.nlargest(top_k, "final_score")

    # ── item-level quantity estimation ────────────────────────────────────────

    def _estimate_item_quantity(
        self,
        row: Any,
        area_sqft: Optional[float],
    ) -> Tuple[Optional[float], Optional[int]]:
        """
        Return (estimated_quantity, estimated_total_cost) for a single product
        row when area_sqft is known; otherwise (None, None).

        Phase-2: prefer `usage_type` + `usage_ratio` from materials_master.csv.
        Falls back to ITEM_QUANTITY_RULES if needed.
        """
        if not area_sqft or area_sqft <= 0:
            return None, None

        # Phase-2: usage ratio (per_sqft / per_marla / per_house / per_site)
        usage_type = str(row.get("usage_type", "") or "").strip().lower()
        try:
            usage_ratio = float(row.get("usage_ratio", 0) or 0)
        except Exception:
            usage_ratio = 0.0

        if usage_ratio > 0 and usage_type:
            if usage_type == "per_sqft":
                qty = usage_ratio * area_sqft
            elif usage_type == "per_marla":
                # default marla sqft; city-specific conversion isn't available here
                qty = usage_ratio * (area_sqft / 272.0)
            elif usage_type in ("per_house", "per_site"):
                qty = usage_ratio
            else:
                qty = usage_ratio * area_sqft
            qty = float(qty)
            if not np.isfinite(qty) or qty <= 0:
                return None, None
            qty = round(qty, 2)
            unit_price = float(row.get("final_price_pkr", 0) or 0)
            if unit_price > 0 and qty > 0:
                return qty, int(qty * unit_price)

        name_lower = str(row.get("item_name", "")).lower()
        cat_lower = str(row.get("category", "")).lower()
        combined = f"{name_lower} {cat_lower}"

        for rule in ITEM_QUANTITY_RULES:
            if any(kw in combined for kw in rule["keywords"]):
                qty = round(rule["per_sqft"] * area_sqft, 1)
                unit_price = float(row.get("final_price_pkr", 0) or 0)
                return qty, int(qty * unit_price)

        return None, None

    # ── row → dict serialiser ─────────────────────────────────────────────────

    def _row_to_item(
        self,
        row: Any,
        area_sqft: Optional[float],
        category: str,
    ) -> Dict[str, Any]:
        def _sf(val: Any, default: float = 0.0) -> float:
            try:
                f = float(val)
                return f if np.isfinite(f) else float(default)
            except Exception:
                return float(default)

        def _ss(val: Any, default: str = "") -> str:
            try:
                if val is None:
                    return default
                # pandas NaN check: NaN != NaN
                if isinstance(val, float) and not np.isfinite(val):
                    return default
                if val != val:  # type: ignore[comparison-overlap]
                    return default
                return str(val)
            except Exception:
                return default

        est_qty, est_cost = self._estimate_item_quantity(row, area_sqft)
        
        # Extract description and phase from materials_master
        description = _ss(row.get("description"), "")
        phase = _ss(row.get("phase")) or None
        
        item: Dict[str, Any] = {
            "material_id":        _ss(row.get("material_id")) or None,
            "item_name":          _ss(row.get("item_name")),
            "description":        description,  # Now included at top level
            "phase":              phase,         # Now included at top level
            "brand":              _ss(row.get("brand"), ""),
            "quality_grade":      _ss(row.get("quality_grade"), "standard"),
            "unit":               _ss(row.get("unit"), "unit"),
            "market_price_pkr":   int(row.get("market_price_pkr") or row.get("final_price_pkr") or 0),
            "final_price_pkr":    int(row.get("final_price_pkr") or 0),
            "availability":       row.get("availability", "N/A"),
            "recommendation_score": round(_sf(row.get("final_score")) * 100, 1),
            "score_breakdown": {
                "cosine":         round(_sf(row.get("semantic_score")), 4),
                "rec_score_norm": round(_sf(row.get("rec_score_norm")), 4),
                "quality_match":  round(_sf(row.get("quality_match")), 2),
                "city_match":     round(_sf(row.get("city_match")), 2),
            },
            "meta": {
                "subcategory": _ss(row.get("subcategory")) or None,
                "functional_tag": _ss(row.get("functional_tag")) or None,
                "room_type": _ss(row.get("room_type"), "general"),
                "finishing_tier_min": _ss(row.get("finishing_tier_min"), "economy"),
            },
        }
        if est_qty is not None:
            item["estimated_quantity"]   = est_qty
            item["estimated_total_cost"] = est_cost
        return item

    # ── main entry point ──────────────────────────────────────────────────────

    def recommend(
        self,
        text: str,
        budget_pkr: Optional[int] = None,
        city: Optional[str] = None,
        quality: str = "Standard",
        area_sqft: Optional[float] = None,
        top_n_per_cat: int = 8,
        use_llm: bool = True,
        finishing_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full-coverage recommendation engine (v2 spec).

        Hard filter: user finishing tier vs each row's `finishing_tier_min`.
        Soft signals: city, budget, semantic relevance (ranking only).
        """
        text = text or ""

        cache_key = (
            text.strip().lower(),
            int(budget_pkr) if budget_pkr else None,
            (city or "").strip().lower() or None,
            (quality or "Standard").strip(),
            float(area_sqft) if area_sqft else None,
            int(top_n_per_cat),
            (finishing_tier or "").strip().lower() or None,
        )
        cached = self._recommend_cache.get(cache_key)
        if cached is not None:
            return cached

        # ── 1) Detect project type & target categories ──────────────────────
        project_type = self.detect_project_type(text)
        target_cats: List[str] = list(dict.fromkeys(   # deduped + ordered
            self.intent_to_categories.get(
                project_type,
                self.intent_to_categories["full_house"],
            )
        ))

        # Hard constraints to avoid irrelevant home-only phases for non-residential projects.
        # (Still allow if the user explicitly asks for kitchen/bathroom/etc.)
        if project_type in ("school", "hospital", "plaza", "marriage_hall", "mosque"):
            deny = ("kitchen", "wardrobe")
            if not any(d in text.lower() for d in deny):
                target_cats = [c for c in target_cats if not any(d in c.lower() for d in deny)]

        # ── 2) Global semantic search (large pool) ──────────────────────────
        # Phase-2: keep query general but informative for semantic retrieval.
        search_query = (
            f"{text} {project_type.replace('_', ' ')} materials Pakistan construction"
        ).strip()
        sem_df = self.semantic_search(search_query, top_k=500)
        if sem_df.empty:
            return {"status": "error", "message": "No products found"}

        sem_df = self._filter_finishing_tier(sem_df, finishing_tier, quality)
        if sem_df.empty:
            return {
                "status": "error",
                "message": "No products match the selected finishing tier.",
            }

        # ── 3) Score ALL results — quality/city as soft signals ONLY ────────
        sem_df = sem_df.copy()
        sem_df["quality_match"] = sem_df["quality_grade"].apply(
            lambda g: self._quality_match_score(g, quality)
        )
        if city:
            sem_df["city_match"] = sem_df["material_id"].apply(
                lambda mid: 1.0 if self.datastore.price_avg(str(mid), city) > 0 else 0.2
            )
        else:
            sem_df["city_match"] = 0.5

        if "confidence_score" in sem_df.columns:
            sem_df["rec_score_norm"] = sem_df["confidence_score"].fillna(0.6).astype(float)
        else:
            sem_df["rec_score_norm"] = 0.6
        sem_df["final_score"] = (
            0.65 * sem_df["semantic_score"]
            + 0.20 * sem_df["rec_score_norm"]
            + 0.10 * sem_df["quality_match"]
            + 0.05 * sem_df["city_match"]
        )

        # Observability — log cosine stats once per call.
        try:
            top20 = sem_df.nlargest(20, "final_score")
            cvs = top20["semantic_score"].astype(float).to_numpy()
            if cvs.size:
                logger.info(
                    "recommend(): top-%d cosine — mean=%.3f min=%.3f max=%.3f | "
                    "pool=%d | cats=%d | type=%s",
                    cvs.size, cvs.mean(), cvs.min(), cvs.max(),
                    len(sem_df), len(target_cats), project_type,
                )
        except Exception:
            pass

        # Encode query vector (for per-category fallback).
        query_vec = self._encode_query(search_query)

        # ── 4) Per-category top-N (with fallback for thin categories) ────────
        results: Dict[str, List[Dict[str, Any]]] = {}
        min_items_from_faiss = 3   # if fewer than this, top-up via fallback

        # Join city pricing into the candidate pool so item cards have prices.
        if city:
            sem_df["final_price_pkr"] = sem_df["material_id"].apply(
                lambda mid: int(self.datastore.price_avg(str(mid), city) or 0)
            )
            sem_df["market_price_pkr"] = sem_df["material_id"].apply(
                lambda mid: int(self.datastore.price_range(str(mid), city)[1] or 0)
            )
        else:
            sem_df["final_price_pkr"] = sem_df["material_id"].apply(
                lambda mid: int(self.datastore.price_avg(str(mid), "Lahore") or 0)
            )
            sem_df["market_price_pkr"] = sem_df["final_price_pkr"]

        # Replace any NaNs in numeric columns to keep JSON encoding strict.
        for col in ("semantic_score", "rec_score_norm", "quality_match", "city_match", "final_score"):
            if col in sem_df.columns:
                sem_df[col] = sem_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        for category in target_cats:
            cat_in_pool = sem_df[sem_df["category"] == category].copy()

            if len(cat_in_pool) < min_items_from_faiss:
                # Fallback: score ALL products in this category via reconstruction.
                cat_scored = self._score_category_products(
                    category,
                    query_vec,
                    quality,
                    city,
                    top_k=15,
                    finishing_tier=finishing_tier,
                )
            else:
                cat_scored = cat_in_pool

            if cat_scored.empty:
                continue

            # Ensure numeric columns are JSON-safe.
            for col in ("semantic_score", "rec_score_norm", "quality_match", "city_match", "final_score"):
                if col in cat_scored.columns:
                    cat_scored[col] = cat_scored[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            cat_top = (
                cat_scored
                .sort_values("final_score", ascending=False)
                .drop_duplicates(subset=["item_name"])
                .head(top_n_per_cat)
            )

            items = [self._row_to_item(row, area_sqft, category)
                     for _, row in cat_top.iterrows()]
            if items:
                results[category] = items

        # ── 5) Rule-based BOQ quantities ─────────────────────────────────────
        quantities = self.estimate_quantities(area_sqft) if area_sqft else None

        # ── 6) LLM explanations (always on, capped at 5 cats) ───────────────
        explanations: Dict[str, str] = {}
        try:
            if use_llm:
                for cat in list(results.keys())[:5]:
                    line = self.llm.explain_recommendation(
                        category=cat, quality=quality, city=city,
                        items=results[cat], max_items=3,
                    )
                    if line:
                        explanations[cat] = line
        except Exception as exc:
            logger.warning("LLM explanations skipped: %s", exc)

        total_products = sum(len(v) for v in results.values())
        eff_tier = normalize_finishing_tier(finishing_tier or quality)

        response = {
            "status":        "success",
            "project_type":  project_type,
            "area":          f"{area_sqft:.0f} sqft" if area_sqft else None,
            # v2 key — use "categories" for clarity; keep "recommendations" as alias.
            "categories":    results,
            "recommendations": results,   # backward-compat for old frontend code
            "total_products": total_products,
            "total_items":   total_products,
            "categories_covered": list(results.keys()),
            "quantities":    quantities,
            "explanations":  explanations or None,
            "city_advisory": self.get_city_advisory(city) if city else None,
            "finishing_tier_effective": eff_tier,
            "finishing_tier_summary": explain_tier(eff_tier),
            "filters_applied": {
                "quality": quality,
                "city":    city,
                "budget_pkr": budget_pkr,
                "area_sqft":  area_sqft,
                "finishing_tier": eff_tier,
            },
            "scoring": {
                "formula": "0.65*cosine + 0.20*rec_score + 0.10*quality_match + 0.05*city_relevance",
                "note": (
                    "Within finishing-tier-filtered pool: city/quality/budget are ranking signals only; "
                    "items are excluded when finishing_tier_min exceeds the selected tier."
                ),
                "metric": "cosine (FAISS IndexFlatIP, normalized)",
            },
            "llm": {
                "enabled":   bool(use_llm),
                "available": self.llm.is_available,
                "model":     self.llm.model_name,
            },
        }

        self._recommend_cache.put(cache_key, response)
        return response

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "products_loaded": len(self.products) if self.products is not None else 0,
            "index_size": self.index.ntotal if self.index else 0,
            "categories": len(self.products["category"].unique()) if self.products is not None else 0,
            "metric": "cosine_ip",
        }

    # ── scoring helpers -------------------------------------------------------

    @staticmethod
    def _quality_match_score(actual: Any, requested: str) -> float:
        """1.0 if exact match, 0.5 if within the same tier set, else 0.0."""
        if not requested:
            return 0.5  # neutral when user didn't specify
        if not isinstance(actual, str):
            return 0.0
        if actual.lower() == requested.lower():
            return 1.0
        allowed = QUALITY_TIERS.get(requested, [])
        return 0.5 if actual in allowed else 0.0

    @staticmethod
    def _city_match_score(actual: Any, requested: Optional[str]) -> float:
        """1.0 if cities match; neutral 0.5 if user didn't pass a city; else 0.0."""
        if not requested:
            return 0.5
        if not isinstance(actual, str):
            return 0.0
        return 1.0 if actual.lower() == requested.lower() else 0.0
