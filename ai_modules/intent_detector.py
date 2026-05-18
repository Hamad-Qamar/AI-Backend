"""
IntentDetector — Module 2
Single responsibility: classify a clean query string into one of 5 intent labels.
Stateless after __init__. Class centroid vectors computed once at startup.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT SEED PHRASES  (~20 per class, held-out test queries must NOT appear here)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_SEEDS: dict[str, list[str]] = {
    "recommendation": [
        "materials for 5 marla house",
        "what materials do i need for foundation",
        "suggest tiles for bathroom",
        "which cement is good for construction",
        "recommend steel for 2000 sqft house",
        "cheap tiles for kitchen",
        "best paint for exterior walls",
        "materials needed for 3 bedroom house",
        "what to use for roofing",
        "plumbing materials list",
        "electrical materials for house wiring",
        "flooring options for living room",
        "good bricks for boundary wall",
        "insulation material for roof",
        "waterproofing material for bathroom",
        "material list for grey structure",
        "marble options for flooring",
        "affordable alternatives to granite",
        "material for double story house",
        "what material for 10 marla construction",
    ],
    "cost_estimation": [
        "how much does it cost to build a house",
        "cost of building 5 marla house",
        "estimate for 2000 sqft construction",
        "how much for 3 bedroom house",
        "budget for grey structure",
        "total cost of construction per sqft",
        "price to build 1 kanal house",
        "how much labour cost for plastering",
        "construction cost estimate lahore",
        "how much will it cost overall",
        "what is the total budget needed",
        "cost breakdown for finishing stage",
        "estimate for foundation work",
        "how much for tiles and flooring",
        "plumbing cost for 5 marla",
        "electrical work budget",
        "compare economy vs standard construction",
        "cost reduction tips for building",
        "material cost for 10 marla house",
        "building cost in karachi",
    ],
    "platform_help": [
        "how to upload listing as seller",
        "where is my order",
        "how to track my delivery",
        "seller dashboard kahan hai",
        "how to register as buyer",
        "how do i reset my password",
        "how to save wishlist",
        "where to find purchase history",
        "how to message a seller",
        "how to post a project as buyer",
        "how do i leave a review",
        "how to submit proposal as freelancer",
        "how to update my profile",
        "where are my saved estimates",
        "how to export cost estimate pdf",
        "payment methods on buildhive",
        "how to report a fake review",
        "how to cancel an order",
        "how to verify my email",
        "what are platform fees for sellers",
    ],
    "navigation": [
        "show tiles category",
        "take me to my dashboard",
        "open plumbing section",
        "go to seller listings",
        "show me cement products",
        "navigate to cost estimator",
        "open ai recommendation tool",
        "show order history",
        "go to payment section",
        "open my profile",
        "show all categories",
        "navigate to flooring products",
        "take me to search",
        "open messages",
        "go to help center",
        "show featured products",
        "open project listings",
        "navigate to buyer dashboard",
        "show wishlist",
        "go to checkout",
    ],
    "general_question": [
        "what is the difference between opc and ppc cement",
        "which brand of cement is best",
        "is marble better than tiles",
        "what type of bricks are strongest",
        "how long does grey structure take",
        "what is a good recommendation score",
        "difference between economy and premium quality",
        "what is pprc pipe",
        "which paint is best for humidity",
        "what is a kanal in sqft",
        "how many bags of cement per 100 sqft",
        "what is sarya steel",
        "types of roofing materials",
        "what is a marla",
        "what does quality grade mean",
        "what is waterproofing chemical",
        "how does faiss search work",
        "what is buildhive",
        "how does the ai recommendation work",
        "what is a bill of quantities",
    ],
}


# Fast keyword lookup — frozen sets per intent for O(1) hit detection
INTENT_KEYWORDS: dict[str, frozenset] = {
    "recommendation": frozenset([
        "suggest", "recommend", "materials", "material", "which",
        "best", "good", "need", "list", "options", "use",
        "tile", "tiles", "cement", "bricks", "steel", "paint",
        "marble", "pipe", "wire", "insulation", "flooring",
        "alternative", "cheap", "affordable", "foundation", "grey structure",
        "finishing", "eco-friendly", "earthquake", "farmhouse", "bedroom",
        "washroom", "bathroom", "toilet", "modular kitchen", "wiring",
        "pvc", "pprc", "sarya", "waterproofing", "sangmarmar", "ziarat",
        "emulsion", "timber", "gypsum", "thermopore", "mcb", "rccb",
    ]),
    "cost_estimation": frozenset([
        "cost", "price", "estimate", "budget", "lakh", "how much",
        "kitna", "kharcha", "total", "breakdown", "labour",
        "per sqft", "per marla", "expensive", "affordable",
        "compare", "rate", "rates", "rupee", "rs", "pkr",
        "fee", "charge", "amount", "boq", "quantity estimate",
    ]),
    "platform_help": frozenset([
        "how to", "upload", "listing", "register", "signup",
        "login", "password", "reset", "order", "track", "delivery",
        "review", "refund", "cancel", "message", "contact",
        "profile", "account", "dashboard", "export", "save",
        "proposal", "milestone", "payment method", "verify",
        "report", "flag", "invoice", "receipt",
    ]),
    "navigation": frozenset([
        "show", "open", "go to", "take me", "navigate",
        "find", "search for", "view", "display", "bring",
        "section", "category", "page", "tab", "marketplace",
    ]),
    "general_question": frozenset([
        "what is", "what are", "difference between", "explain",
        "define", "meaning", "how does", "why", "types of",
        "how long", "how many", "when", "where is", "vs", "better",
    ]),
}


# Confidence threshold below which we return 'clarification_needed'
CONFIDENCE_THRESHOLD = 0.30
# Keyword fast-path threshold (SequenceMatcher ratio)
KEYWORD_FAST_PATH_THRESHOLD = 0.82


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    label: str   # One of the 5 intent labels or "clarification_needed"
    score: float # Confidence score in [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class IntentDetector:
    """
    Two-stage intent classifier:
      Stage 1 — keyword fast path (no model call, O(1))
      Stage 2 — cosine similarity against class centroid embeddings

    Stateless after __init__. Centroids computed once from seed phrases.
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        """
        Args:
            model: A pre-loaded SentenceTransformer instance.
                   Pass the one already loaded in ChatBotModule to avoid
                   loading a second copy into memory.
        """
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.centroids: dict[str, np.ndarray] = {}
        self._compute_centroids()
        logger.info("✓ IntentDetector ready — %d intent classes", len(self.centroids))

    # ── public ────────────────────────────────────────────────────────────────

    def classify(self, clean_query: str) -> IntentResult:
        """
        Classify a pre-processed query string.
        Always returns an IntentResult — never raises.
        """
        if not clean_query or clean_query == "[empty query]":
            return IntentResult(label="clarification_needed", score=0.0)

        # Stage 1: keyword fast path
        fast_result = self._keyword_classify(clean_query)
        if fast_result is not None:
            return fast_result

        # Stage 2: semantic centroid matching
        return self._semantic_classify(clean_query)

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_centroids(self) -> None:
        """Encode all seed phrases and average into per-class centroid vectors."""
        for intent, phrases in INTENT_SEEDS.items():
            embeddings = self.model.encode(phrases, show_progress_bar=False)
            centroid = embeddings.mean(axis=0)
            # L2-normalise so cosine similarity = dot product
            norm = np.linalg.norm(centroid)
            self.centroids[intent] = centroid / (norm + 1e-8)

    def _keyword_classify(self, query: str) -> Optional[IntentResult]:
        """
        Check if the query contains a keyword match above the fast-path threshold.
        Returns an IntentResult if confident, None otherwise.
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scores: dict[str, float] = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            hit_count = 0
            for kw in keywords:
                # Multi-word keyword: check substring
                if " " in kw:
                    if kw in query_lower:
                        hit_count += 2  # bonus for multi-word match
                else:
                    # Single-word: exact token match or fuzzy
                    if kw in query_tokens:
                        hit_count += 1
                    else:
                        best = max(
                            (SequenceMatcher(None, kw, t).ratio() for t in query_tokens),
                            default=0.0,
                        )
                        if best >= KEYWORD_FAST_PATH_THRESHOLD:
                            hit_count += 0.7

            scores[intent] = hit_count

        # Normalise
        total = sum(scores.values())
        if total == 0:
            return None

        best_intent = max(scores, key=lambda k: scores[k])
        best_raw = scores[best_intent]

        # Require at least 2 keyword hits to fire the fast path
        if best_raw < 1.5:
            return None

        # Normalise to a pseudo-probability
        confidence = min(best_raw / max(total, 1) + 0.15, 0.95)

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        return IntentResult(label=best_intent, score=round(confidence, 3))

    def _semantic_classify(self, query: str) -> IntentResult:
        """
        Encode the query and compute cosine similarity against each centroid.
        Returns the best-matching label or 'clarification_needed'.
        """
        try:
            vec = self.model.encode([query], show_progress_bar=False)[0]
            norm = np.linalg.norm(vec)
            vec = vec / (norm + 1e-8)

            best_label, best_score = "clarification_needed", 0.0
            for intent, centroid in self.centroids.items():
                score = float(np.dot(vec, centroid))
                if score > best_score:
                    best_score = score
                    best_label = intent

            if best_score < CONFIDENCE_THRESHOLD:
                return IntentResult(label="clarification_needed", score=round(best_score, 3))

            return IntentResult(label=best_label, score=round(best_score, 3))

        except Exception as exc:
            logger.error("Semantic classification failed: %s", exc)
            return IntentResult(label="clarification_needed", score=0.0)
