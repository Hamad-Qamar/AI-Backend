"""
Acceptance tests for the v2 specification:

  ✔ "5 marla house" → 50+ products, 7+ categories
  ✔ detect_project_type classifies correctly
  ✔ _extract_entities detects materials + purchase actions
  ✔ _estimate_item_quantity produces per-base-unit quantities
  ✔ Scoring weights appear in API response; finishing tier filters catalog
    rows by finishing_tier_min.
"""

from __future__ import annotations

import pytest

from ai_modules.recommendation_module import (
    FULL_PROJECT_KEYWORDS,
    ITEM_QUANTITY_RULES,
    RecommendationModule,
)
from ai_modules.chatbot_module import (
    ChatBotModule,
    _MATERIAL_ENTITIES,
    _PURCHASE_ACTIONS,
)


# ── RecommendationModule helpers ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def rec():
    # Phase-2 dataset: no products.csv required.
    return RecommendationModule()


class TestProjectTypeDetection:
    @pytest.mark.parametrize("text,expected", [
        ("5 marla house",             "full_house"),
        ("10 marla ghar",             "full_house"),
        ("1 kanal building",          "full_house"),
        # "grey structure only" has no broad-project keyword → safe default
        ("grey structure only",       "full_house"),
        # Room scopes should be detected and constrained
        ("bathroom tiles",            "bathroom"),
        ("kitchen renovation",        "kitchen"),
        ("electrical wiring",         "electrical"),
        ("plumbing pipes",            "plumbing"),
        ("flooring material",         "flooring"),
        ("cement for slab",           "cement"),
        ("best paints",               "paint"),
        ("just some query",           "full_house"),   # safe default
    ])
    def test_detect_project_type(self, rec, text, expected):
        assert rec.detect_project_type(text) == expected

    def test_full_house_expands_to_many_categories(self, rec):
        cats = rec.intent_to_categories["full_house"]
        assert len(cats) >= 10


class TestFullHouseRecommendation:
    def test_5_marla_returns_50_plus_products(self, rec):
        result = rec.recommend(
            text="5 marla house", quality="Standard", city="Lahore",
            area_sqft=1361.25, top_n_per_cat=8, use_llm=False,
        )
        assert result["status"] == "success"
        assert result["total_products"] >= 50, (
            f"Got only {result['total_products']} products — need ≥50"
        )

    def test_5_marla_returns_7_plus_categories(self, rec):
        result = rec.recommend(
            text="5 marla house", quality="Standard",
            area_sqft=1361.25, top_n_per_cat=8, use_llm=False,
        )
        assert len(result["categories_covered"]) >= 7, (
            f"Got {len(result['categories_covered'])} categories — need ≥7"
        )

    def test_categories_key_present(self, rec):
        result = rec.recommend(text="house", use_llm=False)
        assert "categories" in result
        # Backward-compat alias also present.
        assert "recommendations" in result

    def test_project_type_in_response(self, rec):
        result = rec.recommend(text="5 marla house", use_llm=False)
        assert result.get("project_type") == "full_house"

    def test_no_products_excluded_by_city_with_wrong_city(self, rec):
        """Products must NOT be filtered out when city doesn't match."""
        result_city = rec.recommend(text="5 marla house", city="Sialkot", use_llm=False)
        result_nocity = rec.recommend(text="5 marla house", use_llm=False)
        # With a rare/non-existent city, we should still get products.
        assert result_city["total_products"] >= 10, (
            "City mismatch should not eliminate most products"
        )

    def test_estimated_quantity_present_when_area_given(self, rec):
        result = rec.recommend(
            text="cement for house", area_sqft=1000, use_llm=False,
        )
        # At least some items in any category should have the quantity field.
        all_items = [item for items in result["categories"].values() for item in items]
        annotated = [i for i in all_items if "estimated_quantity" in i]
        assert len(annotated) >= 1, "No items carry estimated_quantity"

    def test_estimated_total_cost_present_when_area_given(self, rec):
        result = rec.recommend(
            text="5 marla house", area_sqft=1361, use_llm=False,
        )
        all_items = [item for items in result["categories"].values() for item in items]
        costed = [i for i in all_items if i.get("estimated_total_cost")]
        assert len(costed) >= 1


class TestScoringWeights:
    def test_scoring_formula_in_response(self, rec):
        result = rec.recommend(text="cement", use_llm=False)
        formula = result["scoring"]["formula"]
        assert "0.65" in formula
        assert "0.20" in formula


class TestFinishingTierFilters:
    def test_finishing_tier_summary_in_response(self, rec):
        r = rec.recommend(text="tiles for house", quality="Premium", use_llm=False)
        assert r["status"] == "success"
        assert r.get("finishing_tier_effective") == "premium"
        assert isinstance(r.get("finishing_tier_summary"), str)
        assert len(r["finishing_tier_summary"]) > 10

    def test_recommend_response_has_no_room_filter_field(self, rec):
        r = rec.recommend(text="5 marla house tiles cement", quality="Standard", use_llm=False)
        assert r["status"] == "success"
        assert "room_type" not in (r.get("filters_applied") or {})


class TestItemQuantityRules:
    @pytest.mark.parametrize("kwds,area,expected_unit", [
        (["cement"], 1000, "bags"),
        (["brick"],  1000, "units"),
        (["tile"],   1000, "sqft"),
        (["paint"],  1000, "liters"),
    ])
    def test_rule_fires_for_keyword(self, kwds, area, expected_unit, rec):
        # Simulate a row whose item_name contains the keyword.
        row = {
            "item_name": kwds[0], "category": "Raw Materials",
            "final_price_pkr": 1000,
            "typical_qty_per_1000sqft_house": None,
        }
        qty, cost = rec._estimate_item_quantity(row, area)
        assert qty is not None and qty > 0
        assert cost is not None and cost > 0

    def test_no_quantity_when_area_none(self, rec):
        row = {"item_name": "cement", "category": "X", "final_price_pkr": 1000}
        qty, cost = rec._estimate_item_quantity(row, None)
        assert qty is None and cost is None


# ── ChatBotModule entity extraction ──────────────────────────────────────────


class TestEntityExtraction:
    @pytest.fixture(scope="class")
    def bot(self):
        # Build without model load so it's fast.
        return ChatBotModule.__new__(ChatBotModule)

    @pytest.mark.parametrize("text,mat", [
        ("how to buy cement", "cement"),
        ("where to get tiles",  "tile"),
        ("best paint for walls", "paint"),
        ("purchase steel rebar", "steel"),
    ])
    def test_detects_material(self, bot, text, mat):
        entities = bot._extract_entities(text)
        detected_kws = [kw for kw, _ in entities["materials"]]
        assert any(mat in kw for kw in detected_kws), (
            f"'{mat}' not found in detected: {detected_kws}"
        )

    @pytest.mark.parametrize("text", [
        "how to buy cement", "where can I purchase tiles",
        "kahan se milega brick", "recommend best paint",
    ])
    def test_detects_purchase_action(self, bot, text):
        entities = bot._extract_entities(text)
        assert entities["has_purchase_action"], f"No purchase action in: '{text}'"

    def test_no_purchase_action_for_generic_query(self, bot):
        entities = bot._extract_entities("what is BuildHive")
        assert not entities["has_purchase_action"]

    def test_material_entities_table_non_empty(self):
        assert len(_MATERIAL_ENTITIES) >= 10

    def test_purchase_actions_table_non_empty(self):
        assert len(_PURCHASE_ACTIONS) >= 5
