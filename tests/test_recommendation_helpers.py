"""
Pure-logic tests for RecommendationModule helpers. These do NOT load the
SentenceTransformer model or the products.csv / FAISS index — they
exercise scoring / quantity logic in isolation.
"""

import pytest

from ai_modules.recommendation_module import (
    QUALITY_TIERS,
    RecommendationModule,
)


# ── _quality_match_score ─────────────────────────────────────────────────────


def test_quality_exact_match_scores_1():
    assert RecommendationModule._quality_match_score("Premium", "Premium") == 1.0


def test_quality_case_insensitive_exact_match():
    assert RecommendationModule._quality_match_score("premium", "PREMIUM") == 1.0


def test_quality_within_tier_set_scores_half():
    # "Standard" is in QUALITY_TIERS["Premium"]
    assert "Standard" in QUALITY_TIERS["Premium"]
    assert RecommendationModule._quality_match_score("Standard", "Premium") == 0.5


def test_quality_outside_tier_set_scores_zero():
    assert RecommendationModule._quality_match_score("Random Junk", "Premium") == 0.0


def test_quality_empty_request_is_neutral():
    assert RecommendationModule._quality_match_score("Anything", "") == 0.5


def test_quality_non_string_actual_scores_zero():
    assert RecommendationModule._quality_match_score(None, "Premium") == 0.0
    assert RecommendationModule._quality_match_score(42, "Premium") == 0.0


# ── _city_match_score ────────────────────────────────────────────────────────


def test_city_exact_match_scores_1():
    assert RecommendationModule._city_match_score("Lahore", "Lahore") == 1.0


def test_city_case_insensitive_match():
    assert RecommendationModule._city_match_score("LAHORE", "lahore") == 1.0


def test_city_mismatch_scores_zero():
    assert RecommendationModule._city_match_score("Karachi", "Lahore") == 0.0


def test_city_no_request_is_neutral():
    assert RecommendationModule._city_match_score("Karachi", None) == 0.5
    assert RecommendationModule._city_match_score("Karachi", "") == 0.5


def test_city_non_string_actual_scores_zero():
    assert RecommendationModule._city_match_score(None, "Lahore") == 0.0


# ── estimate_quantities (pure rule-based; no I/O) ────────────────────────────


@pytest.fixture(scope="module")
def bare_recommender():
    """
    Bypass __init__ so we don't load the embedding model or 50K products.
    estimate_quantities() is a pure function of area_sqft.
    """
    return RecommendationModule.__new__(RecommendationModule)


def test_estimate_quantities_zero_area_returns_empty(bare_recommender):
    assert bare_recommender.estimate_quantities(0) == {}
    assert bare_recommender.estimate_quantities(-1) == {}
    assert bare_recommender.estimate_quantities(None) == {}


def test_estimate_quantities_returns_expected_keys(bare_recommender):
    out = bare_recommender.estimate_quantities(1361.25)
    assert out["area_sqft"] == 1361.25
    assert out["method"] == "rule_based"
    assert "structural" in out and "finishing" in out

    structural = out["structural"]
    for key in ("cement", "bricks", "steel", "sand", "gravel"):
        assert key in structural, f"missing structural key: {key}"
        assert "quantity" in structural[key]
        assert "unit" in structural[key]

    finishing = out["finishing"]
    for key in ("tiles", "paint", "wood"):
        assert key in finishing, f"missing finishing key: {key}"


def test_estimate_quantities_brick_count_is_realistic(bare_recommender):
    # ~12.5 bricks/sqft × 1000 sqft = ~12,500 bricks (rule of thumb).
    out = bare_recommender.estimate_quantities(1000)
    bricks = out["structural"]["bricks"]["quantity"]
    assert 10_000 <= bricks <= 15_000


def test_estimate_quantities_scales_linearly_with_area(bare_recommender):
    a = bare_recommender.estimate_quantities(1000)
    b = bare_recommender.estimate_quantities(2000)
    # Bricks should roughly double when area doubles.
    ratio = b["structural"]["bricks"]["quantity"] / a["structural"]["bricks"]["quantity"]
    assert ratio == pytest.approx(2.0, rel=0.001)
