"""Unit checks for BOQ metadata sync helpers (Supabase / estimator catalog)."""

import numpy as np

from ai_modules.cost_estimation_module import (
    _coarse_construction_phase,
    _safe_catalog_text,
    normalize_city_key,
)


def test_normalize_city_key_strips_and_collapses():
    assert normalize_city_key(" Islamabad ") == "islamabad"
    assert normalize_city_key("New  York") == "new york"


def test_safe_catalog_text_rejects_sentinels():
    assert _safe_catalog_text(None) == ""
    assert _safe_catalog_text(float("nan")) == ""
    assert _safe_catalog_text("  N/A  ") == ""
    assert _safe_catalog_text("  hello world  ") == "hello world"


def test_safe_catalog_text_numpy_nan():
    assert _safe_catalog_text(np.nan) == ""


def test_coarse_phase_expanded():
    assert _coarse_construction_phase("Paint & Finishing") == "Finishing"
    assert _coarse_construction_phase("RCC & Structural columns") == "Structure"
    assert _coarse_construction_phase("Masonry & Wall Work") == "Masonry"
    assert _coarse_construction_phase("Stone cladding pack") == "Masonry"
    assert _coarse_construction_phase("Site Preparation & Foundation") == "Foundation"
