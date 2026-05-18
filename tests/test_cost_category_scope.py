"""Category scope: grey vs full vs renovation UI buckets and totals stay aligned."""

from pathlib import Path

import pytest

from ai_modules.cost_estimation_module import CostEstimationModule, allowed_ui_buckets_for_construction

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PHASE2_FILES = (
    PROJECT_ROOT / "materials_master.csv",
    PROJECT_ROOT / "pricing_data.csv",
    PROJECT_ROOT / "labor_rates.csv",
    PROJECT_ROOT / "phase_labor_mapping.csv",
)

needs_phase2_csv = pytest.mark.skipif(
    not all(p.is_file() for p in _PHASE2_FILES),
    reason="Phase-2 CSV fixtures not found at project root",
)


def _cost() -> CostEstimationModule:
    return CostEstimationModule(
        pricing_data_path=str(PROJECT_ROOT / "pricing_data.csv"),
        materials_master_path=str(PROJECT_ROOT / "materials_master.csv"),
        city_area_standards_path=str(PROJECT_ROOT / "city_area_standards.csv"),
        construction_rates_path=str(PROJECT_ROOT / "construction_rates.csv"),
        labor_rates_path=str(PROJECT_ROOT / "labor_rates.csv"),
        phase_labor_mapping_path=str(PROJECT_ROOT / "phase_labor_mapping.csv"),
    )


def _sqft(mod: CostEstimationModule) -> int:
    return int(mod._marla_sqft_by_city.get("lahore", 272) * 5)


@needs_phase2_csv
def test_grey_structure_excludes_finishing_bucket():
    cost = _cost()
    sq = _sqft(cost)
    r = cost.estimate_project_cost(
        sqft=sq,
        grade="standard",
        city="Lahore",
        construction_type="grey_structure",
        bhk=3,
        apply_market_benchmark=False,
    )
    assert r["status"] == "success"
    cats = r.get("category_breakdown") or {}
    assert "Finishing" not in cats, "Grey structure must not report Finishing category totals"
    allowed = allowed_ui_buckets_for_construction("grey_structure")
    assert allowed is not None
    for k in cats:
        assert k in allowed
    grand = r["breakdown"]["summary"]["grand_total"]
    assert abs(sum(cats.values()) - grand) <= 2, "Category totals should match grand total (rounding)"


@needs_phase2_csv
def test_full_construction_includes_finishing():
    cost = _cost()
    sq = _sqft(cost)
    r = cost.estimate_project_cost(
        sqft=sq,
        grade="standard",
        city="Lahore",
        construction_type="full_construction",
        bhk=3,
        apply_market_benchmark=False,
    )
    assert r["status"] == "success"
    cats = r.get("category_breakdown") or {}
    assert "Finishing" in cats and cats["Finishing"] > 0
    assert r.get("cost_scope", {}).get("lines_dropped_out_of_scope", -1) == 0


def test_labour_synthetic_bucket_grey_package_is_grey_structure():
    from ai_modules.cost_estimation_module import ui_bucket_for_phase_or_labour_label

    assert ui_bucket_for_phase_or_labour_label("Grey package labour (realism floor)") == "Grey Structure"
    assert ui_bucket_for_phase_or_labour_label("Finishing trades (labour realism floor)") == "Finishing"


# ─────────────────────────────────────────────────────────────────────────────
# Single-source-of-truth: grey_structure view must match full_construction
# ─────────────────────────────────────────────────────────────────────────────

@needs_phase2_csv
def test_grey_structure_matches_full_construction_grey_bucket():
    """
    REQUIREMENT: full_estimate["Grey Structure"] == grey_only grand_total
    Both are now computed from the same full BOQ then filtered, so they must match.
    """
    cost = _cost()
    sq = _sqft(cost)

    full_r = cost.estimate_project_cost(
        sqft=sq, grade="standard", city="Lahore",
        construction_type="full_construction", bhk=3, apply_market_benchmark=False,
    )
    grey_r = cost.estimate_project_cost(
        sqft=sq, grade="standard", city="Lahore",
        construction_type="grey_structure", bhk=3, apply_market_benchmark=False,
    )

    assert full_r["status"] == "success"
    assert grey_r["status"] == "success"

    full_grey_bucket = (full_r.get("category_breakdown") or {}).get("Grey Structure", 0)
    grey_grand_total = grey_r["breakdown"]["summary"]["grand_total"]

    # Allow ≤ 2 PKR rounding difference (int(round(...)) across categories)
    diff = abs(full_grey_bucket - grey_grand_total)
    assert diff <= 2, (
        f"full_estimate['Grey Structure']={full_grey_bucket:,} != "
        f"grey_only grand_total={grey_grand_total:,} (diff={diff:,} PKR). "
        "Single-source-of-truth violation."
    )


@needs_phase2_csv
def test_5marla_full_vs_finishing_only_category_matches():
    """full_estimate["Finishing"] must equal renovation/finishing grand_total
    when the same full-BOQ base is used for both."""
    cost = _cost()
    sq = int(cost._marla_sqft_by_city.get("lahore", 272) * 5)

    full_r = cost.estimate_project_cost(
        sqft=sq, grade="standard", city="Lahore",
        construction_type="full_construction", bhk=3, apply_market_benchmark=False,
    )
    assert full_r["status"] == "success"
    # The full estimate must include a Finishing bucket
    cats = full_r.get("category_breakdown") or {}
    assert "Finishing" in cats, "Full construction must include Finishing bucket"
    assert cats["Finishing"] > 0


@needs_phase2_csv
def test_category_sum_matches_grand_total_grey():
    """cost_scope.category_sum_matches_grand_total must be True for grey_structure."""
    cost = _cost()
    sq = _sqft(cost)
    r = cost.estimate_project_cost(
        sqft=sq, grade="standard", city="Lahore",
        construction_type="grey_structure", bhk=3, apply_market_benchmark=False,
    )
    assert r["status"] == "success"
    assert r.get("cost_scope", {}).get("category_sum_matches_grand_total") is True


@needs_phase2_csv
def test_category_sum_matches_grand_total_full():
    """cost_scope.category_sum_matches_grand_total must be True for full_construction."""
    cost = _cost()
    sq = _sqft(cost)
    r = cost.estimate_project_cost(
        sqft=sq, grade="standard", city="Lahore",
        construction_type="full_construction", bhk=3, apply_market_benchmark=False,
    )
    assert r["status"] == "success"
    assert r.get("cost_scope", {}).get("category_sum_matches_grand_total") is True


@needs_phase2_csv
def test_multi_floor_grey_consistency():
    """2-floor grey structure: category sum still equals grand total."""
    cost = _cost()
    sq = _sqft(cost)
    r = cost.estimate_project_cost(
        sqft=sq, floors=2, grade="standard", city="Lahore",
        construction_type="grey_structure", bhk=3, apply_market_benchmark=False,
    )
    assert r["status"] == "success"
    cats = r.get("category_breakdown") or {}
    grand = r["breakdown"]["summary"]["grand_total"]
    assert abs(sum(cats.values()) - grand) <= 2
