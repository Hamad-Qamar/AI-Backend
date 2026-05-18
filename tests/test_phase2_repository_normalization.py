"""Phase-2 DataFrame column normalization (CSV / Supabase key drift)."""

import pandas as pd

from ai_modules.phase2_repository import _normalize_phase2_dataframe


def test_city_area_aliases_sqft_and_city():
    raw = pd.DataFrame(
        [
            {"City": "Lahore", "SQFT PER MARLA": 225},
            {"Town": "karachi", "marlasqft": 272},
        ]
    )
    norm = _normalize_phase2_dataframe(raw, table="city_area_standards")
    assert "city" in norm.columns and "marla_sqft" in norm.columns
    assert norm.loc[0, "city"] == "Lahore"
    assert norm.loc[0, "marla_sqft"] == 225
    assert norm.loc[1, "city"] == "karachi"


def test_materials_master_aliases_name_description_phase():
    raw = pd.DataFrame(
        [
            {
                "Material_ID": "M1",
                "Item_Name": "Cement Bag",
                "DESC": "OPC grade",
                "Phase_Name": "Grey",
            }
        ]
    )
    norm = _normalize_phase2_dataframe(raw, table="materials_master")
    assert norm.loc[0, "material_id"] == "M1"
    assert norm.loc[0, "name"] == "Cement Bag"
    assert norm.loc[0, "description"] == "OPC grade"
    assert norm.loc[0, "phase"] == "Grey"


def test_materials_fill_name_from_alias_when_sparse():
    raw = pd.DataFrame(
        [{"material_id": "M2", "name": "", "item_name": "Steel bar"}],
    )
    norm = _normalize_phase2_dataframe(raw, table="materials_master")
    assert norm.loc[0, "name"] == "Steel bar"


def test_phase_labor_mapping_phase_alias():
    raw = pd.DataFrame([{"phase_name": "Finishing", "work_type": "Painter", "unit": "day"}])
    norm = _normalize_phase2_dataframe(raw, table="phase_labor_mapping")
    assert "phase" in norm.columns
    assert norm.loc[0, "phase"] == "Finishing"
