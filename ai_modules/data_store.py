"""
Phase-2 Data Store for BuildHive.

Replaces the monolithic `products.csv` architecture with multiple structured files:
  - materials_master.csv
  - pricing_data.csv
  - construction_rates.csv
  - labor_rates.csv
  - city_area_standards.csv
  - phase_labor_mapping.csv

Data is loaded via `phase2_repository` (CSV by default, or Supabase when PHASE2_SOURCE=supabase).

This module centralises loading + lookup logic so other modules (recommendations,
chatbot purchase flow, cost estimation) can share one consistent view.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .phase2_repository import Phase2Paths, get_phase2_repository

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ["Phase2DataStore", "Phase2Paths"]


class Phase2DataStore:
    def __init__(self, paths: Phase2Paths = Phase2Paths()):
        self.paths = paths

        self.city_area: pd.DataFrame = pd.DataFrame()
        self.materials_master: pd.DataFrame = pd.DataFrame()
        self.pricing_data: pd.DataFrame = pd.DataFrame()
        self.construction_rates: pd.DataFrame = pd.DataFrame()
        self.labor_rates: pd.DataFrame = pd.DataFrame()
        self.phase_labor_mapping: pd.DataFrame = pd.DataFrame()

        self._marla_sqft_by_city: Dict[str, float] = {}
        self._price_avg: Dict[Tuple[str, str], float] = {}  # (material_id, city) -> avg
        self._price_minmax: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        bundle = get_phase2_repository(paths=self.paths).load_all()
        self.city_area = bundle.city_area
        self.materials_master = bundle.materials_master
        self.pricing_data = bundle.pricing_data
        self.construction_rates = bundle.construction_rates
        self.labor_rates = bundle.labor_rates
        self.phase_labor_mapping = bundle.phase_labor_mapping

        # City marla sqft
        if not self.city_area.empty:
            for _, r in self.city_area.iterrows():
                city = str(r.get("city", "")).strip().lower()
                sqft = float(r.get("marla_sqft", 0) or 0)
                if abs(sqft - 272.25) < 1e-6 or abs(sqft - 272.5) < 1e-6:
                    sqft = 272.0
                if city and sqft > 0:
                    self._marla_sqft_by_city[city] = sqft

        # Pricing maps
        if not self.pricing_data.empty:
            for _, r in self.pricing_data.iterrows():
                mid = str(r.get("material_id", "")).strip()
                city = str(r.get("city", "")).strip().lower()
                if not mid or not city:
                    continue
                key = (mid, city)
                avg = float(r.get("price_avg", 0) or 0)
                pmin = float(r.get("price_min", 0) or 0)
                pmax = float(r.get("price_max", 0) or 0)
                if avg > 0:
                    self._price_avg[key] = avg
                if pmin > 0 and pmax > 0:
                    self._price_minmax[key] = (pmin, pmax)

        self._loaded = True
        logger.info(
            "Phase2DataStore loaded: materials=%d prices=%d cities=%d",
            len(self.materials_master) if not self.materials_master.empty else 0,
            len(self.pricing_data) if not self.pricing_data.empty else 0,
            len(self._marla_sqft_by_city),
        )

    def marla_sqft(self, city: str, default: float = 272.0) -> float:
        self.load()
        return float(self._marla_sqft_by_city.get((city or "").strip().lower(), default))

    def price_avg(self, material_id: str, city: str) -> float:
        """Return city avg price; fallback to global avg across cities."""
        self.load()
        key = (material_id, (city or "").strip().lower())
        if key in self._price_avg:
            return float(self._price_avg[key])
        vals = [v for (mid, _), v in self._price_avg.items() if mid == material_id]
        return float(np.mean(vals)) if vals else 0.0

    def price_range(self, material_id: str, city: str) -> Tuple[float, float]:
        """Return (min,max) for city; fallback to global min/max."""
        self.load()
        key = (material_id, (city or "").strip().lower())
        if key in self._price_minmax:
            mn, mx = self._price_minmax[key]
            return float(mn), float(mx)
        vals = [v for (mid, _), v in self._price_minmax.items() if mid == material_id]
        if vals:
            mins = [a for a, _ in vals]
            maxs = [b for _, b in vals]
            return float(min(mins)), float(max(maxs))
        return 0.0, 0.0
