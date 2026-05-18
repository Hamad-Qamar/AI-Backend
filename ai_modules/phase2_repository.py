"""
Phase-2 data access: CSV (local) or Supabase Postgres.

Controlled by env:
  PHASE2_SOURCE=csv|supabase   (default: csv)

Supabase (server-side only):
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  SUPABASE_SSL_VERIFY=true|false  (set false on dev machines with broken CA store)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _supabase_httpx_client() -> Any:
    """Build httpx client for Supabase (handles Windows CA / corporate proxy issues)."""
    import httpx

    if not _env_truthy("SUPABASE_SSL_VERIFY", default=True):
        logger.warning(
            "SUPABASE_SSL_VERIFY=false — TLS certificate verification disabled for Supabase"
        )
        return httpx.Client(verify=False)

    verify: Any = True
    try:
        import certifi

        verify = certifi.where()
    except ImportError:
        pass
    return httpx.Client(verify=verify)


def _snake_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase / strip column keys so CSV and PostgREST payloads match engine expectations."""
    if df.empty:
        return df
    out = df.copy()
    new_cols: List[str] = []
    for i, c in enumerate(out.columns):
        s = str(c).strip().lower().replace(" ", "_")
        new_cols.append(s if s else f"_col_{i}")
    out.columns = new_cols
    return out


def _coalesce_columns(
    df: pd.DataFrame, canonical: str, alternates: Sequence[str]
) -> pd.DataFrame:
    """
    Ensure `canonical` exists: prefer existing canonical, fill gaps from alternate column names
    (then drop alternates). Safe for string or numeric cells.
    """
    alts = [a for a in alternates if a in df.columns and a != canonical]
    if canonical not in df.columns:
        if not alts:
            return df
        out = df.rename(columns={alts[0]: canonical})
        alts = alts[1:]
    else:
        out = df
    for a in alts:
        if a not in out.columns:
            continue
        base = out[canonical]
        fill = out[a]
        if base.dtype == object or fill.dtype == object:
            b = base.astype(str).replace({"nan": ""}).str.strip()
            f = fill.astype(str).replace({"nan": ""}).str.strip()
            out[canonical] = b.where(b != "", f)
        else:
            out[canonical] = base.fillna(fill)
        out = out.drop(columns=[a])
    return out


def _normalize_phase2_dataframe(df: pd.DataFrame, *, table: str) -> pd.DataFrame:
    """Apply shared + per-table column aliases (Supabase UI / legacy CSV exports)."""
    if df.empty:
        return df
    out = _snake_column_names(df)
    # Keys used across several rate/city tables
    out = _coalesce_columns(out, "city", ("city_name", "town", "urban_area", "location"))
    if table in ("materials_master", "pricing_data"):
        out = _coalesce_columns(
            out, "material_id", ("materialid", "mat_id", "sku")
        )
    if table == "city_area_standards":
        out = _coalesce_columns(
            out,
            "marla_sqft",
            (
                "sqft_per_marla",
                "sqft_marlas",
                "marlasqft",
                "sqft_marla",
                "marla_size_sqft",
                "area_marla_sqft",
            ),
        )
    if table == "materials_master":
        out = _coalesce_columns(
            out, "name", ("item_name", "material_name", "product_name", "title", "label")
        )
        out = _coalesce_columns(
            out, "description", ("desc", "details", "long_description", "spec")
        )
        out = _coalesce_columns(out, "phase", ("phase_name", "work_phase", "construction_phase"))
    if table == "pricing_data":
        out = _coalesce_columns(
            out, "confidence_score", ("confidence", "conf_score", "score")
        )
    if table == "labor_rates":
        out = _coalesce_columns(out, "work_type", ("worktype", "trade", "labour_type"))
        out = _coalesce_columns(out, "rate_avg", ("avg_rate", "average_rate"))
    if table == "construction_rates":
        out = _coalesce_columns(
            out, "construction_type", ("type", "building_type", "ctype")
        )
    if table == "phase_labor_mapping":
        out = _coalesce_columns(
            out, "phase", ("phase_name", "work_phase", "construction_phase")
        )
    return out


@dataclass(frozen=True)
class Phase2Paths:
    materials_master: str = "materials_master.csv"
    pricing_data: str = "pricing_data.csv"
    construction_rates: str = "construction_rates.csv"
    labor_rates: str = "labor_rates.csv"
    city_area_standards: str = "city_area_standards.csv"
    phase_labor_mapping: str = "phase_labor_mapping.csv"


@dataclass(frozen=True)
class Phase2DataBundle:
    city_area: pd.DataFrame
    materials_master: pd.DataFrame
    pricing_data: pd.DataFrame
    construction_rates: pd.DataFrame
    labor_rates: pd.DataFrame
    phase_labor_mapping: pd.DataFrame


class Phase2Repository(Protocol):
    def load_all(self) -> Phase2DataBundle: ...

    def fetch_materials_metadata_by_ids(self, material_ids: List[str]) -> pd.DataFrame:
        """Default: no remote batch (CSV bundle already includes materials_master)."""
        return pd.DataFrame()


class CsvPhase2Repository:
    """Load Phase-2 tables from CSV files (current behaviour)."""

    def __init__(self, paths: Phase2Paths) -> None:
        self.paths = paths

    def fetch_materials_metadata_by_ids(self, material_ids: List[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def load_all(self) -> Phase2DataBundle:
        def _read(path: str, table: str) -> pd.DataFrame:
            if os.path.exists(path):
                raw = pd.read_csv(path)
            else:
                alt = os.path.join(os.getcwd(), path)
                if os.path.exists(alt):
                    raw = pd.read_csv(alt)
                else:
                    logger.warning("Phase-2 CSV missing: %s", path)
                    return pd.DataFrame()
            return _normalize_phase2_dataframe(raw, table=table)

        return Phase2DataBundle(
            city_area=_read(self.paths.city_area_standards, "city_area_standards"),
            materials_master=_read(self.paths.materials_master, "materials_master"),
            pricing_data=_read(self.paths.pricing_data, "pricing_data"),
            construction_rates=_read(self.paths.construction_rates, "construction_rates"),
            labor_rates=_read(self.paths.labor_rates, "labor_rates"),
            phase_labor_mapping=_read(
                self.paths.phase_labor_mapping, "phase_labor_mapping"
            ),
        )


class SupabasePhase2Repository:
    """Load Phase-2 tables from Supabase Postgres (mirror schema in supabase/migrations)."""

    _TABLES = (
        "city_area_standards",
        "materials_master",
        "pricing_data",
        "construction_rates",
        "labor_rates",
        "phase_labor_mapping",
    )

    def __init__(self, url: str, key: str) -> None:
        self._url = (url or "").strip().rstrip("/")
        self._key = (key or "").strip()
        self._client: Any = None

    @classmethod
    def from_env(cls) -> "SupabasePhase2Repository":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError(
                "SupabasePhase2Repository requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "when PHASE2_SOURCE=supabase."
            )
        return cls(url=url, key=key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from supabase import create_client  # type: ignore
                from supabase.lib.client_options import SyncClientOptions  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Install the Supabase client: pip install supabase"
                ) from e
            self._client = create_client(
                self._url,
                self._key,
                options=SyncClientOptions(httpx_client=_supabase_httpx_client()),
            )
        return self._client

    def _fetch_table(self, name: str) -> pd.DataFrame:
        client = self._get_client()
        resp = client.table(name).select("*").execute()
        rows: List[Dict[str, Any]] = list(resp.data or [])
        if not rows:
            logger.warning("Supabase table %s returned 0 rows", name)
            return pd.DataFrame()
        return _normalize_phase2_dataframe(pd.DataFrame(rows), table=name)

    def load_all(self) -> Phase2DataBundle:
        logger.info("Loading Phase-2 data from Supabase (%d tables)", len(self._TABLES))
        bundle = Phase2DataBundle(
            city_area=self._fetch_table("city_area_standards"),
            materials_master=self._fetch_table("materials_master"),
            pricing_data=self._fetch_table("pricing_data"),
            construction_rates=self._fetch_table("construction_rates"),
            labor_rates=self._fetch_table("labor_rates"),
            phase_labor_mapping=self._fetch_table("phase_labor_mapping"),
        )
        logger.info(
            "Phase-2 Supabase shapes: city_area=%s materials_master=%s pricing_data=%s "
            "construction_rates=%s labor_rates=%s phase_labor_mapping=%s",
            len(bundle.city_area.index),
            len(bundle.materials_master.index),
            len(bundle.pricing_data.index),
            len(bundle.construction_rates.index),
            len(bundle.labor_rates.index),
            len(bundle.phase_labor_mapping.index),
        )
        return bundle

    def fetch_materials_metadata_by_ids(self, material_ids: List[str]) -> pd.DataFrame:
        """Batch-fetch material_id, name, phase, description for BOQ enrichment (one round-trip per chunk)."""
        ids = [str(x).strip() for x in material_ids if str(x).strip()]
        if not ids:
            return pd.DataFrame()
        client = self._get_client()
        out_rows: List[Dict[str, Any]] = []
        chunk = 120
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            resp = (
                client.table("materials_master")
                .select("*")
                .in_("material_id", part)
                .execute()
            )
            out_rows.extend(list(resp.data or []))
        if not out_rows:
            return pd.DataFrame()
        return _normalize_phase2_dataframe(
            pd.DataFrame(out_rows), table="materials_master"
        )


def get_phase2_repository(paths: Optional[Phase2Paths] = None) -> Phase2Repository:
    """
    Factory: CSV (default) or Supabase based on PHASE2_SOURCE.
    When using Supabase, `paths` is ignored.
    """
    src = (os.getenv("PHASE2_SOURCE") or "csv").strip().lower()
    if src == "supabase":
        return SupabasePhase2Repository.from_env()
    return CsvPhase2Repository(paths or Phase2Paths())
