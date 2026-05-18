-- Phase-2 tables mirroring project CSVs (BuildHive cost engine + datastore).
-- Run in Supabase SQL Editor or via `supabase db push` after linking the project.

-- City marla standards
CREATE TABLE IF NOT EXISTS city_area_standards (
  city TEXT PRIMARY KEY,
  marla_sqft DOUBLE PRECISION NOT NULL
);

-- Materials master (one row per material_id in current dataset)
CREATE TABLE IF NOT EXISTS materials_master (
  material_id TEXT PRIMARY KEY,
  name TEXT,
  phase TEXT,
  category TEXT,
  subcategory TEXT,
  description TEXT,
  specifications TEXT,
  unit TEXT,
  usage_type TEXT,
  usage_ratio DOUBLE PRECISION,
  quality_grade TEXT,
  brand TEXT,
  notes TEXT,
  functional_tag TEXT,
  synonyms TEXT,
  room_type TEXT,
  finishing_tier_min TEXT
);

-- Pricing (natural key for upserts from CSV seed)
CREATE TABLE IF NOT EXISTS pricing_data (
  price_id TEXT PRIMARY KEY,
  material_id TEXT NOT NULL,
  city TEXT NOT NULL,
  price_min DOUBLE PRECISION,
  price_max DOUBLE PRECISION,
  price_avg DOUBLE PRECISION,
  confidence_score DOUBLE PRECISION,
  last_updated TEXT,
  source_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_pricing_material ON pricing_data (material_id);
CREATE INDEX IF NOT EXISTS idx_pricing_city ON pricing_data (city);
CREATE INDEX IF NOT EXISTS idx_pricing_material_city ON pricing_data (material_id, city);

-- Benchmark construction rates per city / type
CREATE TABLE IF NOT EXISTS construction_rates (
  city TEXT NOT NULL,
  construction_type TEXT NOT NULL,
  cost_min_per_sqft DOUBLE PRECISION,
  cost_max_per_sqft DOUBLE PRECISION,
  cost_avg_per_sqft DOUBLE PRECISION,
  last_updated TEXT,
  PRIMARY KEY (city, construction_type)
);

-- Labour rates
CREATE TABLE IF NOT EXISTS labor_rates (
  city TEXT NOT NULL,
  work_type TEXT NOT NULL,
  rate_min DOUBLE PRECISION,
  rate_max DOUBLE PRECISION,
  rate_avg DOUBLE PRECISION,
  unit TEXT,
  last_updated TEXT,
  confidence_score DOUBLE PRECISION,
  PRIMARY KEY (city, work_type)
);

-- Phase labour productivity mapping
CREATE TABLE IF NOT EXISTS phase_labor_mapping (
  phase TEXT NOT NULL,
  work_type TEXT NOT NULL,
  unit TEXT NOT NULL,
  productivity_rate DOUBLE PRECISION,
  notes TEXT,
  PRIMARY KEY (phase, work_type, unit)
);
