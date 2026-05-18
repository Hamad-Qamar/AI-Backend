# BuildHive — Project Documentation (Single Source of Truth)

This file consolidates the project documentation into **one** place.

## What this project is

BuildHive is a FastAPI backend + browser UI that provides:

- **Chatbot**: deterministic, UX-friendly responses for greetings/utility/safety + routes to modules for recommendations and cost estimation.
- **Material recommendations**: uses the recommendation engine and prebuilt indexes to suggest materials.
- **Construction cost estimation**: quantity-driven BOQ (first-principles + realism floors/caps) with market benchmark controls.

---

## Quick start (local)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the server

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 3) Open in browser

- **UI**: `http://127.0.0.1:8000`
- **Swagger API docs**: `http://127.0.0.1:8000/docs`

If you don’t see recent changes, do a hard refresh: **Ctrl+Shift+R**.

---

## Project structure (what matters)

### Core runtime files

- `main.py`
  - FastAPI app and API routes.
  - Initializes modules and injects them into the chatbot.
- `index.html`
  - The browser UI for chatbot + cost estimator.
- `ai_modules/`
  - `chatbot_module.py`: chatbot orchestration + deterministic utility/safety layer.
  - `recommendation_module.py`: materials recommendation logic (uses `materials.index`).
  - `cost_estimation_module.py`: BOQ + labour + benchmark logic (Phase-2 CSVs).
  - `data_store.py`: loads Phase-2 tabular data (CSV or Supabase) and city marla standards.
  - `phase2_repository.py`: single loader for Phase-2 tables (`PHASE2_SOURCE=csv|supabase`).

### Data files (runtime inputs)

- `materials_master.csv`: Phase‑2 BOQ ratios (usage_type/usage_ratio).
- `pricing_data.csv`: city-wise material pricing.
- `phase_labor_mapping.csv`: mapping of phases to labour productivity rules.
- `labor_rates.csv`: labour rate card.
- `construction_rates.csv`: benchmark/aux rate card.
- `city_area_standards.csv`: city marla→sqft conversions (**standardized to 272** where applicable).
- `prices.json`: small pricing config (legacy/compat inputs).

### Index files (runtime accelerators)

- `materials.index` + `materials.index.meta.json`: recommendation search index (prebuilt).
- `products.index` + `products.index.meta.json`: product search index (prebuilt).

### Tests

- `tests/`: pytest suite (cost engine + chatbot + advanced material toggles).
- `pytest.ini`: test markers configuration.

---

## API endpoints (FastAPI)

Exact request/response schemas are available in Swagger:

- `GET /docs`

High-level endpoints:

- **Chat**
  - `POST /chat`
  - Request includes:
    - `query` (string)
    - optional `conversation_id` (string)
    - optional `current_page` (string)
    - optional `use_llm` (bool; project defaults to deterministic)
- **Cost estimation**
  - `POST /estimate-cost` (see `/docs` for exact route name/schema)
- **Recommendations**
  - `POST /recommend-materials` (see `/docs` for exact route name/schema)

---

## Chatbot behavior (UX rules)

The chatbot is intentionally **simple and deterministic** for UX:

- **Greetings**: replies with `Hi! How may I help you?`
- **Default menu/clarification**: shows the “What would you like to do?” menu.
- **Module answers**: cost/recommendation answers are short and sourced from modules.
- **Navigation button**: only one relevant navigation action is returned.
- **No follow-up chips spam**: follow-ups are suppressed for routed module answers.
- **Safety**: refuses unsafe content and provides safe guidance.

---

## Cost estimation — quantity logic guarantees

The cost estimator is designed to avoid the common “small house hallucinations”.

Key safeguards implemented in `ai_modules/cost_estimation_module.py`:

- **Structural BOQ minimums (grey + full)**:
  - Prevents bricks/cement collapsing on 1–2 marla footprints.
  - Uses physics-based brick minimum for `full_construction` (\(540 \times \sqrt{\text{sqft}}\)).
- **Tile realism (min + hard cap)**:
  - Floor tiles capped to `floor_area × 1.15` plus wet wall allowance.
- **Sanitary realism**:
  - Sanitary/shower sets are clamped to feasible bathroom count.
- **Window frame realism**:
  - Aluminum window frame rft is capped by room/bath/kitchen openings.
- **Septic realism**:
  - Small homes are capped to pre-cast ring pricing bands.
- **Marla normalization**:
  - Any `272.25` / `272.5` style conversions are normalized to **272**.

---

## Phase-2 data source (CSV vs Supabase)

By default the backend reads the six Phase-2 CSVs from the project root (same as before). To load the same tables from **Supabase Postgres** instead:

1. Create a Supabase project and apply the schema in `supabase/migrations/0001_phase2_tables.sql` (SQL Editor or Supabase migrations).
2. Seed rows from your local CSVs (Table Editor import, or `python scripts/seed_supabase_from_csv.py` with env set).
3. Set environment variables on the server (never expose the service role key to the browser or frontend bundles):

| Variable | Purpose |
| --- | --- |
| `PHASE2_SOURCE` | `csv` (default) or `supabase` |
| `SUPABASE_URL` | Project URL, e.g. `https://<ref>.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-only key with full DB access |

When `PHASE2_SOURCE=supabase`, custom CSV path arguments passed into `CostEstimationModule` are ignored; all six tables are fetched from Supabase.

### Deployment (Option A — Supabase only, no CSVs in the image)

Cloud logs like `Phase-2 CSV missing` mean the server is still on **CSV mode** or the env vars are not set. Do this on your host (Render, Railway, Fly, etc.):

1. In Supabase: apply `supabase/migrations/0001_phase2_tables.sql` and seed the six tables (Table Editor or `scripts/seed_supabase_from_csv.py`).
2. In the platform **Environment / Secrets** UI, set:
   - `PHASE2_SOURCE=supabase`
   - `SUPABASE_URL` — your project URL (`https://<ref>.supabase.co`)
   - `SUPABASE_SERVICE_ROLE_KEY` — **server only**; never expose in frontend or public repos
3. Redeploy / restart the service. Check `GET /health`: `recommendation` and `cost_estimation` should be `online`.

Optional: `SUPABASE_SSL_VERIFY=false` only for broken CA stores (prefer fixing TLS in production). See `.env.example` for a full template.

### Railway — Hugging Face embedding model (`all-MiniLM-L6-v2`)

Recommendation + chatbot need SentenceTransformer weights. If logs show **cannot connect to huggingface.co** / **cannot find them in cached files**, the deployed image never downloaded them (cold runtime often blocks or rate-limits the Hub).

**Recommended:** Deploy with the repo **`Dockerfile`**. The build runs `scripts/prefetch_embedding_model.py` once into `HF_HOME`; runtime uses **`HF_HUB_OFFLINE=1`** and **`TRANSFORMERS_OFFLINE=1`** so startup does not need Hub access.

If the **build** also cannot reach the Hub, set a Hugging Face [access token](https://huggingface.co/settings/tokens) (`HF_TOKEN`) for the Docker **build**.

**Without Docker:** Add a Railway build step that runs `python scripts/prefetch_embedding_model.py` after `pip install`, then mirror the same **`HF_HOME`** plus offline flags at runtime.

---

## Running tests

Run all tests:

```bash
python -m pytest tests/ -q
```

---

## Regenerating Phase‑2 data (only if needed)

If you update your Phase‑2 CSVs and want to regenerate derived artifacts, use the project scripts:

- `generate_phase2.py`
- `sync_prices.py`
- `scripts/seed_supabase_from_csv.py` — upserts CSV rows into Supabase (requires `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`).

Note: these are **build utilities**, not required at runtime unless you use Supabase for Phase-2.

---

## Maintenance notes

- Don’t commit virtual environments (`buildhive_ai/` or `.venv/`) into source control.
- If you see stale UI behavior after backend changes: restart `uvicorn` and hard refresh the browser.

