# AI-Backend

BuildHive FastAPI backend: chatbot, material recommendations, and construction cost estimation.

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Deploy (Docker — recommended)

```bash
docker build -t ai-backend .
docker run -p 8080:8080 --env-file .env ai-backend
```

The image prefetches the SentenceTransformer model at build time so runtime does not need Hugging Face Hub access.

## Cloud environment variables

| Variable | Purpose |
| --- | --- |
| `PHASE2_SOURCE` | `csv` (default) or `supabase` |
| `SUPABASE_URL` | Supabase project URL (if using supabase) |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-only DB key (never expose to frontend) |
| `HF_TOKEN` | Optional; helps Docker build download embedding weights |
| `PORT` | HTTP port (default `8080`) |

For production without CSVs in the image, use Supabase: apply `supabase/migrations/0001_phase2_tables.sql`, seed tables, then set `PHASE2_SOURCE=supabase` and redeploy.

See `DOCUMENTATION.md` for full API and architecture details.
