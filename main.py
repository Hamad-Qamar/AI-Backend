"""
BuildHive AI Backend - Main Application
Modular architecture with 3 AI Systems.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import logging
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()  # .env: PHASE2_SOURCE, SUPABASE_*, etc.
except ImportError:
    pass

# Prefer certifi CA bundle for Hugging Face / httpx when system store is incomplete.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# Import AI Modules
from ai_modules import ChatBotModule, RecommendationModule, CostEstimationModule
from ai_modules.cost_estimation_module import normalize_city_key

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="BuildHive AI Backend",
    description="Modular AI system with Chat, Recommendations, and Cost Estimation",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# INITIALIZE AI MODULES
# ═══════════════════════════════════════════════════════════════

logger.info("Initializing AI modules...")

chatbot = None
recommendation = None
cost_estimation = None

try:
    chatbot = ChatBotModule()
    logger.info("✓ ChatBot Module loaded")
except Exception as e:
    logger.error(f"Failed to load ChatBot: {e}")

try:
    recommendation = RecommendationModule()
    logger.info("✓ Recommendation Module loaded")
except Exception as e:
    logger.error(f"Failed to load Recommendation: {e}")

try:
    cost_estimation = CostEstimationModule()
    logger.info("✓ Cost Estimation Module loaded")
except Exception as e:
    logger.error(f"Failed to load Cost Estimation: {e}")

# Inject modules into chatbot for routing
if chatbot and recommendation and cost_estimation:
    chatbot.inject_modules(recommendation, cost_estimation)

# Ratings file
RATINGS_LOG = "chat_ratings.jsonl"

# ═══════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    query: str
    user_role: str = "buyer"
    current_page: Optional[str] = None
    conversation_id: Optional[str] = None
    # use_llm accepted for backward-compat but always treated as True
    use_llm: bool = True

class ChatRatingRequest(BaseModel):
    response_id: str
    rating: str   # "up" | "down"

class RecommendationRequest(BaseModel):
    text: str
    budget: Optional[str] = None
    city: Optional[str] = None
    quality: Optional[str] = "Standard"
    area_sqft: Optional[float] = None
    area: Optional[str] = None  # e.g. "5 marla", "1 kanal", "2000 sqft"
    top_n_per_cat: int = 8
    finishing_tier: Optional[str] = None  # economy|standard|premium|luxury; defaults from quality
    # use_llm accepted for backward-compat but always True
    use_llm: bool = True

class CostEstimationRequest(BaseModel):
    sqft: Optional[int] = None
    floors: int = 1
    quality: str = "Standard"
    city: str = "Lahore"
    timeline_months: Optional[int] = None
    # v2 fields
    area: Optional[str] = None
    project_type: Optional[str] = "Full Construction"  # Full / Grey / Renovation (UI)
    # Primary layout for houses: 1–6 BHK (maps to default bed/bath/kitchen; see estimator_policy).
    bhk: Optional[int] = None
    bedrooms: int = 0
    washrooms: int = 0
    kitchens: int = 0
    compare: bool = False
    use_llm: bool = True


class UniversalEstimateRequest(BaseModel):
    building_type: str
    city: str = "Lahore"
    # exactly one preferred, but we allow multiple and pick in this order:
    # sqft, marla, kanal, sqm, sqy, acre
    area: Dict[str, Any]
    floors: Optional[int] = None
    bhk: Optional[int] = None
    finishing_tier: Optional[str] = None
    capacity: Optional[Dict[str, Any]] = None
    wall_height_ft: Optional[float] = None
    span_m: Optional[float] = None
    industrial_heavy: bool = False

class SaveRecommendationRequest(BaseModel):
    project_name: str = "My Project"
    input: Dict[str, Any]
    recommendations: Dict[str, Any]
    total_estimated_cost: Optional[float] = None

# ═══════════════════════════════════════════════════════════════
# STATIC CONTENT
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "BuildHive AI Backend v3.0", "status": "online"}

# ═══════════════════════════════════════════════════════════════
# 1️⃣ CHATBOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/chat", tags=["ChatBot"])
async def chat_endpoint(request: ChatRequest):
    if not chatbot:
        return JSONResponse(status_code=503, content={"error": "ChatBot unavailable"})
    return chatbot.answer_query(
        request.query,
        request.user_role,
        request.current_page,
        conversation_id=request.conversation_id,
        use_llm=request.use_llm,
    )

@app.post("/chat/rate", tags=["ChatBot"])
async def rate_chat_response(request: ChatRatingRequest):
    entry = {"id": str(uuid.uuid4()), "response_id": request.response_id, "rating": request.rating}
    with open(RATINGS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return JSONResponse(status_code=204, content=None)

# ═══════════════════════════════════════════════════════════════
# 2️⃣ RECOMMENDATION SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/recommend", tags=["Recommendation"])
async def recommend_endpoint(request: RecommendationRequest):
    if not recommendation:
        return JSONResponse(status_code=503, content={"error": "Recommendation service unavailable"})
    
    budget_pkr = None
    if request.budget:
        try:
            if "lakh" in request.budget.lower():
                budget_pkr = int(float("".join(c for c in request.budget if c.isdigit() or c == ".")) * 100000)
            else:
                budget_pkr = int("".join(c for c in request.budget if c.isdigit()))
        except: pass

    # Resolve area_sqft. Prefer explicit numeric, else parse free-text via the
    # cost module (it already understands marla/kanal/sqft/sqm).
    area_sqft = request.area_sqft
    if area_sqft is None and request.area and cost_estimation is not None:
        try:
            area_sqft = float(cost_estimation.parse_area(request.area))
        except Exception:
            area_sqft = None

    return recommendation.recommend(
        text=request.text,
        budget_pkr=budget_pkr,
        city=request.city,
        quality=request.quality or "Standard",
        area_sqft=area_sqft,
        top_n_per_cat=request.top_n_per_cat,
        use_llm=True,   # always on
        finishing_tier=request.finishing_tier,
    )

@app.post("/save-recommendation", tags=["Recommendation"])
async def save_recommendation(request: SaveRecommendationRequest):
    """Persist a recommendation set to saved_recommendations.json."""
    import json as _json
    from datetime import datetime
    save_path = "saved_recommendations.json"
    try:
        try:
            with open(save_path, encoding="utf-8") as f:
                saved: List[Dict[str, Any]] = _json.load(f)
        except (FileNotFoundError, ValueError):
            saved = []

        entry = {
            "id":                   str(uuid.uuid4()),
            "project_name":         request.project_name,
            "created_at":           datetime.utcnow().isoformat() + "Z",
            "input":                request.input,
            "recommendations":      request.recommendations,
            "total_estimated_cost": request.total_estimated_cost,
        }
        saved.append(entry)
        with open(save_path, "w", encoding="utf-8") as f:
            _json.dump(saved, f, ensure_ascii=False, indent=2)
        return {"status": "saved", "id": entry["id"]}
    except Exception as exc:
        logger.error("save_recommendation error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/saved-recommendations", tags=["Recommendation"])
async def list_saved_recommendations():
    """Return all saved recommendations."""
    import json as _json
    save_path = "saved_recommendations.json"
    try:
        with open(save_path, encoding="utf-8") as f:
            return {"status": "success", "saved": _json.load(f)}
    except FileNotFoundError:
        return {"status": "success", "saved": []}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/categories", tags=["Search"])
def get_all_categories():
    if not recommendation: return {"categories": []}
    cats = recommendation.products["category"].dropna().unique().tolist()
    return {"categories": sorted(cats), "total": len(cats)}

@app.get("/search", tags=["Search"])
def search_items(q: str = Query(...), limit: int = 10):
    if not recommendation: return {"status": "error"}
    df = recommendation.semantic_search(q, top_k=limit*2)
    df = df.drop_duplicates("item_name").head(limit)
    return {"query": q, "results": df[["item_name", "category", "brand", "quality_grade", "unit", "market_price_pkr", "recommendation_score"]].to_dict(orient="records")}


@app.get("/estimator-config", tags=["Cost Estimation"])
def get_estimator_config():
    """Canonical cities (with rate cards), feasibility bands, floor multipliers — for UI + validation."""
    if not cost_estimation:
        return {"cities": [], "feasibility_bands": [], "floor_policy": {}}
    return cost_estimation.get_estimator_catalog()

# ═══════════════════════════════════════════════════════════════
# 3️⃣ COST ESTIMATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/estimate-cost", tags=["Cost Estimation"])
async def estimate_cost_endpoint(request: CostEstimationRequest):
    if not cost_estimation:
        return JSONResponse(status_code=503, content={"error": "Cost Estimation unavailable"})
    
    # Map UI project type → engine construction_type
    pt = (request.project_type or "Full Construction").lower()
    if "grey" in pt:
        ctype = "grey_structure"
    elif "reno" in pt or "finish" in pt:
        ctype = "renovation"
    else:
        ctype = "full_construction"

    if request.area:
        # Allow free-text, but keep it structured for the engine.
        return cost_estimation.estimate_from_text(
            f"{request.area} {ctype} in {request.city} {request.quality}",
            use_llm=request.use_llm,
        )

    city_key = normalize_city_key(request.city or "")
    if cost_estimation._supported_pricing_cities and city_key not in cost_estimation._supported_pricing_cities:
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported_city",
                "message": f"City '{request.city}' has no PKR rate card in pricing_data.",
                "supported_cities": sorted(c.title() for c in cost_estimation._supported_pricing_cities),
            },
        )

    return cost_estimation.estimate_project_cost(
        sqft=request.sqft or 1000,
        floors=request.floors,
        grade=request.quality,
        city=request.city,
        construction_type=ctype,
        bedrooms=request.bedrooms,
        washrooms=request.washrooms,
        kitchens=request.kitchens,
        bhk=request.bhk,
        timeline_months=request.timeline_months,
        compare=request.compare,
        use_llm=request.use_llm,
    )


@app.post("/estimate-universal", tags=["Cost Estimation"])
async def estimate_universal_endpoint(request: UniversalEstimateRequest):
    """
    Universal BOQ engine:
    - supports all building types (residential/commercial/institutional/industrial)
    - derives quantities from first principles (no lump-sum packages)
    - returns structured JSON + validation warnings
    """
    if not cost_estimation:
        return JSONResponse(status_code=503, content={"error": "Cost Estimation unavailable"})
    try:
        return cost_estimation.estimate_universal_boq(request.model_dump())
    except Exception as exc:
        logger.error("estimate_universal error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/estimate-cost/compare", tags=["Cost Estimation"])
async def cost_comparison(sqft: int = 1000, floors: int = 1, city: str = "Lahore"):
    if not cost_estimation: return {"error": "service offline"}
    return cost_estimation.compare_quality_tiers(sqft, floors, city)

# ═══════════════════════════════════════════════════════════════
# GUIDES & DATA
# ═══════════════════════════════════════════════════════════════

PHASE_GUIDE = [
    {"phase": 1, "name": "Site Preparation & Foundation", "categories": ["Raw Materials", "RCC & Structural", "Chemicals & Treatments"]},
    {"phase": 2, "name": "Grey Structure (Columns, Beams, Brickwork)", "categories": ["Raw Materials", "RCC & Structural", "Masonry & Wall Work"]},
    {"phase": 3, "name": "Roof Slab & Waterproofing", "categories": ["Raw Materials", "Roofing Materials", "Chemicals & Treatments"]},
    {"phase": 4, "name": "Plumbing & Electrical Rough-in", "categories": ["Plumbing - Pipes & Fittings", "Electrical - Wiring & Cables"]},
    {"phase": 5, "name": "Plastering & Wall Finishing", "categories": ["Masonry & Wall Work", "Raw Materials"]},
    {"phase": 6, "name": "Flooring", "categories": ["Flooring Materials", "Masonry & Wall Work"]},
    {"phase": 7, "name": "Doors, Windows & Carpentry", "categories": ["Doors & Windows", "Wood & Carpentry"]},
    {"phase": 8, "name": "Sanitary & Fixtures", "categories": ["Sanitary Items", "Plumbing - Taps & Fixtures", "Kitchen Materials"]},
    {"phase": 9, "name": "Electrical Finishing", "categories": ["Electrical - Switchgear", "Electrical - Protection", "Electrical - Lighting & Fixtures"]},
    {"phase": 10, "name": "Painting & Final Finishing", "categories": ["Paint & Finishing", "Hardware & Fasteners"]},
]

@app.get("/phases", tags=["Guides"])
def get_phases():
    return {"phases": PHASE_GUIDE, "total_phases": len(PHASE_GUIDE)}

@app.get("/city-advisory", tags=["Guides"])
def get_city_advice(city: str = Query(...)):
    if not recommendation: return {"city": city, "advisory": "N/A"}
    return {"city": city, "advisory": recommendation.get_city_advisory(city) or "No specific advisory."}

# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "online",
        "modules": {
            "chatbot": "online" if chatbot else "offline",
            "recommendation": "online" if recommendation else "offline",
            "cost_estimation": "online" if cost_estimation else "offline"
        }
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)