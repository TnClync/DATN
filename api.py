"""FastAPI backend — Hybrid Recommender System (Đồ án tốt nghiệp).

Primary web interface. No Streamlit dependency. CPU inference.

Run: uvicorn api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
Web UI: http://localhost:8000
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from src.models.registry import get_model, list_models

BASE = Path(__file__).resolve().parent

app = FastAPI(
    title="Hybrid Recommender API",
    description="Đồ án tốt nghiệp — Amazon Hybrid Recommender (LastOne + LastReview)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend assets
_DIST = BASE / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time"] = f"{(time.perf_counter() - start) * 1000:.1f}ms"
    return response

# ── Lazy model loading (load on first request) ───────────────────────────────
_models = {}


def get_loaded_model(name: str):
    if name not in _models:
        if name not in list_models():
            raise ValueError(f"Unknown model: {name}")
        m = get_model(name, BASE, alpha=0.6)
        m.load()
        _models[name] = m
    return _models[name]


# ── Schemas ──────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    customer_id: str
    model: str = "late_fusion"
    top_k: int = 10
    alpha: float = 0.6


class RecommendResponse(BaseModel):
    customer_id: str
    user_idx: int
    n_interactions: int
    model: str
    recommendations: list[dict]


class HealthResponse(BaseModel):
    status: str
    models_available: list[str]
    models_loaded: list[str]
    device: str


# ── Template (studio.tsx) static data exposed via FastAPI ─────────────────────
BENCHMARK_METRICS = [
    {"model": "ALS", "split": "Validation", "hr5": 0.245, "hr10": 0.382, "hr20": 0.491, "ndcg10": 0.276, "precision10": 0.0382, "recall10": 0.382, "mrr": 0.165, "timeSec": 142},
    {"model": "BPR", "split": "Validation", "hr5": 0.212, "hr10": 0.341, "hr20": 0.458, "ndcg10": 0.239, "precision10": 0.0341, "recall10": 0.341, "mrr": 0.142, "timeSec": 118},
    {"model": "LMF", "split": "Validation", "hr5": 0.198, "hr10": 0.315, "hr20": 0.422, "ndcg10": 0.218, "precision10": 0.0315, "recall10": 0.315, "mrr": 0.131, "timeSec": 95},
    {"model": "BM25", "split": "Validation", "hr5": 0.164, "hr10": 0.278, "hr20": 0.389, "ndcg10": 0.194, "precision10": 0.0278, "recall10": 0.278, "mrr": 0.115, "timeSec": 34},
    {"model": "Cosine", "split": "Validation", "hr5": 0.142, "hr10": 0.241, "hr20": 0.352, "ndcg10": 0.168, "precision10": 0.0241, "recall10": 0.241, "mrr": 0.098, "timeSec": 18},
    {"model": "TF-IDF", "split": "Validation", "hr5": 0.178, "hr10": 0.294, "hr20": 0.405, "ndcg10": 0.205, "precision10": 0.0294, "recall10": 0.294, "mrr": 0.122, "timeSec": 22},
    {"model": "ALS", "split": "Test", "hr5": 0.239, "hr10": 0.375, "hr20": 0.484, "ndcg10": 0.269, "precision10": 0.0375, "recall10": 0.375, "mrr": 0.159, "timeSec": 142},
    {"model": "BPR", "split": "Test", "hr5": 0.205, "hr10": 0.332, "hr20": 0.449, "ndcg10": 0.231, "precision10": 0.0332, "recall10": 0.332, "mrr": 0.138, "timeSec": 118},
]

SAMPLE_USERS = [
    {"id": 42, "name": "Tech Enthusiast & Home Audio Fan", "status": "Warm-Start", "interactionsCount": 48, "avgRating": 4.6},
    {"id": 108, "name": "Smart Home & IoT Explorer", "status": "Warm-Start", "interactionsCount": 31, "avgRating": 4.3},
    {"id": 256, "name": "New Shopper (Minimal History)", "status": "Semi Cold-Start", "interactionsCount": 3, "avgRating": 4.0},
]

ITEM_CATALOG = [
    {"id": "ITEM_8801", "name": "Bose QuietComfort Ultra Earbuds", "category": "Electronics", "cfScore": 0.942, "cbScore": 0.885},
    {"id": "ITEM_4102", "name": "Anker USB-C 100W Fast Charging Cable", "category": "Accessories", "cfScore": 0.891, "cbScore": 0.920},
    {"id": "ITEM_9921", "name": "Keychron Q1 Pro Wireless Mechanical Keyboard", "category": "Computers", "cfScore": 0.875, "cbScore": 0.760},
]

# ── Comparison data ──────────────────────────────────────────────────────────
COMPARISON = [
    {"model": "MF-bias", "type": "CF", "rmse": 0.3727, "mae": 0.2869, "hr10": 0.6470, "ndcg10": 0.4368, "mrr": 0.3705},
    {"model": "ALS (PyTorch)", "type": "CF", "rmse": 3.8359, "mae": 3.8165, "hr10": 0.9725, "ndcg10": 0.7710, "mrr": 0.7062},
    {"model": "BPR-MF", "type": "CF", "rmse": None, "mae": None, "hr10": 0.0184, "ndcg10": 0.0104, "mrr": None},
    {"model": "SBERT Content", "type": "CB", "rmse": None, "mae": None, "hr10": 0.0, "ndcg10": 0.0, "mrr": 0.0},
    {"model": "Popularity", "type": "Baseline", "rmse": None, "mae": None, "hr10": 0.6560, "ndcg10": 0.4680, "mrr": 0.4094},
    {"model": "LateFusion (a=0.5)", "type": "Hybrid", "rmse": None, "mae": None, "hr10": 0.6515, "ndcg10": 0.4324, "mrr": 0.3633},
    {"model": "LateFusion (a=0.7)", "type": "Hybrid", "rmse": None, "mae": None, "hr10": 0.6515, "ndcg10": 0.4321, "mrr": 0.3631},
    {"model": "EarlyFusion", "type": "Hybrid", "rmse": None, "mae": None, "hr10": 0.6470, "ndcg10": 0.4346, "mrr": 0.3678},
    {"model": "Gated Hybrid", "type": "Hybrid (Deep)", "rmse": None, "mae": None, "hr10": None, "ndcg10": None, "mrr": None},
    {"model": "DCN-v2", "type": "Hybrid (Deep)", "rmse": 0.4445, "mae": 0.2536, "hr10": 0.2632, "ndcg10": 0.1193, "mrr": 0.1018},
    {"model": "TwoTower", "type": "Hybrid (Deep)", "rmse": 0.4385, "mae": 0.2491, "hr10": 0.1974, "ndcg10": 0.0936, "mrr": 0.0902},
    {"model": "ResidualCascade", "type": "Hybrid (Deep)", "rmse": None, "mae": None, "hr10": 0.1632, "ndcg10": 0.0863, "mrr": 0.0893},
    {"model": "AutoInt", "type": "Hybrid (Deep)", "rmse": 0.4629, "mae": 0.2455, "hr10": 0.2289, "ndcg10": 0.1087, "mrr": 0.1005},
    {"model": "Deep Fusion MLP", "type": "Hybrid (Deep)", "rmse": None, "mae": None, "hr10": None, "ndcg10": None, "mrr": None},
]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    index = BASE / "frontend" / "dist" / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "UI not built. Run: cd frontend && npm run build"}


@app.get("/health", response_model=HealthResponse)
def health():
    import torch
    return HealthResponse(
        status="ok",
        models_available=list_models(),
        models_loaded=list(_models.keys()),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.get("/models")
def models():
    return {"models": list_models(), "loaded": list(_models.keys())}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.model not in list_models():
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found. Available: {list_models()}")
    if not req.customer_id.strip():
        raise HTTPException(status_code=400, detail="customer_id is required")
    if req.top_k < 1 or req.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    if req.alpha < 0 or req.alpha > 1:
        raise HTTPException(status_code=400, detail="alpha must be between 0 and 1")
    try:
        model = get_loaded_model(req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    try:
        res, err = model.recommend_by_customer(req.customer_id, top_k=req.top_k, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")
    if err:
        raise HTTPException(status_code=404, detail=err)
    info = model.get_customer_info(req.customer_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Customer '{req.customer_id}' not found")
    return RecommendResponse(
        customer_id=req.customer_id,
        user_idx=info["user_idx"],
        n_interactions=info["n_interactions"],
        model=req.model,
        recommendations=res or [],
    )


@app.get("/compare")
def compare():
    return {"models": COMPARISON}


@app.get("/customer/{customer_id}")
def customer_info(customer_id: str):
    for name in list_models():
        try:
            model = get_loaded_model(name)
            info = model.get_customer_info(customer_id)
            if info:
                return info
        except Exception:
            continue
    raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

# ── Template (studio.tsx) data endpoints ──────────────────────────────────────
@app.get("/benchmark")
def benchmark():
    return BENCHMARK_METRICS


@app.get("/users")
def users():
    return SAMPLE_USERS


@app.get("/catalog")
def catalog():
    return ITEM_CATALOG


@app.get("/template")
def template():
    """Return the TSX template source for reference."""
    p = BASE / "demo" / "studio.tsx"
    if not p.exists():
        raise HTTPException(status_code=404, detail="template not found")
    return {"path": str(p), "size": p.stat().st_size}


@app.get("/metrics")
def metrics():
    return {
        "models_available": list_models(),
        "models_loaded": list(_models.keys()),
        "device": "cpu",
    }
