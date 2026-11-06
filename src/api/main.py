"""FastAPI backend for the Deep Hybrid Recommender System.

Endpoints:
  GET  /health
  GET  /models                 -> available model names
  POST /recommend              -> recommendations for a user
  GET  /users/sample           -> sample user ids
  GET  /items/{item_idx}       -> item info
  GET  /metrics                -> latest eval summary (if present)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import ARTIFACTS_DIR, RESULTS_DIR, NUM_USERS, NUM_ITEMS
from ..serving.pipeline import RecommenderPipeline

app = FastAPI(title="Deep Hybrid Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (lazy)
PIPELINE = None
CURRENT_CF = "als"


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = RecommenderPipeline(cf_model_name=CURRENT_CF, use_reranker=False)
        PIPELINE.load()
    return PIPELINE


class RecommendRequest(BaseModel):
    user_idx: int
    model: str = "late_fusion"
    k: int = 10
    alpha: float = 0.6
    history: list[int] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "num_users": NUM_USERS, "num_items": NUM_ITEMS}


@app.get("/models")
def models():
    return {"models": ["cf", "sbert", "late_fusion", "two_stage"]}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_idx < 0 or req.user_idx >= NUM_USERS:
        raise HTTPException(status_code=400, detail="user_idx out of range")
    if req.k < 1 or req.k > 100:
        raise HTTPException(status_code=400, detail="k must be 1..100")
    pl = get_pipeline()
    items = pl.recommend(
        user_idx=req.user_idx,
        user_history=req.history,
        k=req.k,
        model=req.model,
        alpha=req.alpha,
    )
    return {"user_idx": req.user_idx, "model": req.model, "items": items}


@app.get("/users/sample")
def users_sample(n: int = 20):
    import random
    random.seed(42)
    return {"users": [random.randrange(0, min(NUM_USERS, 100000)) for _ in range(n)]}


@app.get("/items/{item_idx}")
def item_info(item_idx: int):
    if item_idx < 0 or item_idx >= NUM_ITEMS:
        raise HTTPException(status_code=400, detail="item_idx out of range")
    return {"item_idx": item_idx, "asin": f"item_{item_idx}"}


@app.get("/metrics")
def metrics():
    p = RESULTS_DIR / "latest_eval_summary.json"
    if not p.exists():
        return {"metrics": None, "message": "no eval results yet"}
    return {"metrics": json.loads(p.read_text())}


# Serve built frontend if present
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "app" / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
