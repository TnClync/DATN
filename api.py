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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>Hybrid Recommender — DATN</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
            padding: 2rem; text-align: center; }
        .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { color: #667eea; margin-bottom: 1rem; }
        .form-row { display: flex; gap: 1rem; align-items: end; flex-wrap: wrap; }
        .form-group { flex: 1; min-width: 180px; }
        .form-group label { display: block; margin-bottom: 0.3rem; font-weight: 600; color: #555; }
        .form-group input, .form-group select { width: 100%; padding: 0.6rem; border: 1px solid #ddd;
            border-radius: 8px; font-size: 1rem; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
            border: none; padding: 0.7rem 2rem; border-radius: 8px; cursor: pointer;
            font-size: 1rem; font-weight: 600; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.7rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .loading { text-align: center; padding: 2rem; color: #667eea; }
        .error { color: #e74c3c; padding: 1rem; background: #fde8e8; border-radius: 8px; }
        .success { color: #27ae60; padding: 0.5rem; }
        .metric { display: inline-block; background: #f8f9fa; border-radius: 8px; padding: 0.8rem 1.2rem;
            margin: 0.3rem; text-align: center; border-left: 4px solid #667eea; }
        .metric h3 { margin: 0; color: #333; font-size: 1.3rem; }
        .metric p { margin: 0.2rem 0 0; color: #666; font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Hybrid Recommender System</h1>
        <p>Đồ án tốt nghiệp — Amazon Hybrid Recommender | LastOne + LastReview (Kaggle 2×T4)</p>
    </div>
    <div class="container">
        <div class="card">
            <h2>🎯 Gợi ý sản phẩm</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>Customer ID (reviewerID)</label>
                    <input type="text" id="customer_id" placeholder="VD: A1JWL00KO67I05" value="A1JWL00KO67I05">
                </div>
                <div class="form-group">
                    <label>Model</label>
                    <select id="model">
                        <option value="gated_hybrid">Gated Hybrid (Deep)</option>
                        <option value="dcnv2">DCN-v2 (Deep & Cross)</option>
                        <option value="twotower">TwoTower (Deep)</option>
                        <option value="residual_cascade">ResidualCascade (Deep)</option>
                        <option value="late_fusion">Late Fusion (Content)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Top-K</label>
                    <input type="number" id="top_k" value="10" min="1" max="50">
                </div>
                <div class="form-group">
                    <label>Alpha</label>
                    <input type="number" id="alpha" value="0.6" min="0" max="1" step="0.1">
                </div>
                <div>
                    <button class="btn" onclick="recommend()">🚀 Recommend</button>
                </div>
            </div>
            <div id="result" style="margin-top:1rem"></div>
        </div>

        <div class="card">
            <h2>📊 So sánh Model</h2>
            <div style="margin-bottom:1rem">
                <span class="metric"><h3>0.3727</h3><p>MF-bias RMSE</p></span>
                <span class="metric"><h3>0.6560</h3><p>Popularity HR@10</p></span>
                <span class="metric"><h3>0.6515</h3><p>LateFusion HR@10</p></span>
                <span class="metric"><h3>0.2632</h3><p>DCN-v2 HR@10</p></span>
            </div>
            <button class="btn" onclick="loadCompare()">Load Comparison Table</button>
            <div id="compare" style="margin-top:1rem"></div>
        </div>
    </div>

    <script>
        const API = '';
        async function recommend() {
            const cid = document.getElementById('customer_id').value;
            const model = document.getElementById('model').value;
            const top_k = parseInt(document.getElementById('top_k').value);
            const alpha = parseFloat(document.getElementById('alpha').value);
            const el = document.getElementById('result');
            el.innerHTML = '<div class="loading">⏳ Đang tải model & gợi ý (lần đầu có thể mất 1-2 phút)...</div>';
            try {
                const res = await fetch(API + '/recommend', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({customer_id: cid, model: model, top_k: top_k, alpha: alpha})
                });
                const data = await res.json();
                if (res.status !== 200) { el.innerHTML = '<div class="error">' + (data.detail || 'Error') + '</div>'; return; }
                let html = '<div class="success">✅ User ' + data.customer_id + ' → user_idx=' + data.user_idx + ' (' + data.n_interactions + ' interactions) | Model: ' + data.model + '</div>';
                html += '<table><tr><th>Rank</th><th>ItemIdx</th><th>ASIN</th><th>Score</th></tr>';
                data.recommendations.forEach(r => {
                    html += '<tr><td>' + r.rank + '</td><td>' + r.item_idx + '</td><td>' + r.asin + '</td><td>' + r.score + '</td></tr>';
                });
                html += '</table>';
                el.innerHTML = html;
            } catch(e) {
                el.innerHTML = '<div class="error">Error: ' + e.message + '</div>';
            }
        }
        async function loadCompare() {
            const el = document.getElementById('compare');
            el.innerHTML = '<div class="loading">Loading...</div>';
            const res = await fetch(API + '/compare');
            const data = await res.json();
            let html = '<table><tr><th>Model</th><th>Type</th><th>RMSE</th><th>MAE</th><th>HR@10</th><th>NDCG@10</th><th>MRR</th></tr>';
            data.models.forEach(m => {
                html += '<tr><td>' + m.model + '</td><td>' + m.type + '</td><td>' + (m.rmse||'-') + '</td><td>' + (m.mae||'-') + '</td><td>' + (m.hr10||'-') + '</td><td>' + (m.ndcg10||'-') + '</td><td>' + (m.mrr||'-') + '</td></tr>';
            });
            html += '</table>';
            el.innerHTML = html;
        }
    </script>
</body></html>
    """


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
    try:
        model = get_loaded_model(req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    res, err = model.recommend_by_customer(req.customer_id, top_k=req.top_k, alpha=req.alpha)
    if err:
        raise HTTPException(status_code=404, detail=err)
    info = model.get_customer_info(req.customer_id)
    return RecommendResponse(
        customer_id=req.customer_id,
        user_idx=info["user_idx"],
        n_interactions=info["n_interactions"],
        model=req.model,
        recommendations=res,
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
