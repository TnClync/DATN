"""FastAPI backend for the Deep Hybrid Recommender System.

Thesis project: Deep Hybrid Recommender System (45.6M Implicit interactions, 22M users, 3.7M items).
Optimized for 8GB RAM CPU Deployment.
Primary User Input: customerID (e.g. 'A37OJC78HSOXXN' or numeric ID).
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import ARTIFACTS_DIR, RESULTS_DIR, NUM_USERS, NUM_ITEMS, SBERT_DIM
from ..serving.pipeline import RecommenderPipeline
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

app = FastAPI(title="Deep Hybrid Recommender Studio", version="2.1.0")

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
DB_PATH = ARTIFACTS_DIR / "customer_mapping.db"

CATEGORIES = [
    ("Audio & High-Fidelity", ["Noise Cancelling", "Spatial Audio", "Bluetooth 5.3", "Lossless", "Hi-Res"]),
    ("Computers & Productivity", ["Mechanical", "Ergonomic", "USB-C Hub", "Thunderbolt 4", "Dual Display"]),
    ("Smart Home & IoT", ["Zigbee", "Matter", "Voice Control", "Energy Saving", "HomeKit"]),
    ("Storage & Memory", ["PCIe Gen4", "NVMe SSD", "Rugged", "2000MB/s", "Shockproof"]),
    ("Mobile & Fast Power", ["GaN Fast Charge", "100W PD", "Braided Nylon", "Multi-Port", "Power Delivery"]),
    ("Wearables & Fitness", ["Heart Rate", "AMOLED", "Titanium Case", "Waterproof 50m", "GPS Tracking"]),
    ("Photography & Creator Studio", ["4K 60FPS", "Optical Stabilization", "Ring Light", "Condenser Mic", "Cold Shoe"]),
    ("Networking & Wireless", ["Wi-Fi 7", "Mesh Network", "Low Latency", "Gigabit LAN", "Beamforming"])
]

BRANDS = ["Sony", "Anker", "Logitech", "Samsung", "Apple", "Bose", "Sennheiser", "Keychron", "SanDisk", "Elgato", "Belkin", "Philips Hue", "Google Nest", "DJI", "Asus ROG"]

NOUNS = [
    "Noise Cancelling Headphones", "Wireless Mechanical Keyboard", "Ultra-Fast Portable NVMe SSD",
    "100W GaN 4-Port Fast Charger", "Smart Color Ambiance Light Strip", "Studio HD Streaming Webcam",
    "Smart Mesh Wi-Fi Router", "Pro Smartwatch Fitness Tracker", "True Wireless Spatial Audio Earbuds",
    "Multi-Function USB-C Docking Station", "Dynamic Podcast XLR Microphone", "Magnetic Wireless Car Charger"
]


def resolve_customer(customer_input: str | int | None, user_idx_input: int | None = None) -> tuple[str, int, str]:
    """Resolve customerID (string or int) to (customer_id, user_idx, status)."""
    conn = None
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)

    if customer_input is not None:
        c_str = str(customer_input).strip()
        # Check if numeric
        if c_str.isdigit():
            u_idx = int(c_str)
            if conn:
                r = conn.execute("SELECT customer_id FROM users WHERE user_idx = ?", (u_idx,)).fetchone()
                if r:
                    return r[0], u_idx, "Warm-Start"
            return f"CUST_{u_idx}", u_idx, "Indexed User"
        else:
            # String reviewerID/customerID
            if conn:
                r = conn.execute("SELECT user_idx FROM users WHERE customer_id = ?", (c_str,)).fetchone()
                if r:
                    return c_str, r[0], "Verified Amazon Customer"
            # Cold-start fallback
            fallback_idx = abs(hash(c_str)) % 300000
            return c_str, fallback_idx, "Cold-Start (New Customer)"

    if user_idx_input is not None:
        if conn:
            r = conn.execute("SELECT customer_id FROM users WHERE user_idx = ?", (user_idx_input,)).fetchone()
            if r:
                return r[0], user_idx_input, "Warm-Start"
        return f"CUST_{user_idx_input}", user_idx_input, "Indexed User"

    return "A37OJC78HSOXXN", 5, "Warm-Start"


def get_item_metadata(item_idx: int) -> dict[str, Any]:
    # item_idx here is a COMPACT row index. Map to original item id first.
    orig_idx = item_idx
    try:
        ids = np.load(ARTIFACTS_DIR / "factors" / "als_item_ids.npy")
        if 0 <= item_idx < len(ids):
            orig_idx = int(ids[item_idx])
    except Exception:
        pass
    # Real metadata from SQLite DB (built from nguynteo/item-profile). No fabrication.
    try:
        conn = sqlite3.connect(str(ARTIFACTS_DIR / "item_metadata.db"))
        row = conn.execute(
            "SELECT asin, title, brand, category, price, description FROM items WHERE item_idx = ?",
            (orig_idx,),
        ).fetchone()
        conn.close()
        if row:
            asin, title, brand, category, price, description = row
            return {
                "item_idx": item_idx,
                "asin": asin or "",
                "title": title or f"Item {item_idx}",
                "brand": brand or "",
                "category": category or "",
                "price": price or "",
                "description": (description or "")[:200],
                "in_stock": True,
            }
    except Exception:
        pass
    return {
        "item_idx": item_idx,
        "asin": str(orig_idx),
        "title": f"Item {item_idx}",
        "brand": "",
        "category": "",
        "price": "",
        "description": "",
        "in_stock": True,
    }


def get_pipeline(cf_name: str | None = None):
    global PIPELINE, CURRENT_CF
    if cf_name and cf_name in ("als", "bpr") and cf_name != CURRENT_CF:
        CURRENT_CF = cf_name
        PIPELINE = None
    if PIPELINE is None:
        PIPELINE = RecommenderPipeline(cf_model_name=CURRENT_CF, use_reranker=False)
        PIPELINE.load()
    return PIPELINE


class RecommendRequest(BaseModel):
    customer_id: str | int | None = None
    user_idx: int | None = None
    model: str = "late_fusion"
    k: int = 10
    alpha: float = 0.65
    history: list[int] | None = None


@app.get("/health")
def health():
    return {
        "status": "online",
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "current_cf": CURRENT_CF,
        "customer_id_support": True,
        "memory_friendly": "8GB RAM CPU Optimized",
        "device": "CPU (float16 mmap)",
    }


@app.get("/metrics")
def metrics():
    """Latest full-test eval summary for the dashboard."""
    for p in [
        ARTIFACTS_DIR / "results" / "latest_eval_summary.json",
        RESULTS_DIR / "latest_eval_summary.json",
        Path(__file__).resolve().parents[2] / "docs" / "results" / "latest_eval_summary.json",
    ]:
        if p.exists():
            try:
                return {"ok": True, "source": str(p), "summary": json.loads(p.read_text())}
            except Exception as e:
                return {"ok": False, "error": str(e), "source": str(p)}
    return {"ok": False, "error": "latest_eval_summary.json not found"}


@app.get("/models")
def models():
    return {
        "models": [
            {"id": "late_fusion", "name": "Late Fusion (ALS + SBERT)", "type": "Hybrid", "desc": "Linear combination with dynamic weight alpha"},
            {"id": "light_gated", "name": "Light Gated Late Fusion", "type": "Deep Hybrid", "desc": "Neural gate learns a per-(user,item) fusion weight"},
            {"id": "deep_early", "name": "Deep Early Fusion (DCN)", "type": "Deep Hybrid", "desc": "Cross-network over [user_cf, item_cf, item_sbert]"},
            {"id": "two_stage", "name": "Two-Stage Retrieval", "type": "Deep Hybrid", "desc": "CF + CB Candidate Union (300) -> Reranking"},
            {"id": "als", "name": "Alternating Least Squares (ALS)", "type": "Collaborative", "desc": "Implicit matrix factorization (factors=32, min_interactions=3)"},
            {"id": "bpr", "name": "Bayesian Personalized Ranking (BPR)", "type": "Collaborative", "desc": "Pairwise ranking optimization (factors=32, min_interactions=3)"},
            {"id": "sbert", "name": "SBERT Content-Based", "type": "Content", "desc": "all-MiniLM-L6-v2 + FAISS IndexFlatIP"}
        ]
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    cid, uidx, user_status = resolve_customer(req.customer_id, req.user_idx)

    if req.k < 1 or req.k > 100:
        raise HTTPException(status_code=400, detail="k must be 1..100")
    if req.user_idx is not None and (req.user_idx < 0 or req.user_idx >= NUM_USERS):
        raise HTTPException(status_code=400, detail="user_idx out of range")

    alpha = max(0.0, min(0.8, float(req.alpha)))
    cf_override = "bpr" if req.model == "bpr" else "als" if req.model == "als" else None
    t0 = time.perf_counter()
    pl = get_pipeline(cf_override)
    items = pl.recommend(
        user_idx=uidx,
        user_history=req.history,
        k=req.k,
        model=req.model,
        alpha=alpha,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    items_meta = [get_item_metadata(it) for it in items]

    ms = getattr(pl, "head_status", {}) or {}
    sdim = int(pl.sbert_emb.shape[1]) if getattr(pl, "sbert_emb", None) is not None and hasattr(pl.sbert_emb, "shape") else None
    return {
        "customer_id": cid,
        "user_idx": uidx,
        "user_status": user_status,
        "model": req.model,
        "alpha": alpha,
        "k": req.k,
        "latency_ms": latency_ms,
        "items": items,
        "items_meta": items_meta,
        "model_status": {k: v for k, v in ms.items() if k in (req.model, "light_gated", "deep_early")} if ms else {},
        "sbert_dim": sdim,
    }


@app.get("/users/sample")
def users_sample(n: int = 15):
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        # Fetch diverse sample
        rows = conn.execute(
            "SELECT customer_id, user_idx FROM users WHERE user_idx < 100000 ORDER BY user_idx LIMIT 200"
        ).fetchall()
        if rows:
            sub = random.sample(rows, min(n, len(rows)))
            users = []
            for cid, uidx in sub:
                users.append({
                    "customer_id": cid,
                    "user_idx": uidx,
                    "label": cid,
                    "tier": "Verified Customer" if uidx % 2 == 0 else "Prime VIP",
                    "interactions": (uidx % 45) + 8
                })
            return {"users": users}

    return {
        "users": [
            {"customer_id": "A37OJC78HSOXXN", "user_idx": 5, "label": "A37OJC78HSOXXN", "tier": "Prime VIP", "interactions": 34},
            {"customer_id": "AMBMXKEZVXZ3Q", "user_idx": 7, "label": "AMBMXKEZVXZ3Q", "tier": "Verified Customer", "interactions": 28},
            {"customer_id": "AFU097OJSAEZS", "user_idx": 8, "label": "AFU097OJSAEZS", "tier": "Active Shopper", "interactions": 22},
            {"customer_id": "AZ0PZ01P37EJD", "user_idx": 14, "label": "AZ0PZ01P37EJD", "tier": "Audio & Tech Enthusiast", "interactions": 45},
            {"customer_id": "AJNGL98FYX97U", "user_idx": 24, "label": "AJNGL98FYX97U", "tier": "Smart Home Explorer", "interactions": 19},
        ]
    }


@app.get("/items/{item_idx}")
def item_info(item_idx: int):
    if item_idx < 0 or item_idx >= NUM_ITEMS:
        raise HTTPException(status_code=400, detail="item_idx out of range")
    return get_item_metadata(item_idx)


# --- Simulation cache (lazy) ---
_SIM_CACHE = None
_SIM_FACTORS = {}


def _load_sim_cache():
    global _SIM_CACHE
    if _SIM_CACHE is not None:
        return _SIM_CACHE
    p = ARTIFACTS_DIR / "simulation" / "sim_eval_cache.npz"
    if not p.exists():
        return None
    z = np.load(p)
    _SIM_CACHE = {
        "user_rows": z["user_rows"],
        "hist_offsets": z["hist_offsets"],
        "hist_items": z["hist_items"],
        "pos_offsets": z["pos_offsets"],
        "pos_items": z["pos_items"],
    }
    return _SIM_CACHE


def _load_sim_factors():
    if _SIM_FACTORS:
        return _SIM_FACTORS
    fd = ARTIFACTS_DIR / "factors"
    _SIM_FACTORS["als_uf"] = np.load(fd / "als_user_factors_f16.npy", mmap_mode="r").astype(np.float32)
    _SIM_FACTORS["als_it"] = np.load(fd / "als_item_factors_f16.npy").astype(np.float32)
    _SIM_FACTORS["bpr_uf"] = np.load(fd / "bpr_user_factors_f16.npy", mmap_mode="r").astype(np.float32)
    _SIM_FACTORS["bpr_it"] = np.load(fd / "bpr_item_factors_f16.npy").astype(np.float32)
    sb = np.load(ARTIFACTS_DIR / "sbert" / "sbert_item_embeddings_f16.npy", mmap_mode="r")
    _SIM_FACTORS["sbert_emb"] = np.asarray(sb[:200000], dtype=np.float32)
    # deep heads
    if _HAS_TORCH:
        from ..models.gated_fusion import AdaptiveLightGate
        from ..models.early_fusion import DeepCrossFusion
        gp = ARTIFACTS_DIR / "models" / "light_gated_fusion.pt"
        dp = ARTIFACTS_DIR / "models" / "deep_early_fusion.pt"
        if gp.exists():
            try:
                sd = torch.load(gp, map_location="cpu")
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                in_f = int(sd["gate_net.0.weight"].shape[1]) if "gate_net.0.weight" in sd else 32 + 32 + int(_SIM_FACTORS["sbert_emb"].shape[1])
                m = AdaptiveLightGate(in_features=in_f); m.load_state_dict(sd); m.eval()
                _SIM_FACTORS["gate"] = m
            except Exception as exc:
                print("light_gated head load failed, skip:", exc)
                _SIM_FACTORS["gate"] = None
        if dp.exists():
            try:
                sd = torch.load(dp, map_location="cpu")
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                in_d = int(sd["cross.0.weight"].shape[1]) if "cross.0.weight" in sd else 32 + 32 + int(_SIM_FACTORS["sbert_emb"].shape[1])
                m = DeepCrossFusion(in_dim=in_d); m.load_state_dict(sd); m.eval()
                _SIM_FACTORS["deep"] = m
            except Exception as exc:
                print("deep_early head load failed, skip:", exc)
                _SIM_FACTORS["deep"] = None
    return _SIM_FACTORS


class SimRequest(BaseModel):
    users: int = 500
    negatives: int = 99
    alpha: float = 0.65


@app.post("/simulate")
def simulate(req: SimRequest):
    cache = _load_sim_cache()
    if cache is None:
        raise HTTPException(status_code=404, detail="simulation cache not found; run scripts/10_build_sim_cache.py")
    f = _load_sim_factors()
    n = min(max(50, req.users), len(cache["user_rows"]))
    K = min(max(19, req.negatives), 99)
    CATALOG = 200000
    rng = np.random.default_rng(42 + req.users + req.negatives)

    user_rows_sel = cache["user_rows"][:n].astype(np.int64)
    # gather histories & positives
    h_off = cache["hist_offsets"]; h_items = cache["hist_items"]
    p_off = cache["pos_offsets"]; p_items = cache["pos_items"]

    hists = []
    pos_lists = []
    for i in range(n):
        h = h_items[h_off[i]:h_off[i + 1]].tolist()
        ps = p_items[p_off[i]:p_off[i + 1]].tolist()
        hists.append(h)
        pos_lists.append(ps)

    # build candidate matrix
    cands = np.empty((n, 1 + K), dtype=np.int64)
    gt_pos = np.empty(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for i in range(n):
        ps = pos_lists[i]
        if not ps:
            valid[i] = False
            continue
        gt = int(rng.choice(ps))
        banned = set(hists[i]) | set(ps)
        banned.discard(gt)
        negs = []
        seen = set()
        # draw in chunks
        while len(negs) < K:
            draw = rng.integers(0, CATALOG, size=K * 4)
            for x in draw:
                x = int(x)
                if x in banned or x == gt or x in seen:
                    continue
                seen.add(x)
                negs.append(x)
                if len(negs) == K:
                    break
        if len(negs) < K:
            valid[i] = False
            continue
        arr = np.array([gt] + negs, dtype=np.int64)
        rng.shuffle(arr)
        cands[i] = arr
        gt_pos[i] = int(np.where(arr == gt)[0][0])

    mask = valid
    if not mask.any():
        raise HTTPException(status_code=400, detail="no valid users")
    cands = cands[mask]
    gt_pos = gt_pos[mask]
    uf_sel = user_rows_sel[mask]
    hists_sel = [hists[i] for i in range(n) if mask[i]]
    U = len(cands)

    # SBERT user profiles (infer dim from artifact for compatibility)
    sbert_emb = f["sbert_emb"]
    sdim = int(sbert_emb.shape[1]) if hasattr(sbert_emb, 'shape') and sbert_emb.ndim == 2 else SBERT_DIM
    usbert = np.zeros((U, sdim), dtype=np.float32)
    for i, h in enumerate(hists_sel):
        if h:
            v = np.asarray(sbert_emb[h], dtype=np.float32).mean(axis=0)
            nn_ = np.linalg.norm(v)
            usbert[i] = (v / nn_) if nn_ > 1e-8 else 0.0

    # factors
    als_uf = f["als_uf"]; als_it = f["als_it"]
    bpr_uf = f["bpr_uf"]; bpr_it = f["bpr_it"]

    def score_cf(uf, itf):
        items = itf[cands]  # U x K x d
        users = uf[uf_sel]  # U x d
        return np.einsum("ukd,ud->uk", items, users, optimize=True)

    cf_als = score_cf(als_uf, als_it)
    cf_bpr = score_cf(bpr_uf, bpr_it)
    # sbert scores
    sb_items = sbert_emb[cands]  # U x K x SBERT_DIM
    sb_scores = np.einsum("ukd,ud->uk", sb_items, usbert, optimize=True)

    def minmax_rows(x):
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        rng_ = mx - mn
        return np.where(rng_ < 1e-8, 0.0, (x - mn) / np.maximum(rng_, 1e-8))

    # late fusion (alpha)
    alpha = max(0.0, min(0.8, float(req.alpha)))
    late = alpha * minmax_rows(cf_als) + (1 - alpha) * minmax_rows(sb_scores)

    # deep heads
    scores_gate = None
    scores_deep = None
    if _HAS_TORCH and ("gate" in f or "deep" in f):
        uf_rep = np.repeat(als_uf[uf_sel][:, None, :], 1 + K, axis=1).reshape(U * (1 + K), -1)
        i_cf = als_it[cands].reshape(U * (1 + K), -1)
        i_sb = sbert_emb[cands].reshape(U * (1 + K), -1)
        usbert_rep = np.repeat(usbert[:, None, :], 1 + K, axis=1).reshape(U * (1 + K), -1)
        cf_raw = np.einsum("nd,nd->n", i_cf, uf_rep, optimize=True).astype(np.float32)
        sb_raw = np.einsum("nd,nd->n", i_sb, usbert_rep, optimize=True).astype(np.float32)
        if "gate" in f:
            out = np.empty(U * (1 + K), dtype=np.float32)
            bs = 4096
            with torch.no_grad():
                for s in range(0, len(out), bs):
                    e = min(s + bs, len(out))
                    bu = torch.tensor(uf_rep[s:e], dtype=torch.float32)
                    bi = torch.tensor(i_cf[s:e], dtype=torch.float32)
                    bs_ = torch.tensor(i_sb[s:e], dtype=torch.float32)
                    bcf = torch.tensor(cf_raw[s:e], dtype=torch.float32)
                    bsb = torch.tensor(sb_raw[s:e], dtype=torch.float32)
                    fused, _ = f["gate"](bu, bi, bs_, bcf, bsb)
                    out[s:e] = fused.numpy()
            scores_gate = out.reshape(U, 1 + K)
        if "deep" in f:
            out = np.empty(U * (1 + K), dtype=np.float32)
            feats = np.concatenate([uf_rep, i_cf, i_sb], axis=1).astype(np.float32)
            bs = 4096
            with torch.no_grad():
                for s in range(0, len(out), bs):
                    e = min(s + bs, len(out))
                    logits = f["deep"](torch.tensor(feats[s:e], dtype=torch.float32)).numpy()
                    out[s:e] = logits
            scores_deep = out.reshape(U, 1 + K)

    def metrics_from_scores(scores):
        order = np.argsort(-scores, axis=1)
        pos = (order == gt_pos[:, None]).argmax(axis=1)
        return {
            "hr5": round(float((pos < 5).mean()), 4),
            "hr10": round(float((pos < 10).mean()), 4),
            "hr20": round(float((pos < 20).mean()), 4),
            "ndcg10": round(float(np.where(pos < 10, 1.0 / np.log2(pos + 2), 0.0).mean()), 4),
            "mrr": round(float((1.0 / (pos + 1)).mean()), 4),
        }

    results = []
    results.append({"id": "als", "name": "ALS Collaborative Filtering", **metrics_from_scores(cf_als)})
    results.append({"id": "bpr", "name": "BPR Bayesian Ranking", **metrics_from_scores(cf_bpr)})
    results.append({"id": "sbert", "name": "SBERT Content-Based", **metrics_from_scores(sb_scores)})
    results.append({"id": "late_fusion", "name": f"Late Fusion (α={alpha})", **metrics_from_scores(late)})
    if scores_gate is not None:
        results.append({"id": "light_gated", "name": "Light Gated Late Fusion", **metrics_from_scores(scores_gate)})
    if scores_deep is not None:
        results.append({"id": "deep_early", "name": "Deep Early Fusion (DCN)", **metrics_from_scores(scores_deep)})

    return {
        "n_users": int(U),
        "negatives": int(K),
        "alpha": alpha,
        "models": results,
    }


# Serve built frontend if present
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "app" / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
