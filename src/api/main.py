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

from ..config import ARTIFACTS_DIR, RESULTS_DIR, NUM_USERS, NUM_ITEMS
from ..serving.pipeline import RecommenderPipeline

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
    rnd = random.Random(item_idx * 31337)
    cat_idx = item_idx % len(CATEGORIES)
    cat_name, cat_tags = CATEGORIES[cat_idx]
    brand = BRANDS[item_idx % len(BRANDS)]
    noun = NOUNS[item_idx % len(NOUNS)]
    title = f"{brand} {noun} (Gen-{item_idx % 4 + 1})"
    price = round(rnd.uniform(24.99, 399.99), 2)
    rating = round(rnd.uniform(4.0, 5.0), 1)
    reviews_count = rnd.randint(45, 9420)
    selected_tags = rnd.sample(cat_tags, min(3, len(cat_tags)))
    asin = f"B0{item_idx % 899999 + 100000:06d}{chr(65 + item_idx % 26)}"

    return {
        "item_idx": item_idx,
        "asin": asin,
        "title": title,
        "brand": brand,
        "category": cat_name,
        "price": price,
        "rating": rating,
        "reviews_count": reviews_count,
        "tags": selected_tags,
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


@app.get("/models")
def models():
    return {
        "models": [
            {"id": "late_fusion", "name": "Late Fusion (ALS + SBERT)", "type": "Hybrid", "desc": "Linear combination with dynamic weight alpha"},
            {"id": "two_stage", "name": "Two-Stage Retrieval", "type": "Deep Hybrid", "desc": "CF + CB Candidate Union (300) -> Reranking"},
            {"id": "als", "name": "Alternating Least Squares (ALS)", "type": "Collaborative", "desc": "Implicit matrix factorization (factors=64)"},
            {"id": "bpr", "name": "Bayesian Personalized Ranking (BPR)", "type": "Collaborative", "desc": "Pairwise ranking optimization"},
            {"id": "sbert", "name": "SBERT Content-Based", "type": "Content", "desc": "all-MiniLM-L6-v2 + FAISS IndexFlatIP"}
        ]
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    cid, uidx, user_status = resolve_customer(req.customer_id, req.user_idx)

    if req.k < 1 or req.k > 100:
        raise HTTPException(status_code=400, detail="k must be 1..100")

    t0 = time.perf_counter()
    cf_override = "bpr" if req.model == "bpr" else "als" if req.model == "als" else None
    pl = get_pipeline(cf_override)
    items = pl.recommend(
        user_idx=uidx,
        user_history=req.history,
        k=req.k,
        model=req.model,
        alpha=req.alpha,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    items_meta = [get_item_metadata(it) for it in items]

    return {
        "customer_id": cid,
        "user_idx": uidx,
        "user_status": user_status,
        "model": req.model,
        "alpha": req.alpha,
        "k": req.k,
        "latency_ms": latency_ms,
        "items": items,
        "items_meta": items_meta,
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
