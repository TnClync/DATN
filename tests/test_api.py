import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    assert "late_fusion" in r.json()["models"]


def test_recommend():
    r = client.post("/recommend", json={"user_idx": 5, "model": "cf", "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["items"], list)


def test_recommend_invalid():
    r = client.post("/recommend", json={"user_idx": -1, "k": 5})
    assert r.status_code == 400


def test_items():
    r = client.get("/items/10")
    assert r.status_code == 200
