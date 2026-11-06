from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    assert "baseline" in r.json()["models"]


def test_recommend_invalid_model():
    r = client.post("/recommend", json={"customer_id": "x", "model": "nope"})
    assert r.status_code == 404


def test_recommend_empty_customer():
    r = client.post("/recommend", json={"customer_id": "  ", "model": "baseline"})
    assert r.status_code == 400


def test_benchmark():
    r = client.get("/benchmark")
    assert r.status_code == 200
    assert isinstance(r.json(), list)