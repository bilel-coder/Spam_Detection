"""
API integration tests using FastAPI TestClient.
Run: pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Test client with a mocked SpamDetector."""
    mock_result = {
        "label": "spam", "label_id": 1,
        "confidence": 0.97, "spam_proba": 0.97, "is_spam": True,
    }
    mock_batch = [
        {"label": "spam", "label_id": 1, "confidence": 0.97, "spam_proba": 0.97, "is_spam": True},
        {"label": "ham",  "label_id": 0, "confidence": 0.90, "spam_proba": 0.10, "is_spam": False},
    ]

    mock_detector = MagicMock()
    mock_detector.predict.return_value       = mock_result
    mock_detector.predict_batch.return_value = mock_batch

    with patch("api.routes.get_detector", return_value=lambda: mock_detector):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_valid(self, client):
        res = client.post("/api/v1/predict", json={"text": "Win a FREE iPhone now!"})
        assert res.status_code == 200
        data = res.json()
        assert "label"      in data
        assert "is_spam"    in data
        assert "confidence" in data
        assert "spam_proba" in data

    def test_predict_empty_text(self, client):
        res = client.post("/api/v1/predict", json={"text": ""})
        assert res.status_code == 422   # Pydantic validation error

    def test_predict_whitespace_only(self, client):
        res = client.post("/api/v1/predict", json={"text": "   "})
        assert res.status_code == 422

    def test_predict_missing_text(self, client):
        res = client.post("/api/v1/predict", json={})
        assert res.status_code == 422

    def test_predict_response_schema(self, client):
        res = client.post("/api/v1/predict", json={"text": "Hello!"})
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data["label"],      str)
        assert isinstance(data["label_id"],   int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_spam"],    bool)


class TestBatchPredictEndpoint:
    def test_batch_valid(self, client):
        res = client.post("/api/v1/predict/batch", json={
            "texts": ["Win FREE prize!", "How are you?"]
        })
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty(self, client):
        res = client.post("/api/v1/predict/batch", json={"texts": []})
        assert res.status_code == 422
