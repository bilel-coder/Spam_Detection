"""
Unit tests for the inference module.
Run: pytest tests/test_inference.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# ── Fixtures ──────────────────────────────────────────────────────────────────

SPAM_TEXT = "WINNER!! Claim your FREE prize now! Call 07900 123456 immediately!"
HAM_TEXT  = "Hey, are we still on for lunch tomorrow?"

@pytest.fixture
def mock_pipeline():
    """A mock sklearn pipeline that always predicts spam for the first message."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([1])
    pipeline.predict_proba.return_value = np.array([[0.05, 0.95]])
    return pipeline


@pytest.fixture
def detector(mock_pipeline, tmp_path):
    """SpamDetector with a mock pipeline."""
    from spamdet.inference import SpamDetector
    model_file = tmp_path / "spam_pipeline.joblib"

    with patch("spamdet.inference._load_pipeline", return_value=mock_pipeline):
        detector = SpamDetector.__new__(SpamDetector)
        detector._pipeline = mock_pipeline
        return detector


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSpamDetector:

    def test_predict_returns_required_keys(self, detector):
        result = detector.predict(SPAM_TEXT)
        assert set(result.keys()) == {"label", "label_id", "confidence", "spam_proba", "is_spam"}

    def test_predict_spam_label(self, detector):
        result = detector.predict(SPAM_TEXT)
        assert result["label"]    == "spam"
        assert result["label_id"] == 1
        assert result["is_spam"]  is True

    def test_predict_confidence_range(self, detector):
        result = detector.predict(SPAM_TEXT)
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["spam_proba"] <= 1.0

    def test_predict_ham(self, detector, mock_pipeline):
        mock_pipeline.predict.return_value = np.array([0])
        mock_pipeline.predict_proba.return_value = np.array([[0.92, 0.08]])
        result = detector.predict(HAM_TEXT)
        assert result["label"]    == "ham"
        assert result["label_id"] == 0
        assert result["is_spam"]  is False

    def test_predict_batch_length(self, detector, mock_pipeline):
        mock_pipeline.predict.return_value = np.array([1, 0, 1])
        mock_pipeline.predict_proba.return_value = np.array([
            [0.1, 0.9], [0.85, 0.15], [0.05, 0.95]
        ])
        texts  = [SPAM_TEXT, HAM_TEXT, SPAM_TEXT]
        results = detector.predict_batch(texts)
        assert len(results) == 3

    def test_predict_batch_mixed_labels(self, detector, mock_pipeline):
        mock_pipeline.predict.return_value = np.array([1, 0])
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])
        results = detector.predict_batch([SPAM_TEXT, HAM_TEXT])
        assert results[0]["is_spam"] is True
        assert results[1]["is_spam"] is False


class TestCleanText:

    def test_lowercase(self):
        from spamdet.preprocessing import clean_text
        result = clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_url_replaced(self):
        from spamdet.preprocessing import clean_text
        result = clean_text("Visit https://free-prize.com now!")
        assert "url" in result
        assert "https" not in result

    def test_empty_string(self):
        from spamdet.preprocessing import clean_text
        result = clean_text("")
        assert isinstance(result, str)

    def test_phone_replaced(self):
        from spamdet.preprocessing import clean_text
        result = clean_text("Call 07900 123456 now")
        assert "phone" in result
