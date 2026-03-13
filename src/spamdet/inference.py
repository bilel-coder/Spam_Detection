"""
Inference engine — loads the trained pipeline and serves predictions.

Usage (programmatic)
────────────────────
    from spamdet.inference import SpamDetector
    detector = SpamDetector()
    result = detector.predict("Congratulations! You won a FREE prize. Call now!")
    print(result)
    # {'label': 'spam', 'label_id': 1, 'confidence': 0.97, 'is_spam': True}
"""

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from spamdet.config import PIPELINE_PATH
from spamdet.preprocessing import clean_text

logger = logging.getLogger(__name__)


class SpamDetector:
    """Wrapper around the persisted sklearn Pipeline."""

    def __init__(self, model_path: Path = PIPELINE_PATH):
        self._pipeline = _load_pipeline(model_path)
        logger.info("SpamDetector ready (model: %s)", model_path.name)

    # ── Public methods ────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict whether *text* is spam or ham.

        Returns
        ───────
        dict with keys:
            label       : 'spam' | 'ham'
            label_id    : 1 | 0
            confidence  : float in [0, 1]
            is_spam     : bool
        """
        cleaned = clean_text(text)
        label_id = int(self._pipeline.predict([cleaned])[0])

        try:
            proba = self._pipeline.predict_proba([cleaned])[0]
            confidence = float(np.max(proba))
            spam_proba = float(proba[1])
        except AttributeError:
            # LinearSVC / SGD with hinge loss has no predict_proba
            confidence = 1.0
            spam_proba = float(label_id)

        return {
            "label":      "spam" if label_id == 1 else "ham",
            "label_id":   label_id,
            "confidence": round(confidence, 4),
            "spam_proba": round(spam_proba, 4),
            "is_spam":    bool(label_id),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict for a list of messages (more efficient than looping)."""
        cleaned = [clean_text(t) for t in texts]
        label_ids = self._pipeline.predict(cleaned).tolist()

        try:
            probas = self._pipeline.predict_proba(cleaned)
            spam_probas = probas[:, 1].tolist()
            confidences = np.max(probas, axis=1).tolist()
        except AttributeError:
            spam_probas = [float(l) for l in label_ids]
            confidences = [1.0] * len(label_ids)

        return [
            {
                "label":      "spam" if lid == 1 else "ham",
                "label_id":   lid,
                "confidence": round(conf, 4),
                "spam_proba": round(sp, 4),
                "is_spam":    bool(lid),
            }
            for lid, conf, sp in zip(label_ids, confidences, spam_probas)
        ]


# ── Singleton loader (cached) ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python -m spamdet.train` first."
        )
    logger.info("Loading pipeline from %s", model_path)
    return joblib.load(model_path)


def get_detector() -> SpamDetector:
    """FastAPI dependency — returns a singleton SpamDetector."""
    return SpamDetector()
