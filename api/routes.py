"""
FastAPI router — all prediction endpoints live here.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from spamdet.config import API_VERSION, METRICS_PATH, PIPELINE_PATH
from spamdet.inference import SpamDetector, get_detector
from spamdet.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Health check ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health():
    """Returns API status and model info."""
    model_name = PIPELINE_PATH.name if PIPELINE_PATH.exists() else "not loaded"
    return HealthResponse(status="ok", model=model_name, version=API_VERSION)


# ── Single prediction ─────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single message",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict(
    request: PredictRequest,
    detector: SpamDetector = Depends(get_detector),
):
    """
    Classify a single SMS / email message as **spam** or **ham**.

    - **text**: the message to analyse (1 – 5 000 characters)
    """
    try:
        result = detector.predict(request.text)
        logger.info("predict | label=%s | confidence=%.4f", result["label"], result["confidence"])
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


# ── Batch prediction ──────────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify multiple messages",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_batch(
    request: BatchPredictRequest,
    detector: SpamDetector = Depends(get_detector),
):
    """
    Classify up to **100** messages in a single request.

    - **texts**: list of messages to analyse
    """
    try:
        predictions = detector.predict_batch(request.texts)
        return BatchPredictResponse(
            predictions=[PredictResponse(**p) for p in predictions],
            count=len(predictions),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


# ── Model info ────────────────────────────────────────────────────────────────

@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metrics and comparison",
    tags=["Utility"],
)
def model_info():
    """Returns training metrics for the selected model and all candidates."""
    if not Path(METRICS_PATH).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="metrics.json not found. Run training first.",
        )
    data = json.loads(Path(METRICS_PATH).read_text())
    return ModelInfoResponse(**data)
