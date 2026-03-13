"""
Pydantic schemas for request / response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The SMS / email text to classify.",
        examples=["Congratulations! You've won a FREE iPhone. Call now!"],
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of messages to classify (max 100).",
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    label:      str   = Field(..., description="'spam' or 'ham'")
    label_id:   int   = Field(..., description="1 for spam, 0 for ham")
    confidence: float = Field(..., description="Probability of the predicted class")
    spam_proba: float = Field(..., description="Probability that the message is spam")
    is_spam:    bool  = Field(..., description="True if the message is classified as spam")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status:  str = "ok"
    model:   str
    version: str


class ModelInfoResponse(BaseModel):
    best_model:   str
    best_metrics: dict
    all_models:   dict
