"""
Centralized configuration for the spam detection pipeline.
All paths and hyperparameters are defined here.
"""

import os
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
# En Docker (Render) : fichier est dans /app/src/spamdet/config.py → ROOT = /app
# En local           : fichier est dans .../Spam_Detection/src/spamdet/config.py → ROOT = .../Spam_Detection
_THIS = Path(__file__).resolve()

if _THIS.parts[1] == "app":
    # Docker / Render
    ROOT_DIR = Path("/app")
else:
    # Local
    ROOT_DIR = _THIS.parents[2]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW_DIR        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
RAW_CSV             = DATA_RAW_DIR / "spam.csv"
PROCESSED_CSV       = DATA_PROCESSED_DIR / "spam_clean.csv"

# ── Model artifacts ───────────────────────────────────────────────────────────
MODELS_DIR    = ROOT_DIR / "models" / "artifacts"
METRICS_DIR   = ROOT_DIR / "models" / "metrics"
PIPELINE_PATH = MODELS_DIR / "spam_pipeline.joblib"
METRICS_PATH  = METRICS_DIR / "metrics.json"

# ── Debug (visible dans les logs Render) ──────────────────────────────────────
print(f"DEBUG ROOT_DIR     = {ROOT_DIR}")
print(f"DEBUG PIPELINE_PATH= {PIPELINE_PATH}")
print(f"DEBUG EXISTS       = {PIPELINE_PATH.exists()}")

# ── Feature engineering ───────────────────────────────────────────────────────
TEXT_COLUMN  = "text"
LABEL_COLUMN = "label"          # 0 = ham, 1 = spam

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_SUBLINEAR_TF = True

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── API ───────────────────────────────────────────────────────────────────────
API_TITLE       = "Spam Detection API"
API_DESCRIPTION = "Détecte si un message SMS/email est un spam ou non."
API_VERSION     = "1.0.0"
API_HOST        = "0.0.0.0"
API_PORT        = 8000