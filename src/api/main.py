"""
FastAPI application — entry point.
"""

import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# ── Fix imports ───────────────────────────────────────────────────────────────
# /app/src/api/main.py → parent = /app/src/api → parent.parent = /app/src
SRC_DIR = Path(__file__).resolve().parent.parent
print(f"DEBUG SRC_DIR = {SRC_DIR}")
print(f"DEBUG SRC_DIR exists = {SRC_DIR.exists()}")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print(f"DEBUG sys.path[0] = {sys.path[0]}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from spamdet.config import API_DESCRIPTION, API_TITLE, API_VERSION, PIPELINE_PATH
from api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not PIPELINE_PATH.exists():
        logger.warning("⚠  Model not found at %s.", PIPELINE_PATH)
    else:
        logger.info("✅ Model pipeline found: %s", PIPELINE_PATH)
    yield
    logger.info("API shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# ── Frontend statique ─────────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    logger.info("Frontend served from %s", _FRONTEND_DIR)
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")