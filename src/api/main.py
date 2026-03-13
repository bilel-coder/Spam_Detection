"""
FastAPI application — entry point.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from spamdet.config import API_DESCRIPTION, API_TITLE, API_VERSION, PIPELINE_PATH
from src.api.routes import router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not PIPELINE_PATH.exists():
        logger.warning(
            "⚠  Model not found at %s. "
            "Run `python -m spamdet.train` before starting the API.",
            PIPELINE_PATH,
        )
    else:
        logger.info("✅ Model pipeline found: %s", PIPELINE_PATH)
    yield
    # Shutdown (nothing to clean up)
    logger.info("API shutdown.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in dev (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Static frontend ───────────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    logger.info("Frontend served from %s", _FRONTEND_DIR)

# ── Root redirect (when no static files) ─────────────────────────────────────
else:
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")
