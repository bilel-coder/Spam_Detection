# ╔══════════════════════════════════════════════════════════╗
# ║         Spam Detection API — Production Dockerfile       ║
# ╚══════════════════════════════════════════════════════════╝

# ── Stage 1 : builder (install deps) ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install OS build tools (needed for some Python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix folder
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir \
        $(grep -v "^python" requirements.txt | grep -v "^#" | grep -v "^$" | grep -v "jupyter" | grep -v "matplotlib" | grep -v "seaborn")


# ── Stage 2 : runtime (lean image) ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="spam-detection-project"
LABEL description="Spam Detection ML API"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY src/          ./src/
COPY models/       ./models/
COPY frontend/     ./frontend/
COPY data/         ./data/

# Ensure models dir exists (in case it's empty at build time)
RUN mkdir -p models/artifacts models/metrics

# Download NLTK data at build time (avoids runtime latency)
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet',   quiet=True); \
nltk.download('omw-1.4',   quiet=True)"

# Set PYTHONPATH so `src/` packages are importable
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
