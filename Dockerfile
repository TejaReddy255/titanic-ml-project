# ──────────────────────────────────────────────────────────────────────────────
#  Titanic ML API  —  Dockerfile
#  Multi-stage build:
#    builder  → install deps + train model
#    runtime  → lean production image
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL maintainer="you@example.com"
LABEL description="Titanic survival prediction REST API"

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Train the model at build time so the image ships with a ready model
# (Alternatively, mount the model file via a Docker volume at runtime)
RUN python src/train.py


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Re-install only runtime dependencies (no test/dev packages)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir flask gunicorn scikit-learn pandas numpy joblib

# Copy application code + trained model from builder
COPY --from=builder /app/src      ./src
COPY --from=builder /app/api      ./api
COPY --from=builder /app/data     ./data
COPY --from=builder /app/model    ./model

# ── environment ───────────────────────────────────────────────────────────────
ENV PORT=5000
ENV DEBUG=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 5000



# ── start API via gunicorn ────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000"]
