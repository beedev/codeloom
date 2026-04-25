# ============================================================
# CodeLoom Dockerfile — Multi-stage build
# Stage 1: Build React/Vite frontend
# Stage 2: Python backend serving API + static frontend
# ============================================================

# ── Stage 1: Frontend build ──────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /build/frontend

# Base path for sub-path deployment (e.g. /codeloom)
# Passed at build time so Vite rewrites asset URLs
ARG APP_BASE_PATH=/codeloom
ENV VITE_BASE_PATH=${APP_BASE_PATH}

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Backend + bundled frontend ──────────────────────
FROM python:3.12-slim AS runtime

# System deps required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY codeloom/        ./codeloom/
COPY alembic/         ./alembic/
COPY alembic.ini      ./
COPY config/          ./config/
COPY tools/           ./tools/
COPY dev.sh           ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Create runtime directories
RUN mkdir -p data/data uploads outputs/images outputs/migrations logs

# Environment defaults (overridden by docker-compose / .env)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 9005

# Entrypoint: wait for Postgres, run migrations, start backend
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
