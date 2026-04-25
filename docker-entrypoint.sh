#!/bin/bash
set -e

echo "╔══════════════════════════════════════╗"
echo "║       CodeLoom — Starting Up         ║"
echo "╚══════════════════════════════════════╝"

# ── Wait for PostgreSQL ──────────────────────────────────────
PG_HOST="${POSTGRES_HOST:-localhost}"
PG_PORT="${POSTGRES_PORT:-5432}"
MAX_RETRIES=30
RETRY=0

echo "⏳ Waiting for PostgreSQL at ${PG_HOST}:${PG_PORT}..."

while ! python3 -c "
import socket, sys
try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(('${PG_HOST}', int('${PG_PORT}')))
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "✗ PostgreSQL not reachable after ${MAX_RETRIES} attempts. Exiting."
        exit 1
    fi
    echo "  Attempt ${RETRY}/${MAX_RETRIES} — retrying in 2s..."
    sleep 2
done

echo "✓ PostgreSQL is ready"

# ── Construct DATABASE_URL safely ────────────────────────────
# We URL-encode the password in case it contains special characters (like @ or :)
if [ -n "$POSTGRES_PASSWORD" ]; then
    SAFE_PASS=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$POSTGRES_PASSWORD")
    export DATABASE_URL="postgresql://${POSTGRES_USER:-postgres}:${SAFE_PASS}@${PG_HOST}:${PG_PORT}/${POSTGRES_DB:-codeloom_dev}"
fi

# ── Run Alembic migrations ───────────────────────────────────
echo "▶ Running database migrations..."
cd /app
alembic upgrade head
echo "✓ Migrations complete"

# ── Start backend ────────────────────────────────────────────
echo "▶ Starting CodeLoom backend on port ${BACKEND_PORT:-9005}..."
exec python3 -m codeloom --host 0.0.0.0 --port "${BACKEND_PORT:-9005}"
