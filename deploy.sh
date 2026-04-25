#!/bin/bash
# ============================================================
# CodeLoom — Server Deployment Script (Ubuntu 24 LTS)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh            # Build + start
#   ./deploy.sh build      # Build only
#   ./deploy.sh up         # Start container (no rebuild)
#   ./deploy.sh down       # Stop container
#   ./deploy.sh logs       # Follow logs
#   ./deploy.sh restart    # Rebuild + restart
#   ./deploy.sh status     # Show status
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()  { echo -e "${BLUE}▶${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

# ── Pre-flight checks ───────────────────────────────────────
preflight() {
    if ! command -v docker &>/dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    if ! docker info &>/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    if [ ! -f ".env" ]; then
        print_error ".env file not found. Copy .env.example to .env and configure it first."
        echo "  cp .env.example .env"
        exit 1
    fi
    print_success "Pre-flight checks passed"
}

# ── Build ────────────────────────────────────────────────────
do_build() {
    print_status "Building CodeLoom Docker image..."
    docker compose build --no-cache
    print_success "Image built successfully"
}

# ── Start ────────────────────────────────────────────────────
do_up() {
    print_status "Starting CodeLoom container..."
    docker compose up -d

    # Read base path from .env or default
    local base_path=$(grep -E '^APP_BASE_PATH=' .env 2>/dev/null | cut -d= -f2 | tr -d '"'"'"' ' || echo "/codeloom")
    [ -z "$base_path" ] && base_path="/codeloom"

    echo ""
    print_success "CodeLoom is running!"
    echo -e "  ${YELLOW}App:${NC}      http://your-server${base_path}/"
    echo -e "  ${YELLOW}Backend:${NC}  http://localhost:9005 (internal)"
    echo -e "  ${YELLOW}Health:${NC}   http://localhost:9005/api/health"
    echo -e "  ${YELLOW}API docs:${NC} http://localhost:9005/docs"
    echo -e "  ${YELLOW}Login:${NC}    admin / admin123"
    echo -e "  ${YELLOW}Logs:${NC}     ./deploy.sh logs"
    echo -e "  ${YELLOW}Stop:${NC}     ./deploy.sh down"
    echo ""
    echo -e "  ${BLUE}Nginx:${NC} Add this to your server block:"
    echo ""
    echo "    location ${base_path}/ {"
    echo "        proxy_pass http://127.0.0.1:9005/;"
    echo "        proxy_set_header Host \$host;"
    echo "        proxy_set_header X-Real-IP \$remote_addr;"
    echo "        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;"
    echo "        proxy_set_header X-Forwarded-Proto \$scheme;"
    echo "        proxy_http_version 1.1;"
    echo "        proxy_set_header Connection \"\";"
    echo "        proxy_buffering off;"
    echo "        proxy_read_timeout 600s;"
    echo "        client_max_body_size 100M;"
    echo "    }"
    echo ""
}

# ── Stop ─────────────────────────────────────────────────────
do_down() {
    print_status "Stopping CodeLoom..."
    docker compose down
    print_success "Stopped"
}

# ── Logs ─────────────────────────────────────────────────────
do_logs() {
    docker compose logs -f
}

# ── Status ───────────────────────────────────────────────────
do_status() {
    echo -e "\n${BLUE}═══ CodeLoom Deployment Status ═══${NC}\n"

    # Container
    if docker ps --format '{{.Names}}' | grep -q '^codeloom$'; then
        local status=$(docker inspect --format='{{.State.Status}}' codeloom 2>/dev/null)
        local health=$(docker inspect --format='{{.State.Health.Status}}' codeloom 2>/dev/null || echo "no healthcheck")
        print_success "Container: ${status} (health: ${health})"
    else
        print_error "Container: not running"
    fi

    # Backend health
    if curl -sf http://localhost:9005/api/health &>/dev/null; then
        print_success "Backend API: responding"
    else
        print_warning "Backend API: not responding"
    fi

    echo ""
}

# ── Full deploy ──────────────────────────────────────────────
do_deploy() {
    preflight

    echo -e "\n${BLUE}═══ Deploying CodeLoom ═══${NC}\n"

    # Stop existing container if running
    if docker ps -q -f name=codeloom 2>/dev/null | grep -q .; then
        do_down
    fi

    do_build
    do_up

    echo -e "\n${GREEN}═══ Deployment Complete ═══${NC}\n"
}

# ── Main ─────────────────────────────────────────────────────
case "${1:-}" in
    build)    preflight; do_build ;;
    up)       preflight; do_up ;;
    down)     do_down ;;
    logs)     do_logs ;;
    status)   do_status ;;
    restart)  preflight; do_down; do_build; do_up ;;
    *)        do_deploy ;;
esac
