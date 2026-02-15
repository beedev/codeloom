#!/bin/bash
# DBNotebook Development Script
# Usage: ./dev.sh [local|docker|stop|status|logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}▶${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Check if PostgreSQL is running
check_postgres() {
    if /opt/homebrew/opt/postgresql@17/bin/pg_isready -q 2>/dev/null; then
        return 0
    else
        print_error "PostgreSQL is not running"
        echo "  Start with: brew services start postgresql@17"
        return 1
    fi
}

# Check if Docker is running
check_docker() {
    if docker info >/dev/null 2>&1; then
        return 0
    else
        print_error "Docker is not running"
        echo "  Start Docker Desktop first"
        return 1
    fi
}

# Stop all services
stop_all() {
    print_status "Stopping all services..."

    # Stop local Flask
    if lsof -ti:7860 >/dev/null 2>&1; then
        lsof -ti:7860 | xargs kill -9 2>/dev/null || true
        print_success "Stopped local Flask server (port 7860)"
    fi

    # Stop Docker container
    if docker ps -q -f name=dbnotebook 2>/dev/null | grep -q .; then
        docker compose down 2>/dev/null
        print_success "Stopped Docker container"
    fi

    print_success "All services stopped"
}

# Show status
show_status() {
    echo -e "\n${BLUE}═══ DBNotebook Status ═══${NC}\n"

    # PostgreSQL
    if /opt/homebrew/opt/postgresql@17/bin/pg_isready -q 2>/dev/null; then
        print_success "PostgreSQL: Running on localhost:5432"
    else
        print_error "PostgreSQL: Not running"
    fi

    # Local Flask
    if lsof -ti:7860 >/dev/null 2>&1; then
        print_success "Local Flask: Running on http://localhost:7860"
    else
        echo -e "  ${YELLOW}○${NC} Local Flask: Not running"
    fi

    # Docker
    if docker ps -q -f name=dbnotebook 2>/dev/null | grep -q .; then
        print_success "Docker: Running on http://localhost:7007"
    else
        echo -e "  ${YELLOW}○${NC} Docker: Not running"
    fi

    echo ""
}

# Start local development
start_local() {
    print_status "Starting local development environment..."

    # Check PostgreSQL
    check_postgres || exit 1

    # Stop Docker if running
    if docker ps -q -f name=dbnotebook 2>/dev/null | grep -q .; then
        print_warning "Stopping Docker container first..."
        docker compose down 2>/dev/null
    fi

    # Check if port is in use
    if lsof -ti:7860 >/dev/null 2>&1; then
        print_warning "Port 7860 in use, killing existing process..."
        lsof -ti:7860 | xargs kill -9 2>/dev/null || true
        sleep 1
    fi

    # Load environment
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
        # Replace host.docker.internal with localhost for local dev
        export DATABASE_URL="${DATABASE_URL//host.docker.internal/localhost}"
        export POSTGRES_HOST="${POSTGRES_HOST//host.docker.internal/localhost}"
    fi

    # Activate venv and sync dependencies
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Syncing Python dependencies..."
        python3 -m pip install -q -r requirements.txt
    else
        print_error "Virtual environment not found. Run: python3 -m venv venv && pip install -r requirements.txt"
        exit 1
    fi

    # Build frontend
    if [ -d "frontend" ]; then
        print_status "Building frontend..."
        cd frontend
        npm install --silent
        npm run build
        cd "$SCRIPT_DIR"
        print_success "Frontend built"
    fi

    # Run migrations
    print_status "Running database migrations..."
    PYTHONPATH="$SCRIPT_DIR" alembic upgrade head

    # Start Flask
    print_success "Starting Flask on http://localhost:7860"
    echo -e "  ${YELLOW}Login:${NC} admin / admin123"
    echo -e "  ${YELLOW}Stop:${NC}  ./dev.sh stop"
    echo ""

    PYTHONPATH="$SCRIPT_DIR" python3 -m dbnotebook --host 0.0.0.0 --port 7860
}

# Start Docker
start_docker() {
    print_status "Starting Docker environment..."

    # Check Docker
    check_docker || exit 1

    # Check PostgreSQL (Docker connects to host's PostgreSQL)
    check_postgres || exit 1

    # Stop local Flask if running
    if lsof -ti:7860 >/dev/null 2>&1; then
        print_warning "Stopping local Flask server first..."
        lsof -ti:7860 | xargs kill -9 2>/dev/null || true
    fi

    # Remove existing container if exists
    if docker ps -aq -f name=dbnotebook 2>/dev/null | grep -q .; then
        docker rm -f dbnotebook 2>/dev/null
    fi

    # Build and start
    print_status "Building and starting container..."
    docker compose up --build -d

    # Wait for startup
    print_status "Waiting for container to initialize..."
    sleep 10

    # Check health
    if curl -s http://localhost:7007/api/auth/me >/dev/null 2>&1; then
        print_success "Docker running on http://localhost:7007"
        echo -e "  ${YELLOW}Login:${NC} admin / admin123"
        echo -e "  ${YELLOW}Logs:${NC}  ./dev.sh logs"
        echo -e "  ${YELLOW}Stop:${NC}  ./dev.sh stop"
    else
        print_warning "Container started but health check pending..."
        echo "  Check logs: docker logs dbnotebook"
    fi
    echo ""
}

# Show logs
show_logs() {
    if docker ps -q -f name=dbnotebook 2>/dev/null | grep -q .; then
        docker logs -f dbnotebook
    else
        print_error "Docker container is not running"
    fi
}

# Main
case "${1:-}" in
    local|l)
        start_local
        ;;
    docker|d)
        start_docker
        ;;
    stop|s)
        stop_all
        ;;
    status|st)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo -e "\n${BLUE}DBNotebook Development Script${NC}\n"
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  local, l    Start local Flask development server (port 7860)"
        echo "  docker, d   Start Docker container (port 7007)"
        echo "  stop, s     Stop all services"
        echo "  status, st  Show status of all services"
        echo "  logs        Follow Docker container logs"
        echo ""
        echo "Database: PostgreSQL on localhost:5432 (shared by both modes)"
        echo ""
        exit 1
        ;;
esac
