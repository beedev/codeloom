#!/bin/bash
# CodeLoom Development Script
# Usage: ./dev.sh [local|docker|stop|status|logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ports — read from appregistry via python, fall back to defaults
APPREG="$HOME/.appregistry/registry.yaml"
if [ -f "$APPREG" ] && command -v python3 &>/dev/null; then
    _ports=$(python3 -c "
import yaml
with open('$APPREG') as f:
    d = yaml.safe_load(f)
a = d.get('apps',{}).get('codeloom',{})
print(a.get('backend_port',''))
print(a.get('frontend_port',''))
" 2>/dev/null)
    BACKEND_PORT=${BACKEND_PORT:-$(echo "$_ports" | sed -n '1p')}
    FRONTEND_PORT=${FRONTEND_PORT:-$(echo "$_ports" | sed -n '2p')}
fi
BACKEND_PORT=${BACKEND_PORT:-9005}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

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

# Check if PostgreSQL is reachable (generic)
is_postgres_running() {
    local host="localhost"
    local port="5432"
    
    if [ -f .env ]; then
        local env_host=$(grep -E '^POSTGRES_HOST=' .env | cut -d= -f2 | tr -d '""'"''"'\r ')
        local env_port=$(grep -E '^POSTGRES_PORT=' .env | cut -d= -f2 | tr -d '""'"''"'\r ')
        [ -n "$env_host" ] && host="$env_host"
        [ -n "$env_port" ] && port="$env_port"
    fi
    
    if command -v pg_isready >/dev/null 2>&1; then
        pg_isready -h "$host" -p "$port" -q 2>/dev/null
        return $?
    elif command -v nc >/dev/null 2>&1; then
        nc -z "$host" "$port" 2>/dev/null
        return $?
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$host', int('$port')))" 2>/dev/null
        return $?
    else
        # Return success if we have no tools to test it, let it fail naturally down the line
        return 0
    fi
}

# Check if PostgreSQL is running
check_postgres() {
    if is_postgres_running; then
        return 0
    else
        local host="localhost"
        local port="5432"
        [ -f .env ] && host=$(grep -E '^POSTGRES_HOST=' .env | cut -d= -f2 | tr -d '""'"''"'\r ') || host="localhost"
        [ -f .env ] && port=$(grep -E '^POSTGRES_PORT=' .env | cut -d= -f2 | tr -d '""'"''"'\r ') || port="5432"
        
        print_error "PostgreSQL is not running on $host:$port"
        echo "  Make sure your local or Docker database is started."
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

    # Stop local backend
    if lsof -ti:$BACKEND_PORT >/dev/null 2>&1; then
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        print_success "Stopped backend server (port $BACKEND_PORT)"
    fi

    # Stop frontend dev server
    if lsof -ti:$FRONTEND_PORT >/dev/null 2>&1; then
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
        print_success "Stopped frontend dev server (port $FRONTEND_PORT)"
    fi

    # Stop Docker container
    if docker ps -q -f name=codeloom 2>/dev/null | grep -q .; then
        docker compose down 2>/dev/null
        print_success "Stopped Docker container"
    fi

    print_success "All services stopped"
}

# Show status
show_status() {
    echo -e "\n${BLUE}═══ CodeLoom Status ═══${NC}\n"

    # PostgreSQL
    local pg_host="localhost"
    local pg_port="5432"
    [ -f .env ] && pg_host=$(grep -E '^POSTGRES_HOST=' .env | cut -d= -f2 | tr -d '""'"''"'\r ') || pg_host="localhost"
    [ -f .env ] && pg_port=$(grep -E '^POSTGRES_PORT=' .env | cut -d= -f2 | tr -d '""'"''"'\r ') || pg_port="5432"
    
    if is_postgres_running; then
        print_success "PostgreSQL: Reachable on $pg_host:$pg_port"
    else
        print_error "PostgreSQL: Not reachable on $pg_host:$pg_port"
    fi

    # Backend
    if lsof -ti:$BACKEND_PORT >/dev/null 2>&1; then
        print_success "Backend: Running on http://localhost:$BACKEND_PORT"
    else
        echo -e "  ${YELLOW}○${NC} Backend: Not running"
    fi

    # Frontend
    if lsof -ti:$FRONTEND_PORT >/dev/null 2>&1; then
        print_success "Frontend: Running on http://localhost:$FRONTEND_PORT"
    else
        echo -e "  ${YELLOW}○${NC} Frontend: Not running"
    fi

    # Docker
    if docker ps -q -f name=codeloom 2>/dev/null | grep -q .; then
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
    if docker ps -q -f name=codeloom 2>/dev/null | grep -q .; then
        print_warning "Stopping Docker container first..."
        docker compose down 2>/dev/null
    fi

    # Check if port is in use
    if lsof -ti:$BACKEND_PORT >/dev/null 2>&1; then
        print_warning "Port $BACKEND_PORT in use, killing existing process..."
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi

    # Load environment
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
        # Replace host.docker.internal with localhost for local dev
        export DATABASE_URL="${DATABASE_URL//host.docker.internal/localhost}"
        export POSTGRES_HOST="${POSTGRES_HOST//host.docker.internal/localhost}"
    fi

    # Activate venv — prefer shared appreg venv, fall back to local
    SHARED_VENV="$HOME/.appregistry/venvs/ai-full"
    if [ -d "$SHARED_VENV" ]; then
        source "$SHARED_VENV/bin/activate"
        print_success "Using shared venv: $SHARED_VENV"
    elif [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Syncing Python dependencies..."
        python3 -m pip install -q -r requirements.txt
    else
        print_error "No venv found. Run: appreg venv-create ai-full"
        exit 1
    fi

    # Run migrations
    print_status "Running database migrations..."
    PYTHONPATH="$SCRIPT_DIR" alembic upgrade head

    # Start frontend dev server in background
    if [ -d "frontend" ]; then
        print_status "Starting frontend dev server on http://localhost:$FRONTEND_PORT..."
        cd frontend
        npm install --silent 2>/dev/null
        PORT=$FRONTEND_PORT BACKEND_PORT=$BACKEND_PORT npm run dev &
        FRONTEND_PID=$!
        cd "$SCRIPT_DIR"
        print_success "Frontend dev server started (PID $FRONTEND_PID)"
    fi

    # Start backend
    print_success "Starting backend on http://localhost:$BACKEND_PORT"
    echo -e "  ${YELLOW}Frontend:${NC} http://localhost:$FRONTEND_PORT"
    echo -e "  ${YELLOW}Backend:${NC}  http://localhost:$BACKEND_PORT"
    echo -e "  ${YELLOW}API docs:${NC} http://localhost:$BACKEND_PORT/docs"
    echo -e "  ${YELLOW}Login:${NC}    admin / admin123"
    echo -e "  ${YELLOW}Stop:${NC}     ./dev.sh stop"
    echo ""

    # Trap to kill frontend when backend exits
    trap 'kill $FRONTEND_PID 2>/dev/null; exit' INT TERM EXIT

    PYTHONPATH="$SCRIPT_DIR" python3 -m codeloom --host 0.0.0.0 --port $BACKEND_PORT
}

# Start Docker
start_docker() {
    print_status "Starting Docker environment..."

    # Check Docker
    check_docker || exit 1

    # Check PostgreSQL (Docker connects to host's PostgreSQL)
    check_postgres || exit 1

    # Stop local backend if running
    if lsof -ti:$BACKEND_PORT >/dev/null 2>&1; then
        print_warning "Stopping local backend first..."
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    fi

    # Remove existing container if exists
    if docker ps -aq -f name=codeloom 2>/dev/null | grep -q .; then
        docker rm -f codeloom 2>/dev/null
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
        echo "  Check logs: docker logs codeloom"
    fi
    echo ""
}

# Build frontend + backend
build_all() {
    print_status "Building CodeLoom..."

    # Frontend
    if [ -d "frontend" ]; then
        print_status "Installing frontend dependencies..."
        cd frontend
        npm install --silent 2>/dev/null
        print_status "Building frontend (Vite production build)..."
        npm run build
        cd "$SCRIPT_DIR"
        print_success "Frontend built → frontend/dist/"
    else
        print_warning "frontend/ directory not found — skipping"
    fi

    # Backend: venv + deps + typecheck
    SHARED_VENV="$HOME/.appregistry/venvs/ai-full"
    if [ -d "$SHARED_VENV" ]; then
        source "$SHARED_VENV/bin/activate"
        print_success "Using shared venv: $SHARED_VENV"
    elif [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Syncing Python dependencies..."
        python3 -m pip install -q -r requirements.txt
        print_success "Python dependencies up to date"
    else
        print_error "No venv found. Run: appreg venv-create ai-full"
        exit 1
    fi

    # Run migrations if Postgres is available
    if is_postgres_running; then
        print_status "Running database migrations..."
        PYTHONPATH="$SCRIPT_DIR" alembic upgrade head
        print_success "Database migrations applied"
    else
        print_warning "PostgreSQL not running — skipping migrations"
    fi

    echo ""
    print_success "Build complete"
    echo ""
}

# Show logs
show_logs() {
    if docker ps -q -f name=codeloom 2>/dev/null | grep -q .; then
        docker logs -f codeloom
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
    build|b)
        build_all
        ;;
    setup-mcp)
        print_status "Registering CodeLoom MCP server with Claude Code CLI..."

        if [ ! -f ".env" ]; then
            print_error ".env file not found. Copy .env.example to .env first."
            exit 1
        fi

        source .env 2>/dev/null || true

        CODELOOM_DIR="$(cd "$(dirname "$0")" && pwd)"
        MCP_CONFIG="$HOME/.claude/mcp.json"

        # Ensure ~/.claude directory exists
        mkdir -p "$HOME/.claude"

        # Use Python in venv to safely merge JSON
        if [ -d "venv" ]; then
            source venv/bin/activate 2>/dev/null || true
        fi
        python3 - <<PYEOF
import json, os, sys

config_path = os.path.expanduser("~/.claude/mcp.json")
existing = {}
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            existing = json.load(f)
    except Exception:
        pass

codeloom_dir = "${CODELOOM_DIR}"
db_url = "${DATABASE_URL:-postgresql://codeloom:codeloom@localhost:5432/codeloom_dev}"
llm_provider = "${LLM_PROVIDER:-ollama}"

entry = {
    "codeloom": {
        "command": sys.executable,
        "args": ["-m", "codeloom.mcp"],
        "cwd": codeloom_dir,
        "env": {
            "DATABASE_URL": db_url,
            "LLM_PROVIDER": llm_provider,
            "PYTHONPATH": codeloom_dir,
            "PYTHONUNBUFFERED": "1",
        }
    }
}

# Merge mcpServers key (Claude Code uses "mcpServers" in mcp.json)
if "mcpServers" not in existing:
    existing["mcpServers"] = {}
existing["mcpServers"].update(entry)

with open(config_path, "w") as f:
    json.dump(existing, f, indent=2)
print(f"  ✓ MCP config written to {config_path}")
print(f"  ✓ Server: codeloom → python -m codeloom.mcp")
PYEOF

        print_success "CodeLoom MCP server registered with Claude Code"
        echo "  Next: Start a Claude Code session and run: /migrate"
        ;;
    setup-tools)
        echo -e "\n${BLUE}Building optional enrichment tools...${NC}\n"

        # JavaParser CLI (requires Maven + JDK)
        print_status "Building JavaParser CLI..."
        if command -v mvn &>/dev/null && command -v java &>/dev/null; then
            (cd "$SCRIPT_DIR/tools/javaparser-cli" && mvn -q package -DskipTests)
            if [ -f "$SCRIPT_DIR/tools/javaparser-cli/target/javaparser-cli.jar" ]; then
                print_success "javaparser-cli.jar built"
            else
                print_error "JavaParser CLI build failed"
            fi
        else
            print_warning "Maven or JDK not found — skipping (install Maven + JDK to enable Java enrichment)"
        fi

        # Roslyn Analyzer (requires .NET SDK)
        print_status "Building Roslyn Analyzer..."
        if command -v dotnet &>/dev/null; then
            (cd "$SCRIPT_DIR/tools/roslyn-analyzer" && dotnet build -c Release -q)
            if [ -f "$SCRIPT_DIR/tools/roslyn-analyzer/bin/Release/net8.0/roslyn-analyzer.dll" ]; then
                print_success "roslyn-analyzer.dll built"
            else
                print_error "Roslyn Analyzer build failed"
            fi
        else
            print_warning "dotnet not found — skipping (install .NET 8 SDK to enable C# enrichment)"
        fi

        # PlantUML JAR (requires JDK — for local diagram rendering)
        print_status "Setting up PlantUML JAR..."
        PLANTUML_DIR="$SCRIPT_DIR/tools/plantuml"
        PLANTUML_JAR="$PLANTUML_DIR/plantuml.jar"
        PLANTUML_VERSION="1.2024.8"
        if [ -f "$PLANTUML_JAR" ]; then
            print_success "plantuml.jar already present"
        elif command -v java &>/dev/null; then
            mkdir -p "$PLANTUML_DIR"
            PLANTUML_URL="https://github.com/plantuml/plantuml/releases/download/v${PLANTUML_VERSION}/plantuml-${PLANTUML_VERSION}.jar"
            print_status "Downloading PlantUML v${PLANTUML_VERSION}..."
            if curl -fSL -o "$PLANTUML_JAR" "$PLANTUML_URL"; then
                print_success "plantuml.jar downloaded (v${PLANTUML_VERSION})"
            else
                print_error "PlantUML download failed — diagram rendering will use HTTP fallback"
                rm -f "$PLANTUML_JAR"
            fi
        else
            print_warning "Java not found — skipping PlantUML JAR (diagram rendering will use HTTP fallback)"
        fi
        # Graphviz is optional — PlantUML's built-in Smetana engine works without it
        if ! command -v dot &>/dev/null; then
            print_warning "Graphviz (dot) not found — PlantUML will use built-in Smetana layout engine"
            echo -e "  ${YELLOW}Hint:${NC} brew install graphviz for better layout quality"
        else
            print_success "Graphviz available — PlantUML will use native dot layout"
        fi

        echo ""
        print_success "Tool setup complete"
        echo -e "  ${YELLOW}Note:${NC} These tools are optional — tree-sitter enrichment works without them"
        echo ""
        ;;
    *)
        echo -e "\n${BLUE}CodeLoom Development Script${NC}\n"
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  local, l       Start backend (port $BACKEND_PORT) + frontend dev server (port $FRONTEND_PORT)"
        echo "  build, b       Build frontend + sync backend deps + run migrations"
        echo "  docker, d      Start Docker container (port 7007)"
        echo "  stop, s        Stop all services"
        echo "  status, st     Show status of all services"
        echo "  logs           Follow Docker container logs"
        echo "  setup-tools    Build optional Java/C# enrichment tools"
        echo "  setup-mcp      Register CodeLoom MCP server with Claude Code CLI"
        echo ""
        echo "Database: PostgreSQL on localhost:5432 (shared by both modes)"
        echo ""
        exit 1
        ;;
esac
