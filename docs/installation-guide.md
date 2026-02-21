# CodeLoom Installation Guide

CodeLoom is a code intelligence and migration platform. You upload a codebase as a zip file, and CodeLoom parses it into an AST + ASG graph, embeds it into a vector store, and lets you query it through an AI-powered chat interface. This guide walks you through every step to get a working installation.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Database Setup](#3-database-setup)
4. [Environment Configuration](#4-environment-configuration)
5. [Frontend Setup](#5-frontend-setup)
6. [Optional: Java Enrichment (JavaParser CLI)](#6-optional-java-enrichment-javaparser-cli)
7. [Optional: C# Enrichment (Roslyn Analyzer)](#7-optional-c-enrichment-roslyn-analyzer)
8. [Optional: PlantUML Diagram Rendering](#8-optional-plantuml-diagram-rendering)
9. [Docker Deployment](#9-docker-deployment)
10. [Verification Checklist](#10-verification-checklist)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

Install the following before you begin. The core stack (Python + PostgreSQL) is required. The language enrichment tools are optional but improve analysis quality for Java and C# codebases.

### Required

**Python 3.11 or higher**

CodeLoom uses Python 3.11+ language features and type annotations. Check your version:

```bash
python3 --version
# Python 3.11.x or higher required
```

Install on macOS:

```bash
brew install python@3.11
```

Install on Ubuntu/Debian:

```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Node.js 18 or higher**

Required for the React frontend. Check your version:

```bash
node --version
# v18.x or higher required

npm --version
# 9.x or higher
```

Install on macOS:

```bash
brew install node
```

Install on Ubuntu/Debian:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
```

**PostgreSQL 17**

CodeLoom requires PostgreSQL 17 with the pgvector extension for vector similarity search. pgvector is installed automatically by the Alembic migrations, but PostgreSQL itself must be installed first.

Install on macOS:

```bash
brew install postgresql@17
brew services start postgresql@17
```

Install on Ubuntu/Debian:

```bash
sudo apt install -y postgresql-17 postgresql-server-dev-17

# Install pgvector from source (Ubuntu 22.04+)
sudo apt install -y build-essential git
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

> On Ubuntu, pgvector packages are also available for recent releases:
> `sudo apt install postgresql-17-pgvector`

**Git**

```bash
git --version
# git version 2.x
```

Install on macOS: `brew install git`
Install on Ubuntu/Debian: `sudo apt install git`

### Optional

**JDK 11+ and Maven** — required only for deep Java semantic enrichment. Without these, tree-sitter handles Java parsing without fully qualified type names or checked exception resolution.

**dotnet SDK 8** — required only for deep C# semantic enrichment via the Roslyn analyzer. Without it, tree-sitter handles C# parsing without fully qualified names or nullable annotations.

**Graphviz** — optional companion to PlantUML for improved diagram layout. PlantUML's built-in Smetana layout engine works without it.

---

## 2. Quick Start

If you have Python 3.11+, Node.js 18+, and PostgreSQL 17 installed, these commands get CodeLoom running in under five minutes.

```bash
# 1. Clone the repository
git clone <repo-url> codeloom
cd codeloom

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create the database (PostgreSQL must already be running)
createdb codeloom_dev

# 5. Configure your environment
cp .env.example .env
# Open .env and set your LLM API keys — see Section 4 for details

# 6. Start backend and frontend together
./dev.sh local
```

After `./dev.sh local` succeeds, open http://localhost:3000 and log in with:

- Username: `admin`
- Password: `admin123`

The backend API is available at http://localhost:9005 and its interactive documentation at http://localhost:9005/docs.

---

## 3. Database Setup

### macOS

PostgreSQL 17 via Homebrew is the recommended local setup on macOS.

```bash
# Install
brew install postgresql@17

# Start and enable on login
brew services start postgresql@17

# Verify it is running
pg_isready
# localhost:5432 - accepting connections

# Create the database
createdb codeloom_dev
```

If `createdb` fails with a "role does not exist" error, create a superuser role first:

```bash
psql postgres -c "CREATE ROLE $(whoami) SUPERUSER LOGIN;"
createdb codeloom_dev
```

### Ubuntu / Debian

```bash
# Install PostgreSQL 17
sudo apt install -y postgresql-17

# Install pgvector
sudo apt install -y postgresql-17-pgvector
# If the package is not available, build from source (see Section 1)

# Start the service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create the database user and database
sudo -u postgres psql -c "CREATE USER codeloom WITH PASSWORD 'codeloom';"
sudo -u postgres psql -c "CREATE DATABASE codeloom_dev OWNER codeloom;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE codeloom_dev TO codeloom;"
```

Update your `.env` to match these credentials:

```
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
```

### Creating the Database Schema

CodeLoom uses Alembic for schema migrations. `./dev.sh local` runs migrations automatically before starting the server. To run them manually:

```bash
# Activate your virtual environment first
source venv/bin/activate

alembic upgrade head
```

The migration creates all tables (`users`, `projects`, `code_files`, `code_units`, `code_edges`, `migration_plans`, etc.) and installs the pgvector extension automatically via `CREATE EXTENSION IF NOT EXISTS vector`.

### Default Credentials

The initial migration seeds a default admin account:

- Username: `admin`
- Password: `admin123`

Change this password immediately in any environment other than local development.

---

## 4. Environment Configuration

Copy `.env.example` to `.env` and edit the values for your setup. The sections below describe each group of settings.

```bash
cp .env.example .env
```

### LLM Provider

Set `LLM_PROVIDER` to one of the supported backends:

| Provider | Value | Notes |
|---|---|---|
| Ollama (local) | `ollama` | No API key required; runs models locally |
| OpenAI | `openai` | Requires `OPENAI_API_KEY` |
| Anthropic | `anthropic` | Requires `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | Requires `GOOGLE_API_KEY` |
| Groq | `groq` | Requires `GROQ_API_KEY`; fastest cloud option |

Example for Ollama (no API keys needed):

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:latest
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

Example for OpenAI:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

Example for Groq (fastest cloud inference):

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
```

### Embedding Provider and Dimension

The embedding dimension in `PGVECTOR_EMBED_DIM` must exactly match the dimension of the model you select. If they do not match, ingestion will fail with a dimension mismatch error.

| Model | Provider value | Dimension |
|---|---|---|
| `text-embedding-3-small` | `openai` | 1536 |
| `text-embedding-3-large` | `openai` | 3072 |
| `nomic-ai/nomic-embed-text-v1.5` | `huggingface` | 768 |

Example for OpenAI embeddings (default):

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
PGVECTOR_EMBED_DIM=1536
OPENAI_API_KEY=sk-...
```

Example for local HuggingFace embeddings (no API key needed):

```bash
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
PGVECTOR_EMBED_DIM=768
```

> If you switch embedding models after ingesting data, you must re-ingest all projects because existing embeddings were stored at the old dimension. Drop and recreate the `data_embeddings` table or delete all projects through the UI first.

### Database

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=codeloom_dev
POSTGRES_USER=codeloom
POSTGRES_PASSWORD=codeloom
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
```

When running inside Docker, replace `localhost` with `host.docker.internal` so the container can reach the PostgreSQL instance on the host machine.

### Session Secret

```bash
FLASK_SECRET_KEY=change-me-to-a-long-random-string
```

This key signs the session cookie. Use a cryptographically random string in production:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

The variable name `FLASK_SECRET_KEY` is a legacy name from the project's Flask origins. It is still used by FastAPI's `SessionMiddleware` — do not rename it.

### Reranker

The reranker re-scores retrieval results for better relevance. Choose a local model (requires download) or a Groq-hosted model (fast, requires `GROQ_API_KEY`):

```bash
# Local CPU reranker (downloads on first run)
RERANKER_MODEL=base          # ~30s on 4-core CPU

# Groq-hosted reranker (fastest, requires GROQ_API_KEY)
RERANKER_MODEL=groq:scout    # ~300ms

# Disable reranking (fastest responses, lower quality)
RERANKER_MODEL=disabled
```

### Complete Minimal .env Example

This is the smallest viable configuration for local development with Ollama and OpenAI embeddings:

```bash
# LLM
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:latest

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-key-here

# Database
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
PGVECTOR_TABLE_NAME=data_embeddings
PGVECTOR_EMBED_DIM=1536

# Session
FLASK_SECRET_KEY=replace-with-random-string

# Reranker
RERANKER_MODEL=disabled
RETRIEVAL_STRATEGY=hybrid
```

---

## 5. Frontend Setup

The frontend is a React 19 + TypeScript + Vite application. It runs on port 3000 and proxies all `/api` requests to the backend at port 9005.

### Development Mode

`./dev.sh local` starts the frontend automatically alongside the backend. To run it in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. The Vite dev server proxies the following paths to the backend:

| Path prefix | Backend target |
|---|---|
| `/api` | http://localhost:9005 |
| `/chat` | http://localhost:9005 |
| `/upload` | http://localhost:9005 |
| `/image` | http://localhost:9005 |

### Production Build

```bash
cd frontend
npm run build
# Output written to frontend/dist/
```

The production build outputs static files to `frontend/dist/`. Serve these files through a web server (nginx, Caddy, etc.) or configure your deployment to serve them from the FastAPI app.

### Lint

```bash
cd frontend
npm run lint
```

ESLint runs with TypeScript-aware rules. Fix lint errors before committing — the CI pipeline runs the same check.

### Frontend Dependencies

Key packages and their roles:

| Package | Purpose |
|---|---|
| `react` 19 | UI framework |
| `react-router-dom` 7 | Client-side routing |
| `tailwindcss` 4 | Utility-first CSS |
| `lucide-react` | Icon library |
| `react-markdown` + `remark-gfm` | Renders markdown in chat responses |
| `prism-react-renderer` | Syntax highlighting in code blocks |
| `react-force-graph-2d` | ASG graph visualization |
| `motion` | Animation library |

---

## 6. Optional: Java Enrichment (JavaParser CLI)

By default, CodeLoom parses Java files using tree-sitter, which provides accurate structural information (classes, methods, fields, imports). The optional JavaParser CLI bridge adds a second enrichment pass that resolves fully qualified type names, generic type parameters, and checked exception declarations — information that tree-sitter cannot provide without type resolution.

### Requirements

- JDK 11 or higher (`java --version`)
- Maven 3.6 or higher (`mvn --version`)

### Build

```bash
./dev.sh setup-tools
```

This command runs `mvn -q package -DskipTests` inside `tools/javaparser-cli/` and produces:

```
tools/javaparser-cli/target/javaparser-cli.jar
```

### Verify the Build

```bash
java -jar tools/javaparser-cli/target/javaparser-cli.jar --help
```

### How It Works

When the JAR is present, the bridge in `codeloom/core/ast_parser/bridges/` invokes it as a subprocess during code ingestion. The CLI accepts a `.java` file path and returns enriched JSON containing:

- Fully qualified class and import names
- Generic type bounds on method parameters and return types
- Checked exceptions declared in `throws` clauses
- Field declarations with resolved types

This enriched metadata is merged with the tree-sitter parse output and stored in `code_units.metadata`. The ASG builder uses it to generate more precise `type_dep` and `calls` edges.

If Maven or JDK is not found, `./dev.sh setup-tools` skips the JavaParser build and prints a warning. Java files are still parsed using tree-sitter — you lose the deeper type resolution but not the structural analysis.

---

## 7. Optional: C# Enrichment (Roslyn Analyzer)

The Roslyn Analyzer bridge provides the same enrichment role for C# that JavaParser provides for Java. It uses the .NET Compiler Platform (Roslyn) to perform a full semantic compilation of C# source files and extract fully qualified names, nullable annotations, interface implementations, and generic constraints.

### Requirements

- .NET SDK 8 (`dotnet --version`)

Install on macOS:

```bash
brew install dotnet@8
```

Install on Ubuntu/Debian:

```bash
sudo apt install dotnet-sdk-8.0
```

### Build

```bash
./dev.sh setup-tools
```

This command runs `dotnet build -c Release` inside `tools/roslyn-analyzer/` and produces:

```
tools/roslyn-analyzer/bin/Release/net8.0/roslyn-analyzer.dll
```

### Verify the Build

```bash
dotnet tools/roslyn-analyzer/bin/Release/net8.0/roslyn-analyzer.dll --help
```

### How It Works

The bridge in `codeloom/core/ast_parser/bridges/` invokes the DLL as a subprocess for each `.cs` file during ingestion. The analyzer compiles the file using Roslyn's semantic model and returns JSON with:

- Fully qualified type names for all symbols
- Nullable reference type annotations (`string?`, `List<T>?`)
- Generic type constraints
- Interface implementation metadata
- Method override information

If the DLL is not found, CodeLoom falls back to tree-sitter for C# parsing. Structural information (classes, methods, properties, fields) is still extracted correctly.

---

## 8. Optional: PlantUML Diagram Rendering

CodeLoom can generate UML diagrams (class diagrams, sequence diagrams, activity diagrams, and others) from your codebase. It uses PlantUML to render diagrams from its text-based DSL.

### Setup

```bash
./dev.sh setup-tools
```

This downloads PlantUML v1.2024.8 to `tools/plantuml/plantuml.jar`. Java must be available on your PATH for the download step.

### Graphviz (Optional)

Graphviz provides a higher-quality layout engine for PlantUML diagrams. Without it, PlantUML uses its built-in Smetana engine, which handles most diagrams well.

Install on macOS:

```bash
brew install graphviz
```

Install on Ubuntu/Debian:

```bash
sudo apt install graphviz
```

### Configuration

Uncomment the following line in your `.env` to use the local JAR:

```bash
PLANTUML_JAR_PATH=tools/plantuml/plantuml.jar
```

If the JAR path is not set, CodeLoom falls back to PlantUML's public HTTP server at `https://www.plantuml.com/plantuml`. The public server works but has file size limits and requires an internet connection.

### Verify

```bash
java -jar tools/plantuml/plantuml.jar -version
# PlantUML version 1.2024.8 ...
```

---

## 9. Docker Deployment

Docker deployment packages the backend and frontend into a single container and connects to a PostgreSQL instance running on the host machine.

### Prerequisites

- Docker Desktop installed and running
- PostgreSQL running on the host (same requirement as local mode)
- `.env` file configured (see Section 4)

### Start

```bash
./dev.sh docker
```

This runs `docker compose up --build -d` and starts CodeLoom on port 7007. After a 10-second startup wait, the script checks the health endpoint and prints the URL.

Access the application at http://localhost:7007.

### Database Connectivity in Docker

The Docker container uses `host.docker.internal` to reach the host's PostgreSQL instance. Set these values in your `.env` for Docker mode:

```bash
POSTGRES_HOST=host.docker.internal
DATABASE_URL=postgresql://codeloom:codeloom@host.docker.internal:5432/codeloom_dev
```

When switching between `./dev.sh local` and `./dev.sh docker`, the `local` mode automatically rewrites `host.docker.internal` back to `localhost` in the loaded environment, so a single `.env` file works for both modes.

### Docker Commands

```bash
./dev.sh docker     # Build and start container on port 7007
./dev.sh stop       # Stop the container
./dev.sh logs       # Follow container logs (docker logs -f codeloom)
./dev.sh status     # Show status of all services including Docker
```

### Stopping Services

```bash
./dev.sh stop
```

This stops the backend on port 9005, the frontend dev server on port 3000, and the Docker container if running.

---

## 10. Verification Checklist

Follow these steps in order after installation to confirm everything is working.

**Step 1: Health endpoint**

```bash
curl http://localhost:9005/api/health
```

Expected response: `{"status": "ok"}` or similar. A non-200 response means the backend did not start correctly — check terminal output for errors.

**Step 2: Frontend loads**

Open http://localhost:3000 in a browser. You should see the CodeLoom login page.

**Step 3: Login**

Enter username `admin` and password `admin123`. You should land on the Dashboard page showing an empty project list.

**Step 4: Create a project**

Click "New Project" (or the equivalent button on the Dashboard). Enter a name and description. The project should appear in the project list.

**Step 5: Upload source code**

Open the project. Use the upload button to submit a zip file containing source code. CodeLoom will:
- Extract the zip
- Parse each file using tree-sitter
- Chunk the code into ~1024-token segments
- Generate embeddings
- Store everything in PostgreSQL

Watch the backend logs (`./dev.sh local` keeps them in the terminal) to observe ingestion progress. Large codebases may take several minutes.

**Step 6: Ask a question in code chat**

Navigate to the project's chat view. Type a question about the codebase, such as "What does the authentication system do?" or "List all API endpoints." The backend retrieves relevant code chunks via hybrid search (BM25 + vector + reranking) and generates a response. Streaming output should appear token by token.

**Step 7: Check ASG status**

Return to the project view. The ASG (Abstract Semantic Graph) panel shows relationships between code units — classes, functions, imports, calls, and inheritance. If the graph is populated, ingestion completed successfully.

---

## 11. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `could not load library "vector.so"` or pgvector extension not found | pgvector is not installed in PostgreSQL | Install pgvector: `sudo apt install postgresql-17-pgvector` on Ubuntu, or build from source. Then re-run `alembic upgrade head`. |
| `EMBEDDING_MODEL dimension mismatch` or `expected 1536, got 768` during ingestion | `PGVECTOR_EMBED_DIM` in `.env` does not match the actual embedding model output dimension | Update `PGVECTOR_EMBED_DIM` to match your model (1536 for OpenAI `text-embedding-3-small`, 768 for `nomic-embed-text`). If you changed models after ingesting data, delete all projects and re-ingest. |
| `OMP_NUM_THREADS` or `TOKENIZERS_PARALLELISM` warnings at startup | Normal startup behavior | These environment variables are set intentionally at startup to prevent segfaults in torch and onnxruntime under multi-threaded FastAPI. The warnings are informational and do not indicate a problem. |
| `address already in use: 0.0.0.0:9005` | Another instance of the backend is running | Run `./dev.sh stop` to kill all running services, then restart with `./dev.sh local`. |
| `tree_sitter` build errors during `pip install -r requirements.txt` | Missing C compiler | On macOS: `xcode-select --install`. On Ubuntu/Debian: `sudo apt install build-essential`. Then re-run `pip install -r requirements.txt`. |
| RAPTOR worker does not start or background tasks are skipped | `DISABLE_BACKGROUND_WORKERS=true` is set | Remove or unset `DISABLE_BACKGROUND_WORKERS` from your `.env`. This variable is used in Gunicorn multi-worker deployments to prevent multiple workers from competing on the asyncio event loop. In local single-worker mode it should not be set. |
| Frontend shows "Network Error" or API calls fail with CORS or 404 errors | Vite proxy is not forwarding requests correctly | Check `frontend/vite.config.ts`. The proxy target must be `http://localhost:9005`. Confirm the backend is running on that port with `curl http://localhost:9005/api/health`. |
| `createdb: error: role "username" does not exist` | Your system user does not have a PostgreSQL role | Run `psql postgres -c "CREATE ROLE $(whoami) SUPERUSER LOGIN;"` and then retry `createdb codeloom_dev`. |
| `ModuleNotFoundError` after activating venv | Dependencies not installed or wrong virtual environment | Ensure you are in the project root and have activated the venv: `source venv/bin/activate`. Then run `pip install -r requirements.txt`. |
| Ollama connection refused at startup | Ollama is not running | Start Ollama: `ollama serve`. Confirm it is listening: `curl http://localhost:11434/`. Pull your model if needed: `ollama pull llama3.1`. |
| `java.lang.UnsupportedClassVersionError` when running JavaParser CLI | JDK version is too old | The JavaParser CLI requires JDK 11 or higher. Check with `java --version` and upgrade if needed. |
| `dotnet: command not found` when running setup-tools | .NET SDK is not installed | Install .NET 8 SDK from https://dotnet.microsoft.com/download or via `brew install dotnet@8` on macOS. |
| PlantUML diagram generation produces blank output | Java not on PATH or JAR not downloaded | Run `./dev.sh setup-tools` to download the JAR. Confirm Java is available: `java --version`. |
| Sessions expire immediately or login redirects loop | `FLASK_SECRET_KEY` is not set or changes between restarts | Set a stable, random value for `FLASK_SECRET_KEY` in `.env`. Generate one with: `python3 -c "import secrets; print(secrets.token_hex(32))"`. |

---

## Additional Resources

- **API documentation**: http://localhost:9005/docs (Swagger UI, available when backend is running)
- **Architecture overview**: `docs/architecture.md`
- **Project CLAUDE.md**: `CLAUDE.md` — development conventions, component map, and gotchas for contributors
- **Alembic migrations**: `alembic/versions/` — schema change history
- **Configuration reference**: `config/codeloom.yaml` — ingestion parameters, retrieval settings, RAPTOR configuration
