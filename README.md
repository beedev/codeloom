# CodeLoom

Code intelligence and migration platform powered by AST + ASG + RAG. Upload entire codebases, query them through AI-powered chat, analyze blast radius of changes, and migrate to new architectures through a guided pipeline.

## What It Does

1. **Parses** source code into semantic units (functions, classes, methods) using tree-sitter and regex parsers
2. **Builds** an Abstract Semantic Graph (ASG) capturing 8 relationship types between code units
3. **Embeds** everything into pgvector for hybrid BM25 + vector retrieval with reranking
4. **Chat** with your codebase — ask questions, trace dependencies, understand architecture
5. **Blast radius** — see exactly what breaks when you change a function, with impact scoring
6. **Migrate** codebases through a phased pipeline with human approval gates and MVP clustering
7. **Deep understanding** — automatic entry point detection, call tree tracing, and tiered analysis

## Supported Languages

| Language | Parser | Enrichment |
|----------|--------|------------|
| Python | tree-sitter | Built-in |
| JavaScript | tree-sitter | Built-in |
| TypeScript | tree-sitter | Built-in |
| Java | tree-sitter | Optional JavaParser bridge (deep type resolution) |
| C# | tree-sitter | Optional Roslyn bridge (semantic analysis) |
| VB.NET | Regex-based | Built-in (no tree-sitter grammar available) |
| SQL | Regex-based | Built-in |

## Architecture

```
Browser (React 19, Vite 7, Tailwind CSS 4)
    |
    | HTTP/SSE on :3000 (Vite dev proxy)
    v
FastAPI Backend (:9005)
    |
    +-- api/routes/
    |       fastapi_auth.py   Login, logout, session check
    |       projects.py       Project CRUD, zip/git/local upload, file browsing
    |       code_chat.py      RAG chat and blast radius analysis (SSE streaming)
    |       migration.py      Migration plan lifecycle, background MVP analysis
    |       diagrams.py       UML diagram generation (7 types)
    |       understanding.py  Deep understanding job management
    |
    +-- core/
    |       pipeline.py       Central orchestrator (LLM, embeddings, retrieval)
    |       ingestion/        Upload -> parse -> chunk -> embed -> store
    |       ast_parser/       tree-sitter + regex parsers (Strategy pattern)
    |       asg_builder/      8-type edge detection (calls, inherits, implements...)
    |       code_chunker/     AST-informed chunking with preamble injection
    |       vector_store/     PGVectorStore (pgvector + BM25 hybrid)
    |       raptor/           Hierarchical retrieval tree (background worker)
    |       stateless/        Thread-safe retrieval for multi-user API
    |       migration/        6-phase (V1) and 4-phase (V2) migration engine
    |       understanding/    Deep analysis daemon, call tree tracer
    |       diagrams/         Structural + behavioral UML generation
    |
    +-- PostgreSQL 17 + pgvector
            users, projects, code_files, code_units, code_edges,
            migration_plans, functional_mvps, conversations,
            deep_analysis_jobs, deep_analyses, query_logs
```

### Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + uvicorn |
| Frontend | React 19, Vite 7, Tailwind CSS 4, TypeScript |
| Database | PostgreSQL 17 + pgvector |
| ORM | SQLAlchemy 2.0 + Alembic |
| LLM Framework | LlamaIndex Core 0.14.x |
| AST Parsing | tree-sitter (Python, JS/TS, Java, C#), regex (SQL, VB.NET) |
| Embedding | OpenAI text-embedding-3-small (1536d) or HuggingFace |
| Reranker | mixedbread-ai/mxbai-rerank-base-v1 or Groq LLM rerankers |
| LLM Providers | Ollama, OpenAI, Anthropic, Gemini, Groq |

## Installation

### Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| PostgreSQL | 17 | `pg_isready` |
| Git | 2.x | `git --version` |

**Optional** (for deeper language analysis):
- JDK 11+ and Maven — deep Java semantic enrichment
- .NET SDK 8 — deep C# semantic enrichment via Roslyn
- Graphviz — improved PlantUML diagram layout

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/beedev/codeloom.git
cd codeloom
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Database
brew services start postgresql@17    # macOS
createdb codeloom_dev

# 3. Environment
cp .env.example .env
# Edit .env — set your LLM_PROVIDER and API keys

# 4. Run migrations
alembic upgrade head

# 5. Start
./dev.sh local    # Backend :9005 + Frontend :3000
```

Open http://localhost:3000. Login with `admin` / `admin123`.

### Database Setup

**macOS:**

```bash
brew install postgresql@17
brew services start postgresql@17
createdb codeloom_dev
```

If `createdb` fails with "role does not exist":

```bash
psql postgres -c "CREATE ROLE $(whoami) SUPERUSER LOGIN;"
createdb codeloom_dev
```

**Ubuntu/Debian:**

```bash
sudo apt install -y postgresql-17 postgresql-17-pgvector
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE USER codeloom WITH PASSWORD 'codeloom';"
sudo -u postgres psql -c "CREATE DATABASE codeloom_dev OWNER codeloom;"
```

The Alembic migrations install the pgvector extension and create all tables automatically:

```bash
alembic upgrade head
```

### Environment Configuration

Copy `.env.example` to `.env`. The critical settings:

**LLM Provider** — choose one:

| Provider | `LLM_PROVIDER` | API Key Required |
|----------|----------------|------------------|
| Ollama (local) | `ollama` | No |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |

**Embedding dimension must match the model** (most common setup error):

| Model | Provider | `PGVECTOR_EMBED_DIM` |
|-------|----------|----------------------|
| `text-embedding-3-small` | `openai` | 1536 |
| `text-embedding-3-large` | `openai` | 3072 |
| `nomic-ai/nomic-embed-text-v1.5` | `huggingface` | 768 |

**Minimal `.env` for local development** (Ollama + OpenAI embeddings):

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:latest
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
PGVECTOR_TABLE_NAME=data_embeddings
PGVECTOR_EMBED_DIM=1536
FLASK_SECRET_KEY=replace-with-random-string
RERANKER_MODEL=disabled
RETRIEVAL_STRATEGY=hybrid
```

> `FLASK_SECRET_KEY` is a legacy variable name — it's used by FastAPI's `SessionMiddleware`, not Flask.

### Frontend Setup

`./dev.sh local` starts the frontend automatically. To run separately:

```bash
cd frontend
npm install
npm run dev     # :3000, proxies /api to :9005
```

### Docker Deployment

```bash
./dev.sh docker    # Builds and runs on :7007
```

The Docker build automatically substitutes `localhost` with `host.docker.internal` for database access.

### Verification

After setup, verify everything works:

```bash
# 1. Health check
curl http://localhost:9005/api/health

# 2. Login
curl -X POST http://localhost:9005/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# 3. Open the UI
open http://localhost:3000
```

For the complete installation guide with troubleshooting, see [docs/installation-guide.md](docs/installation-guide.md).

## Ingestion Pipeline

Three ways to get code into CodeLoom:

```
Source (zip upload / git clone / local directory)
    |
    v
Walk directory tree (skip node_modules, .git, __pycache__, venv)
    |
    v
Per file:
    detect_language()     ->  language string
    parse_file()          ->  AST -> CodeUnit records
    SemanticEnricher      ->  metadata (params, types, modifiers, fields)
    CodeChunker.chunk()   ->  TextNode chunks (~1024 tokens each)
    vector_store.add()    ->  embeddings in pgvector
    |
    v
Per project:
    ASGBuilder.build_edges()    ->  CodeEdge records (8 edge types)
    RAPTORWorker.build_tree()   ->  Hierarchical summary nodes
    |
    v
Auto-trigger deep understanding analysis (background worker)
```

## ASG Edge Types

The Abstract Semantic Graph captures 8 relationship types:

| Edge Type | Direction | Example |
|-----------|-----------|---------|
| `contains` | class -> method | `UserService` contains `getUserById` |
| `imports` | file -> imported unit | `api.py` imports `AuthService` |
| `calls` | function -> function | `process_request` calls `validate_token` |
| `inherits` | class -> base class | `AdminService` inherits `BaseService` |
| `implements` | class -> interface | `UserRepository` implements `IRepository` |
| `overrides` | method -> parent method | `save` overrides `AbstractRepository.save` |
| `calls_sp` | app code -> stored procedure | `prepareCall("{ call usp_GetUser }")` |
| `type_dep` | consumer -> referenced type | `OrderService` depends on `PaymentGateway` |

## Chat With Code

Query your codebase through SSE-streaming RAG chat:

```
POST /api/projects/{id}/chat/stream
```

**Two modes:**
- `chat` — Ask questions about code structure, trace dependencies, understand patterns
- `impact` — Blast radius analysis: "What happens if I change UserService.authenticate?"

**Impact scoring** (0.0-1.0) across five dimensions:
- **Reach** (40%): How many units are affected (direct + indirect dependents)
- **Spread** (20%): How many files are touched
- **Depth** (15%): Transitive dependency chain length
- **Coupling** (15%): Edge type diversity (inherits > calls > imports)
- **Criticality** (10%): Unit type weight (interface > class > method)

Score classification: `>= 0.8` critical, `>= 0.5` high, `>= 0.25` moderate, `< 0.25` low.

## Migration Pipeline

MVP-centric migration with human approval gates:

```
Phase 1: Architecture  - Define target architecture and technology stack
Phase 2: Discovery     - Map source code, cluster into Functional MVPs
    |
    v  (per MVP, background analysis)
Phase 3: Transform     - Generate migrated code
Phase 4: Test          - Generate test suite for migrated code
```

Each phase requires explicit human approval before proceeding. MVP analysis runs as non-blocking background tasks with status polling.

## Deep Understanding

Automatic deep analysis of codebases via background daemon workers:

1. **Entry point detection** — Heuristic (zero incoming calls) + annotation patterns
2. **Call tree tracing** — Depth-10 traversal from each entry point through the ASG
3. **Tiered analysis** — Token budget tiers determine analysis depth per entry point
4. **Auto-trigger** — Analysis starts automatically after code ingestion completes

## UML Diagrams

Seven diagram types generated from ASG data:

| Type | Category | Data Source |
|------|----------|-------------|
| Class | Structural | ASG `contains`, `inherits`, `implements` edges |
| Package | Structural | File grouping + `imports` edges |
| Component | Structural | High-level module dependencies |
| Sequence | Behavioral | Deterministic from call tree data |
| Activity | Behavioral | Deterministic from call tree data |
| Use Case | Behavioral | Entry points + call tree |
| Deployment | Behavioral | LLM-assisted, grounded with ASG infrastructure |

Structural and behavioral (sequence, activity, use case) diagrams are fully deterministic — every arrow corresponds to a real code path. Deployment diagrams use LLM assistance.

## Configuration

Primary config: `config/codeloom.yaml`. Environment overrides via `.env`.

Key environment variables:

```bash
LLM_PROVIDER=ollama              # ollama|openai|anthropic|gemini|groq
EMBEDDING_PROVIDER=openai        # openai|huggingface
EMBEDDING_MODEL=text-embedding-3-small
PGVECTOR_EMBED_DIM=1536          # Must match embedding model
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
```

**Embedding dimension must match the model:**

| Model | Dimension |
|-------|-----------|
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| HuggingFace nomic-embed-text-v1.5 | 768 |

## Optional: Java & C# Enrichment

For deeper type resolution beyond tree-sitter:

```bash
./dev.sh setup-tools    # Builds JavaParser CLI + Roslyn Analyzer + downloads PlantUML
```

- **JavaParser CLI** — Requires JDK 11+ and Maven. Resolves generics, fully qualified types, constructor injection patterns.
- **Roslyn Analyzer** — Requires .NET SDK 8. Resolves interface implementations, nullable annotations, generic constraints.
- **PlantUML** — Auto-downloaded JAR for local diagram rendering (no external service dependency).

## API Routes

All routes prefixed with `/api`:

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/auth/login` | POST | Session authentication |
| `/api/projects` | GET, POST | List / create projects |
| `/api/projects/{id}` | GET, PATCH, DELETE | Project CRUD |
| `/api/projects/{id}/upload` | POST | Zip upload with ingestion |
| `/api/projects/{id}/ingest/git` | POST | Git clone ingestion |
| `/api/projects/{id}/ingest/local` | POST | Local directory ingestion |
| `/api/projects/{id}/build-asg` | POST | Build/rebuild ASG edges |
| `/api/projects/{id}/chat/stream` | POST | SSE streaming RAG chat |
| `/api/projects/{id}/files` | GET | List project files |
| `/api/projects/{id}/tree` | GET | Nested file tree |
| `/api/projects/{id}/units` | GET | List code units |
| `/api/migration/{plan_id}/...` | various | Migration plan lifecycle |
| `/api/diagrams/{plan_id}/...` | GET, POST | UML diagram generation |
| `/api/understanding/{project_id}/...` | various | Deep analysis jobs |
| `/api/settings` | GET, PATCH | Runtime configuration |
| `/api/health` | GET | Health check |

## Documentation

| Document | Description |
|----------|-------------|
| [Platform Guide](docs/platform-guide.md) | Comprehensive guide to all 12 subsystems |
| [Installation Guide](docs/installation-guide.md) | Setup including Java/C#/PlantUML tools |
| [Background Analysis](docs/background-analysis.md) | Architecture of 3 background worker patterns |
| [Deep Understanding Architecture](docs/deep-understanding-architecture.md) | Entry point detection, call tree tracing, tiered analysis |

## Development

```bash
./dev.sh local          # Start backend (:9005) + frontend (:3000)
./dev.sh stop           # Stop all services
./dev.sh status         # Check service status
./dev.sh setup-tools    # Build optional enrichment tools

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Frontend
cd frontend && npm install && npm run dev
npm run build           # Production build

# Tests
pytest -v -x
```

## License

Private repository. All rights reserved.
