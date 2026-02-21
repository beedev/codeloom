# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CodeLoom** is a code intelligence and migration platform powered by AST + ASG + RAG. Users upload entire codebases, query them via AI-powered chat, and migrate to new architectures through a 6-phase pipeline.

**Forked from**: DBNotebook (dbn-v2). Reuses LLM providers, embeddings, pgvector, hybrid retrieval, reranking, RAPTOR, auth, and SSE streaming. All Python imports renamed from `dbnotebook` to `codeloom`.

## Development Commands

```bash
# First-time setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew services start postgresql@17
createdb codeloom_dev

# Copy .env.example to .env, then:
./dev.sh local          # Starts backend (:9005) + frontend (:3000)
./dev.sh stop           # Stop all services
./dev.sh status         # Check service status
./dev.sh setup-tools    # Build optional Java/C# enrichment tools (requires JDK+Maven / .NET SDK)

# Frontend (separate terminal)
cd frontend && npm install && npm run dev    # :3000, proxies /api to :9005
npm run build           # Production build
npm run lint            # ESLint

# Database
alembic upgrade head
alembic revision --autogenerate -m "description"

# Tests
pytest                  # All tests
pytest -v -x            # Verbose, stop on first failure
pytest codeloom/tests/  # Test directory
```

**Default login**: admin / admin123

## Architecture

### Stack

- **Backend**: FastAPI + uvicorn (NOT Flask — migrated from Flask)
- **Frontend**: React 19, Vite 7, Tailwind CSS 4, TypeScript, react-router-dom
- **Database**: PostgreSQL 17 + pgvector (SQLAlchemy 2.0 + Alembic)
- **LLM Framework**: LlamaIndex Core 0.14.x
- **AST Parsing**: tree-sitter (Python, JS/TS, Java, C#) + optional JavaParser/Roslyn bridges

### Entry Point and Startup

`python -m codeloom` (`codeloom/__main__.py`) orchestrates startup:
1. Initializes `LocalRAGPipeline` (LLM, embeddings, vector store, RAPTOR worker)
2. Creates `DatabaseManager`, `ProjectManager`, `CodeIngestionService`, `ConversationStore`
3. Calls `create_app()` in `codeloom/api/app.py` to build the FastAPI application
4. Injects all services onto `app.state` for dependency injection
5. Runs uvicorn on port 9005

### Request Flow

```
React Frontend (:3000) --proxy /api--> FastAPI (:9005) --> Pipeline/Services
                                          |
                                     api/deps.py (FastAPI Depends())
                                     extracts services from app.state
```

### Key Architectural Patterns

**Dependency Injection**: Services are stored on `app.state` (pipeline, db_manager, project_manager, code_ingestion, conversation_store) and accessed via `api/deps.py` dependency functions with FastAPI's `Depends()`.

**Pipeline Dual-Mode**: `LocalRAGPipeline` has two query patterns:
- `stateless_query()` / `stateless_query_streaming()` — thread-safe, multi-user safe, used by API routes
- `query()` / `switch_project()` — mutates global state, single-user session mode

**Lazy Imports**: `codeloom/core/__init__.py` uses `__getattr__` with an `_IMPORT_MAP` to avoid pulling in heavy dependencies (torch, LlamaIndex, etc.) on simple imports.

**Plugin Architecture**: LLM/embedding/retrieval providers registered via `core/plugins.py` + `core/registry.py`. Selected by env vars: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `RETRIEVAL_STRATEGY`.

### Core Components

| Component | Path | Purpose |
|-----------|------|---------|
| Pipeline | `codeloom/pipeline.py` | Central orchestrator — manages LLM, embeddings, retrieval, caching |
| FastAPI App | `codeloom/api/app.py` | App factory, CORS, session middleware, route registration |
| Dependencies | `codeloom/api/deps.py` | FastAPI Depends() injection (auth, services from app.state) |
| API Routes | `codeloom/api/routes/` | `fastapi_auth`, `projects`, `code_chat`, `fastapi_settings` |
| DB Models | `codeloom/core/db/models.py` | SQLAlchemy ORM (User, Project, CodeFile, CodeUnit, CodeEdge, MigrationPlan, etc.) |
| AST Parser | `codeloom/core/ast_parser/` | tree-sitter parsing: `base.py` (Strategy pattern), language-specific parsers (Python, JS, TS, Java, C#), `enricher.py` (semantic metadata) |
| Bridges | `codeloom/core/ast_parser/bridges/` | Optional subprocess bridges for deep Java (JavaParser) and C# (Roslyn) semantic analysis |
| Code Chunker | `codeloom/core/code_chunker/` | AST-informed chunking with preamble injection (~1024 tokens/chunk) |
| Code Ingestion | `codeloom/core/ingestion/code_ingestion.py` | Upload → extract → AST parse → chunk → embed → store pipeline |
| Vector Store | `codeloom/core/vector_store/pg_vector_store.py` | PGVectorStore wrapping pgvector |
| Retrieval Engine | `codeloom/core/engine/` | Hybrid retrieval (BM25 + vector + reranking) |
| RAPTOR | `codeloom/core/raptor/` | Hierarchical retrieval with background tree building |
| Stateless Query | `codeloom/core/stateless/` | Thread-safe fast retrieval functions for API routes |
| Config | `config/codeloom.yaml` | Unified YAML config (ingestion, retrieval, RAPTOR, SQL chat) |
| Settings | `codeloom/setting/` | Settings loader merging YAML config + env vars |

### Frontend Structure

React SPA with route-based code splitting:
- `/login` → `Login.tsx`
- `/` → `Dashboard.tsx` (project list)
- `/project/:id` → `ProjectView.tsx` (file tree, code browser)
- `/project/:id/chat` → `CodeChatPage.tsx` (RAG chat)

Auth via `contexts/AuthContext.tsx` with session cookies. API client in `services/api.ts`.

### Data Model (Implemented)

Core tables in `core/db/models.py`: `users`, `projects`, `code_files`, `code_units`, `code_edges`, `migration_plans`, `migration_phases`, `conversations`, `query_logs`, `embedding_config`, `roles`, `user_roles`, `project_access`.

### API Routes (Implemented)

All routes prefixed with `/api`:
- **Auth**: `/api/auth/*` — login, logout, session check
- **Projects**: `/api/projects/*` — CRUD, zip upload with ingestion, file/unit browsing
- **Code Chat**: `/api/projects/{id}/query/stream` — SSE streaming RAG chat
- **Settings**: `/api/settings/*` — runtime configuration
- **Health**: `GET /api/health`

## Environment Configuration

Copy `.env.example` to `.env`. Critical variables:

```bash
LLM_PROVIDER=ollama              # ollama|openai|anthropic|gemini|groq
EMBEDDING_PROVIDER=openai        # openai|huggingface
EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
PGVECTOR_EMBED_DIM=1536          # Must match embedding model (1536=OpenAI, 768=nomic)
FLASK_SECRET_KEY=change-me       # Used for session middleware (name is legacy)
```

## Key Defaults

- Backend: http://localhost:9005 (not 7860 — updated from DBNotebook)
- Frontend dev: http://localhost:3000 (Vite proxies `/api` to :9005)
- PostgreSQL: localhost:5432 (database: `codeloom_dev`)
- Embedding dimension: 1536 (OpenAI text-embedding-3-small)
- Reranker: `mixedbread-ai/mxbai-rerank-base-v1`

## Common Modifications

**Add new API route**: Create in `api/routes/`, register router in `api/app.py` via `app.include_router()`
**Add new service**: Initialize in `__main__.py`, attach to `app.state`, add dependency in `api/deps.py`
**Add AST parser for new language**: Subclass `BaseLanguageParser` in `core/ast_parser/`, register in `core/ast_parser/__init__.py`
**Modify DB schema**: Edit `core/db/models.py`, run `alembic revision --autogenerate -m "description"`
**Frontend component**: Add to `frontend/src/components/`, types in `frontend/src/types/`

## ASG Edge Types

The ASG builder (`core/asg_builder/builder.py`) detects these relationship types between code units:

| Edge Type | Meaning | Source |
|-----------|---------|--------|
| `contains` | Parent contains child (class -> method) | Structural nesting |
| `imports` | File imports another file/module | Import statements (Python, JS/TS, Java, C#) |
| `calls` | Function/method calls another | Call detection (regex + qualified) |
| `inherits` | Class extends another class | `extends` metadata or signature regex |
| `implements` | Class/struct implements interface | `implements` metadata from parser |
| `overrides` | Method overrides parent class method | `@Override` / `override` modifier |
| `type_dep` | Consumer depends on referenced type | Structured metadata: field types, param types, return types |

**Enrichment layers**: (1) tree-sitter enricher (`enricher.py`) runs on all files, adding `parsed_params`, `return_type`, `modifiers`, and `fields` (class field declarations) to metadata. (2) Optional bridges (`bridges/`) provide deeper type resolution when Java/dotnet runtimes are available. Build with `./dev.sh setup-tools`.

## Phased Delivery

- **Phase 1 (MVP)**: Code upload + AST parsing + basic RAG chat (Python) — implemented
- **Phase 2**: ASG builder + relationship-aware retrieval + JS/TS support — implemented
- **Phase 3**: Migration engine (6-phase pipeline) + Java support — implemented
- **Phase 4**: C# parser + semantic enrichment + implements/overrides edges + JavaParser/Roslyn bridges — implemented
- **Phase 5**: Diff views, test generation, VSCode extension

See `docs/architecture.md` for the full architecture document.

## Important Gotchas

- Thread safety: API routes must use `stateless_query*()` methods, never `pipeline.query()` directly
- The RAPTOR background worker uses asyncio — it's disabled in multi-worker (Gunicorn) mode via `DISABLE_BACKGROUND_WORKERS=true`
- `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` are set at startup to prevent segfaults with torch/onnxruntime in multi-threaded contexts
- Node cache has 5-minute TTL — call `pipeline.invalidate_node_cache(project_id)` after document changes
- The `FLASK_SECRET_KEY` env var name is legacy but still used for FastAPI's `SessionMiddleware`
