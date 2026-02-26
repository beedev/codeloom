# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CodeLoom** is a code intelligence and migration platform powered by AST + ASG + RAG. Users upload entire codebases, query them via AI-powered chat, and migrate to new architectures through a multi-phase pipeline with agentic execution.

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
./dev.sh build          # Build frontend + sync deps + run migrations
./dev.sh setup-tools    # Build optional enrichment tools (JavaParser, Roslyn, PlantUML JAR)

# Frontend (separate terminal)
cd frontend && npm install && npm run dev    # :3000, proxies /api to :9005
npm run build           # Production build
npm run lint            # ESLint

# Database
alembic upgrade head
alembic revision --autogenerate -m "description"

# Tests
pytest                              # All tests
pytest -v -x                        # Verbose, stop on first failure
pytest codeloom/tests/              # All tests in directory
pytest codeloom/tests/test_ast_parser.py          # Single test file
pytest codeloom/tests/test_understanding/         # Understanding engine tests
```

**Default login**: admin / admin123

## Architecture

### Stack

- **Backend**: FastAPI + uvicorn (NOT Flask -- migrated from Flask)
- **Frontend**: React 19, Vite 7, Tailwind CSS 4, TypeScript, react-router-dom
- **Database**: PostgreSQL 17 + pgvector (SQLAlchemy 2.0 + Alembic)
- **LLM Framework**: LlamaIndex Core 0.14.x
- **AST Parsing**: tree-sitter (Python, JS/TS, Java, C#) + regex-based (ASP, JSP, VB.NET, SQL, XML Config, Properties) + optional JavaParser/Roslyn bridges
- **Diagrams**: PlantUML (local JAR preferred, HTTP server fallback) + optional Graphviz

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
- `stateless_query()` / `stateless_query_streaming()` -- thread-safe, multi-user safe, used by API routes
- `query()` / `switch_project()` -- mutates global state, single-user session mode

**Lazy Imports**: `codeloom/core/__init__.py` uses `__getattr__` with an `_IMPORT_MAP` to avoid pulling in heavy dependencies (torch, LlamaIndex, etc.) on simple imports.

**Plugin Architecture**: LLM/embedding/retrieval providers registered via `core/plugins.py` + `core/registry.py`. Selected by env vars: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `RETRIEVAL_STRATEGY`.

**LLM Gateway**: `codeloom/core/gateway.py` wraps all LLM calls as a `CustomLLM` subclass set as `Settings.llm`. All ~30+ call sites flow through transparently. Provides:
- Call logging (prompt/response length, latency, model)
- Token tracking extracted from provider responses
- Retry with exponential backoff (all providers)
- Per-call purpose tagging: `"migration"`, `"query"`, `"raptor"`, `"understanding"`
- Cost estimation by model (USD per 1M tokens)
- Thread-safe in-memory metrics via `LLMMetrics` (access via `pipeline.get_gateway_metrics()`)

### Core Components

| Component | Path | Purpose |
|-----------|------|---------|
| Pipeline | `codeloom/pipeline.py` | Central orchestrator -- manages LLM, embeddings, retrieval, caching |
| LLM Gateway | `codeloom/core/gateway.py` | Transparent LLM wrapper with logging, retry, cost tracking |
| FastAPI App | `codeloom/api/app.py` | App factory, CORS, session middleware, route registration |
| Dependencies | `codeloom/api/deps.py` | FastAPI Depends() injection (auth, services from app.state) |
| API Routes | `codeloom/api/routes/` | Auth, projects, code_chat, settings, migration, understanding, analytics |
| DB Models | `codeloom/core/db/models.py` | SQLAlchemy ORM (User, Project, CodeFile, CodeUnit, CodeEdge, MigrationPlan, etc.) |
| AST Parser | `codeloom/core/ast_parser/` | tree-sitter + regex parsers, `enricher.py` (semantic metadata) |
| Bridges | `codeloom/core/ast_parser/bridges/` | Optional subprocess bridges for deep Java (JavaParser) and C# (Roslyn) analysis |
| Code Chunker | `codeloom/core/code_chunker/` | AST-informed chunking with preamble injection (~1024 tokens/chunk) |
| Code Ingestion | `codeloom/core/ingestion/code_ingestion.py` | Upload -> extract -> AST parse -> chunk -> embed -> store pipeline |
| Vector Store | `codeloom/core/vector_store/pg_vector_store.py` | PGVectorStore wrapping pgvector |
| Retrieval Engine | `codeloom/core/engine/` | Hybrid retrieval (BM25 + vector + reranking) |
| RAPTOR | `codeloom/core/raptor/` | Hierarchical retrieval with background tree building |
| Stateless Query | `codeloom/core/stateless/` | Thread-safe fast retrieval functions for API routes |
| Migration Engine | `codeloom/core/migration/` | V1 (6-phase) and V2 (4-phase) migration pipelines |
| Migration Lanes | `codeloom/core/migration/lanes/` | Framework-specific migration logic (Struts->Spring, StoredProc->ORM, VB.NET->.NET Core) |
| Agentic Loop | `codeloom/core/migration/agent/` | Multi-turn LLM tool-calling for migration phases |
| Understanding | `codeloom/core/understanding/` | Deep analysis engine with tiered token budgets and entry-point detection |
| Diagrams | `codeloom/core/diagrams/` | PlantUML diagram generation (7 types: class, package, component, sequence, activity, use case, deployment) |
| ASG Builder | `codeloom/core/asg_builder/builder.py` | Builds Abstract Semantic Graph from code units |
| Config | `config/codeloom.yaml` | Unified YAML config (ingestion, retrieval, RAPTOR, migration, SQL chat) |
| Settings | `codeloom/setting/` | Settings loader merging YAML config + env vars |

### Migration Engine

Two pipeline versions coexist, selected per plan via `pipeline_version` field on `MigrationPlan`:

**V2 Pipeline (Default)** -- Architecture-first, 4-phase:
- Phase 1: Architecture (plan-level)
- Phase 2: Discovery (plan-level)
- Phase 3: Transform (per-MVP)
- Phase 4: Test (per-MVP)
- On-demand: `analyze_mvp()` for deep analysis (merges old Analyze+Design)

**V1 Pipeline (Legacy)** -- Discovery-first, 6-phase (backward compatible)

**Phase type resolution**: `get_phase_type(phase_number, version)` dynamically determines phase meaning.

**Approval gates**: Each phase has a human approval gate. Configurable via `approval_policy`:
- `"manual"`: User must approve after each phase
- `"auto"`: Auto-approve on success
- `"auto_non_blocking"`: Continue even if some MVPs fail

**Batch execution**: `batch_execute(plan_id, phase_number, mvp_ids, approval_policy, use_agent, max_agent_turns)` orchestrates multi-MVP execution.

### Agentic Migration Loop

`codeloom/core/migration/agent/loop.py` implements multi-turn LLM conversation with tool-calling for migration phases:
- **Tools**: Read source files, search codebase, look up documentation, validate syntax
- **Controls**: `max_agent_turns` (default 10), `use_agent` flag in batch execute requests
- **Streaming**: Yields `AgentEvent` objects (AgentStartEvent, ThinkingEvent, ToolCallEvent, OutputEvent) for SSE to frontend
- **Related files**: `agent/events.py` (event types), `agent/tools.py` (tool definitions)

### Migration Lanes

Plugin architecture for framework-specific migration logic in `codeloom/core/migration/lanes/`:
- **Registry**: `LaneRegistry` with class-level `@register` decorator
- **Implemented**: Struts->Spring Boot, Stored Procedures->ORM, VB.NET->.NET Core
- **Capabilities**: Deterministic transform rules (pattern->template), quality gates, LLM prompt augmentation, asset strategy overrides
- **Auto-detection**: `LaneRegistry.detect_lane(source_framework, target_stack)` returns best-matching lane with confidence score

**Quality Gates** (per lane):
- Gate categories: PARITY, COMPILE, UNIT_TEST, INTEGRATION, CONTRACT, REGRESSION
- Gates can be blocking (prevent continuation) or advisory
- Confidence scoring: per-transform `TransformResult.confidence` (0.0-1.0), aggregated via `aggregate_confidence()` and `confidence_tier()`

**Add a new lane**: Subclass `BaseLane` in `core/migration/lanes/`, use `@LaneRegistry.register` decorator, define `lane_id`, `source_frameworks`, `target_frameworks`, transform rules, and quality gates.

### Understanding Engine (Deep Analysis)

`codeloom/core/understanding/` provides deep codebase analysis:

**Tiered token budgets**:
- Tier 1: Full source if <= 100K tokens
- Tier 2: Depth-prioritized truncation if <= 200K tokens
- Tier 3: Summarization fallback (above 200K)

**Entry point detection**: Pass 1 (zero-incoming-calls heuristic) + Pass 2 (annotation pattern matching). Skips `test_*` functions.

**Background worker** (`worker.py`): poll_interval=15s, max_concurrent=2, heartbeat + stale job reclamation, exponential backoff retry (max_retries=2).

**Coverage metrics**: warn_below=50%, target=80%.

### Diagram Generation

`codeloom/core/diagrams/service.py` generates 7 diagram types:
- **Structural** (deterministic from ASG): Class, Package, Component
- **Behavioral** (deterministic from call tree): Sequence, Activity, UseCase
- **Behavioral** (LLM-assisted): Deployment

Output: PlantUML source + SVG rendering. Cached by default. Requires PlantUML JAR (installed via `./dev.sh setup-tools`) or HTTP server fallback.

### AST Parsers

| Parser Type | Languages | Implementation |
|-------------|-----------|----------------|
| tree-sitter | Python, JavaScript, TypeScript, Java, C# | Native AST, high fidelity |
| Regex-based | ASP.NET Web Forms, JSP, VB.NET, SQL, XML Config, Properties | Pattern matching |
| Bridges | Java (JavaParser), C# (Roslyn) | Optional subprocess for deep type resolution |
| Fallback | All other file types | Blank-line splitting |

### Frontend Structure

React SPA with route-based code splitting:
- `/login` -> `Login.tsx`
- `/` -> `Dashboard.tsx` (project list)
- `/project/:id` -> `ProjectView.tsx` (file tree, code browser)
- `/project/:id/chat` -> `CodeChatPage.tsx` (RAG chat)

Auth via `contexts/AuthContext.tsx` with session cookies. API client in `services/api.ts`.

Key frontend libraries: `react-force-graph-2d` (ASG visualization), `react-markdown` + `remark-gfm` (chat rendering), `react-pdf` (PDF viewing), `motion` (animations), `diff` (diff visualization).

### Data Model

Core tables in `core/db/models.py`: `users`, `projects`, `code_files`, `code_units`, `code_edges`, `migration_plans`, `migration_phases`, `conversations`, `query_logs`, `embedding_config`, `roles`, `user_roles`, `project_access`.

### API Routes

All routes prefixed with `/api`:
- **Auth**: `/api/auth/*` -- login, logout, session check
- **Projects**: `/api/projects/*` -- CRUD, zip upload with ingestion, file/unit browsing
- **Code Chat**: `/api/projects/{id}/query/stream` -- SSE streaming RAG chat
- **Settings**: `/api/settings/*` -- runtime configuration
- **Analytics**: `/api/projects/{id}/analytics` -- aggregated project metrics (code breakdown, migration progress, LLM usage, coverage)
- **Graph**: `/api/projects/{id}/graph/*` -- ASG queries (callers, callees, dependencies)
- **Migration**: `/api/migration/*` -- plan CRUD, asset inventory, discovery, MVP management, phase execution, batch execute/retry, diff/download
- **Understanding**: `/api/understanding/{project_id}/*` -- deep analysis (analyze, status, entry-points, results)
- **Diagrams**: `/api/migration/{plan_id}/mvps/{mvp_id}/diagrams/*` -- list, get/generate, refresh
- **Health**: `GET /api/health`

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

Additional notable env vars:
```bash
# Diagrams
PLANTUML_JAR_PATH=...            # Local JAR (preferred over server)
PLANTUML_SERVER_URL=...          # HTTP fallback

# Vision / Image generation
VISION_PROVIDER=...              # For image analysis
IMAGE_GENERATION_PROVIDER=...    # gemini|imagen

# RBAC
RBAC_STRICT_MODE=true            # Strict role-based access control

# LLM tuning
GROQ_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
CONTEXT_WINDOW=...               # Override default context window
CHAT_TOKEN_LIMIT=...             # Override chat token limit
```

## Key Defaults

- Backend: http://localhost:9005
- Frontend dev: http://localhost:3000 (Vite proxies `/api` to :9005)
- PostgreSQL: localhost:5432 (database: `codeloom_dev`)
- Embedding dimension: 1536 (OpenAI text-embedding-3-small)
- Reranker: `mixedbread-ai/mxbai-rerank-base-v1`

## Common Modifications

**Add new API route**: Create in `api/routes/`, register router in `api/app.py` via `app.include_router()`
**Add new service**: Initialize in `__main__.py`, attach to `app.state`, add dependency in `api/deps.py`
**Add AST parser for new language**: Subclass `BaseLanguageParser` in `core/ast_parser/`, register in `core/ast_parser/__init__.py`. Tree-sitter for high fidelity, regex-based for legacy/markup languages.
**Add migration lane**: Subclass `BaseLane` in `core/migration/lanes/`, use `@LaneRegistry.register`, define transforms and quality gates.
**Modify DB schema**: Edit `core/db/models.py`, run `alembic revision --autogenerate -m "description"`
**Frontend component**: Add to `frontend/src/components/`, types in `frontend/src/types/`

## Phased Delivery

- **Phase 1 (MVP)**: Code upload + AST parsing + basic RAG chat (Python) -- implemented
- **Phase 2**: ASG builder + relationship-aware retrieval + JS/TS support -- implemented
- **Phase 3**: Migration engine (6-phase pipeline) + Java support -- implemented
- **Phase 4**: C# parser + semantic enrichment + implements/overrides edges + JavaParser/Roslyn bridges -- implemented
- **Phase 4.5**: V2 migration pipeline, agentic loop, migration lanes, quality gates, LLM gateway, analytics API, understanding engine, diagram generation, ASP/JSP/VB.NET/SQL parsers -- implemented
- **Phase 5**: Diff views, test generation, VSCode extension

See `docs/architecture.md` for the full architecture document.

## Important Gotchas

- Thread safety: API routes must use `stateless_query*()` methods, never `pipeline.query()` directly
- The RAPTOR background worker uses asyncio -- it's disabled in multi-worker (Gunicorn) mode via `DISABLE_BACKGROUND_WORKERS=true`
- `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` are set at startup to prevent segfaults with torch/onnxruntime in multi-threaded contexts
- Node cache has 5-minute TTL -- call `pipeline.invalidate_node_cache(project_id)` after document changes
- The `FLASK_SECRET_KEY` env var name is legacy but still used for FastAPI's `SessionMiddleware`
- Migration V1 vs V2: Check `plan.pipeline_version` to determine phase meanings -- phase numbers map to different phase types depending on version
- LLM Gateway intercepts all LLM calls transparently -- never bypass `Settings.llm` by instantiating LLMs directly
- Understanding worker has max 2 concurrent jobs with stale job reclamation -- don't assume immediate execution
- Deep analysis tier selection is automatic based on token count -- don't override unless you understand the budget implications
