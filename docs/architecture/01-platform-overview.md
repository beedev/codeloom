<!-- Architecture Doc: Platform Overview -->

# Platform Overview

**Series**: CodeLoom Architecture Documentation
**Document**: 01 of 05
**Scope**: Full platform — runtime topology, technology choices, data model, plugin system, API surface, and frontend structure

---

## What is CodeLoom

CodeLoom is a code intelligence and migration platform that treats an entire codebase as a first-class queryable artifact. Engineers upload a project as a zip archive, and the platform parses every file with a language-aware Abstract Syntax Tree (AST) engine, constructs a typed Abstract Semantic Graph (ASG) of relationships between code units, embeds the resulting chunks into a pgvector store, and makes the whole corpus available for AI-powered chat and structured migration workflows. The result is a single environment where a team can answer deep questions about an unfamiliar codebase and plan, execute, and track a migration to a new architecture — all without leaving a browser tab.

---

## System Diagram

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   React SPA  (:3000)                         │
  │                                                              │
  │   /login         /          /project/:id   /project/:id/chat │
  │   Login.tsx  Dashboard.tsx  ProjectView.tsx  CodeChatPage.tsx│
  │                                                              │
  │   /migrations     /migration/:planId     /settings           │
  │   MigrationPlans  MigrationWizard        Settings            │
  │                                                              │
  │   AuthContext  ──  session cookie  ──  services/api.ts       │
  └───────────────────────────┬──────────────────────────────────┘
                              │  Vite dev proxy  /api → :9005
                              │  (production: reverse proxy)
  ┌───────────────────────────▼──────────────────────────────────┐
  │                 FastAPI + uvicorn  (:9005)                    │
  │                                                              │
  │  api/app.py  create_app()                                    │
  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
  │  │  /auth  │ │/projects │ │/query/   │ │  /migration     │  │
  │  │         │ │          │ │ stream   │ │  /understanding  │  │
  │  │         │ │          │ │  (SSE)   │ │  /graph /diagrams│  │
  │  └────┬────┘ └────┬─────┘ └────┬─────┘ └────────┬────────┘  │
  │       │           │            │                 │           │
  │       └───────────┴────────────┴─────────────────┘           │
  │                          │                                   │
  │           api/deps.py  Depends()  ←  app.state               │
  └───────────────────────────┬──────────────────────────────────┘
                              │
         ┌────────────────────┼───────────────────────┐
         │                    │                       │
  ┌──────▼──────┐   ┌─────────▼────────┐   ┌─────────▼────────┐
  │  Ingestion  │   │   Query Engine   │   │ Migration Engine  │
  │  Pipeline   │   │                  │   │                   │
  │             │   │  LocalRAGPipeline│   │  MigrationEngine  │
  │  AST Parser │   │  stateless_query │   │  6-phase pipeline │
  │  ASG Builder│   │  BM25 + vector   │   │  FunctionalMVP    │
  │  Code Chunker│  │  + reranking     │   │  clustering       │
  │  Embedder   │   │  RAPTOR tree     │   │                   │
  └──────┬──────┘   └─────────┬────────┘   └─────────┬────────┘
         │                    │                       │
         └────────────────────┼───────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────────┐
  │              PostgreSQL 17 + pgvector                        │
  │                                                              │
  │  users  projects  code_files  code_units  code_edges         │
  │  migration_plans  migration_phases  functional_mvps          │
  │  conversations  query_logs  embedding_config                 │
  │  roles  user_roles  project_access                           │
  │  deep_analysis_jobs  deep_analyses  analysis_units           │
  └──────────────────────────────────────────────────────────────┘

  Deep Understanding Engine (background worker)
  ┌──────────────────────────────────────────────────────────────┐
  │  UnderstandingEngine  ←  DeepAnalysisJob queue               │
  │  Entry point detection → call-tree tracing → tier analysis   │
  │  Results written to: deep_analyses, analysis_units           │
  └──────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend framework | FastAPI + uvicorn | Async HTTP, SSE streaming, OpenAPI docs |
| Frontend | React 19 + Vite 7 + TypeScript | Single-page application, route-based code splitting |
| Styling | Tailwind CSS 4 | Utility-first CSS, design tokens |
| Database | PostgreSQL 17 + pgvector | Relational storage and vector similarity search |
| ORM / migrations | SQLAlchemy 2.0 + Alembic | Declarative models, schema versioning |
| LLM framework | LlamaIndex Core 0.14.x | RAG orchestration, node management, retrieval fusion |
| AST parsing | tree-sitter | Language-aware parse trees for Python, JS/TS, Java, C# |
| Deep parsing bridges | JavaParser (Maven), Roslyn (.NET) | Optional subprocess bridges for richer semantic metadata |
| Embedding | OpenAI text-embedding-3-small (default) / HuggingFace | 1536-dim or 768-dim vectors stored in pgvector |
| Reranking | mixedbread-ai/mxbai-rerank-base-v1 | Cross-encoder re-scoring of candidate chunks |
| Session auth | Starlette SessionMiddleware | Server-side session cookie, no JWT |

---

## Service Architecture

### App Factory Pattern

The application is constructed through an explicit factory function rather than module-level singletons. This separation means that `__main__.py` initializes all heavy services before the application object exists, making startup sequencing deterministic and dependencies explicit.

The startup sequence is:

```
python -m codeloom
  └── __main__.py  main()
        │
        ├── 1. Set thread safety env vars
        │      OMP_NUM_THREADS=1, TOKENIZERS_PARALLELISM=false
        │      torch.set_num_threads(1), nest_asyncio.apply()
        │
        ├── 2. Initialize LocalRAGPipeline
        │      LLM provider, embedding model, PGVectorStore,
        │      RAPTOR background worker
        │
        ├── 3. Derive shared services from pipeline
        │      db_manager  ← pipeline._db_manager
        │      project_manager ← pipeline._project_manager
        │
        ├── 4. Initialize domain services
        │      CodeIngestionService(db_manager, vector_store)
        │      ConversationStore(db_manager)
        │
        ├── 5. Seed RBAC roles and default admin user
        │
        ├── 6. create_app(pipeline, db_manager, ...)
        │      Attaches all services to app.state
        │      Registers all routers with /api prefix
        │      Adds SessionMiddleware and CORSMiddleware
        │
        └── 7. uvicorn.run(app, host="0.0.0.0", port=9005)
```

### Dependency Injection

Route handlers do not import services directly. Instead, `api/deps.py` exposes a set of async dependency functions that extract the required service from `request.app.state`:

```
Route handler
  └── Depends(get_pipeline)
        └── async def get_pipeline(request: Request)
              └── return request.app.state.pipeline
```

Three additional services — `MigrationEngine`, `DiagramService`, and `UnderstandingEngine` — are initialized lazily on first request rather than at startup. This keeps startup time low for deployments that do not use those subsystems.

### Lazy Module Imports

`codeloom/core/__init__.py` uses a `__getattr__` hook with a declared `_IMPORT_MAP` to defer heavy imports (torch, LlamaIndex, ONNX) until the first access. This allows lightweight modules such as `codeloom.core.db.models` to be imported without triggering the full dependency chain.

### Pipeline Dual-Mode

`LocalRAGPipeline` exposes two query paths to handle concurrency safely:

- `stateless_query()` and `stateless_query_streaming()` — accept a `project_id` as an argument, build retrieval context on each call, and do not mutate shared state. All API routes use this path.
- `query()` and `switch_project()` — legacy single-user session mode that mutates internal state. Not used in the multi-user API.

---

## Data Model Overview

The entity-relationship diagram below shows table-level relationships. Arrow direction indicates the foreign key dependency (child points to parent).

```
  users ──────────────────────────────────────────────────┐
    │                                                      │
    ├── projects ─────────────────────────────────────┐   │
    │     │                                           │   │
    │     ├── code_files                              │   │
    │     │     └── code_units ─────────────────┐    │   │
    │     │                                     │    │   │
    │     ├── code_units (project FK)            │    │   │
    │     │                                     │    │   │
    │     ├── code_edges                        │    │   │
    │     │     ├── source_unit_id → code_units  │    │   │
    │     │     └── target_unit_id → code_units  │    │   │
    │     │                                     │    │   │
    │     ├── migration_plans                   │    │   │
    │     │     ├── migration_phases             │    │   │
    │     │     └── functional_mvps             │    │   │
    │     │           └── migration_phases      │    │   │
    │     │                                     │    │   │
    │     ├── conversations                     │    │   │
    │     ├── query_logs                        │    │   │
    │     │                                     │    │   │
    │     ├── deep_analysis_jobs                │    │   │
    │     │     └── deep_analyses               │    │   │
    │     │           └── analysis_units ───────┘    │   │
    │     │                                          │   │
    │     └── project_access                         │   │
    │                                                │   │
    ├── user_roles → roles                           │   │
    ├── conversations ────────────────────────────────┘   │
    ├── query_logs ───────────────────────────────────────┘
    └── project_access

  embedding_config  (singleton row — tracks active embedding model)
  roles             (admin, editor, viewer — seeded at startup)
```

Key design decisions in the schema:

- `code_units.qualified_name` stores the dotted path (e.g., `com.example.service.UserService.createUser`) to support cross-language relationship resolution in the ASG builder.
- `code_edges` carries an `edge_type` column (`contains`, `imports`, `calls`, `inherits`, `implements`, `overrides`, `type_dep`) which the query engine uses for graph expansion during retrieval.
- `functional_mvps` groups code units into migration slices discovered by the Phase 1 clustering algorithm; each MVP tracks its own migration phase progression independently.
- `deep_analyses` stores a full `DeepContextBundle` as JSONB plus a pre-rendered narrative string that the chat route injects directly into the LLM context window.

---

## Plugin System

All LLM, embedding, and retrieval providers are registered through a central `PluginRegistry` in `codeloom/core/registry.py`. The registration step happens inside `register_default_plugins()` which is called lazily by any of the `get_configured_*` helper functions in `codeloom/core/plugins.py`.

```
PluginRegistry
  ├── LLM providers:        ollama | openai | anthropic | groq
  ├── Embedding providers:  huggingface (openai delegated via env)
  ├── Retrieval strategies: hybrid | semantic | keyword
  ├── Image providers:      gemini
  └── Vision providers:     gemini | openai
```

Provider selection is driven entirely by environment variables. No code change is required to swap providers:

| Variable | Controls | Example values |
|---|---|---|
| `LLM_PROVIDER` | Which LLM class is instantiated | `ollama`, `openai`, `anthropic`, `groq` |
| `LLM_MODEL` | Model name passed to the provider | `llama3.1:latest`, `gpt-4o` |
| `EMBEDDING_PROVIDER` | Embedding backend | `huggingface`, `openai` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `RETRIEVAL_STRATEGY` | Retrieval algorithm | `hybrid`, `semantic`, `keyword` |

The `HybridRetrievalStrategy` (default) fuses BM25 keyword retrieval and dense vector retrieval with configurable weights, then applies the cross-encoder reranker to produce a final ranked list. The weights default to `[0.5, 0.5]` and are overridable per-request through `QueryTimeSettings`.

---

## Configuration

Configuration flows from three sources, applied in order of increasing specificity:

```
config/codeloom.yaml           (static defaults — checked into source control)
        ↓
.env / environment variables   (deployment-specific values — not checked in)
        ↓
QueryTimeSettings              (per-request overrides from the chat UI)
```

`config/codeloom.yaml` is the single source of truth for subsystem defaults: ingestion chunking parameters, retrieval weights, RAPTOR tree-building settings, reranker model selection, LLM sampling parameters, and migration deep-analysis thresholds. The `codeloom/setting/setting.py` module loads the YAML via `codeloom/core/config/config_loader.py` and exposes a `RAGSettings` Pydantic model. `get_settings()` returns a cached singleton.

`QueryTimeSettings` handles per-request tuning from the frontend chat controls (search style slider, result depth selector, temperature slider). These map to `bm25_weight`, `vector_weight`, `similarity_top_k`, and `temperature` fields passed alongside each query. They are validated but not persisted.

The `migration.llm_overrides` section in the YAML allows routing different migration phases to different LLM providers (for example, using a larger reasoning model for analysis and a faster model for code generation) without touching application code.

---

## Authentication and RBAC

Authentication uses server-side sessions backed by Starlette's `SessionMiddleware`. The session secret is read from the `FLASK_SECRET_KEY` environment variable (the name is a legacy artifact from the project's DBNotebook origin).

On login, the server writes `user_id`, `username`, and `roles` into the encrypted session cookie. Subsequent requests present the cookie; `api/deps.py` extracts the session fields and constructs the user context dict without a database round-trip.

RBAC is implemented through three tables:

```
roles            — named roles with a JSONB permissions array
user_roles       — many-to-many join between users and roles
project_access   — per-project access grants (owner | editor | viewer)
```

Three built-in roles are seeded at startup: `admin` (full access including user management), `editor` (assigned project access with edit rights), and `viewer` (read-only access to assigned projects). The default admin user (`admin` / `admin123`) receives the `admin` role automatically on first boot.

Route protection is enforced through dependency functions in `api/deps.py`:
- `get_current_user` — requires a valid session or raises HTTP 401
- `require_editor` — requires `admin` or `editor` role or raises HTTP 403
- `require_admin` — requires `admin` role or raises HTTP 403

---

## API Surface

All routes are registered under the `/api` prefix. The table below lists each route group, its router module, and its primary responsibility.

| Route group | Module | Responsibility |
|---|---|---|
| `/api/auth/*` | `routes/fastapi_auth.py` | Login, logout, session check, password change |
| `/api/projects/*` | `routes/projects.py` | Project CRUD, zip upload + ingestion, file tree, code unit browsing |
| `/api/projects/{id}/query/stream` | `routes/code_chat.py` | SSE streaming RAG chat with conversation history |
| `/api/settings/*` | `routes/fastapi_settings.py` | Runtime configuration read and update |
| `/api/graph/*` | `routes/graph.py` | ASG node and edge queries for graph visualization |
| `/api/migration/*` | `routes/migration.py` | Migration plan CRUD, phase execution, MVP management |
| `/api/understanding/*` | `routes/understanding.py` | Deep analysis job submission and result retrieval |
| `/api/diagrams/*` | `routes/diagrams.py` | UML diagram generation (sequence, class, component) |
| `GET /api/health` | `app.py` inline | Liveness probe — returns `{"status": "ok"}` |

The `/api/projects/{id}/query/stream` endpoint uses Server-Sent Events (SSE). The client receives a stream of `data:` lines; each line carries a partial response token or a structured metadata event. The stream terminates with a `[DONE]` sentinel. This pattern avoids WebSocket complexity while supporting long-running LLM responses.

---

## Frontend Architecture

### Route Map

```
BrowserRouter
  └── AuthProvider (session state, auth API calls)
        └── AppRoutes
              ├── /login                 Login.tsx           (public)
              ├── /                      Dashboard.tsx        (protected)
              ├── /project/:id           ProjectView.tsx      (protected)
              ├── /project/:id/chat      CodeChatPage.tsx     (protected)
              ├── /migrations            MigrationPlans.tsx   (protected)
              ├── /migration/:planId     MigrationWizard.tsx  (protected)
              ├── /settings              Settings.tsx         (protected)
              └── *                      → redirect to /
```

All protected routes are wrapped in a `ProtectedRoute` guard component that reads authentication state from `AuthContext`. If the session check returns unauthenticated, the user is redirected to `/login` with the intended destination preserved in router state.

### Component Hierarchy

```
App
  AuthProvider
    AppRoutes
      Dashboard
        ProjectCard (list of projects)
        NewProjectModal (zip upload)
      ProjectView
        FileTree
        CodeUnitPanel
        ASGGraphViewer
      CodeChatPage
        ChatInput
        MessageList
          AssistantMessage (streaming partial render)
          SourcePanel (retrieved code citations)
      MigrationWizard
        PhaseStepper
        MVPList
        TransformOutputViewer
```

### State Management

The frontend does not use a global state library such as Redux. State is managed at two levels:

- **AuthContext** — wraps the entire application and holds session state (`isAuthenticated`, `user`, `isLoading`). It makes a single `/api/auth/me` call on mount to rehydrate state after page reload.
- **Page-level hooks** — each page component manages its own data fetching and local state with `useState` and `useEffect`. The API client in `services/api.ts` handles fetch calls, response parsing, and error normalization.

### SSE Streaming Pattern

The chat page opens an `EventSource`-compatible fetch with the `Accept: text/event-stream` header. As the server emits partial tokens, the component appends each chunk to a buffer and re-renders incrementally. A `useRef`-based scroll anchor keeps the latest token in view. On stream close, the component commits the final assembled message to conversation state and requests the updated conversation history.

---

## Cross-References

This document describes the platform topology. The following documents cover each major subsystem in depth:

- **02 — Ingestion Pipeline**: `docs/architecture/02-ingestion-pipeline.md`
  Upload handling, AST parsing per language, ASG edge construction, code chunking, embedding, and pgvector storage.

- **03 — Query Engine**: `docs/architecture/03-query-engine.md`
  Hybrid retrieval (BM25 + vector), RAPTOR hierarchical tree, reranking, ASG graph expansion, stateless query path, and SSE response streaming.

- **04 — Migration Engine**: `docs/architecture/04-migration-engine.md`
  Six-phase pipeline, functional MVP clustering, per-MVP phase execution, LLM routing overrides, and migration lane detection.

- **05 — Deep Understanding Engine**: `docs/architecture/05-deep-understanding.md`
  Entry point detection, call-tree tracing, analysis tiers, background job worker, `DeepContextBundle` schema, and chat context injection.
