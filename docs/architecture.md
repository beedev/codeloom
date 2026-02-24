# CodeLoom Architecture

> Code intelligence and migration platform powered by AST + ASG + RAG

---

## 1. Vision

CodeLoom enables developers to upload an entire codebase, understand it
through AI-powered code intelligence, and migrate it to a new architecture
or tech stack with confidence.

**Core capabilities**:
- **Code RAG** - Upload a codebase, ask questions, get code snippets with context
- **Code Intelligence** - AST parsing + ASG relationship mapping + deep understanding narratives
- **Code Migration** - MVP-centric pipeline with agentic LLM execution and human approval gates
- **Observability** - LLM gateway with cost tracking, retry, and per-purpose analytics

**Supported languages**: Python, JavaScript/TypeScript, Java, C#/.NET (via
tree-sitter), VB.NET, SQL, JSP, ASPX, XML config, Properties (via regex
parsers) -- 12 languages/file types total.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      React Frontend (:3000)                           │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │ Code Chat │ │ Project  │ │   Migration    │ │  Project Wiki  │   │
│  │ (RAG)     │ │ View +   │ │   Wizard +     │ │  & Analytics   │   │
│  │           │ │ ASG Graph│ │   Agent Panel  │ │  Dashboard     │   │
│  └─────┬─────┘ └────┬─────┘ └───────┬────────┘ └───────┬────────┘   │
└────────┼────────────┼───────────────┼───────────────────┼────────────┘
         │            │               │                   │
         ▼            ▼               ▼                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FastAPI (:9005)                                  │
│  /api/projects  /api/.../query  /api/migration  /api/understanding   │
│  /api/auth      /api/graph      /api/settings   /api/.../analytics   │
└──────┬──────────────┬───────────────┬───────────────────┬────────────┘
       │              │               │                   │
       ▼              ▼               ▼                   ▼
┌──────────────┐ ┌──────────┐ ┌─────────────────┐ ┌────────────────┐
│ Code         │ │ Code RAG │ │ Migration       │ │ Deep           │
│ Ingestion    │ │ Pipeline │ │ Engine          │ │ Understanding  │
│              │ │          │ │                 │ │                │
│ Zip Upload   │ │ Hybrid   │ │ V2 Pipeline:   │ │ Entry Point    │
│ tree-sitter  │ │ Retrieval│ │ 1. Architecture│ │ Detection      │
│ + regex      │ │ (BM25+   │ │ 2. Discovery   │ │ Call Chain     │
│ AST Parse    │ │  Vector) │ │ 3. Transform   │ │ Tracing        │
│ Semantic     │ │ + RAPTOR │ │ 4. Test        │ │ Tiered LLM     │
│ Enrichment   │ │ + Rerank │ │                │ │ Analysis       │
│ ASG Build    │ │ + Deep   │ │ Agentic Loop   │ │ Narrative      │
│ Chunking     │ │ Analysis │ │ (10 tools,     │ │ Generation     │
│ Embedding    │ │ Context  │ │  multi-turn)   │ │                │
└──────┬───────┘ └────┬─────┘ └───────┬─────────┘ └───────┬────────┘
       │              │               │                    │
       │              ▼               ▼                    │
       │        ┌──────────────────────────────────┐       │
       │        │         LLM Gateway              │       │
       │        │  Retry + Metrics + Cost Tracking │◄──────┘
       │        │  (wraps all LLM providers)       │
       │        └──────────────┬───────────────────┘
       │                       │
       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL + pgvector                              │
│ ┌──────────────┐ ┌───────────┐ ┌────────────┐ ┌───────────────────┐ │
│ │data_embeddings│ │code_units │ │code_edges  │ │migration_plans    │ │
│ │(code vectors) │ │(AST nodes)│ │(ASG graph) │ │functional_mvps    │ │
│ └──────────────┘ └───────────┘ └────────────┘ │migration_phases   │ │
│ ┌──────────────┐ ┌───────────┐ ┌────────────┐ │deep_analysis_jobs │ │
│ │conversations │ │code_files │ │query_logs  │ │deep_analyses      │ │
│ └──────────────┘ └───────────┘ └────────────┘ └───────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 AST Parser (`core/ast_parser/`)

Parses source code into structural units. Two-tier architecture:

| Tier | Languages | Parser | Metadata Depth |
|------|-----------|--------|----------------|
| Tree-sitter | Python, JavaScript, TypeScript, Java, C# | `BaseLanguageParser` subclasses | Full AST + semantic enrichment |
| Regex-based | VB.NET, SQL, JSP, ASPX, XML config, Properties | Custom parsers | Pattern-extracted metadata |

**Semantic Enrichment** (`enricher.py`): Second-pass AST walk extracting
`parsed_params` (name + type), `return_type`, `modifiers`, `fields`,
`is_async`, `is_override`, `annotations`. Feeds ASG edge detection and
migration type generation.

**Optional Bridges** (`bridges/`): JavaParser (JVM) and Roslyn (.NET SDK)
for deep type resolution. Built via `./dev.sh setup-tools`.

### 3.2 ASG Builder (`core/asg_builder/`)

Builds a semantic relationship graph from parsed code units.

**Edge types** (8 core + framework-specific):

| Module | Edges | Detection Strategy |
|--------|-------|--------------------|
| `structural.py` | `contains`, `imports`, `calls` | Nesting, import statements, identifier intersection |
| `oop.py` | `inherits`, `implements`, `overrides`, `type_dep` | Metadata + regex + inheritance chain walking |
| `stored_proc.py` | `calls_sp` | Language-specific SP invocation patterns |
| `struts.py` | `struts_action_*`, `jsp_includes` | Struts XML config + JSP tag parsing |

`EdgeContext` (context.py) pre-indexes all units by name, qualified name,
and file for O(1) resolution during edge detection.

### 3.3 Code Chunker (`core/code_chunker/`)

AST-informed chunking with preamble injection. Each `CodeUnit` becomes one
chunk with file path, imports, and parent class prepended. Token counting
via tiktoken (`cl100k_base`), default 1024 tokens/chunk. Oversized units
split at blank line boundaries with preamble replicated.

### 3.4 Code RAG Pipeline (`core/engine/`, `core/raptor/`, `core/stateless/`)

Hybrid retrieval stack:

```
User Query
    │
    ▼
BM25 sparse search (keyword matching)
    + Vector search via pgvector (semantic similarity)
    │
    ▼
Reranking (mxbai-rerank-base-v1)
    │
    ▼
RAPTOR hierarchical summaries (L0: units, L1: files, L2+: clusters)
    │
    ▼
Deep understanding narratives (when available)
    │
    ▼
LLM Generation → SSE streaming response
```

Two API patterns: `stateless_query*()` (thread-safe, multi-user) and
`query()` (single-user session mode). Node cache with 5-minute TTL.
Session memory up to 100 messages, 24-hour TTL.

### 3.5 LLM Gateway (`core/gateway.py`)

Transparent observability proxy wrapping all LLM providers:

- **Retry**: Exponential backoff (1s, 2s, 4s) for rate limits and timeouts
- **Metrics**: Per-call latency, token counts (real or estimated), cost
- **Purpose tagging**: `migration`, `migration_agent`, `understanding`,
  `raptor`, `query`, `general`
- **Cost estimation**: 45+ models with USD pricing across all providers
- **Thread-safe**: Lock-protected in-memory counters

Subclasses LlamaIndex `CustomLLM` so it sets as `Settings.llm` --
all 30+ call sites flow through automatically with zero code changes.

### 3.6 Deep Understanding Engine (`core/understanding/`)

Background analysis system:

```
ChainTracer → detect entry points (HTTP, CLI, scheduled, events)
    │
    ▼
ChainTracer → trace call chains via ASG (max depth 5)
    │
    ▼
ChainAnalyzer → tiered LLM analysis:
    Tier 1 (<=100K tokens): Full source
    Tier 2 (100-200K): Full shallow + signatures deep
    Tier 3 (>200K): LLM-summarized branches
    │
    ▼
DeepContextBundle → narrative, business_rules, data_entities,
    integrations, side_effects, evidence_refs, confidence, coverage
```

**Worker**: Background daemon thread with asyncio loop, distributed lease
protocol (`FOR UPDATE SKIP LOCKED`), 30s heartbeat, 120s stale reclaim.

**Downstream**: Narratives feed into RAG chat context and migration phase
prompts via `MigrationContextBuilder.get_deep_analysis_context()`.

### 3.7 Migration Engine (`core/migration/`)

MVP-centric pipeline with two execution modes.

**Pipeline V2 (default)**:

```
Plan-level:    Phase 1 (Architecture) → Phase 2 (Discovery/Clustering)
Per-MVP:       Phase 3 (Transform) → Phase 4 (Test)
On-demand:     analyze_mvp() for deep analysis of individual MVPs
```

**Execution modes**:

| Mode | Mechanism | Use Case |
|------|-----------|----------|
| Single-shot | One LLM call with pre-built context | Fast, simple MVPs |
| Agentic | Multi-turn tool-use loop (max 10 turns) | Complex MVPs needing exploration |

**Agentic tool arsenal** (10 tools, phase-gated):

| Tool | Purpose |
|------|---------|
| `get_source_code` | MVP units ordered by connectivity (12K default budget) |
| `read_source_file` | Full file content for reference context |
| `get_unit_details` | Enriched metadata (params, types, modifiers) |
| `get_functional_context` | Business rules, entities, integrations |
| `get_dependencies` | Cross-boundary ASG edges (blast radius) |
| `get_module_graph` | File-level import graph |
| `get_deep_analysis` | Deep understanding narratives for MVP |
| `search_codebase` | Semantic RAG search across full project |
| `lookup_framework_docs` | Context7 primary, Tavily fallback |
| `validate_syntax` | tree-sitter parse check on generated code |

**Agent events** stream via SSE: `ThinkingEvent`, `ToolCallEvent`,
`ToolResultEvent`, `OutputEvent`, `AgentDoneEvent`, `ErrorEvent`.

**Supporting infrastructure**:
- `MvpClusterer`: RAPTOR-driven or package-based clustering with
  cohesion/coupling from ASG edges
- `MigrationContextBuilder`: Phase-specific context assembly within token
  budgets, integrating source code, ASG edges, functional context, deep
  analysis narratives, and framework patterns
- `MigrationLane` ABC: Pluggable framework-specific migration lanes
  (Struts→Spring Boot, StoredProc→ORM, VB.NET→.NET Core)
- Quality gates, confidence scoring, retry with checkpoints, batch
  execution with configurable approval policies

---

## 4. Data Model

### Core Tables

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| `projects` | name, languages, ast_status, asg_status, deep_analysis_status | Project container |
| `code_files` | file_path, language, line_count, raptor_status | Source files |
| `code_units` | unit_type, qualified_name, signature, source, unit_metadata (JSONB) | Parsed code elements |
| `code_edges` | source_unit_id, target_unit_id, edge_type, edge_metadata | ASG relationships |
| `migration_plans` | target_brief, target_stack, pipeline_version, lane_versions, migration_lane_id | Migration configuration |
| `migration_phases` | phase_type, status, output, approved, phase_metadata (JSONB), run_id | Pipeline phase records |
| `functional_mvps` | unit_ids, cohesion_score, coupling_score, migration_readiness, analysis_output | MVP groupings |
| `deep_analysis_jobs` | project_id, status, total/completed entry_points, retry_count | Background analysis jobs |
| `deep_analyses` | entry_unit_id, tier, narrative, result_json (JSONB), confidence, coverage | Entry point analyses |
| `analysis_units` | analysis_id, unit_id, min_depth, path_count | Units in analysis chains |

### Reused Tables (from DBNotebook)

- `users`, `roles`, `user_roles`, `project_access` -- Auth + RBAC
- `data_embeddings` -- pgvector storage for code chunk embeddings
- `conversations` -- Chat history per project
- `query_logs` -- Token usage tracking
- `embedding_config` -- Embedding model configuration

---

## 5. API Surface

All routes prefixed with `/api`.

### Auth & Projects
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Session login |
| POST | `/api/auth/logout` | Session logout |
| GET | `/api/auth/session` | Session check |
| GET | `/api/projects` | List user's projects |
| POST | `/api/projects/upload` | Upload zip codebase (triggers ingestion) |
| GET | `/api/projects/{id}` | Project details + stats |
| DELETE | `/api/projects/{id}` | Delete project |
| GET | `/api/projects/{id}/files` | File tree |
| GET | `/api/projects/{id}/units` | Browse code units |
| POST | `/api/projects/{id}/build-asg` | Trigger ASG building |

### Code Chat (RAG)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/{id}/query/stream` | SSE streaming RAG chat |

### Migration
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/migration/plan` | Create plan (source + target) |
| GET | `/api/migration/{id}/asset-inventory` | File-type breakdown + strategies |
| POST | `/api/migration/{id}/discover` | Run clustering + create MVPs |
| GET | `/api/migration/{id}/mvps` | List MVPs with metrics |
| POST | `/api/migration/{id}/mvps/merge` | Merge MVPs |
| POST | `/api/migration/{id}/mvps/{mid}/split` | Split MVP |
| POST | `/api/migration/{id}/phase/N/execute` | Run phase (single-shot or agentic) |
| POST | `/api/migration/{id}/phase/N/approve` | Human approval gate |
| POST | `/api/migration/{id}/phase/N/reject` | Reject and reset phase |
| GET | `/api/migration/{id}/phase/N/diff-context` | Source vs. migrated |
| GET | `/api/migration/{id}/phase/N/download` | Download generated files |
| POST | `/api/migration/{id}/batch/execute` | Batch phase execution |
| GET | `/api/migration/{id}/batch/{bid}/status` | Batch progress |
| GET | `/api/migration/{id}/scorecard` | Plan-level quality metrics |

### Deep Understanding
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/understanding/{id}/analyze` | Start background analysis job |
| GET | `/api/understanding/{id}/status/{job_id}` | Job progress |
| GET | `/api/understanding/{id}/entry-points` | Preview detectable entry points |
| GET | `/api/understanding/{id}/results` | List all analyses with coverage |
| GET | `/api/understanding/{id}/chain/{analysis_id}` | Full narrative + evidence |

### Analytics & Settings
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects/{id}/analytics` | Unified project metrics (code, migration, LLM) |
| GET | `/api/settings` | Runtime configuration |
| POST | `/api/settings` | Update configuration |
| GET | `/api/graph/{id}` | ASG visualization data |
| GET | `/api/health` | Health check |

---

## 6. Frontend

React 19 SPA with Vite 7, Tailwind CSS 4, TypeScript, react-router-dom.

### Pages

| Page | Route | Purpose |
|------|-------|---------|
| **Login** | `/login` | Authentication |
| **Dashboard** | `/` | Project list, status badges |
| **Project View** | `/project/{id}` | File tree, code browser, ASG graph |
| **Project Wiki** | `/project/{id}/wiki` | Multi-section intelligence dashboard |
| **Code Chat** | `/project/{id}/chat` | RAG chat with deep analysis context |
| **Migration** | `/migration/{id}` | Phase wizard with approval gates |
| **Batch Panel** | `/migration/{id}/batch` | Multi-MVP batch execution |

### Migration UI Components

| Component | Purpose |
|-----------|---------|
| `PhaseViewer` | Phase-by-phase output with approve/reject/re-execute |
| `AgentExecutionPanel` | Real-time SSE stream rendering of agentic execution |
| `AgentStepCard` | Individual step cards (thinking, tool call, result, output, error) |
| `BatchExecutionPanel` | Multi-MVP orchestration: configure → monitor → results |

### Wiki Sections

| Section | Content |
|---------|---------|
| Overview | Status badges, metric cards, language distribution |
| Architecture | Dependency graph, module hierarchy, interface contracts |
| Migration | Plan timeline, MVP breakdown, confidence trends |
| Understanding | Entry point catalog, narrative summaries, evidence links |
| Generated Code | Target architecture snippets, design patterns |
| MVP Catalog | MVP cards with unit composition and business context |
| Diagrams | Mermaid diagrams (ASG, MVP structure, call chains) |

---

## 7. Phased Delivery

### Phase 1: Code Upload + Basic RAG -- COMPLETE
- Zip upload + file extraction
- tree-sitter AST parsing (Python)
- Code-aware chunking with preamble injection
- Embed + store in pgvector
- Code Chat via hybrid retrieval + RAPTOR
- Basic project view (file tree + code browser)

### Phase 2: ASG + Relationship-Aware Retrieval -- COMPLETE
- ASG builder (8 edge types across 3 detector modules)
- Graph storage in PostgreSQL
- ASG-expanded retrieval in RAG pipeline
- Dependency graph visualization
- JavaScript/TypeScript support

### Phase 3: Migration Engine -- COMPLETE
- V2 4-phase pipeline (Architecture → Discovery → Transform → Test)
- MVP clustering (RAPTOR-driven + package-based fallback)
- Per-MVP phase execution with approval gates
- Migration lanes (Struts→Spring Boot, StoredProc→ORM)
- Batch execution with configurable approval policies
- Java support

### Phase 4: Advanced Parsing + Semantic Enrichment -- COMPLETE
- C#/.NET parser with semantic enrichment
- VB.NET, SQL, JSP, ASPX, XML, Properties regex parsers
- `implements`, `overrides`, `type_dep` edge types
- JavaParser/Roslyn bridges for deep type resolution
- VB.NET→.NET Core migration lane
- Cross-technology view layer migration (JSP→React, ASPX→React)

### Phase 5: Intelligence + Observability -- COMPLETE
- LLM Gateway (retry, metrics, cost tracking, purpose tagging)
- Deep Understanding (entry point detection, call chain tracing, tiered analysis)
- Agentic migration (10-tool multi-turn execution)
- Project Wiki and Analytics dashboard
- Enterprise quality (gates, retry, confidence, scorecard)
- Deep analysis → chat context and migration context integration

### Phase 6: Future
- Diff views for migrated code
- Test generation from ASG paths
- VSCode extension (query + debug from editor)

---

## 8. Forked from DBNotebook

### Reused components
- `core/providers/` -- All LLM providers (Groq, OpenAI, Anthropic, Gemini, Ollama)
- `core/vector_store/` -- pgvector store
- `core/engine/` -- Hybrid retrieval engine (BM25 + vector + reranker)
- `core/embedding/` -- Embedding layer
- `core/config/` -- YAML config loader
- `core/auth/` -- Auth + RBAC
- `core/db/` -- SQLAlchemy DB layer (models adapted for code domain)
- `core/raptor/` -- Hierarchical retrieval (adapted for code hierarchy)
- `core/memory/` -- Session memory
- `core/observability/` -- Query logger + token metrics
- `core/interfaces/` -- Base interfaces
- `core/registry.py` -- Plugin registry
- `pipeline.py` -- Adapted for code RAG

### Not carried over
- SQL Chat, Analytics, Quiz, Studio, Document transformations
- Vision/image processing
- Document-specific ingestion and agents

---

*Created: February 6, 2026*
*Last updated: February 24, 2026*
*Status: Phase 5 Complete*
