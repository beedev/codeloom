# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**CodeLoom** - A code intelligence and migration platform powered by AST (Abstract Syntax Tree) + ASG (Abstract Semantic Graph) + RAG. Upload an entire codebase, understand it through AI-powered code intelligence, and migrate it to a new architecture or tech stack with confidence.

**Core capabilities**:
- **Code RAG** - Upload a codebase, ask questions, get code snippets with context
- **Code Intelligence** - AST parsing + ASG relationship mapping for deep understanding
- **Code Migration** - 6-phase pipeline from current state to target architecture

**Supported languages**: Python, JavaScript/TypeScript, Java, C#/.NET (via tree-sitter)

**Forked from**: DBNotebook (dbn-v2) - reuses LLM providers, embeddings, pgvector, hybrid retrieval, reranking, RAPTOR, auth, Flask API, and SSE streaming infrastructure.

## Architecture Reference

See `docs/architecture.md` for the complete system architecture, data model, API surface, and phased delivery plan.

## Development Commands

**LOCAL DEVELOPMENT**: Use `./dev.sh` with local PostgreSQL on port 5432.

```bash
# First-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew services start postgresql@17        # macOS - start PostgreSQL
createdb codeloom_dev                     # Create database if needed

# Local development
./dev.sh local                # Start Flask backend locally
./dev.sh                      # Shows usage help
./dev.sh status               # Check status of all services
./dev.sh stop                 # Stop all services

# Local PostgreSQL setup
# - PostgreSQL running on localhost:5432
# - Database: codeloom_dev
# - User/Password: dbnotebook/dbnotebook (inherited from dbn-v2, update as needed)
# - Default login: admin / admin123
# - .env uses host.docker.internal but dev.sh replaces with localhost

# Frontend development (React + Vite)
cd frontend
npm install
npm run dev                   # Dev server on :3000 (proxies to Flask :7860)
npm run build                 # Production build
npm run lint                  # ESLint

# Database migrations (Alembic)
alembic upgrade head
alembic revision --autogenerate -m "description"

# Tests
pytest                        # All tests
pytest -v -x                  # Verbose, stop on first failure
```

## Quick Reference

**Add new provider**: Create class in `core/providers/`, register in `core/plugins.py`
**Add new route**: Create in `api/routes/`, register in `ui/web.py` via `create_*_routes()`
**Modify DB schema**: Edit `core/db/models.py`, then `alembic revision --autogenerate`
**Frontend component**: Add to `frontend/src/components/`, types in `frontend/src/types/`

## Architecture

### Core Data Flow

```
React Frontend (:3000) → Flask API (:7860) → CodeLoom Pipeline
                                                    |
                    +-------------------------------+-------------------------------+
                    |                               |                               |
            Code Ingestion                    Code RAG Chat                  Migration Engine
            (zip upload)                    (hybrid retrieval)              (6-phase pipeline)
                    |                               |                               |
            tree-sitter AST                 ASG-Expanded                    LLM Analysis +
            ASG Builder                     Retrieval                       ASG-Informed
            Code Chunker                    (BM25+Vector+Graph)            Code Transform
            Embedding                       Reranking                       Test Generation
                    |                               |                               |
                    +-------------------------------+-------------------------------+
                                                    |
                                    PostgreSQL + pgvector
                        (projects, code_files, code_units, code_edges,
                         data_embeddings, migration_plans, migration_phases)
```

### Key Components

**`codeloom/pipeline.py`** - Central orchestrator (inherited from DBNotebook):
- Manages LLM/embedding providers, project context
- Key methods: `switch_notebook()`, `store_nodes()`, `stream_chat()`

**`codeloom/core/ast_parser/`** - NEW - AST parsing via tree-sitter:
- Unified parser for Python, JS/TS, Java, C#
- Extracts functions, classes, methods, imports, module structure
- Output: list of code units with boundaries, signatures, docstrings

**`codeloom/core/asg_builder/`** - NEW - Abstract Semantic Graph:
- Builds relationship graph from AST output
- Edge types: calls, imports, inherits, implements, uses, contains
- Storage: PostgreSQL `code_edges` table
- Queries: "what calls X", "what depends on X", "blast radius of changing X"

**`codeloom/core/code_chunker/`** - NEW - Code-aware chunking:
- AST-informed: each chunk = one logical code unit (function, class, module block)
- Preamble injection: file path + imports + class context prepended
- ~1024 token chunks (larger than document chunks - code needs more context)
- Fallback: blank-line splitting for unknown languages

**`codeloom/core/migration/`** - NEW - 6-phase migration engine:
1. `approach.py` - Analyze source, compare with target brief, produce strategy
2. `architecture.py` - Design migration architecture, module mapping
3. `mvp.py` - Identify MVP scope using ASG dependency ordering
4. `design.py` - Detailed design per module
5. `transform.py` - Code transformation using AST
6. `testing.py` - Test generation from ASG paths

### Reused from DBNotebook (codeloom/core/)

- **`providers/`** - LLM providers (Ollama, OpenAI, Anthropic, Gemini, Groq)
- **`vector_store/`** - PGVectorStore (PostgreSQL + pgvector)
- **`engine/`** - Hybrid retrieval engine (BM25 + vector + reranking)
- **`embedding/`** - Embedding layer (HuggingFace, OpenAI)
- **`config/`** - YAML config loader
- **`auth/`** - Authentication + RBAC
- **`db/`** - Database layer (SQLAlchemy 2.0 + Alembic)
- **`raptor/`** - Hierarchical retrieval (to adapt for code hierarchy)
- **`memory/`** - Session memory
- **`observability/`** - Query logger, token metrics
- **`interfaces/`** - Base interfaces
- **`registry.py`** - Plugin registry
- **`plugins.py`** - Plugin registration
- **`api/core/`** - Decorators, response builders, exceptions

### Plugin Architecture

```
-- LLM Providers: Ollama, OpenAI, Anthropic, Gemini, Groq
-- Embedding Providers: HuggingFace, OpenAI
-- Retrieval Strategies: Hybrid, Semantic, Keyword
```

Providers selected via env vars: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `RETRIEVAL_STRATEGY`

### Data Model

**Inherited tables**: `users`, `data_embeddings`

**New CodeLoom tables** (to be created):
- `projects` - Codebase projects (replaces notebooks)
- `code_files` - Individual files in project
- `code_units` - AST-parsed units (functions, classes, methods)
- `code_edges` - ASG relationships between code units
- `migration_plans` - Migration configurations
- `migration_phases` - Phase tracking and output

### API Surface (Planned)

**Project Management**:
- `POST /api/projects/upload` - Upload zip/tar.gz codebase
- `GET /api/projects/{id}/structure` - File tree + ASG overview
- `GET /api/projects/{id}/graph` - ASG visualization data

**Code Chat (RAG)**:
- `POST /api/projects/{id}/query` - Query against codebase
- `POST /api/projects/{id}/query/stream` - SSE streaming query

**Migration**:
- `POST /api/migration/plan` - Create migration plan
- `GET /api/migration/{id}/status` - Migration progress
- `POST /api/migration/{id}/phase/{n}/execute` - Execute phase
- `GET /api/migration/{id}/phase/{n}/output` - Get phase output

### Frontend Pages (Planned)

- **Dashboard** - List projects, recent queries, active migrations
- **Project View** - File tree, ASG graph visualization, code browser
- **Code Chat** - RAG chat against uploaded codebase
- **Migration Wizard** - 6-phase pipeline with approval gates
- **Diff View** - Side-by-side source vs migrated code
- **Dependency Graph** - Interactive ASG visualization

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:

```bash
# Core providers
LLM_PROVIDER=ollama              # ollama|openai|anthropic|gemini|groq
LLM_MODEL=llama3.1:latest
EMBEDDING_PROVIDER=openai        # openai|huggingface
EMBEDDING_MODEL=text-embedding-3-small
RETRIEVAL_STRATEGY=hybrid        # hybrid|semantic|keyword

# Database
DATABASE_URL=postgresql://dbnotebook:dbnotebook@localhost:5432/codeloom_dev
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=codeloom_dev

# pgvector embedding dimension (must match embedding model)
PGVECTOR_EMBED_DIM=1536          # 1536 for OpenAI, 768 for nomic

# Authentication
FLASK_SECRET_KEY=change-me       # Required for sessions
```

## Frontend

Located in `/frontend/`:
- **Stack**: React 19, Vite 7, Tailwind CSS 4, TypeScript
- **Proxy**: Vite proxies `/api` to Flask :7860
- **State**: React Context pattern

## Key Defaults

- Flask: http://localhost:7860
- Frontend dev: http://localhost:3000
- PostgreSQL: localhost:5432
- Default login: admin / admin123
- Embedding dimension: 1536 (OpenAI text-embedding-3-small)
- Reranker: `mixedbread-ai/mxbai-rerank-large-v1`

## Phased Delivery

**Phase 1 (MVP)**: Code upload + AST parsing + basic RAG chat (Python only)
**Phase 2**: ASG + relationship-aware retrieval + JS/TS support
**Phase 3**: Migration engine (6-phase pipeline) + Java, C# support
**Phase 4**: Advanced features (diff views, test gen, VSCode extension)

## Important Notes

- All Python imports currently reference `dbnotebook` - must be renamed to `codeloom` before running
- New modules (ast_parser, asg_builder, code_chunker, migration) are empty directories awaiting implementation
- `api/routes/` is empty - new CodeLoom-specific routes to be built
- tree-sitter dependencies need to be added to requirements.txt
- DB models need updating: strip document-specific tables, add CodeLoom tables
