# CodeLoom Architecture

> Code intelligence and migration platform powered by AST + ASG + RAG

---

## 1. Vision

CodeLoom enables developers to upload an entire codebase, understand it through AI-powered code intelligence, and migrate it to a new architecture or tech stack with confidence.

**Core capabilities**:
- **Code RAG** - Upload a codebase, ask questions, get code snippets with context
- **Code Intelligence** - AST parsing + ASG relationship mapping for deep understanding
- **Code Migration** - 6-phase pipeline from current state to target architecture

**Supported languages**: Python, JavaScript/TypeScript, Java, C#/.NET (via tree-sitter)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (:3000)                        │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Code Chat │  │ Project View │  │ Migration Wizard         │  │
│  │ (RAG)     │  │ (ASG Graph)  │  │ (6-phase pipeline)       │  │
│  └─────┬─────┘  └──────┬───────┘  └────────────┬─────────────┘  │
└────────┼───────────────┼────────────────────────┼───────────────┘
         │               │                        │
         ▼               ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask API (:7860)                           │
│  /api/projects  │  /api/projects/{id}/query  │  /api/migration  │
└────────┬────────────────┬────────────────────────┬──────────────┘
         │                │                        │
         ▼                ▼                        ▼
┌────────────────┐  ┌──────────────┐  ┌────────────────────────┐
│ Code Ingestion │  │ Code RAG     │  │ Migration Engine       │
│                │  │ Pipeline     │  │                        │
│ Zip Upload     │  │ Hybrid       │  │ 1. Approach            │
│ tree-sitter    │  │ Retrieval    │  │ 2. Architecture        │
│ AST Parse      │  │ (BM25+Vector)│  │ 3. MVP Scope           │
│ ASG Build      │  │ + ASG Expand │  │ 4. Design              │
│ Code Chunking  │  │ + Reranking  │  │ 5. Code Transform      │
│ Embedding      │  │              │  │ 6. Test Generation     │
└────────┬───────┘  └──────┬───────┘  └────────────┬───────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL + pgvector                          │
│  ┌──────────────┐  ┌───────────┐  ┌────────────┐  ┌──────────┐ │
│  │data_embeddings│ │code_units │  │code_edges  │  │migration │ │
│  │(code vectors) │ │(AST nodes)│  │(ASG graph) │  │_plans    │ │
│  └──────────────┘  └───────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 AST Parser (`core/ast_parser/`)

Parses source code into structural units using tree-sitter.

**Input**: Source file (any supported language)
**Output**: List of `CodeUnit` objects

```python
@dataclass
class CodeUnit:
    unit_id: str              # Unique ID
    file_id: str              # Parent file
    unit_type: str            # function | class | method | module | interface | enum
    name: str                 # Function/class name
    language: str             # python | javascript | typescript | java | csharp
    start_line: int
    end_line: int
    signature: str            # Full signature (def foo(x: int) -> str)
    docstring: str            # Extracted docstring/JSDoc/Javadoc
    source: str               # Raw source code
    imports: List[str]        # What this unit imports/uses
    parent: Optional[str]     # Parent class/module if nested
```

**Parser strategy**:
- **Primary**: tree-sitter for all 4 languages (consistent cross-language approach)
- **Enhancement**: Python `ast` module for Python-specific depth (type inference, decorator parsing)
- **Fallback**: Blank-line splitting with larger chunks (~1024 tokens) for unsupported languages

**tree-sitter queries per language**:
| Language | Functions | Classes | Methods | Imports |
|----------|-----------|---------|---------|---------|
| Python | `function_definition` | `class_definition` | nested `function_definition` | `import_statement`, `import_from_statement` |
| JS/TS | `function_declaration`, `arrow_function` | `class_declaration` | `method_definition` | `import_statement` |
| Java | `method_declaration` | `class_declaration` | `method_declaration` in class | `import_declaration` |
| C# | `method_declaration` | `class_declaration` | `method_declaration` in class | `using_directive` |

### 3.2 ASG Builder (`core/asg_builder/`)

Builds a semantic relationship graph from AST-parsed code units.

**Edge types**:
| Edge | Meaning | Example |
|------|---------|---------|
| `calls` | Function A calls function B | `ingest_file()` calls `store_nodes()` |
| `imports` | Module A imports from module B | `pipeline.py` imports from `retriever.py` |
| `inherits` | Class A extends class B | `MyService(BaseService)` |
| `implements` | Class implements interface | `GroqProvider implements LLMProvider` |
| `uses` | Function uses a type/class | `def foo(x: UserModel)` |
| `contains` | Module contains class/function | `pipeline.py` contains `LocalRAGPipeline` |
| `overrides` | Method overrides parent method | `get_llm()` overrides `LLMProvider.get_llm()` |

**Storage**: PostgreSQL adjacency table

```sql
CREATE TABLE code_edges (
    id SERIAL PRIMARY KEY,
    project_id UUID NOT NULL,
    source_unit_id VARCHAR NOT NULL,
    target_unit_id VARCHAR NOT NULL,
    edge_type VARCHAR(50) NOT NULL,  -- calls, imports, inherits, etc.
    metadata JSONB DEFAULT '{}',      -- line number, context, confidence
    UNIQUE(project_id, source_unit_id, target_unit_id, edge_type)
);
CREATE INDEX idx_code_edges_source ON code_edges(source_unit_id);
CREATE INDEX idx_code_edges_target ON code_edges(target_unit_id);
CREATE INDEX idx_code_edges_type ON code_edges(edge_type);
```

**Graph queries**:
- `get_callers(unit_id)` - What calls this function?
- `get_callees(unit_id)` - What does this function call?
- `get_dependencies(unit_id, depth=2)` - Transitive dependency tree
- `get_dependents(unit_id, depth=2)` - Transitive reverse dependency (blast radius)
- `get_import_graph(project_id)` - Full module dependency graph
- `get_class_hierarchy(project_id)` - Inheritance tree

### 3.3 Code Chunker (`core/code_chunker/`)

Produces self-contained chunks for embedding, using AST boundaries instead of token windows.

**Strategy**:
1. Each `CodeUnit` (function, class, method) becomes one chunk
2. Prepend a **preamble** to each chunk:
   ```
   # File: src/core/engine/retriever.py
   # Imports: from llama_index import VectorIndexRetriever, ...
   # Class: LocalRetriever(BaseRetriever)

   def _build_two_stage_retriever(self, index, top_k):
       ...
   ```
3. Large units (>1024 tokens) get split at logical boundaries (inner functions, comment blocks)
4. Module-level code (imports, constants, globals) bundled as one chunk per file

**Chunk metadata** (stored in JSONB alongside embedding):
```json
{
    "file_path": "src/core/engine/retriever.py",
    "language": "python",
    "unit_type": "method",
    "unit_name": "_build_two_stage_retriever",
    "class_name": "LocalRetriever",
    "signature": "def _build_two_stage_retriever(self, index, top_k)",
    "start_line": 145,
    "end_line": 198,
    "imports": ["VectorIndexRetriever", "BM25Retriever"],
    "unit_id": "abc123"
}
```

### 3.4 Code RAG Pipeline

Extends DBNotebook's retrieval with ASG-augmented expansion.

**Query flow**:
```
User Query
    │
    ▼
Hybrid Retrieval (BM25 + Vector)          ← Existing DBNotebook engine
    │
    ▼
Top-K code chunks (initial matches)
    │
    ▼
ASG Expansion                              ← NEW
    │  For each matched chunk:
    │  - Get callers/callees (depth=1)
    │  - Get imports
    │  - Get parent class/module
    │
    ▼
Expanded chunk set (initial + related)
    │
    ▼
Reranker                                   ← Existing DBNotebook reranker
    │
    ▼
Final Top-K chunks with full context
    │
    ▼
LLM Generation (code-specific system prompt)
    │
    ▼
Response with file paths + code snippets
```

**Code-specific system prompt**:
- Always include file paths with code snippets
- Show function signatures
- Explain relationships ("X calls Y which...")
- Format code blocks with language tags

### 3.5 Migration Engine (`core/migration/`)

6-phase pipeline, each phase takes input from the previous and produces artifacts.

```
┌──────────────────────────────────────────────────────────────────┐
│                     MIGRATION PIPELINE                           │
│                                                                  │
│  INPUTS:                                                         │
│  - Source project (uploaded, AST+ASG built)                      │
│  - Target architecture brief (text description)                  │
│  - Target tech stack (languages, frameworks, versions)           │
│  - Constraints (timeline, team size, risk tolerance)             │
│                                                                  │
│  ┌─────────┐   ┌──────────────┐   ┌─────┐   ┌────────┐         │
│  │1.Approach│──▶│2.Architecture│──▶│3.MVP│──▶│4.Design│         │
│  └─────────┘   └──────────────┘   └─────┘   └───┬────┘         │
│                                                   │              │
│                    ┌─────────┐   ┌───────────┐    │              │
│                    │6.Testing│◀──│5.Migration │◀───┘              │
│                    └─────────┘   └───────────┘                   │
│                                                                  │
│  Each phase has:                                                 │
│  - LLM-powered analysis using source ASG + target brief          │
│  - Human approval gate before proceeding                         │
│  - Persisted output artifacts                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Phase details**:

| Phase | Input | LLM + ASG Role | Output |
|-------|-------|---------------|--------|
| **1. Approach** | Source ASG + target brief | ASG reveals coupling/complexity; LLM recommends rewrite vs refactor vs strangler fig | Strategy document with risk assessment |
| **2. Architecture** | Strategy + target stack | ASG dependency graph drives module mapping old→new; LLM designs target architecture | Architecture design with module mapping |
| **3. MVP** | Architecture + priorities | ASG identifies leaf nodes (low dependency) for safe early migration | MVP scope: ordered list of modules |
| **4. Design** | MVP scope | AST provides exact signatures/types; LLM designs new interfaces per module | Detailed design per module |
| **5. Migration** | Design per module | AST transforms code structures; ASG ensures all call sites updated | Migrated code, module by module |
| **6. Testing** | Migrated code | ASG generates tests covering real call paths; LLM writes test code | Test suite: unit + integration + equivalence |

---

## 4. Data Model

### New Tables

```sql
-- Projects (replaces DBNotebook's notebooks for code context)
CREATE TABLE projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    primary_language VARCHAR(50),        -- python, javascript, java, csharp
    languages JSONB DEFAULT '[]',        -- all detected languages
    file_count INTEGER DEFAULT 0,
    total_lines INTEGER DEFAULT 0,
    ast_status VARCHAR(20) DEFAULT 'pending',  -- pending, parsing, complete, error
    asg_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Individual files in a project
CREATE TABLE code_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    file_path VARCHAR(1024) NOT NULL,    -- relative path within project
    language VARCHAR(50),
    file_hash VARCHAR(64),               -- MD5 for change detection
    line_count INTEGER,
    size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id, file_path)
);

-- AST-parsed code units
CREATE TABLE code_units (
    unit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES code_files(file_id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    unit_type VARCHAR(50) NOT NULL,      -- function, class, method, module, interface
    name VARCHAR(255) NOT NULL,
    qualified_name VARCHAR(1024),        -- module.class.method
    language VARCHAR(50),
    start_line INTEGER,
    end_line INTEGER,
    signature TEXT,
    docstring TEXT,
    source TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- ASG relationship edges
CREATE TABLE code_edges (
    id SERIAL PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id) ON DELETE CASCADE,
    source_unit_id UUID REFERENCES code_units(unit_id) ON DELETE CASCADE,
    target_unit_id UUID REFERENCES code_units(unit_id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    UNIQUE(project_id, source_unit_id, target_unit_id, edge_type)
);

-- Migration plans
CREATE TABLE migration_plans (
    plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    source_project_id UUID REFERENCES projects(project_id),
    target_brief TEXT NOT NULL,          -- Target architecture description
    target_stack JSONB NOT NULL,         -- {languages, frameworks, versions}
    constraints JSONB DEFAULT '{}',     -- {timeline, team_size, risk_tolerance}
    status VARCHAR(20) DEFAULT 'draft', -- draft, in_progress, complete, abandoned
    current_phase INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Migration phase outputs
CREATE TABLE migration_phases (
    phase_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID REFERENCES migration_plans(plan_id) ON DELETE CASCADE,
    phase_number INTEGER NOT NULL,       -- 1-6
    phase_type VARCHAR(50) NOT NULL,     -- approach, architecture, mvp, design, migration, testing
    status VARCHAR(20) DEFAULT 'pending',
    input_summary TEXT,
    output TEXT,                          -- LLM-generated output (markdown)
    output_files JSONB DEFAULT '[]',     -- Generated code files
    approved BOOLEAN DEFAULT FALSE,
    approved_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(plan_id, phase_number)
);

-- Reuse existing data_embeddings table for code chunk vectors
-- Reuse existing users, conversations tables
```

### Reused Tables (from DBNotebook)
- `users` - Authentication, roles, API keys
- `data_embeddings` - pgvector storage for code chunk embeddings
- `conversations` - Chat history per project
- `query_logs` - Token usage tracking

---

## 5. API Surface

### Project Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/upload` | Upload zip/tar.gz codebase |
| GET | `/api/projects` | List user's projects |
| GET | `/api/projects/{id}` | Project details + stats |
| DELETE | `/api/projects/{id}` | Delete project |
| GET | `/api/projects/{id}/files` | File tree |
| GET | `/api/projects/{id}/file/{path}` | View file content |

### Code Intelligence
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects/{id}/structure` | AST overview (units, stats per file) |
| GET | `/api/projects/{id}/graph` | ASG data for visualization |
| GET | `/api/projects/{id}/unit/{unit_id}` | Code unit details + relationships |
| GET | `/api/projects/{id}/unit/{unit_id}/callers` | What calls this? |
| GET | `/api/projects/{id}/unit/{unit_id}/callees` | What does this call? |
| GET | `/api/projects/{id}/unit/{unit_id}/blast-radius` | Impact of changing this |

### Code Chat (RAG)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/{id}/query` | Query codebase (JSON response) |
| POST | `/api/projects/{id}/query/stream` | Query codebase (SSE streaming) |

### Migration
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/migration/plan` | Create migration plan |
| GET | `/api/migration/{id}` | Plan status + phase summaries |
| POST | `/api/migration/{id}/phase/{n}/execute` | Execute phase N |
| GET | `/api/migration/{id}/phase/{n}` | Phase output |
| POST | `/api/migration/{id}/phase/{n}/approve` | Approve phase to proceed |
| GET | `/api/migration/{id}/diff` | Source vs migrated diff |

---

## 6. Frontend Pages

| Page | Route | Purpose |
|------|-------|---------|
| **Dashboard** | `/` | List projects, active migrations, recent queries |
| **Project View** | `/project/{id}` | File tree, stats, ASG graph, code browser |
| **Code Chat** | `/project/{id}/chat` | RAG chat against codebase |
| **Dependency Graph** | `/project/{id}/graph` | Interactive ASG visualization |
| **Migration Wizard** | `/migration/{id}` | Step through 6 phases with approval gates |
| **Diff View** | `/migration/{id}/diff` | Side-by-side source vs migrated code |
| **Login** | `/login` | Authentication |
| **Admin** | `/admin` | User management, token metrics |

---

## 7. Phased Delivery

### Phase 1: Code Upload + Basic RAG (MVP)
- Zip upload + file extraction
- tree-sitter AST parsing (Python first)
- Code-aware chunking (by function/class)
- Embed + store in pgvector
- Code Chat via existing retrieval engine
- Basic project view (file tree + code browser)

### Phase 2: ASG + Relationship-Aware Retrieval
- ASG builder (call graphs, imports, inheritance)
- Graph storage in PostgreSQL
- ASG-expanded retrieval (pull related code with matches)
- Dependency graph visualization
- Add JavaScript/TypeScript support

### Phase 3: Migration Engine
- 6-phase migration pipeline
- Target brief input UI
- Migration planning (approach + architecture + MVP)
- Code transformation (design + migration + testing)
- Diff view for migrated code
- Add Java support

### Phase 4: Advanced
- C#/.NET support
- VSCode extension (query + debug from editor)
- Multi-project comparison
- Test generation from ASG paths
- Migration templates for common patterns (Flask->FastAPI, React class->hooks, etc.)

---

## 8. Forked from DBNotebook

### Reused components (from dbnotebook/)
- `core/providers/` - All LLM providers (Groq, OpenAI, Anthropic, Gemini, Ollama)
- `core/vector_store/` - pgvector store
- `core/engine/` - Hybrid retrieval engine (BM25 + vector + reranker)
- `core/embedding/` - Embedding layer
- `core/config/` - YAML config loader
- `core/auth/` - Auth + RBAC
- `core/db/` - SQLAlchemy DB layer (models adapted for code domain)
- `core/raptor/` - Hierarchical retrieval (adapted for code hierarchy)
- `core/memory/` - Session memory
- `core/observability/` - Query logger + token metrics
- `core/interfaces/` - Base interfaces
- `core/registry.py` - Plugin registry
- `api/core/` - Decorators, response builders, exceptions
- `pipeline.py` - Adapted for code RAG

### Not carried over
- SQL Chat, Analytics, Quiz, Studio, Document transformations
- Vision/image processing
- Document-specific ingestion
- Document-specific agents
- All DBNotebook API routes (replaced with code-specific routes)

---

*Created: February 6, 2026*
*Status: Architecture Design*
