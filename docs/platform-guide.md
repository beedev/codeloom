# CodeLoom Platform Guide

## Table of Contents

1. [Overview](#1-overview)
2. [Source Code Ingestion](#2-source-code-ingestion)
3. [AST Parsing](#3-ast-parsing)
4. [Abstract Semantic Graph (ASG)](#4-abstract-semantic-graph-asg)
5. [Code Chunking and Embedding](#5-code-chunking-and-embedding)
6. [Vector Store and RAPTOR](#6-vector-store-and-raptor)
7. [Chat with Code](#7-chat-with-code)
8. [Migration Pipeline](#8-migration-pipeline)
9. [Deep Understanding](#9-deep-understanding)
10. [Blast Radius and Impact Analysis](#10-blast-radius-and-impact-analysis)
11. [UML Diagrams](#11-uml-diagrams)
12. [Configuration](#12-configuration)

---

## 1. Overview

CodeLoom is a code intelligence and migration platform. You upload an entire codebase, and the platform makes it queryable via AI-powered chat and prepares it for structured migration to a new architecture or framework.

The platform parses source code into semantic units, builds a graph of their relationships, embeds them for retrieval, and exposes that knowledge through a chat interface and a guided migration pipeline.

**Supported languages**: Python, JavaScript, TypeScript, Java, C#, VB.NET, SQL

### Platform Architecture

```
Browser (React 19, Vite 7)
    |
    | HTTP/SSE on :3000 (Vite dev proxy)
    |
    v
FastAPI Backend (:9005)
    |
    +-- api/app.py            App factory, CORS, session middleware
    |
    +-- api/deps.py           FastAPI Depends() injection from app.state
    |
    +-- api/routes/
    |       fastapi_auth.py   Login, logout, session check
    |       projects.py       Project CRUD, zip upload, file/unit browsing
    |       code_chat.py      RAG chat and impact analysis (SSE streaming)
    |       migration.py      Migration plan lifecycle
    |       diagrams.py       UML diagram generation
    |       understanding.py  Deep understanding job management
    |
    +-- core/
    |       pipeline.py       Central orchestrator (LLM, embeddings, retrieval)
    |       ingestion/        Upload -> parse -> chunk -> embed -> store
    |       ast_parser/       Tree-sitter parsers (Strategy pattern)
    |       asg_builder/      Abstract Semantic Graph edge detection
    |       code_chunker/     Preamble injection, token-aware chunking
    |       vector_store/     PGVectorStore (pgvector)
    |       raptor/           Hierarchical retrieval tree
    |       stateless/        Thread-safe retrieval for API routes
    |       migration/        6-phase (V1) and 4-phase (V2) migration engine
    |       understanding/    Deep analysis worker and call tree tracer
    |       diagrams/         Structural and behavioral UML generation
    |
    +-- PostgreSQL 17 + pgvector
            users, projects, code_files, code_units, code_edges,
            migration_plans, migration_phases, conversations,
            deep_analyses, analysis_units, query_logs
```

### Request Flow

```
User query
    |
    v
React frontend (:3000)
    |  POST /api/projects/{id}/chat/stream
    v
FastAPI (:9005)
    |
    +-- api/deps.py           Extracts pipeline, db_manager from app.state
    |
    +-- code_chat.py          Validates project, loads conversation history
    |
    +-- stateless/            fast_retrieve() -> hybrid BM25 + vector search
    |
    +-- asg_builder/expander  ASG graph neighbor expansion
    |
    +-- stateless/            build_context_with_history() -> context string
    |
    +-- Settings.llm          execute_query_streaming() -> SSE token stream
    |
    v
Browser receives SSE events:
    type=sources  (citations list)
    type=content  (streaming LLM tokens)
    type=done     (metadata: elapsed_ms, retrieval_count)
```

### Stack Summary

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Frontend | React 19, Vite 7, Tailwind CSS 4, TypeScript |
| Database | PostgreSQL 17 + pgvector |
| ORM | SQLAlchemy 2.0 + Alembic |
| LLM Framework | LlamaIndex Core 0.14.x |
| AST Parsing | tree-sitter (Python, JS/TS, Java, C#) |
| Embedding | OpenAI text-embedding-3-small (1536d) or HuggingFace |
| Reranker | mixedbread-ai/mxbai-rerank-base-v1 |
| Default LLM | Configurable: Ollama, OpenAI, Anthropic, Gemini, Groq |

---

## 2. Source Code Ingestion

**Entry point**: `codeloom/core/ingestion/code_ingestion.py`

The `CodeIngestionService` class orchestrates the full pipeline from raw source to searchable embeddings. The pipeline runs as a background task after project creation.

```
Source (zip / git clone / local dir)
    |
    v
Extract to temp directory
    |
    v
Walk directory tree
  (skip: node_modules, .git, __pycache__, venv, build, dist, .mypy_cache)
    |
    v
Per file:
    detect_language(file_path)  ->  language string or None
    parse_file(file_path)       ->  ParseResult (AST)
    CodeChunker.chunk_file()    ->  List[TextNode]
    vector_store.add_documents  ->  embeddings stored in pgvector
    |
    v
Per project:
    ASGBuilder.build_edges()    ->  CodeEdge records in PostgreSQL
    RAPTORWorker.build_tree()   ->  Summary nodes in pgvector
```

### Ingestion Limits

| Limit | Default | Setting |
|---|---|---|
| Max file size | 50 MB | `MAX_FILE_SIZE_MB` in `code_ingestion.py` |
| Max files per project | 500 | `MAX_FILES` in `code_ingestion.py` |
| Max tokens per chunk | 1024 | `max_tokens_per_chunk` constructor param |

### Upload Methods

**Method 1: Zip upload**

The most common path. The user uploads a `.zip` file via the project creation dialog or the `/api/projects/{id}/upload` endpoint.

```python
# codeloom/core/ingestion/code_ingestion.py
def ingest_zip(self, zip_path: str, project_id: str, user_id: str) -> IngestionResult:
    temp_dir = tempfile.mkdtemp(prefix="codeloom_ingest_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    return self._ingest_directory(temp_dir, project_id, user_id)
    # temp_dir is cleaned up in the finally block
```

**Method 2: Git clone (shallow)**

Clones the specified branch with `--depth 1` to keep the clone fast. Requires `git` on the server PATH. Clone timeout is 300 seconds.

```python
def ingest_git(self, repo_url: str, branch: str, project_id: str, user_id: str) -> IngestionResult:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir],
        timeout=300,
    )
    return self._ingest_directory(temp_dir, project_id, user_id)
```

**Method 3: Local directory**

Reads directly from a server-side path without copying. Suitable for development or when the codebase is already available on the host.

```python
def ingest_local(self, dir_path: str, project_id: str, user_id: str) -> IngestionResult:
    # No copy — reads files in-place
    return self._ingest_directory(dir_path, project_id, user_id)
```

### IngestionResult

All three methods return an `IngestionResult` dataclass:

```python
@dataclass
class IngestionResult:
    project_id: str
    files_processed: int   # Files successfully parsed
    files_skipped: int     # Unrecognized or oversized files
    units_extracted: int   # Total code units (functions, classes, methods)
    chunks_created: int    # TextNodes created from units
    embeddings_stored: int # Embeddings written to pgvector
    errors: List[str]      # Non-fatal errors (file-level failures)
    elapsed_seconds: float
```

### Project Status Transitions

The `projects.ast_status` column tracks ingestion progress:

```
draft  ->  parsing  ->  complete
                    ->  error
```

Once `ast_status = complete`, the project becomes available for chat. ASG build happens separately, setting `asg_status`:

```
pending  ->  building  ->  complete
                       ->  error
```

---

## 3. AST Parsing

**Entry point**: `codeloom/core/ast_parser/base.py`

All language parsers implement the `BaseLanguageParser` abstract class following the Strategy pattern. This allows `parse_file()` in `codeloom/core/ast_parser/__init__.py` to dispatch to the correct parser based on file extension without the caller knowing which parser is used.

### Parser Hierarchy

```
BaseLanguageParser (ABC)        codeloom/core/ast_parser/base.py
    |
    +-- PythonParser             python_parser.py     tree-sitter
    +-- JavaScriptParser         javascript_parser.py tree-sitter
    +-- TypeScriptParser         typescript_parser.py tree-sitter
    +-- JavaParser               java_parser.py       tree-sitter
    +-- CSharpParser             csharp_parser.py     tree-sitter
    |
    +-- SqlParser                sql_parser.py        regex-based
    +-- VBNetParser              vbnet_parser.py (via fallback_parser.py)
    |
    +-- FallbackParser           fallback_parser.py   line-count only
```

### BaseLanguageParser Interface

```python
class BaseLanguageParser(ABC):
    @abstractmethod
    def get_language(self) -> str:
        """Return language identifier string ('python', 'java', etc.)."""

    @abstractmethod
    def get_tree_sitter_language(self) -> tree_sitter.Language:
        """Return the compiled tree-sitter Language object."""

    @abstractmethod
    def extract_units(
        self, tree: tree_sitter.Tree, source: bytes, file_path: str
    ) -> List[CodeUnit]:
        """Walk the AST and extract CodeUnit objects."""

    @abstractmethod
    def extract_imports(self, tree: tree_sitter.Tree, source: bytes) -> List[str]:
        """Return raw import statement strings from the AST."""

    def parse_file(self, file_path: str, project_root: str = "") -> ParseResult:
        """Shared logic: read file, run tree-sitter, delegate to subclass."""

    def parse_source(self, source_text: str, file_path: str) -> ParseResult:
        """Parse a source string (used in tests and bridged parsers)."""
```

The `parse_file` and `parse_source` methods are implemented once on the base class and call the abstract methods. Subclasses only need to implement the four abstract methods.

### ParseResult and CodeUnit

```python
@dataclass
class ParseResult:
    file_path: str
    language: str
    units: List[CodeUnit]      # All extracted code entities
    imports: List[str]         # All import statements in file
    module_docstring: Optional[str]
    line_count: int
    errors: List[ParseError]   # Non-fatal parse warnings/errors

@dataclass
class CodeUnit:
    unit_type: str             # "function" | "class" | "method" | "constructor" | ...
    name: str                  # "parse_file"
    qualified_name: str        # "codeloom.core.ast_parser.parser.parse_file"
    language: str              # "python"
    start_line: int
    end_line: int
    source: str                # Raw source code of this unit
    file_path: str             # Relative path within project
    signature: Optional[str]   # "def parse_file(path: str) -> ParseResult:"
    docstring: Optional[str]
    parent_name: Optional[str] # Class name, for methods
    decorators: List[str]
    imports: List[str]         # Module-level imports (injected after parse)
    metadata: Dict[str, Any]   # Language-specific extras
```

### Semantic Enricher

**Path**: `codeloom/core/ast_parser/enricher.py`

After tree-sitter parsing, the `SemanticEnricher` runs a second pass over each `CodeUnit` to extract structured metadata that the ASG builder uses for edge detection. This enrichment runs on all tree-sitter-supported languages.

The enricher adds to `unit.metadata`:

| Key | Type | Description |
|---|---|---|
| `parsed_params` | `List[{name, type}]` | Method/function parameters with types |
| `return_type` | `str` | Return type annotation |
| `modifiers` | `List[str]` | `public`, `static`, `abstract`, `override`, etc. |
| `fields` | `List[{name, type}]` | Class field declarations |
| `annotations` | `List[str]` | Java/C# annotations and attributes |
| `extends` | `str \| List[str]` | Base class name(s) |
| `implements` | `List[str]` | Interface names |
| `is_override` | `bool` | Whether the method overrides a parent |

### Optional Bridges

**Path**: `codeloom/core/ast_parser/bridges/`

For deeper semantic analysis beyond what tree-sitter provides, two optional subprocess bridges are available. They run external runtimes and communicate via JSON over stdin/stdout.

**JavaParser bridge** (`bridges/java_bridge.py`):
- Runs a Maven-built JAR using JavaParser
- Resolves generic type parameters and fully qualified class names
- Detects constructor injection patterns (Spring `@Autowired`, etc.)
- Build: `./dev.sh setup-tools` (requires JDK 11+ and Maven)

**Roslyn bridge** (`bridges/csharp_bridge.py`):
- Runs a .NET console app using Microsoft.CodeAnalysis (Roslyn)
- Resolves interface implementations across assemblies
- Detects attribute-based patterns (ASP.NET `[ApiController]`, etc.)
- Build: `./dev.sh setup-tools` (requires .NET SDK 6+)

Both bridges are optional. If unavailable, the system falls back to tree-sitter enrichment, which covers the majority of edge detection cases.

### Extension-to-Language Mapping

Defined in `codeloom/core/ast_parser/utils.py`:

```
.py                -> python
.js, .mjs, .cjs    -> javascript
.ts, .tsx          -> typescript
.java              -> java
.cs                -> csharp
.vb                -> vbnet
.sql               -> sql
.jsx               -> javascript
```

Files with unrecognized extensions are skipped during ingestion.

---

## 4. Abstract Semantic Graph (ASG)

**Entry point**: `codeloom/core/asg_builder/builder.py`

The `ASGBuilder` runs as a post-ingestion step. It reads all `CodeUnit` records for a project from the database, detects relationships between them using metadata-first then regex-fallback strategies, and bulk-inserts `CodeEdge` records.

### When ASG Runs

ASG is triggered explicitly via `POST /api/projects/{id}/build-asg` after ingestion completes. The UI shows a "Build Graph" button that becomes available once `ast_status = complete`. Alternatively, ingestion can be configured to trigger ASG build automatically.

### The 8 Edge Types

```
CodeUnit A  ---edge_type--->  CodeUnit B
```

| Edge Type | Direction | Detection Source | Example |
|---|---|---|---|
| `contains` | class -> method | `parent_name` metadata or qualified name parsing | `UserService` contains `getUserById` |
| `imports` | file unit -> imported unit | `file_imports` metadata then regex scan | `api.py` imports `AuthService` |
| `calls` | function -> function | Regex scan of function body source | `process_request` calls `validate_token` |
| `inherits` | class -> base class | `extends` metadata (Java/C#) or signature regex (Python/JS) | `AdminService` inherits `BaseService` |
| `implements` | class -> interface | `implements` metadata list | `UserRepository` implements `IRepository` |
| `overrides` | method -> parent method | `@Override` annotation or `override` modifier | `save` overrides `AbstractRepository.save` |
| `calls_sp` | app code -> stored procedure | Language-specific SQL invocation patterns | Java `prepareCall("{ call usp_GetUser }")` |
| `type_dep` | consumer -> referenced type | Structured `parsed_params`, `return_type`, `fields` metadata | `OrderService` depends on `PaymentGateway` |

### Edge Detection Order

The builder processes all 8 edge types in sequence for each project:

```python
# codeloom/core/asg_builder/builder.py
def build_edges(self, project_id: str) -> int:
    # 1. contains   (class -> method/property/constructor)
    # 2. inherits   (class -> base class)
    # 3. implements (class/struct/record -> interface)
    # 4. calls      (function -> function via identifier matching)
    # 5. imports    (units importing other units)
    # 6. overrides  (method -> parent method)
    # 7. calls_sp   (app code -> stored procedure)  # Only if SQL units exist
    # 8. type_dep   (field types, param types, return types)

    # Deduplicate then bulk-insert with ON CONFLICT DO NOTHING
    stmt = pg_insert(CodeEdge).values(unique_edges)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_code_edge")
```

The `ON CONFLICT DO NOTHING` makes `build_edges` idempotent. Running it twice on the same project produces the same graph.

### Contains Edge Detection (3 Strategies)

The `_detect_contains` method uses three fallback strategies in order:

```
Strategy 1: unit_metadata["parent_name"]
    Set during AST parsing for methods whose parent class is known.
    Most reliable — no ambiguity.

Strategy 2: qualified_name parsing
    "com.example.UserService.getUserById" -> parent is "UserService"
    Works when qualified names follow dotted-path conventions.

Strategy 3: Line range containment
    Method start_line falls within class start_line..end_line in the same file.
    Fallback for languages where metadata is incomplete.
```

### Calls Edge Detection

Regex-based, with a broad-to-specific pattern:

```python
# Simple calls: myFunc()
_CALL_RE = re.compile(r"(?<!\w)(\w+)\s*\(")

# Qualified calls: obj.method() or Package.Class.method()
_QUALIFIED_CALL_RE = re.compile(r"(?:\w+\.)+(\w+)\s*\(")
```

After matching, the set of call targets is intersected with the set of known unit names in the project. Common builtins (`print`, `len`, `Console.WriteLine`, etc.) are excluded via a frozenset defined at the bottom of `builder.py`.

### Stored Procedure Call Detection

Four language-specific patterns detect SP invocations:

```python
# Java: CallableStatement / prepareCall("{ call usp_Name(...) }")
_JAVA_SP_CALL_RE = re.compile(r"""(?:prepareCall|callproc)\s*\(\s*["']\{?\s*call\s+(\w+)""")

# Java: @Procedure(name = "usp_Name") or @Procedure("usp_Name")
_JAVA_PROC_ANNOTATION_RE = re.compile(r"""@Procedure\s*\(\s*(?:name\s*=\s*)?["'](\w+)["']""")

# C#: SqlCommand(..., "usp_Name") with CommandType.StoredProcedure in context
_CSHARP_SP_RE = re.compile(r"""CommandText\s*=\s*["'](\w+)["']|new\s+SqlCommand\s*\(\s*["'](\w+)["']""")

# Python: cursor.callproc("usp_Name")
_PYTHON_CALLPROC_RE = re.compile(r"""callproc\s*\(\s*["'](\w+)["']""")

# Generic: EXEC[UTE] usp_Name in string literals (any language)
_EXEC_IN_STRING_RE = re.compile(r"""["'].*?\bEXEC(?:UTE)?\s+(\w+).*?["']""")
```

### Type Dependency Detection

`_detect_type_deps` reads pre-structured enricher metadata (no regex parsing of raw source):

```
Strategy A: parsed_params[*].type  -> method/constructor depends on each param type
Strategy B: return_type            -> method depends on its return type
Strategy C: fields[*].type         -> class depends on each field's declared type
```

Generic containers and standard library types are excluded via the `_PRIMITIVE_TYPES` frozenset (`String`, `List`, `Optional`, `Task`, `IEnumerable`, etc.).

### ASG Database Schema

```sql
CREATE TABLE code_edges (
    edge_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id    UUID NOT NULL REFERENCES projects(project_id),
    source_unit_id UUID NOT NULL REFERENCES code_units(unit_id),
    target_unit_id UUID NOT NULL REFERENCES code_units(unit_id),
    edge_type     VARCHAR(32) NOT NULL,
    edge_metadata JSONB DEFAULT '{}',
    created_at    TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_code_edge UNIQUE (source_unit_id, target_unit_id, edge_type)
);
```

### ASG Graph Expansion During Chat

**Path**: `codeloom/core/asg_builder/expander.py`

When a chat query returns retrieval results, the `ASGExpander` enriches those results by traversing one hop in the ASG graph. This pulls in semantically related units that the vector search might have missed. The expansion score decays by a configurable factor (default 0.7) per hop.

```python
expander = ASGExpander(db_manager)
retrieval_results = expander.expand(
    results=retrieval_results,
    project_id=project_id,
    cached_nodes=nodes,
    max_expansion=data.max_sources * 2,
    score_decay=0.7,
)
```

---

## 5. Code Chunking and Embedding

**Entry point**: `codeloom/core/code_chunker/`

The `CodeChunker` converts `ParseResult` objects into `TextNode` objects ready for embedding. Each `CodeUnit` maps to one `TextNode` (or multiple if the unit exceeds the token limit).

### Preamble Injection

**Path**: `codeloom/core/code_chunker/preamble.py`

Before embedding, each code chunk receives a context header that gives the embedding model and the LLM situational awareness:

```python
class PreambleBuilder:
    def build(
        self,
        file_path: str,
        imports: List[str],
        parent_class: Optional[str] = None,
        max_imports: int = 10,
    ) -> str:
        lines = [f"# File: {file_path}"]
        if imports:
            import_text = ", ".join(imports[:max_imports])
            lines.append(f"# Imports: {import_text}")
        if parent_class:
            lines.append(f"# Class: {parent_class}")
        return "\n".join(lines)
```

Example preamble for a method inside a class:

```
# File: codeloom/core/auth/auth_service.py
# Imports: from sqlalchemy.orm import Session, from ..db.models import User
# Class: AuthService
```

The final chunk text is `preamble + "\n\n" + unit.source`.

### Token-Aware Splitting

The chunker uses `tiktoken` with the `cl100k_base` encoding (GPT-4/embedding-compatible) to count tokens. The default target is 1024 tokens per chunk.

```python
class CodeChunker:
    def __init__(self, max_tokens: int = 1024, encoding_name: str = "cl100k_base"):
        ...

    def chunk_file(self, parse_result: ParseResult, project_id: str, file_id: str) -> List[TextNode]:
        for unit in parse_result.units:
            preamble = self._preamble_builder.build(...)
            text = f"{preamble}\n\n{unit.source}"
            token_count = self._token_counter.count(text)

            if token_count <= self._max_tokens:
                nodes.append(self._create_text_node(unit, text, ...))
            else:
                nodes.extend(self._split_unit(unit, preamble, ...))
```

For oversized units, `_split_unit` splits at blank lines, keeping the preamble on each sub-chunk. If a sub-chunk still cannot fit, it is emitted as-is rather than dropped.

### TextNode Metadata

Each `TextNode` carries metadata that the retrieval layer and UI use for citation and filtering:

```python
TextNode(
    text=text,  # preamble + source
    metadata={
        "unit_id":        unit.metadata.get("unit_id"),  # UUID for ASG lookup
        "project_id":     project_id,
        "source_id":      file_id,           # CodeFile UUID
        "file_name":      parse_result.file_path,
        "node_type":      "code",
        "unit_type":      unit.unit_type,    # "function" | "class" | "method" | ...
        "unit_name":      unit.name,
        "qualified_name": unit.qualified_name,
        "class_name":     unit.parent_name,
        "language":       unit.language,
        "start_line":     unit.start_line,
        "end_line":       unit.end_line,
        "signature":      unit.signature,
        "has_docstring":  bool(unit.docstring),
    },
    excluded_embed_metadata_keys=[
        "unit_id", "project_id", "source_id",
        "node_type", "start_line", "end_line", "has_docstring",
    ],
)
```

The `excluded_embed_metadata_keys` list prevents UUIDs and line numbers from influencing the embedding vector — only semantically meaningful fields like `file_name`, `unit_name`, `class_name`, `language`, and `signature` are included.

### Embedding Providers

Provider selection is controlled by the `EMBEDDING_PROVIDER` environment variable:

| Provider | Model | Dimension | Speed |
|---|---|---|---|
| `openai` (default) | `text-embedding-3-small` | 1536 | Fast, paid API |
| `huggingface` | `nomic-embed-text-v1.5` or configurable | 768 | Local, free |

The dimension must match the `PGVECTOR_EMBED_DIM` environment variable. If you switch providers, you must re-ingest all projects (vectors with different dimensions are incompatible).

---

## 6. Vector Store and RAPTOR

### Vector Store

**Path**: `codeloom/core/vector_store/pg_vector_store.py`

CodeLoom uses `PGVectorStore` from LlamaIndex, which wraps PostgreSQL's `pgvector` extension. Embeddings are stored in the `data_llamaindex` table (managed by LlamaIndex) alongside the metadata JSONB blob.

Hybrid retrieval combines two signals:

```
BM25 (keyword)  +  vector (semantic)
       weight: 0.5        weight: 0.5
               |
               v
         Fusion: dist_based_score (RRF variant)
               |
               v
         Reranker: mxbai-rerank-base-v1
         (cross-encoder, top_k=10)
               |
               v
         Final ranked results to LLM
```

Weights and top-k values are configurable in `config/codeloom.yaml` under `retrieval`.

### RAPTOR Hierarchical Retrieval

**Path**: `codeloom/core/raptor/`

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval, arxiv.org/abs/2401.18059) builds a hierarchical summary tree over the code chunks after ingestion. This allows queries about high-level architectural concepts to find relevant context even when no single chunk matches the query precisely.

**Tree construction** (bottom-up):

```
Level 0:  Raw code chunks (leaf nodes)
              |
              | UMAP dimensionality reduction (10 components)
              | GMM soft clustering (probability threshold 0.3)
              v
Level 1:  Cluster summaries (LLM-generated)
              |
              | Re-cluster summaries
              v
Level 2:  Summary-of-summaries
              |
              | (repeat until single cluster or max_tree_depth)
              v
Level N:  Root summary node
```

**Clustering parameters** (from `config/codeloom.yaml`):

```yaml
raptor:
  clustering:
    umap_n_components: 10
    umap_n_neighbors: 15
    gmm_probability_threshold: 0.3
    min_cluster_size: 3
    max_cluster_size: 10
    max_tree_depth: 4
    min_nodes_to_cluster: 5
```

**Worker**: `codeloom/core/raptor/worker.py`

The RAPTOR tree is built by a background daemon thread. It uses `asyncio` internally for concurrent LLM summarization calls. In multi-worker Gunicorn deployments, set `DISABLE_BACKGROUND_WORKERS=true` to prevent multiple workers from building the tree simultaneously.

**Retrieval from tree**:

The retrieval strategy adapts based on query intent. Summary-oriented queries ("overview of the authentication system") retrieve from higher tree levels. Detail-oriented queries ("what does the `validate_token` function return") retrieve from level 0 (raw chunks) with a level-1 boost.

```yaml
raptor:
  level_retrieval:
    summary_query_levels: [0, 1, 2, 3]  # All levels
    detail_query_levels: [0, 1]          # Raw chunks + first summary level
    top_k_per_level: 6
    summary_level_boost: 1.5
    detail_level_boost: 1.3
```

**Node cache**: `pipeline.py` maintains a 5-minute in-memory node cache per project. After ingestion or document changes, call `pipeline.invalidate_node_cache(project_id)` to force a refresh.

---

## 7. Chat with Code

**Entry points**:
- Non-streaming: `POST /api/projects/{id}/chat` (returns JSON)
- Streaming: `POST /api/projects/{id}/chat/stream` (returns SSE)

**Retrieval functions**: `codeloom/core/stateless/`

The `stateless` module contains thread-safe retrieval functions that are safe to call from multiple FastAPI workers concurrently. API routes must use `stateless_query*()` methods and never call `pipeline.query()` directly (which mutates global state).

### Chat Request Schema

```python
class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None
    session_id: str | None = None
    include_history: bool = True    # Load last 10 conversation turns
    max_sources: int = 6            # Max retrieved chunks
    temperature: float | None = None
    response_type: str = "detailed" # "detailed" | "concise"
    mode: str = "chat"              # "chat" | "impact"
```

### Chat Processing Pipeline

```
1. Validate project (ast_status must be "complete")
2. Load conversation history (last 10 turns from conversations table)
3. fast_retrieve(query, project_id, top_k=max_sources)
        BM25 + vector hybrid search -> reranker -> ranked nodes
4. ASG expansion (if asg_status == "complete")
        ASGExpander.expand() adds graph neighbors with score_decay=0.7
5. Impact detection (if mode=="impact" OR query matches impact patterns)
        _build_blast_radius_context() -> markdown context + structured data
6. Deep analysis narrative enrichment (if deep_analysis_status=="completed")
        _get_relevant_narratives() for retrieved unit_ids
7. build_context_with_history()
        Combines: blast_radius_context + deep_narrative + retrieval chunks
8. execute_query_streaming() -> SSE token stream
9. save_conversation_turn()
```

### Conversation System Prompt

```python
CODE_SYSTEM_PROMPT = """You are a code intelligence assistant for the CodeLoom platform.
When answering questions about code:
- Cite file paths with line numbers (e.g., src/auth/login.py:42)
- Show function signatures when referencing code
- Format code blocks with appropriate language tags
- Explain how different pieces of code connect to each other
- Be precise about which functions call which, and what dependencies exist
"""
```

### SSE Event Stream Format

The streaming endpoint emits events in this sequence:

```
data: {"type": "sources",  "sources": [{...}]}              # Citations list

data: {"type": "content",  "content": "Here is..."}         # Token chunk
data: {"type": "content",  "content": " the answer..."}     # (many events)

data: {"type": "done",     "metadata": {
    "execution_time_ms": 1420,
    "session_id": "abc123",
    "node_count": 847,
    "retrieval_count": 6
}}
```

In impact mode, an additional event is emitted before `sources`:

```
data: {"type": "impact",   "impact": [{...}]}               # Blast radius data
```

Or, if the ASG is not available:

```
data: {"type": "impact_status", "status": "unavailable", "message": "..."}
```

### Auto-Detection of Impact Intent

Even in `mode="chat"`, the system automatically switches to impact analysis if the query contains known impact-analysis patterns:

```python
_IMPACT_PATTERNS = [
    r"impact",          r"blast\s*radius",  r"affected",
    r"depends?\s+on",   r"what\s+happens\s+if",
    r"change.*affect",  r"ripple",          r"downstream",
    r"upstream",        r"who\s+calls",     r"who\s+uses",
    r"breaking\s+change",
]
```

---

## 8. Migration Pipeline

**Entry point**: `codeloom/core/migration/engine.py`

The `MigrationEngine` class orchestrates an MVP-centric migration pipeline. The source codebase is partitioned into Functional MVPs (Minimum Viable Products), each migrated independently through a sequence of phases.

### Pipeline Versions

CodeLoom supports two pipeline versions. New plans always use V2.

**V1 (6-phase, legacy)**:

```
Plan-level:
  Phase 1: Discovery    - Inventory codebase, identify units and dependencies
  Phase 2: Architecture - Define target architecture and technology stack

Per-MVP:
  Phase 3: Analyze      - Deep analysis of MVP scope and complexity
  Phase 4: Design       - Design target-side architecture for MVP
  Phase 5: Transform    - Generate migrated code
  Phase 6: Test         - Generate test suite for migrated code
```

**V2 (4-phase, default)**:

```
Plan-level:
  Phase 1: Architecture - Define target architecture first
  Phase 2: Discovery    - Map source code to target architecture (MVP clustering)

Per-MVP:
  Phase 3: Transform    - Generate migrated code
  Phase 4: Test         - Generate test suite

On-demand:
  analyze_mvp()         - Merges old Analyze+Design into FunctionalMVP.analysis_output
```

### Creating a Migration Plan

```python
engine.create_plan(
    user_id=user_id,
    source_project_id=project_id,
    target_brief="Migrate from Spring Boot 2 to Quarkus 3 with reactive endpoints",
    target_stack={
        "framework": "quarkus",
        "version": "3.x",
        "language": "java",
        "build": "maven",
    },
    migration_type="framework_migration",  # "version_upgrade" | "framework_migration" | "rewrite"
)
```

### Human Approval Gates

Each phase requires explicit human approval before the next phase begins. The API exposes `POST /api/migration/{plan_id}/phases/{phase_num}/approve`. The frontend presents a diff view of the phase output and a confirm button.

```
Phase N output produced
    |
    v  (human reviews)
    |
    v  POST /approve
Phase N+1 begins
```

### MVP Clustering

**Path**: `codeloom/core/migration/mvp_clusterer.py`

During the Discovery phase, the `MvpClusterer` partitions the project's code units into functional groups. Each group becomes a `FunctionalMVP` — an independently migratable slice of the codebase.

Clustering considers:
- ASG edge density (highly connected units belong together)
- File and package co-location
- Configurable maximum cluster size (`_MAX_CLUSTER_SIZE`)

### Framework Documentation Enrichment

**Path**: `codeloom/core/migration/doc_enricher.py`

The `DocEnricher` fetches framework documentation from the web using Tavily search and injects it into the migration context. This gives the LLM accurate, up-to-date API documentation for the target framework during the Transform phase, reducing hallucinations about new APIs.

### Migration Plan Database Schema

```
migration_plans          Top-level plan record
    |
    +-- migration_phases  Individual phase records with outputs
    |
    +-- functional_mvps   Code unit clusters (one per functional slice)
```

---

## 9. Deep Understanding

**Entry point**: `codeloom/core/understanding/engine.py`

The Deep Understanding system performs a comprehensive, structured analysis of a codebase's behavior by tracing execution chains from entry points. The results enrich chat responses and migration analysis.

### What It Produces

For each entry point detected in the codebase, the system traces the call chain up to depth 10, analyzes what the chain does, and produces a structured narrative. These narratives are stored in the `deep_analyses` and `analysis_units` tables and surfaced in chat responses when relevant retrieved units overlap with analyzed chains.

### Processing Flow

```
POST /api/projects/{id}/understanding/start
    |
    v
UnderstandingEngine.start_analysis()
    Creates DeepAnalysisJob row (status=pending)
    Ensures UnderstandingWorker daemon is running
    |
    v
UnderstandingWorker (background thread)
    Polls for pending jobs (interval: 15s)
    Acquires semaphore (max_concurrent=2)
    |
    v
Entry point detection (Pass 1 + Pass 2):
    Pass 1 (heuristic): Functions with zero incoming "calls" edges in ASG
        (excludes test_* functions if skip_test_functions=true)
    Pass 2 (annotations): @RestController, @RequestMapping, @app.route, etc.
    Limit: max_entry_points=200 per project
    |
    v
ChainTracer.trace(entry_point, max_depth=10)
    BFS/DFS over ASG "calls" edges from entry point
    Collects code unit sources at each hop
    |
    v
Tiered analysis (based on token budget):
    Tier 1 (<=100,000 tokens): Full source included in LLM prompt
    Tier 2 (<=200,000 tokens): Depth-prioritized truncation
    Tier 3 (>200,000 tokens):  Summarization fallback
    |
    v
LLM produces structured AnalysisUnit
    Stored in analysis_units table
    Linked to deep_analyses job record
```

### Token Budget Tiers

```yaml
# config/codeloom.yaml
migration:
  deep_analysis:
    tier_1_max_tokens: 100000   # Full source
    tier_2_max_tokens: 200000   # Depth-prioritized truncation
    # Above 200k tokens -> Tier 3 (summarization fallback)
    max_trace_depth: 10
    max_entry_points: 200
```

### Job Status

```
pending  ->  running  ->  completed
                      ->  failed
                      ->  cancelled
```

Jobs that stall beyond `stale_threshold` (120s) are automatically reclaimed and retried up to `max_retries` (2) times with exponential backoff.

### Chat Integration

When a chat query retrieves units that overlap with completed deep analysis chains, the analysis narrative is prepended to the context:

```python
if project.get("deep_analysis_status") == "completed":
    narratives = _get_relevant_narratives(
        db_manager, project_id, result_unit_ids
    )
    if narratives:
        deep_narrative = "\n\n## FUNCTIONAL NARRATIVE\n" + "\n\n".join(narratives)
        context = deep_narrative + "\n\n" + context
```

---

## 10. Blast Radius and Impact Analysis

**Entry point**: `codeloom/api/routes/code_chat.py` (`_build_blast_radius_context`, `_compute_impact_score`)

Impact analysis answers the question: "If I change this code, what else will break?" It uses the ASG to trace dependent units and assigns a numerical impact score.

### Activation

Impact analysis runs when:
1. `mode="impact"` is set in the `ChatRequest`, OR
2. The query matches any pattern in `_IMPACT_PATTERNS` (auto-detection), AND
3. `asg_status == "complete"` for the project

### 4-Phase Expansion

```
Phase 1: Collect retrieval roots
    For each retrieved node, extract unit_id from metadata
    Deduplicate by unit_name

Phase 1b: Parent class resolution
    For method-level units, traverse incoming "contains" edges
    to find the parent class (implements/inherits edges live on the class)

Phase 2: ASG edge expansion (implements / inherits)
    Expand roots to include classes that implement or extend them
    Captures indirect dependents through polymorphism

Phase 3: Multi-root traversal (get_dependents)
    For all roots, query get_dependents(unit_id, project_id, depth=3)
    Walks outgoing "calls", "imports", "type_dep", "inherits", "implements" edges
    Collects: direct_count, indirect_count, affected_files

Phase 4: Build enriched LLM context
    Format as markdown with impact scores
    Emit as SSE "impact" event with structured data
```

### Impact Score Formula

The impact score is a weighted sum of five dimensions:

```python
def _compute_impact_score(
    direct: int,        # Number of direct dependents
    indirect: int,      # Number of indirect dependents (transitive)
    files_affected: int, # Number of distinct files touched
    total_files: int,   # Total files in project (for normalization)
    max_depth: int,     # Deepest dependency chain found
    edge_types: set,    # Which edge types were involved
    unit_type: str,     # Type of the source unit
) -> tuple[float, str]:  # (score, level)
```

| Dimension | Weight | Calculation |
|---|---|---|
| Reach | 40% | `min(1.0, (direct * 2 + indirect) / 50)` |
| Spread | 20% | `min(1.0, files_affected / (total_files * 0.2))` |
| Depth | 15% | `min(1.0, max_depth / 5)` |
| Coupling | 15% | Edge type diversity: inherits/implements (0.5), calls (0.3), type_dep (0.15), imports (0.05) |
| Criticality | 10% | Unit type: interface (1.0), class (0.8), method (0.5), property (0.3) |

**Score classification**:

| Score | Level | Meaning |
|---|---|---|
| >= 0.8 | `critical` | Changes here risk widespread breakage |
| >= 0.5 | `high` | Significant portion of codebase affected |
| >= 0.25 | `moderate` | Local subsystem impact |
| < 0.25 | `low` | Isolated change, minimal blast radius |

### Structured Impact Event

The SSE `type=impact` event carries this structure per retrieved unit:

```json
{
  "type": "impact",
  "impact": [
    {
      "unit_name": "AuthService",
      "file_path": "codeloom/core/auth/auth_service.py",
      "source": "retrieval",
      "direct": 8,
      "indirect": 23,
      "files_affected": 5,
      "impact_score": 0.74,
      "impact_level": "high",
      "dependents": [
        {
          "unit_id": "...",
          "name": "LoginHandler",
          "unit_type": "class",
          "edge_type": "calls",
          "depth": 1
        }
      ]
    }
  ]
}
```

---

## 11. UML Diagrams

**Entry point**: `codeloom/core/diagrams/service.py`

The `DiagramService` generates PlantUML diagrams scoped to a `FunctionalMVP`. Diagrams are cached in the project record and regenerated on demand.

### 7 Diagram Types

| Type | Category | Source | PlantUML |
|---|---|---|---|
| `class` | Structural | ASG: inherits, implements, contains, depends | `@startuml` class diagram |
| `package` | Structural | ASG: file groupings, inter-package edges | `@startuml` package diagram |
| `component` | Structural | ASG: calls, imports between components | `@startuml` component diagram |
| `sequence` | Behavioral | ASG call tree (deterministic, no LLM) | `@startuml` sequence diagram |
| `activity` | Behavioral | ASG call tree (deterministic, no LLM) | `@startuml` activity diagram |
| `usecase` | Behavioral | ASG entry points + actor detection | `@startuml` use case diagram |
| `deployment` | Behavioral | LLM-assisted, grounded by ASG infrastructure | `@startuml` deployment diagram |

### Two Generation Strategies

**Structural diagrams** (`class`, `package`, `component`): Generated deterministically from ASG data. No LLM calls. Queries in `codeloom/core/diagrams/queries.py` fetch class/interface/component data and their edges, then `codeloom/core/diagrams/structural.py` renders them as PlantUML.

**Behavioral diagrams** (`sequence`, `activity`, `usecase`): Also deterministic, driven by the `ChainTracer` call tree. Every arrow in the sequence or activity diagram corresponds to a real `calls` edge in the ASG. No hallucination about control flow.

**Deployment diagram**: LLM-assisted. The system detects infrastructure clues in the ASG (database access classes, HTTP clients, message queue references) and provides them as grounding context. The LLM then produces the PlantUML.

### PlantUML Rendering

**Path**: `codeloom/core/diagrams/renderer.py`

PlantUML text is converted to SVG via two methods in priority order:

```
1. Local JAR (preferred):
   java -jar tools/plantuml/plantuml.jar -tsvg -pipe
   No size limits. Requires JDK.
   JAR location: PLANTUML_JAR_PATH env var or tools/plantuml/plantuml.jar

2. Public HTTP API (fallback):
   https://www.plantuml.com/plantuml/svg/{encoded}
   Deflate compression + custom base64 URL encoding.
   Subject to PlantUML server availability and diagram size limits.
```

Download the JAR: `./dev.sh setup-tools`

### Diagram API

```
GET  /api/migration/{plan_id}/mvps/{mvp_id}/diagrams/{diagram_type}
     Returns: {diagram_type, category, puml, svg, title, cached, generated_at}

POST /api/migration/{plan_id}/mvps/{mvp_id}/diagrams/{diagram_type}/refresh
     Force regenerate (bypasses cache)
```

### Caching

Diagrams are stored as JSONB on the `functional_mvps` record. The cache key is `(plan_id, mvp_id, diagram_type)`. Structural diagrams are regenerated on `force_refresh`. Behavioral diagrams also check whether the ASG or deep analysis data has changed since the cache was written.

---

## 12. Configuration

CodeLoom uses two configuration layers: a YAML file for defaults and environment variables for secrets and provider selection.

### YAML Configuration

**Path**: `config/codeloom.yaml`

This file is the single source of truth for tunable parameters. It is divided into sections:

```yaml
ingestion:          # Chunk size, overlap, contextual retrieval toggle
retrieval:          # Strategy, BM25/vector weights, reranker model and top_k
llm:                # Temperature, top_k, keep_alive, request_timeout
raptor:             # Enabled, clustering params, summarization prompts, tree depth
sql_chat:           # Connection pool, query limits, security, masking
migration:          # Deep analysis tiers, worker concurrency, LLM overrides
```

Notable retrieval configuration:

```yaml
retrieval:
  strategy: "hybrid"              # hybrid | semantic | keyword
  similarity_top_k: 20            # Candidates before reranking
  retriever_weights: [0.5, 0.5]   # [BM25, vector]
  reranker:
    enabled: true
    model: "base"                 # xsmall | base | large
    top_k: 10                     # Final results to LLM
```

### Environment Variables

Copy `.env.example` to `.env` and set these:

```bash
# Provider selection
LLM_PROVIDER=ollama              # ollama | openai | anthropic | gemini | groq
EMBEDDING_PROVIDER=openai        # openai | huggingface
EMBEDDING_MODEL=text-embedding-3-small

# Database
DATABASE_URL=postgresql://codeloom:codeloom@localhost:5432/codeloom_dev
PGVECTOR_EMBED_DIM=1536          # Must match embedding model dimension

# Security
FLASK_SECRET_KEY=change-me       # Session middleware secret (legacy name)
OPENAI_API_KEY=sk-...            # Required if LLM_PROVIDER=openai
ANTHROPIC_API_KEY=sk-ant-...     # Required if LLM_PROVIDER=anthropic

# Optional overrides
RERANKER_MODEL=base              # xsmall | base | large | disabled
PLANTUML_JAR_PATH=/opt/plantuml.jar  # Custom JAR location
DISABLE_BACKGROUND_WORKERS=true  # For multi-worker Gunicorn deployments
```

### Plugin Architecture

**Paths**: `codeloom/core/plugins.py`, `codeloom/core/registry.py`

LLM providers, embedding providers, and retrieval strategies are registered as plugins. This allows new providers to be added without modifying the core engine.

Registration example:

```python
# codeloom/core/plugins.py
register_llm_provider("openai", OpenAILLMProvider)
register_llm_provider("anthropic", AnthropicLLMProvider)
register_llm_provider("ollama", OllamaLLMProvider)

register_embedding_provider("openai", OpenAIEmbeddingProvider)
register_embedding_provider("huggingface", HuggingFaceEmbeddingProvider)
```

The active provider is selected at startup from `LLM_PROVIDER` and `EMBEDDING_PROVIDER` env vars.

### Lazy Import System

**Path**: `codeloom/core/__init__.py`

To keep startup time fast and avoid importing heavy dependencies (PyTorch, LlamaIndex, ONNX Runtime) when only lightweight modules are needed, the core package uses `__getattr__` with an `_IMPORT_MAP`:

```python
# codeloom/core/__init__.py
_IMPORT_MAP = {
    "LocalRAGPipeline": "codeloom.pipeline",
    "DatabaseManager": "codeloom.core.db",
    "CodeIngestionService": "codeloom.core.ingestion.code_ingestion",
    # ...
}

def __getattr__(name):
    module_path = _IMPORT_MAP.get(name)
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module 'codeloom.core' has no attribute {name!r}")
```

Heavy imports only happen when the corresponding class is first accessed.

### Adding a New LLM Provider

1. Create `codeloom/core/providers/your_provider.py` implementing the provider interface
2. Register it in `codeloom/core/plugins.py`
3. Set `LLM_PROVIDER=your_provider` in `.env`
4. Restart the server

### Adding a New AST Parser

1. Subclass `BaseLanguageParser` in `codeloom/core/ast_parser/your_language_parser.py`
2. Implement `get_language()`, `get_tree_sitter_language()`, `extract_units()`, `extract_imports()`
3. Register the parser and file extensions in `codeloom/core/ast_parser/__init__.py`
4. No other changes required — the ingestion service will pick it up automatically

### Database Migrations

CodeLoom uses Alembic for schema migrations:

```bash
# Apply all pending migrations
alembic upgrade head

# Generate a new migration after editing core/db/models.py
alembic revision --autogenerate -m "add_my_column"

# Rollback one migration
alembic downgrade -1
```

Migration scripts live in `alembic/versions/`. The Alembic configuration is in `alembic.ini` with the `env.py` file in `alembic/`.

### Operational Notes

**Thread safety**: API routes must call `stateless_query()` and `stateless_query_streaming()` from `codeloom/core/stateless/`. Never call `pipeline.query()` or `pipeline.switch_project()` from route handlers — these methods mutate global pipeline state and are not safe for concurrent use.

**Node cache TTL**: The in-memory node cache in `pipeline.py` expires after 5 minutes. After any ingestion or document update, call `pipeline.invalidate_node_cache(project_id)` explicitly. The ingestion service does this automatically.

**Background workers**: Both the RAPTOR worker and the Understanding worker run as daemon threads. In multi-process deployments (Gunicorn with multiple workers), set `DISABLE_BACKGROUND_WORKERS=true` and run workers as separate processes.

**Segfault prevention**: Startup sets `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` to prevent crashes from PyTorch and HuggingFace tokenizers in multi-threaded FastAPI contexts. Do not remove these settings.

**Default credentials**: The default admin account is `admin` / `admin123`. Change this immediately in any non-development deployment by updating the users table or setting `ADMIN_PASSWORD` before first startup.
