<!-- Architecture Doc: Ingestion Pipeline -->

# Ingestion Pipeline

## Overview

The ingestion pipeline is the entry point for every codebase in CodeLoom. When a user provides a source — whether a zip archive, a Git repository URL, or a local directory path — the pipeline walks the file tree, parses each supported file into structured code units using tree-sitter, enriches those units with semantic metadata, breaks them into embeddable chunks, generates vector embeddings, persists everything to PostgreSQL, and finally builds the Abstract Semantic Graph (ASG) that maps the relationships between units. The result is a fully indexed, queryable representation of the codebase that powers both the RAG chat interface and the migration engine.

---

## Pipeline Flow

The following diagram shows the full ingestion sequence. AST parsing and ASG construction run sequentially; embedding is batched across all files.

```
 Source Input
      |
      v
 +--------------------+
 |  Extract / Clone   |  zip -> temp dir
 |                    |  git clone --depth 1 -> temp dir
 |                    |  local dir -> used in place
 +--------------------+
      |
      v
 +--------------------+
 |  Walk File Tree    |  skip vendor dirs, hidden dirs
 |  Collect Files     |  language detection by extension
 |                    |  size check: <= 50 MB per file
 |                    |  count check: <= 1000 files total
 +--------------------+
      |
      | (per file, sequential)
      v
 +--------------------+     +-------------------------+
 |  AST Parsing       | --> |  Semantic Enrichment     |
 |  (tree-sitter)     |     |  (enricher.py)           |
 |  BaseLanguageParser|     |  parsed_params           |
 |  -> ParseResult    |     |  return_type             |
 |     (CodeUnits)    |     |  modifiers, fields       |
 +--------------------+     +-------------------------+
      |                              |
      |                              | (optional, if runtime available)
      |                             +-------------------------+
      |                             |  Bridge Enrichment      |
      |                             |  JavaParser (Java)      |
      |                             |  Roslyn (C#)            |
      |                             +-------------------------+
      |
      v
 +--------------------+
 |  DB Storage        |  CodeFile record
 |                    |  CodeUnit records (one per unit)
 |                    |  file_imports stamped on first unit
 +--------------------+
      |
      v
 +--------------------+
 |  Code Chunking     |  one CodeUnit -> one TextNode
 |  (code_chunker)    |  preamble injection
 |                    |  split at ~1024 tokens if oversized
 +--------------------+
      |
      | (all files collected, then batched)
      v
 +--------------------+
 |  Embedding         |  batch size: 100 chunks per call
 |  (LlamaIndex)      |  default: OpenAI text-embedding-3-small
 |                    |  1536 dimensions
 +--------------------+
      |
      v
 +--------------------+
 |  Vector Store      |  PGVectorStore (pgvector)
 |  (pg_vector_store) |  metadata on each TextNode
 +--------------------+
      |
      v
 +--------------------+
 |  ASG Construction  |  reads all CodeUnits from DB
 |  (asg_builder)     |  domain-gated detectors
 |                    |  bulk insert CodeEdge records
 |                    |  ON CONFLICT DO NOTHING
 +--------------------+
      |
      v
 Project ready for query
```

---

## Source Types

`CodeIngestionService` exposes three public entry points. All three converge on the same internal `_ingest_directory()` method once a directory is available.

**Zip Upload** (`ingest_zip`): The uploaded `.zip` file is extracted to a temporary directory using Python's `zipfile` module. After ingestion completes, the temporary directory is removed unconditionally, even on error.

**Git Clone** (`ingest_git`): A shallow clone (`git clone --depth 1`) is performed into a temporary directory. The clone is limited to 300 seconds. If `git` is not on the system PATH, the error is surfaced cleanly. The branch name is recorded on the project record alongside the repository URL.

**Local Directory** (`ingest_local`): The directory is read in place — no copy is made. The path is validated as an existing directory before proceeding. This source type is intended for server-local codebases where copying would be wasteful.

After ingestion, `source_type`, `source_url`, and `repo_branch` are written to the project record for provenance tracking.

---

## AST Parsing

### Strategy Pattern

All language-specific parsing is handled by concrete subclasses of `BaseLanguageParser`, defined in `core/ast_parser/base.py`. Each parser implements four methods:

- `get_language()` — returns the language identifier string
- `get_tree_sitter_language()` — returns the tree-sitter `Language` object
- `extract_units()` — walks the parsed AST and produces a list of `CodeUnit` objects
- `extract_imports()` — extracts raw import statement strings from the AST

The base class `parse_file()` method handles file reading, tree-sitter parser creation, error checking, and assembly of the final `ParseResult`. Language-specific logic is entirely isolated in subclasses. Adding a new language means subclassing `BaseLanguageParser` and registering the parser in `core/ast_parser/__init__.py`.

### Supported Languages

| Language   | Parser File         | Bridge Available |
|------------|---------------------|-----------------|
| Python     | `python_parser.py`  | No              |
| JavaScript | `js_parser.py`      | No              |
| TypeScript | `ts_parser.py`      | No              |
| Java       | `java_parser.py`    | Yes (JavaParser)|
| C#         | `csharp_parser.py`  | Yes (Roslyn)    |

Language detection uses file extension. The `detect_language()` function in `core/ast_parser/__init__.py` maps extensions to language identifiers and returns `None` for unsupported files, which are skipped during file collection.

### ParseResult and CodeUnit

Each parser returns a `ParseResult` containing:

- `file_path` — relative path within the project root
- `language` — language identifier
- `units` — list of `CodeUnit` objects
- `imports` — list of raw import statement strings
- `line_count` — total lines in the file
- `module_docstring` — module-level docstring if present
- `errors` — list of `ParseError` objects for non-fatal issues

Each `CodeUnit` carries:

- `name` — short identifier (e.g., `save_user`)
- `qualified_name` — fully qualified name (e.g., `com.example.UserService.save_user`)
- `unit_type` — one of `function`, `method`, `class`, `interface`, `constructor`, `module`, `struct`, `record`, `stored_procedure`, etc.
- `language` — language identifier
- `start_line` / `end_line` — 1-indexed line range in the source file
- `signature` — declaration line (used for inheritance detection)
- `docstring` — doc comment if present
- `source` — full source text of the unit
- `parent_name` — containing class name for methods
- `metadata` — extensible dict for enrichment output

### Semantic Enrichment

After the initial parse, `SemanticEnricher` in `core/ast_parser/enricher.py` runs a second pass over the AST to add structured type information to each `CodeUnit.metadata`. This runs on Java, C#, Python, and TypeScript.

For callable units (methods, functions, constructors), the enricher adds:

- `parsed_params` — list of `{"name", "type", "default", "optional"}` dicts
- `return_type` — declared return type as a string
- `modifiers` — list of modifier keywords (e.g., `["public", "static"]`)
- `is_async` — boolean
- `is_override` — boolean (from `@Override` annotation or `override` modifier)
- `is_abstract` — boolean

For class, interface, and struct units, the enricher adds:

- `fields` — list of `{"name", "type"}` dicts representing field declarations

The `fields` metadata is consumed later by the ASG builder's `type_dep` detector. Without it, cross-class type dependency edges cannot be produced.

### Bridge Enrichment

For Java and C#, optional subprocess bridges provide deeper type resolution than tree-sitter alone can offer.

**JavaParser bridge** (`bridges/java_bridge.py`): Invokes a compiled JAR that runs JavaParser on the file and returns structured JSON with fully resolved types. Used when the JDK and Maven are installed and `./dev.sh setup-tools` has been run.

**Roslyn bridge** (`bridges/dotnet_bridge.py`): Invokes a compiled .NET executable that runs Roslyn on the file and returns structured JSON. Used when the .NET SDK is installed and `setup-tools` has been run.

Both bridges are subprocess-based and check `is_available()` before attempting to run. If the runtime is absent or the bridge process fails, the enrichment step is silently skipped — tree-sitter enrichment is the fallback. This means bridges are strictly additive and never required for successful ingestion.

---

## Code Chunking

### One Unit per TextNode

The `CodeChunker` in `core/code_chunker/chunker.py` converts each `CodeUnit` from a `ParseResult` into a LlamaIndex `TextNode`. The design is deliberately one-to-one: each code unit becomes exactly one node (or a small number of nodes if it is very large). This preserves the semantic boundary of the unit during retrieval — a query for a method returns that entire method, not an arbitrary slice of the file.

If a unit's token count (measured with tiktoken using the `cl100k_base` encoding) exceeds 1024 tokens, the unit source is split at line boundaries. Each sub-chunk receives the same preamble and a sequential name suffix (`_part1`, `_part2`, etc.).

### Preamble Injection

Before the unit source, `PreambleBuilder` prepends a short context header:

```
# File: src/core/engine/retriever.py
# Imports: from llama_index.core import VectorStoreIndex, from .models import CodeUnit (+3 more)
# Class: HybridRetriever
```

The preamble includes the file path, up to ten import statements (with a count of any omitted), and the parent class name if the unit is a method. The preamble is included in the text that is embedded, which grounds the embedding in file and class context. A query for "how does HybridRetriever load nodes" will match against chunks that carry `HybridRetriever` in their preamble even if that name does not appear in the method body itself.

The preamble lines are not included in the metadata keys passed to the embedding model separately — the full concatenated text (`preamble + unit source`) is the embedding input.

---

## Embedding and Storage

### Batch Embedding

After all files are processed, the list of `TextNode` objects is embedded in batches of 100. Each batch calls `embed_model.get_text_embedding_batch()` via LlamaIndex's `Settings.embed_model`, which dispatches to the configured provider.

The embedding model is selected at startup via environment variables:

| Variable            | Default                         | Notes                          |
|---------------------|---------------------------------|-------------------------------|
| `EMBEDDING_PROVIDER`| `openai`                        | `openai` or `huggingface`     |
| `EMBEDDING_MODEL`   | `text-embedding-3-small`        | Must match `PGVECTOR_EMBED_DIM`|
| `PGVECTOR_EMBED_DIM`| `1536`                          | 768 for nomic-embed-text       |

Batching avoids rate limit errors on large codebases and allows progress logging at each batch boundary.

### Vector Store

Embeddings and metadata are stored via `PGVectorStore` in `core/vector_store/pg_vector_store.py`, which wraps the pgvector extension in PostgreSQL. Each `TextNode` is stored as a row containing the embedding vector, the full text, and a JSONB metadata column.

The metadata written to each stored node includes:

| Field            | Description                                      |
|------------------|--------------------------------------------------|
| `unit_id`        | UUID linking back to the `code_units` DB record  |
| `project_id`     | Project UUID for scoped retrieval                |
| `source_id`      | `CodeFile` UUID                                  |
| `file_name`      | Relative file path                               |
| `node_type`      | Always `code` for code chunks                    |
| `unit_type`      | `function`, `method`, `class`, etc.              |
| `unit_name`      | Short name                                       |
| `qualified_name` | Fully qualified name                             |
| `class_name`     | Parent class if applicable                       |
| `language`       | Language identifier                              |
| `start_line`     | Start line in source file                        |
| `end_line`       | End line in source file                          |
| `signature`      | Declaration signature                            |
| `has_docstring`  | Boolean                                          |

Fields marked as `excluded_embed_metadata_keys` (unit_id, project_id, source_id, node_type, start_line, end_line, has_docstring) are stored but not concatenated into the embedding input text, keeping the embedding focused on semantic content.

---

## ASG Construction

### What the ASG Represents

The Abstract Semantic Graph is a directed graph where nodes are `CodeUnit` records and edges are typed semantic relationships between them. It captures structure, control flow at the declaration level, and type dependencies — producing a queryable representation of how code units relate to each other across the entire project. Edges are stored as `CodeEdge` records in the `code_edges` table and are used by the migration engine's impact analysis and by relationship-aware retrieval strategies.

### Edge Types

| Edge Type   | Meaning                                         | Detection Source                                    |
|-------------|--------------------------------------------------|-----------------------------------------------------|
| `contains`  | Parent unit contains child unit                  | `parent_name` field on CodeUnit                     |
| `imports`   | File-level unit imports another module or file   | Import statements resolved within project           |
| `calls`     | Function or method invokes another               | Identifier matching in unit source (regex)          |
| `inherits`  | Class extends a base class                       | `extends` metadata or signature pattern matching    |
| `implements`| Class or struct implements an interface          | `implements` metadata from parser                   |
| `overrides` | Method overrides a parent class method           | `@Override` annotation or `override` modifier       |
| `type_dep`  | Unit depends on a referenced type                | Field types, parameter types, return types in metadata |

### Domain-Gated Detectors

`ASGBuilder.build_edges()` in `core/asg_builder/builder.py` delegates edge detection to four domain-specific modules. The structural and OOP modules always run. The stored procedure and Struts modules are gated on whether relevant unit types exist in the project:

```
structural.detect_contains(ctx)   -- always runs
structural.detect_imports(ctx)    -- always runs
structural.detect_calls(ctx)      -- always runs

oop.detect_inherits(ctx)          -- always runs
oop.detect_implements(ctx)        -- always runs
oop.detect_overrides(ctx)         -- always runs
oop.detect_type_deps(ctx)         -- always runs

stored_proc.detect_sp_calls(ctx)  -- only if stored_procedure or sql_function units exist
struts_edges.detect_struts_edges(ctx)  -- only if struts_* or jsp_page units exist
```

Gating means a pure Python codebase does not pay the cost of scanning for stored procedure invocation patterns. The gate checks unit types present in the project before dispatching.

### EdgeContext: Shared Lookup Optimization

`EdgeContext` (defined in `core/asg_builder/context.py`) is built once at the start of `build_edges()` and passed to every detector. It holds:

- `units` — the full list of `CodeUnit` records for the project
- `unit_by_name` — index from short name to first matching unit
- `unit_by_qualified` — index from qualified name to unit
- `units_by_file` — units grouped by file ID
- `all_names` — set of all short names for fast membership checks

Without this shared context, each detector would need to build its own lookup tables or issue repeated database queries. Building the indexes once and sharing them keeps the ASG build pass to a single database read followed by in-memory analysis.

### Deduplication

After all detectors have run, the edge list is deduplicated in memory using a `(source_unit_id, target_unit_id, edge_type)` key set. The deduplicated batch is then inserted with PostgreSQL's `ON CONFLICT DO NOTHING` against the `uq_code_edge` unique constraint. This makes `build_edges()` idempotent — it can be called again on the same project without creating duplicate edges.

---

## Limits and Status Tracking

### Ingestion Limits

| Limit             | Value  | Enforcement Point                          |
|-------------------|--------|--------------------------------------------|
| Files per project | 500    | Checked in `_collect_files()` before parsing |
| File size         | 50 MB  | Checked per file in `_collect_files()`     |
| Git clone timeout | 300 s  | `subprocess.run(timeout=300)`              |

Files exceeding the size limit are silently skipped. If the total file count exceeds 1,000, ingestion is aborted with an error before any parsing begins.

### Status Tracking

The `projects` table carries two status columns that reflect pipeline progress:

| Column      | Values                                          | Updated By                         |
|-------------|--------------------------------------------------|-------------------------------------|
| `ast_status`| `parsing`, `complete`, `error`                  | `_ingest_directory()` stages        |
| `asg_status`| `building`, `complete`, `error`                 | After ASG construction completes    |

These columns are written at each major transition and are surfaced in the frontend as status indicators on the project card. A project with `ast_status = complete` and `asg_status = complete` is fully indexed and ready for query.

---

## Design Decisions

**Why tree-sitter for parsing**: tree-sitter provides a single, consistent parsing API across all supported languages. Parsers are compiled native libraries with no language runtime requirements — no JVM or CLR needed to parse Java or C# at the AST level. Parse speed is typically under 50 ms per file even for large files, which keeps ingestion throughput acceptable for codebases with hundreds of files. The tradeoff is that tree-sitter operates on syntax rather than semantics; it cannot resolve types across compilation units. This is why the optional bridges exist for Java and C#.

**Why preamble injection**: Embedding a method body in isolation loses the context that matters most for retrieval. A method named `process()` embedded without its file path and class name is nearly indistinguishable from any other `process()` in the codebase. Prepending the file path, parent class, and imports into the embedding input grounds the vector in the unit's structural location. Retrieval experiments show that preamble injection significantly improves precision for queries that reference file paths, class names, or library names that do not appear in the method body itself.

**Why pgvector over a dedicated vector database**: Storing embeddings in pgvector keeps the entire CodeLoom data model in a single PostgreSQL database. This enables metadata filtering in vector queries (e.g., restrict search to a specific project or language), ACID guarantees across the `code_units` and vector rows, and simpler operations — no second database to provision, back up, or monitor. The pgvector extension supports both exact and approximate nearest-neighbor search (HNSW and IVFFlat indexes) and is sufficient for codebases at the scale CodeLoom targets.

**Why domain-gated detectors in the ASG builder**: Running all possible edge detectors on every project wastes time on patterns that cannot exist in the project's language mix. A TypeScript frontend application will never contain stored procedure calls. Gating on unit types present in the project means the stored procedure and Struts detectors are skipped entirely for projects that do not need them. This keeps ASG construction fast for common cases while preserving full detection capability for enterprise Java or legacy Struts applications that do require it.

---

## Cross-References

- [01-platform-overview.md](./01-platform-overview.md) — Platform context, stack summary, and component map
- [03-query-engine.md](./03-query-engine.md) — How ingested data is retrieved: hybrid search, reranking, RAPTOR, and ASG-aware retrieval
