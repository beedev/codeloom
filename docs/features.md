# CodeLoom: Platform Features & Architecture

> Code intelligence and migration platform powered by AST + ASG + RAG.
> Users upload entire codebases, query them via AI-powered chat, and migrate
> to new architectures through a phased pipeline with human approval gates.

---

## Platform Overview

CodeLoom sits at the intersection of three technologies that don't normally
talk to each other:

```
                    +--------------------+
                    |    LLM / RAG       |  <-- Natural language querying
                    |    (LlamaIndex)    |
                    +---------+----------+
                              |
              +---------------+---------------+
              |               |               |
     +--------+-------+ +----+-----+ +-------+-----------+
     |  AST Parsing    | |   ASG    | |  Migration         |
     |  (tree-sitter)  | |  Graph   | |  Engine + Lanes    |
     +----------------+ +----------+ +-------------------+
              |               |               |
              +---------------+---------------+
                              |
                    +---------+----------+
                    |  PostgreSQL +       |
                    |  pgvector          |
                    +--------------------+
```

The system is designed so that improving any layer -- adding a new language
parser, adding a new edge type, adding a new migration lane -- automatically
improves every downstream capability. A new parser means new code is
queryable, graphable, and migratable. A new edge type means better clustering,
better context building, and better quality gates. A new lane means a new
migration path with the full infrastructure already in place.

---

## Feature 1: Multi-Language AST Parsing with Semantic Enrichment

**What it does**: Parses source code into structured `CodeUnit` records with
rich metadata -- not just syntax, but types, parameters, inheritance, and
annotations.

**Implementation**: `codeloom/core/ast_parser/`

The system supports 12 languages/file types through a two-tier parser
architecture:

| Tier | Languages | Parser | Metadata Depth |
|------|-----------|--------|----------------|
| Tree-sitter | Python, JavaScript, TypeScript, Java, C# | `BaseLanguageParser` subclasses | Full AST + enrichment |
| Regex-based | VB.NET, SQL, JSP, ASPX, XML config, Properties | Custom parsers | Pattern-extracted metadata |

The `BaseLanguageParser` (base.py) establishes a Strategy pattern -- each
language implements `extract_units()` and `extract_imports()` against a
tree-sitter parse tree. The `SemanticEnricher` (enricher.py) then runs a
second pass, walking the AST to extract structured type information:

- **For callables**: `parsed_params` (name + type per param), `return_type`,
  `modifiers`, `is_async`, `is_override`, `annotations`
- **For classes**: `fields` (name + type per field), `extends`, `implements`

This two-pass design matters. The first pass gives the *shape* of code (what
functions/classes exist). The second pass gives the *types* (what they consume
and produce). Without both, the ASG builder can't detect `type_dep` edges,
and the migration engine can't generate type-correct target code.

Optional enrichment bridges (`bridges/`) shell out to JavaParser (JVM) or
Roslyn (.NET SDK) for deeper semantic analysis when those runtimes are
available -- things like full type resolution that tree-sitter alone can't
provide. Built via `./dev.sh setup-tools`.

**How it enhances the platform**: Every downstream system -- chunking,
retrieval, ASG building, migration -- operates on these structured CodeUnit
records. The richer the metadata, the more intelligent everything else
becomes. A migration lane that knows a method takes `ByRef Integer` can
generate `ref int` in C#. A retrieval engine that knows a class implements
`IDisposable` can surface it when someone asks about resource management.

---

## Feature 2: Abstract Semantic Graph (ASG) -- Relationship Intelligence

**What it does**: Detects and stores relationships between code units -- who
calls whom, who inherits from whom, who depends on what types.

**Implementation**: `codeloom/core/asg_builder/`

After parsing, the `ASGBuilder` runs over all CodeUnits for a project and
produces `CodeEdge` records. This is the intelligence layer that transforms
a bag of parsed files into a navigable graph.

Eight core edge types detected through domain-specific modules:

| Module | Edges | Detection Strategy |
|--------|-------|--------------------|
| `structural.py` | `contains` | Parent-child nesting (metadata -> qualified name -> line range fallback) |
| `structural.py` | `imports` | Import statements (metadata first -> regex fallback) |
| `structural.py` | `calls` | Identifier intersection against known units + builtin filtering |
| `oop.py` | `inherits` | Metadata `extends` -> regex signature parsing fallback |
| `oop.py` | `implements` | Metadata `implements` list -> interface resolution |
| `oop.py` | `overrides` | `@Override`/`override` modifier + inheritance chain walking |
| `oop.py` | `type_dep` | Three strategies: param types, return types, field types |
| `stored_proc.py` | `calls_sp` | Language-specific SP invocation patterns (Java `prepareCall`, C# `SqlCommand`, Python `callproc`) |

Plus framework-specific edges (`struts.py`):

- `struts_action_class`, `struts_action_form`, `struts_action_forward` --
  from Struts XML config
- `struts_tile_include`, `jsp_includes` -- view layer relationships

The `EdgeContext` (context.py) is built once and shared across all detectors
-- it pre-indexes units by name, qualified name, and file, with a fast
`resolve_unit()` function that handles generics stripping, namespace
resolution, and fallback matching.

**How it enhances the platform**: The ASG is what makes CodeLoom's migration
qualitatively different from a text-based translator. When the migration
engine asks "what will break if I change this class?", the answer comes from
walking the graph -- incoming `calls` edges, `type_dep` edges, `inherits`
chains. When the MVP clusterer groups code into migration units, it uses ASG
edge density to measure cohesion (how tightly coupled is this group
internally?) and coupling (how many edges cross group boundaries?). Without
the ASG, migration is guesswork. With it, it's graph analysis.

---

## Feature 3: AST-Informed Code Chunking with Preamble Injection

**What it does**: Converts parsed CodeUnits into embeddable `TextNode` objects
where each chunk is self-contained with enough context to be useful in
isolation.

**Implementation**: `codeloom/core/code_chunker/`

The `CodeChunker` takes each CodeUnit and wraps it with a preamble:

```
# File: src/core/engine.py
# Imports: FooService, BarRepository, (+2 more)
# Class: EngineClass

def process_order(self, order_id: int) -> OrderResult:
    ...
```

Token counting uses tiktoken (`cl100k_base` encoding) with a default budget
of 1024 tokens per chunk. Oversized units split at blank line boundaries,
with the preamble replicated on each sub-chunk.

Each `TextNode` carries structured metadata: `unit_type`, `language`,
`qualified_name`, `signature`, `start_line`, `end_line`. This metadata is
excluded from embedding computation but available at retrieval time for
filtering and reranking.

**How it enhances the platform**: Standard code chunking (split at N
characters) produces chunks that are meaningless without their surrounding
file. CodeLoom's chunks carry their own context -- you know what file they're
from, what imports are in scope, what class they belong to. When the RAG
engine retrieves a chunk in response to a question, the LLM can understand
it without needing the full file. It's the difference between handing someone
a random paragraph vs. a paragraph with a header that says "Chapter 5:
Authentication -- LoginController class".

---

## Feature 4: Hybrid RAG with RAPTOR Hierarchical Retrieval

**What it does**: Powers the AI chat with a retrieval engine that combines
multiple search strategies and hierarchical document summaries.

**Implementation**: `codeloom/core/engine/`, `codeloom/core/raptor/`,
`codeloom/core/stateless/`

The retrieval stack is layered:

1. **BM25 sparse search** -- keyword matching (good for exact symbol names)
2. **Vector search** -- semantic similarity via pgvector (good for conceptual
   queries)
3. **Reranking** -- `mixedbread-ai/mxbai-rerank-base-v1` reorders results
   by relevance
4. **RAPTOR** -- hierarchical summaries at multiple abstraction levels

RAPTOR builds a tree of summaries in the background:

- **L0**: Individual code units (the raw chunks)
- **L1**: File-level summaries (semantic summary of all units in a file)
- **L2+**: Cluster summaries (groups of related files, built via UMAP + GMM
  clustering of L1 embeddings)

The `LocalRAGPipeline` orchestrates this with two API patterns:

- `stateless_query()` / `stateless_query_streaming()` -- thread-safe, used
  by API routes for multi-user access
- `query()` -- single-user mode with mutable global state

The pipeline maintains a node cache with 5-minute TTL to avoid reloading from
the database on every query. Session memory (`SessionMemoryService`) preserves
conversation context across requests (up to 100 messages, 24-hour TTL).

**How it enhances the platform**: A developer asking "how does authentication
work in this codebase?" needs answers from multiple files across multiple
abstraction levels. BM25 finds the files that mention "auth". Vector search
finds conceptually related code (session management, token validation).
RAPTOR's hierarchical summaries provide the architectural overview that no
single chunk contains. The combination means CodeLoom can answer both "what
does line 47 of auth.py do?" and "explain the authentication architecture"
-- from the same index.

---

## Feature 5: MVP-Centric Migration Pipeline with Human Approval Gates

**What it does**: Orchestrates full-codebase migration through a phased
pipeline where code is grouped into functional vertical slices (MVPs), each
migrated independently with LLM assistance and human oversight.

**Implementation**: `codeloom/core/migration/engine.py`, `phases.py`,
`mvp_clusterer.py`, `context_builder.py`

### Pipeline Versions

**V2 (default for new plans)** -- 4 phases:

```
Plan-level:     Phase 1 (Architecture) -> Phase 2 (Discovery/Clustering)
Per-MVP:        Phase 3 (Transform) -> Phase 4 (Test)
On-demand:      analyze_mvp() for deep analysis of individual MVPs
```

**V1 (legacy)** -- 6 phases:

```
Plan-level:     Phase 1 (Discovery) -> Phase 2 (Architecture)
Per-MVP:        Phase 3 (Analyze) -> Phase 4 (Design) -> Phase 5 (Transform) -> Phase 6 (Test)
```

Every phase has an approval gate -- output is presented to the user, who must
approve before the next phase runs. Phases can be rejected and re-executed.

### MVP Clustering

The `MvpClusterer` decides how to slice a monolith into migratable units.
Two strategies:

**Primary (RAPTOR-driven)**:
1. Takes L1 file summaries
2. UMAP dimensionality reduction
3. GMM clustering into ~15-25 functional groups
4. Assigns orphan files by embedding similarity
5. Computes cohesion/coupling from ASG edges

**Fallback (package-based)**:
1. Seeds from package/namespace structure
2. Mechanical merge/split based on cohesion threshold (0.3) and coupling
   threshold (0.7)
3. Size constraints: min 3, max 120 units per MVP

### Phase Context Building

The `MigrationContextBuilder` assembles phase-specific context strings from
ASG data within a token budget:

| Phase | Context Includes |
|-------|-----------------|
| Discovery | Project stats, edge statistics, hot spots |
| Architecture | Full dependency graph, class hierarchies, interface contracts |
| Transform | MVP source code, framework patterns, existing interfaces |
| Test | Code structure, test artifacts, coverage |

Budget management via `_join_within_budget()` greedily includes sections
until budget exhausted, prioritizing overview before detail.

**How it enhances the platform**: Most migration tools either translate
file-by-file (losing architectural context) or try to migrate everything at
once (overwhelming the LLM). CodeLoom's MVP clustering finds the natural
functional boundaries in the codebase -- a group of files that work together
to implement one feature -- and migrates each group as a self-contained unit.
The ASG edges ensure the clustering respects actual code dependencies, not
just directory structure. And the approval gates mean a human reviews each
step, catching errors before they propagate.

---

## Feature 6: Migration Lanes -- Pluggable Framework Intelligence

**What it does**: Encapsulates framework-specific migration knowledge
(transform rules, prompt augmentation, quality gates) in swappable lane
modules.

**Implementation**: `codeloom/core/migration/lanes/`

The `MigrationLane` ABC defines a contract that each lane implements:

| Method | Purpose |
|--------|---------|
| `detect_applicability(source, target)` | Score 0.0-1.0 for source/target match |
| `get_transform_rules()` | Deterministic source -> target templates |
| `apply_transforms(units, context)` | Generate target code from source units |
| `augment_prompt(phase, base_prompt, context)` | Inject domain knowledge into LLM prompts |
| `get_gates()` | Quality gate definitions |
| `run_gate(name, source, target, context)` | Execute quality gate checks |
| `get_asset_strategy_overrides()` | Per-unit-type migration strategy |

### Registered Lanes

| Lane | Source | Target | Rules | Key Capability |
|------|--------|--------|-------|----------------|
| `struts_to_springboot` | Struts 1/2, JSP | Spring Boot, React | 13+ | Action -> Controller+Service+Repo, JSP -> React+API+Types |
| `storedproc_to_orm` | SQL stored procedures | Service + Data Access | 5+ | SP -> Service method + Repository |
| `vbnet_to_dotnetcore` | VB.NET, WebForms, WinForms | .NET Core C#, React | 13 | CodeBehind -> Controller+Service+Repo, ASPX -> React+API+Types |

The `LaneRegistry` auto-detects which lane to use based on source framework
and target stack. Lanes score themselves --
`detect_applicability("struts", {"framework": "spring"})` returns 0.95 while
`detect_applicability("java", {"framework": "spring"})` returns 0.0.

### Concern Decomposition Philosophy

The core architectural philosophy shared across all lanes: a single monolithic
source file (Struts Action, WebForms code-behind, VB.NET class with mixed
concerns) produces 2-3 layered output files:

```
Source (one monolith)
    +-- Controller    (thin HTTP routing)
    +-- Service       (business logic with interface)
    +-- Repository    (data access, conditional on DB patterns)
```

This isn't syntactic translation -- it's architectural decomposition. The lane
analyzes the source to determine *what concerns are mixed*, then separates
them into the layered architecture the target framework expects.

### Confidence Scores and Quality Gates

Every transform carries a calibrated confidence score:

| Transform | Confidence | Rationale |
|-----------|-----------|-----------|
| Enum -> enum | 0.98 | Near-identical syntax |
| Module -> static class | 0.95 | Direct mapping |
| Interface -> interface | 0.95 | Direct mapping |
| Class -> C# class | 0.90 | Inheritance + modifiers need conversion |
| Code-behind -> Controller | 0.80 | Structural mapping but needs DI wiring |
| Code-behind -> Service | 0.75 | Business logic extraction is judgment call |
| ASPX -> React | 0.70 | Cross-technology, needs manual refinement |
| Code-behind -> Repository | 0.65 | ADO.NET -> EF Core is a paradigm shift |
| WinForms -> Blazor | 0.55 | Desktop -> web is a fundamental redesign |

Quality gates validate migration completeness:

- **Blocking gates**: class parity (every type mapped), method parity (every
  public method mapped)
- **Non-blocking gates**: page-component parity, event handler coverage,
  namespace preservation

### Prompt Augmentation

Each migration phase gets domain-specific context injected into the LLM
prompt:

- **Architecture phase**: Reference tables (VB.NET -> C# keywords, WebForms
  -> ASP.NET Core concepts, ASP controls -> React elements)
- **Transform phase**: Pre-completed deterministic transforms, service
  cross-references, state migration notes
- **Test phase**: Framework-specific test scaffolding (WebApplicationFactory
  for controllers, Moq for services, React Testing Library for components)

**How it enhances the platform**: Without lanes, the migration engine would
pass raw code to an LLM and say "convert this." With lanes, the LLM receives
a pre-completed code scaffold, domain-specific reference tables, and quality
gates that catch omissions. The LLM's job shifts from "figure out the
migration" to "fill in the business logic in this pre-built scaffold" -- a
much more tractable task.

---

## Feature 7: Framework-Aware Understanding

**What it does**: Detects which frameworks a codebase uses and extracts
framework-specific architectural knowledge.

**Implementation**: `codeloom/core/understanding/frameworks/`

Registered analyzers:

- **StrutsAnalyzer** -- Detects Struts 1/2 from unit types, extracts
  form-bean declarations, servlet filters, security config, interceptors,
  validation rules
- **SpringAnalyzer** -- Spring Boot patterns
- **AspNetAnalyzer** -- ASP.NET patterns

Each analyzer produces a `FrameworkContext` with structured fields:
`di_registrations`, `middleware_pipeline`, `security_config`,
`transaction_boundaries`, `aop_pointcuts`, `analysis_hints`.

The `DocEnricher` fetches target framework documentation dynamically --
topics generated from detected source patterns (not hardcoded), using Tavily
search API with graceful degradation when unavailable. Phase-specific:
Architecture phase fetches DI/routing docs, Transform phase fetches code
idiom docs, Test phase fetches test framework docs.

**How it enhances the platform**: A codebase isn't just files and functions
-- it's an application built on a framework. Knowing that "this project uses
Struts 1 with Tiles" means the migration engine can look for form-beans (DI
precursors), action forwards (routing equivalents), and tile definitions
(layout components). This framework-level understanding feeds into both the
LLM prompts and the lane selection.

---

## Feature 8: Cross-Technology View Layer Migration

**What it does**: Converts server-rendered markup (JSP, ASPX) into React SPA
components with API services and TypeScript types.

**Implementation**: Struts lane (`struts_to_springboot.py`) and VB.NET lane
(`vbnet_to_dotnetcore.py`)

This is the most ambitious transform pattern -- crossing from server-rendered
pages to a client-side SPA stack. One source page produces up to 3 output
files:

| Output | Condition | Contents |
|--------|-----------|----------|
| `src/components/{Name}.tsx` | Always | Functional component with useState, useEffect, handlers |
| `src/services/{name}Api.ts` | When event handlers exist | fetch-based API client with typed methods |
| `src/types/{name}.types.ts` | When data bindings exist | TypeScript interfaces from data expressions |

**ASP.NET server control mapping** (30+ controls):

| ASP Control | React Element | Notes |
|------------|---------------|-------|
| `asp:TextBox` | `<input type="text">` | value + onChange |
| `asp:Button` | `<button type="submit">` | onClick |
| `asp:DropDownList` | `<select>` | value + onChange |
| `asp:GridView` | `<table>` + `.map()` | Data-driven rows |
| `asp:Repeater` | `<div>` + `.map()` | Template items |
| `asp:HyperLink` | `<Link>` | react-router-dom |
| `asp:UpdatePanel` | `<div>` | Remove -- React handles updates |
| `asp:ScriptManager` | *(remove)* | Not needed in React |

**State migration**:
- `ViewState["key"]` -> `useState<string>('')` hooks
- `Session["key"]` -> Auth context / IDistributedCache migration comments
- `PostBack` -> `handleSubmit` with `fetch()` calls
- Validation controls -> HTML5 validation attributes

**Struts tag mapping** (JSP):
- `<html:text property="...">` -> `<input>` with controlled state
- `<html:form action="...">` -> `<form onSubmit={...}>`
- `<logic:iterate>` -> `.map()` iteration
- `<bean:write>` -> `{variable}` expression
- Struts 2: `<s:textfield>`, `<s:form>`, `<s:iterator>`

**How it enhances the platform**: This crosses a technology boundary that no
simple AST transformation can handle. The lane doesn't just translate syntax
-- it changes the entire rendering paradigm from server-side postback to
client-side SPA. The generated API service file creates the bridge between
the old server-rendered model and the new React + API architecture.

---

## Feature 9: Complete REST API with Approval Workflow

**Implementation**: `codeloom/api/routes/`

The REST API exposes the full platform through organized route groups:

| Route Group | Key Endpoints | Purpose |
|-------------|---------------|---------|
| `/api/auth/*` | login, logout, session | Session-based authentication |
| `/api/projects/*` | CRUD, upload, files, units, build-asg | Project and code management |
| `/api/projects/{id}/query/stream` | SSE streaming | RAG chat |
| `/api/migration/*` | plan, discover, mvps, phases, batch, diff, download | Full migration workflow |
| `/api/settings/*` | runtime config | LLM/embedding configuration |
| `/api/graph/*` | ASG visualization | Graph browsing |
| `/api/understanding/*` | deep analysis | Framework analysis |

### Migration API Workflow

The migration routes implement the full MVP lifecycle:

```
1. POST /migration/plan               -- Create plan (source + target)
2. GET  /migration/{id}/asset-inventory -- File-type breakdown + strategies
3. POST /migration/{id}/discover       -- Run clustering + create MVPs
4. GET  /migration/{id}/mvps           -- List MVPs with metrics
5. POST /migration/{id}/mvps/merge     -- Merge MVPs
6. POST /migration/{id}/mvps/{mid}/split -- Split MVP
7. POST /migration/{id}/phase/N/execute -- Run LLM phase
8. POST /migration/{id}/phase/N/approve -- Human approval gate
9. GET  /migration/{id}/phase/N/diff-context -- Source vs. migrated
10. GET /migration/{id}/phase/N/download -- Download generated files
11. POST /migration/{id}/batch/execute  -- Batch phase execution
```

**How it enhances the platform**: The API makes CodeLoom a complete product,
not a library. A React frontend (or any client) can drive the entire workflow
-- upload code, explore the ASG, chat about the codebase, plan a migration,
review each phase's output, approve or reject, download the result. The batch
operations make it practical for large codebases with 50+ MVPs.

---

## Feature 10: Pluggable LLM/Embedding Providers

**Implementation**: `codeloom/core/plugins.py`, `codeloom/core/registry.py`,
`codeloom/setting/`

Five LLM providers and three embedding providers, selected by environment
variable:

| `LLM_PROVIDER` | Models | Use Case |
|-----------------|--------|----------|
| `ollama` | Local models | Air-gapped, free |
| `openai` | GPT-4, etc. | Best quality |
| `anthropic` | Claude | Alternative |
| `gemini` | Gemini | Google ecosystem |
| `groq` | Mixtral, etc. | Fast inference |

| `EMBEDDING_PROVIDER` | Dimension | Use Case |
|----------------------|-----------|----------|
| `openai` | 1536 | Default, high quality |
| `huggingface` | 768 | Local, free |
| `nomic` | 768 | Local alternative |

Migration phases can use different LLMs for different purposes -- the phase
executor supports context-type overrides (`understanding_llm`,
`generation_llm`), so you could use a cheaper model for analysis and a more
capable one for code generation.

**How it enhances the platform**: Organizations have different constraints --
some can't send code to external APIs, some want the best quality regardless
of cost, some need fast iteration. The plugin architecture means CodeLoom
adapts to the environment rather than dictating it.

---

## Data Model

### Core Tables

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| `projects` | name, languages, ast_status, asg_status | Project container |
| `code_files` | file_path, language, line_count, raptor_status | Source files |
| `code_units` | unit_type, name, qualified_name, signature, unit_metadata (JSONB) | Parsed code elements |
| `code_edges` | source_unit_id, target_unit_id, edge_type, edge_metadata | ASG relationships |
| `migration_plans` | target_brief, target_stack, pipeline_version, migration_lane_id | Migration configuration |
| `migration_phases` | phase_type, status, output, approved | Pipeline phase records |
| `functional_mvps` | unit_ids, cohesion_score, coupling_score, migration_readiness | MVP groupings |

### Unit Types

```
function, class, method, constructor, property, event,
module, interface, struct, enum,
stored_procedure, sql_function,
struts_action, struts2_action, struts_form_bean, struts_forward,
struts_tile_def, struts_filter,
jsp_page, aspx_page, aspx_master, aspx_control,
xml_config, properties_file
```

### Edge Types

```
contains, imports, calls, inherits, implements, overrides,
type_dep, calls_sp,
struts_action_class, struts_action_form, struts_action_forward,
struts_action_input, struts2_action_class, struts2_result_view,
struts_tile_include, struts_validation_form, jsp_includes
```

---

## Data Flow Diagrams

### Code Ingestion Pipeline

```
User ZIP Upload
    |
    v
CodeIngestionService
    |
    +-- Extract files from ZIP
    +-- For each file:
    |     +-- Detect language (12 supported)
    |     +-- AST Parse (tree-sitter or regex) -> ParseResult
    |     +-- Semantic Enrich (type metadata)
    |     +-- Store CodeFile + CodeUnits in DB
    |     +-- Create TextNodes (chunker with preamble)
    +-- Bulk insert records
    +-- Create embeddings + store in pgvector
    +-- Trigger background tasks:
          +-- Build RAPTOR tree (hierarchical summaries)
          +-- Run deep understanding analysis
```

### ASG Building Pipeline

```
CodeUnits in DB (with source + metadata)
    |
    v
ASGBuilder.build_edges(project_id)
    |
    +-- Enrich class fields (if needed)
    +-- Load all units, build EdgeContext
    +-- Run edge detectors:
    |     +-- detect_contains()    -> contains edges
    |     +-- detect_imports()     -> imports edges
    |     +-- detect_calls()       -> calls edges
    |     +-- detect_inherits()    -> inherits edges
    |     +-- detect_implements()  -> implements edges
    |     +-- detect_overrides()   -> overrides edges
    |     +-- detect_type_deps()   -> type_dep edges
    |     +-- detect_sp_calls()    -> calls_sp edges (conditional)
    |     +-- detect_struts_edges()-> framework edges (conditional)
    +-- Deduplicate
    +-- Bulk insert CodeEdge records (ON CONFLICT DO NOTHING)
```

### Migration Pipeline (V2)

```
User creates migration plan
    |
    v
create_plan() -> MigrationPlan + Phase 1 (Architecture) + Phase 2 (Discovery)
    |
    v  [User approves Phase 1]
execute_phase(plan_id, phase_number=1)
    +-- build_phase_context() -> context from ASG + code
    +-- assemble prompt + lane augmentation
    +-- _call_llm() -> architecture output
    +-- store in MigrationPhase.output
    |
    v  [User approves Phase 2]
run_discovery(plan_id)
    +-- MvpClusterer.cluster() -> List[MVP]
    +-- Store FunctionalMVP records
    +-- Create per-MVP phases (Phase 3 + 4)
    |
    v  For each MVP:
    +-- execute_phase(phase_number=3, mvp_id=X) -> Transform
    +-- Lane applies deterministic transforms
    +-- LLM fills in business logic
    |   [User approves Phase 3]
    +-- execute_phase(phase_number=4, mvp_id=X) -> Test
    |   [User approves Phase 4]
    +-- Download migrated code (zip)
```

### RAG Chat Pipeline

```
User query
    |
    v
LocalRAGPipeline
    |
    +-- Choose engine:
    |     SimpleChatEngine (no docs) or
    |     CondensePlusContextChatEngine (RAG)
    |
    +-- For RAG path:
    |     +-- Hybrid search (BM25 + vector)
    |     +-- Reranking (mxbai-rerank-base-v1)
    |     +-- RAPTOR summaries (hierarchical context)
    |     +-- Return top-k chunks
    |
    +-- LLM response with retrieved context
    +-- Stream to user (SSE)
    +-- Store in session memory
```

---

## The Integration Effect

What makes CodeLoom more than the sum of its parts is how these features
compose:

1. **Parsing feeds everything**: AST metadata -> ASG edges -> chunk quality
   -> retrieval accuracy -> migration intelligence.

2. **ASG enables clustering**: Edge density metrics -> cohesion/coupling
   scores -> MVP boundaries -> incremental migration.

3. **Lanes leverage metadata**: Parser-extracted types + ASG relationships ->
   type-correct code generation + quality gates.

4. **RAG uses the full stack**: Chunked code + RAPTOR summaries + ASG context
   -> accurate answers at any abstraction level.

5. **Human approval closes the loop**: Every automated step has a checkpoint
   where domain expertise can correct course.

6. **Framework detection informs everything**: Detected patterns -> lane
   selection -> prompt augmentation -> doc enrichment -> transform rules.

The system follows a principle of **progressive enrichment**: raw files become
parsed units, parsed units become graph nodes with edges, graph nodes become
clustered MVPs, MVPs become migration targets with lane-specific transforms.
Each layer adds intelligence that the next layer consumes.

---

## Feature 11: Enterprise Migration Quality

**What it does**: Provides the reliability, auditability, and observability
infrastructure that makes migration output trustworthy enough for production
codebases -- deterministic reproducibility, taxonomy-based quality gates,
exponential-backoff retry, per-phase execution metrics, lane lifecycle
governance, and a confidence model with actionable tiers.

**Implementation**: `codeloom/core/migration/engine.py`,
`codeloom/core/migration/lanes/base.py`

---

### 11.1 Deterministic Execution and Reproducibility

Every phase execution is uniquely identified and auditable. The engine assigns
a `run_id` (UUID) to `MigrationPhase.run_id` at the start of each execution,
distinguishing re-runs of the same phase from one another.

On the first execution of any phase in a plan, the engine snapshots the active
lane versions into `MigrationPlan.lane_versions`:

```python
# engine.py — recorded once, on first execution
if plan.lane_versions is None and active_lanes_snapshot:
    plan.lane_versions = {
        lane.lane_id: lane.version for lane in active_lanes_snapshot
    }
```

The resulting JSONB field looks like:

```json
{
  "struts_to_springboot": "1.0.0",
  "storedproc_to_orm": "1.0.0"
}
```

LLM configuration (model, temperature) is recorded in
`phase_metadata.execution_metrics` per phase. Temperature is currently a
placeholder pending LLM callback integration; the model is read from
`LLM_PROVIDER` at execution time.

**Reproducibility boundary**: Deterministic transforms (same rules + same
source = same output) are fully reproducible. LLM-driven phases depend on
model temperature and are reproducible only when temperature is 0.

---

### 11.2 Quality Gate Taxonomy

The `GateCategory` enum establishes a structured taxonomy that every lane
uses to classify its gates:

| Category | Value | Meaning |
|----------|-------|---------|
| `PARITY` | `"parity"` | Structural migration completeness (endpoint counts, class counts) |
| `COMPILE` | `"compile"` | Target code compiles or builds without errors |
| `UNIT_TEST` | `"unit_test"` | Generated unit tests pass |
| `INTEGRATION` | `"integration"` | Integration test validation |
| `CONTRACT` | `"contract"` | API or interface contract verification |
| `REGRESSION` | `"regression"` | Regression detection against known baselines |

Each gate is defined with a `blocking` flag. Blocking gates prevent phase
approval; non-blocking gates are advisory:

```python
@dataclass
class GateDefinition:
    name: str          # e.g. "compile_check"
    description: str
    blocking: bool = True
    category: GateCategory = GateCategory.PARITY
```

Every registered lane must provide at minimum two baseline gates:

| Gate | Category | Blocking | Behavior |
|------|----------|----------|----------|
| `compile_check` | `COMPILE` | True | Pass-through when no external build system is available |
| `unit_test_check` | `UNIT_TEST` | False | Pass-through baseline; real results require external runner |

**Gate execution flow** (Transform phase only):

```
execute_phase()
    |
    +-- lane.apply_transforms()       -- produces transform result dicts
    |
    +-- _run_gates(active_lanes, ...) -- iterates all gates from all lanes
    |     |
    |     +-- lane.get_gates()        -- returns GateDefinition list
    |     +-- lane.run_gate(name, ...) -- executes and returns GateResult
    |
    +-- gates_all_passed = all(g["blocking"] and g["passed"])
    +-- store gate_results in phase_metadata
    |
    approve_phase()
    +-- block if any blocking gate failed
```

Gate results stored per phase:

```json
{
  "gate_results": [
    {
      "gate_name": "compile_check",
      "lane_id": "struts_to_springboot",
      "passed": true,
      "blocking": true,
      "category": "compile",
      "details": {}
    }
  ],
  "gates_all_passed": true
}
```

---

### 11.3 Reliability: Retry, Checkpoints, and Clean Rejection

**Exponential backoff retry**:

```
Constants:  MAX_RETRIES = 3
            BASE_BACKOFF_SECONDS = 5
            backoff = BASE_BACKOFF_SECONDS * (2 ^ retry_count)
```

On each failure the engine increments `phase_metadata.retry_count` and
stores `last_error`, `last_error_at`, `retryable=True`, and
`next_retry_after` (ISO 8601 timestamp). After `MAX_RETRIES`:

```
terminal_failure = True
retryable = False
phase.status = "error"
```

A phase with `terminal_failure` raises `ValueError` on any re-execution
attempt until `reject_phase()` is called.

**Transform-phase checkpoints**: If the lane's `apply_transforms()` raises
mid-way, the engine saves progress before propagating the error:

```json
{
  "checkpoint": {
    "processed_unit_ids": ["uuid1", "uuid2"],
    "total_units": 47,
    "last_error": "...",
    "timestamp": "2026-02-23T10:00:00"
  }
}
```

On re-execution the engine skips any `unit_id` already in
`processed_unit_ids`, resuming from where it failed rather than starting
over.

**Clean rejection via `reject_phase()`**: Clears `checkpoint`,
`terminal_failure`, `retryable`, `next_retry_after`, and resets
`retry_count` to 0. This produces a fully idempotent re-execution on the
next call -- the phase behaves as if it were brand new.

---

### 11.4 Observability and Measurement

**Per-phase execution metrics** stored in `phase_metadata.execution_metrics`:

| Field | Type | Source |
|-------|------|--------|
| `run_id` | UUID string | Generated at execution start |
| `started_at` | ISO 8601 | `datetime.utcnow()` before LLM call |
| `completed_at` | ISO 8601 | `datetime.utcnow()` after success |
| `duration_ms` | int | `time.monotonic()` delta |
| `llm_model` | string | `LLM_PROVIDER` env var |
| `llm_temperature` | null | Placeholder (requires LLM callback) |
| `token_usage` | null | Placeholder (requires LLM callback) |

**Migration scorecard** via `get_migration_scorecard(plan_id)`:

```python
{
    "plan_id": "...",
    "total_phases": 8,
    "phases_complete": 6,
    "phases_approved": 5,
    "phases_rejected": 1,
    "avg_confidence": 0.8421,
    "avg_confidence_tier": "standard",
    "gate_pass_rate": 0.9333,    # passed / total gate checks
    "total_tokens": 0,           # populated when LLM callback wired
    "total_cost": 0.0,
    "rework_rate": 0.1429,       # rejected / (complete + rejected)
    "total_duration_ms": 142300,
    "time_per_phase_ms": 17787.5,
    "lane_versions": {"struts_to_springboot": "1.0.0"},
    "pipeline_version": 2
}
```

**UI-surfaced fields** on every `get_phase_output()` response, available
for immediate frontend display without unpacking `phase_metadata`:

```json
{
  "confidence_tier": "standard",
  "phase_confidence": 0.812,
  "gates_all_passed": true,
  "requires_manual_review": false
}
```

---

### 11.5 Lane Lifecycle Governance

The `MigrationLane` ABC enforces lifecycle contracts on every registered lane:

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| `version` | `str` | Abstract (mandatory) | Semantic version; bump when rules or gates change |
| `deprecated` | `bool` | Default `False` | Deprecated lanes are skipped by `LaneRegistry.detect_lane()` |
| `min_source_version` | `Optional[str]` | Default `None` | Minimum source framework version supported (inclusive) |
| `max_source_version` | `Optional[str]` | Default `None` | Maximum source framework version supported (inclusive) |

All four properties are surfaced by `LaneRegistry.list_lanes()` for
operator inspection. Lane versions are committed to `MigrationPlan.lane_versions`
at first execution, creating an audit trail that ties every generated artifact
to the exact lane version that produced it.

**Deprecation flow**:

```
lane.deprecated = True
    -> LaneRegistry.detect_lane() skips this lane
    -> Existing plans referencing this lane_id continue to use it
       (version already recorded on plan.lane_versions)
    -> New plan creation will not auto-select a deprecated lane
```

---

### 11.6 Confidence Model

Confidence flows from individual transform rules through aggregation to a
plan-level scorecard.

**Constants and tiers** (`codeloom/core/migration/lanes/base.py`):

```python
CONFIDENCE_HIGH     = 0.90   # auto-approvable; UI flags green
CONFIDENCE_STANDARD = 0.75   # normal review required
# below STANDARD     = low   # mandatory deep review; UI shows warning
```

**Aggregation**: `aggregate_confidence(transform_results, weights=None)` computes
a weighted average of the `confidence` field across all transform result dicts.
The optional `weights` dict maps `rule_name` to a weight multiplier; unweighted
results default to weight 1.0. Returns 0.0 for an empty result list.

**Review escalation**: `TransformRule.requires_review = True` propagates from
the rule definition through the transform result dict into
`phase_metadata.requires_manual_review`. This flag is independent of the
confidence score -- a high-confidence transform can still be flagged for review
if the rule author deems the output inherently judgment-sensitive.

**End-to-end confidence flow**:

```
TransformRule.confidence (per rule, calibrated at rule-definition time)
    |
    +-- lane.apply_transforms() -> TransformResult.confidence (copied from rule)
    |
    +-- aggregate_confidence(all_transforms) -> phase_confidence_score (float)
    +-- confidence_tier(score)               -> "high" | "standard" | "low"
    |
    +-- stored in phase_metadata:
    |     phase_confidence, confidence_tier, requires_manual_review
    |
    +-- surfaced in get_phase_output() for UI consumption
    +-- aggregated across phases in get_migration_scorecard()
```

**How the tiers drive behavior**:

| Tier | Score Range | Engine Behavior | Recommended UI Treatment |
|------|-------------|-----------------|--------------------------|
| high | >= 0.90 | No gate escalation | Green indicator, one-click approve available |
| standard | 0.75 -- 0.89 | Normal review flow | Neutral indicator, review checklist shown |
| low | < 0.75 | `requires_manual_review` set | Warning indicator, deep review required |

**How it enhances the platform**: These six quality subsystems compose into
a single coherent story -- every run is uniquely identified (`run_id`),
the lane that produced it is version-pinned (`lane_versions`), failures
retry without data loss (checkpoint resume), every phase publishes timing
and gate metrics (scorecard), lanes declare their own compatibility and
deprecation lifecycle, and confidence tiers surface the right level of
human attention for each output. Together they shift migration from "LLM
generates code, hope for the best" to "measurable, auditable, progressively
reviewable engineering process."
