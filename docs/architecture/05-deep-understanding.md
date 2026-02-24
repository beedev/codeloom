<!-- Architecture Doc: Deep Understanding -->

# Deep Understanding Engine

**Series**: CodeLoom Architecture Documentation
**Document**: 05 of 05
**Scope**: `codeloom/core/understanding/` — entry point detection, call chain tracing, framework analysis, impact scoring, narrative generation, background worker, and migration integration

---

## Overview

The Deep Understanding Engine extends CodeLoom beyond structural parsing. Where the ingestion pipeline produces an AST node graph and the ASG builder maps typed relationships between code units, the understanding engine asks a different question: what does this code actually do? It detects the points where execution enters the application, traces complete call chains from those entry points through the ASG, identifies framework-specific patterns that govern behavior, and feeds the resulting full execution paths to an LLM for business rule extraction. The output is a structured `DeepContextBundle` per entry point — machine-readable knowledge about business rules, data entities, integrations, and side effects — paired with a plain-language narrative that enriches both chat responses and migration phase prompts.

---

## Analysis Lifecycle

User-triggered analysis is asynchronous. Submitting a deep analysis request creates a job record and returns immediately; a background worker picks up the job, runs the full pipeline, and writes results back to the database. The frontend polls job status and renders results when the worker completes.

```
  User clicks "Run Deep Analysis"
          |
          v
  POST /api/understanding/{project_id}/analyze
          |
          v
  ┌──────────────────────────────────┐
  │  UnderstandingEngine.start_      │
  │  analysis(project_id)            │
  │                                  │
  │  1. Create DeepAnalysisJob row   │
  │     status = "pending"           │
  │  2. Submit to job queue          │
  └────────────────┬─────────────────┘
                   │
                   │  (returns immediately — HTTP 202 Accepted)
                   │
                   v
  ┌──────────────────────────────────┐
  │  Background Worker               │
  │  (asyncio task + semaphore)      │
  │                                  │
  │  status = "running"              │
  │  ┌────────────────────────────┐  │
  │  │ 1. detect_entry_points()   │  │
  │  │    heuristic + annotation  │  │
  │  ├────────────────────────────┤  │
  │  │ 2. detect_and_analyze()    │  │
  │  │    framework detection     │  │
  │  ├────────────────────────────┤  │
  │  │ 3. trace_call_tree()       │  │
  │  │    per entry point         │  │
  │  ├────────────────────────────┤  │
  │  │ 4. analyze_chain()         │  │
  │  │    LLM extraction          │  │
  │  ├────────────────────────────┤  │
  │  │ 5. store results           │  │
  │  │    deep_analyses + embed   │  │
  │  └────────────────────────────┘  │
  │  status = "complete" | "failed"  │
  └──────────────────────────────────┘
          |
          v
  GET /api/understanding/{project_id}/results
  (frontend polls; renders narratives and bundles)
```

Job status transitions: `pending` → `running` → `complete` or `failed`. A failed job records the error phase and message and is eligible for retry with exponential backoff.

---

## Entry Point Detection

Entry points are the roots of the application's execution graph — the code units that receive control from outside the process. Identifying them correctly is the precondition for meaningful call chain tracing. The `ChainTracer` runs a two-pass detection strategy.

### Pass 1: Zero-Incoming-Calls Heuristic

The ASG builder records a `calls` edge for every function invocation it can resolve. A code unit with no inbound `calls` edges is not called by anything in the project — it is either unused or it is an entry point that receives control from an external caller (an HTTP framework, a job scheduler, a message broker, or the runtime itself). The heuristic queries `code_edges` for the set of `target_unit_id` values reached by `calls` edges, then takes the complement: units that never appear as a target.

This is a conservative heuristic. It produces false positives (genuinely unused code) and misses entry points that are called internally as well as externally. Pass 2 corrects both categories.

### Pass 2: Annotation Pattern Matching

Framework annotations declare intent explicitly. A method decorated with `@GetMapping("/orders")` is an HTTP endpoint regardless of whether anything in the project calls it. Pass 2 scans each code unit's source text and metadata against a table of known annotation patterns per language:

| Language | Entry point patterns (examples) |
|---|---|
| Java | `@RequestMapping`, `@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping`, `@PatchMapping`, `@RestController`, `@Controller`, `@Scheduled`, `@KafkaListener`, `@JmsListener`, `@EventListener`, `@PostConstruct` |
| C# | `[HttpGet]`, `[HttpPost]`, `[HttpPut]`, `[HttpDelete]`, `[Route(...]`, `: ControllerBase`, `: Controller`, `IHostedService`, `BackgroundService` |
| Python | `@app.get(`, `@router.post(`, `@click.command`, `def main(`, `if __name__ == "__main__"`, `@celery_app.task` |
| JavaScript / TypeScript | `router.get(`, `app.post(`, `export async function GET`, `export async function POST`, `@Get(`, `@MessagePattern` |

Each matched unit receives a typed `EntryPointType` classification: `HTTP_ENDPOINT`, `MESSAGE_HANDLER`, `SCHEDULED_TASK`, `CLI_COMMAND`, `EVENT_LISTENER`, `STARTUP_HOOK`, or `PUBLIC_API`. When a unit appears in both passes, the annotation-derived type takes precedence over the heuristic's `UNKNOWN` classification. The two result sets are merged by `unit_id` and deduplicated.

### Why Entry Points Matter

Entry points serve three downstream purposes. First, they are the roots of call chain tracing — the starting node for the BFS/DFS traversal. Second, they define the scope of a deep analysis: each entry point gets its own `DeepContextBundle`, so the LLM analyzes one complete execution path at a time rather than the entire codebase at once. Third, they feed into migration MVP clustering, where entry points act as natural grouping centers for related code units.

---

## Call Chain Tracing

Once entry points are identified, `ChainTracer.trace_call_tree()` follows the execution graph outward from each root. The goal is to collect, in traversal order, every code unit that participates in handling a request or event that arrives at that entry point.

### Graph Traversal Strategy

The traversal uses a recursive Common Table Expression (CTE) against the `code_edges` table, extending the `_traverse()` pattern established in `core/asg_builder/`. The CTE carries two columns beyond the standard edge walk: an `ARRAY` path accumulator that records every `unit_id` seen on the current path, and a `depth` counter. An edge is only followed if its `target_unit_id` is not already in the path array — this prevents cycles without requiring a separate visited-set maintained in application memory.

Three edge types are followed during tracing:

- `calls` — the primary traversal edge: function A calls function B
- `contains` — descends into a class to reach its methods when the entry point is a controller class rather than a specific action method
- `imports` — crossed when a file import leads to a module that is itself a callable unit

The edges `inherits`, `implements`, `overrides`, and `type_dep` are not followed during call tree tracing. They carry structural information relevant to impact analysis but would expand the traversal scope far beyond actual execution paths.

### Depth Limit and Cycle Detection

The default maximum depth is 10 call levels. Most application call stacks resolve within 5 to 7 levels; the depth cap prevents runaway traversal in codebases with deep utility hierarchies or indirect recursion that the cycle detector cannot catch across multiple paths. When the depth limit is reached, the tree node is recorded as a leaf even if the target unit has outgoing edges.

The resulting structure is a `CallTreeNode` tree: the root is the entry point, and each node holds a list of children corresponding to units it directly calls. Each node carries the full source text of the code unit, its depth in the traversal, and the edge type that connected it to its parent.

### Token Budget Awareness

Before passing a call tree to the LLM analyzer, the worker calculates the total token count across all nodes. Three tiers govern how the chain is presented:

| Tier | Condition | Strategy |
|---|---|---|
| Tier 1 | Total tokens <= 100K | Full source for every node — no truncation |
| Tier 2 | Total tokens <= 200K | Depth-prioritized truncation — shallower nodes kept complete, deeper nodes summarized |
| Tier 3 | Total tokens > 200K | Summarization fallback — LLM receives pre-summarized sub-trees rather than raw source |

The `DeepContextBundle` records which tier was used and whether the chain was truncated, so downstream consumers can calibrate confidence accordingly.

---

## Framework Detection

Different frameworks impose different conventions on code structure. A Spring Boot `@Service` bean, an ASP.NET Core `IHostedService`, and a Struts 1 `ActionForm` all represent dependency injection registrations, but they look nothing alike in source text. The framework detection layer surfaces this semantic layer so that LLM prompts receive framework-aware context rather than treating all code as generic Java or C#.

### FrameworkAnalyzer Abstract Base Class

Every framework analyzer implements the `FrameworkAnalyzer` ABC from `core/understanding/frameworks/base.py`. The contract has three methods:

- `detect(project_id) -> bool` — returns True if this framework is present in the project. Detection is fast: it queries `code_units.unit_type` for framework-specific unit types or scans for characteristic file names in `code_files`.
- `analyze(project_id) -> FrameworkContext` — runs a deeper pass to extract framework-specific metadata. Returns a `FrameworkContext` dataclass.
- `get_context(project_id) -> FrameworkContext` — convenience wrapper; calls `detect` and `analyze` together.

The registry in `frameworks/__init__.py` maintains an ordered list of analyzers. `detect_and_analyze()` runs each analyzer's `detect()` method in priority order (Struts, then Spring, then ASP.NET Core). For every framework that is detected, it calls `analyze()` and returns a list of serialized `FrameworkContext` dicts. Multiple frameworks can be detected in the same project (for example, a monorepo with both a Spring backend and a .NET gateway).

### Built-In Framework Analyzers

**Spring / Spring Boot** (`frameworks/spring.py`)

The Spring analyzer detects the presence of `@Service`, `@Repository`, `@Component`, and `@Controller` annotated classes in `code_units.unit_metadata`. It extracts:

- DI registrations: beans declared via component scanning annotations — each is recorded as a `name -> stereotype` mapping (e.g., `UserService -> @Service`)
- AOP pointcuts: `@Aspect` classes and their `@Around`, `@Before`, `@After` declarations — these represent cross-cutting concerns that operate outside the normal call chain
- Transaction boundaries: methods annotated with `@Transactional`, including propagation and isolation attributes — critical for migration because transaction semantics must be preserved
- Security configuration: `@PreAuthorize`, `@Secured`, and Spring Security filter chain registrations

Analysis hints injected into LLM prompts include guidance on Spring's proxy-based AOP (self-invocation does not trigger advice), `@Transactional` propagation rules, and the implications of prototype vs. singleton bean scoping.

**ASP.NET Core** (`frameworks/aspnet.py`)

The ASP.NET Core analyzer identifies middleware pipeline registrations in the `Configure` and `ConfigureServices` methods, which are the canonical entry points for ASP.NET Core application setup. It extracts:

- Middleware pipeline: ordered list of `app.Use*()` and `app.Map*()` calls — ordering matters because middleware executes in registration order
- DI registrations: `services.AddSingleton`, `services.AddScoped`, and `services.AddTransient` calls from `ConfigureServices` — each maps an interface to an implementation
- Controller routes: `[Route(...)]` attributes and conventional route patterns
- Security configuration: `services.AddAuthentication`, `services.AddAuthorization`, and policy definitions

**Apache Struts** (`frameworks/struts.py`)

The Struts analyzer detects `struts_action` and `struts2_action` unit types, which are produced by the XML config parser when it processes `struts-config.xml` (Struts 1) or `struts.xml` (Struts 2). It distinguishes between versions by checking which unit type dominates and cross-references `pom.xml` dependency declarations for a version string. It extracts:

- Action mappings: the path-to-action-class bindings that constitute the application's URL surface
- Form beans (`struts_form_bean` units): the request binding objects — analogous to DTOs in modern frameworks
- Servlet filters (`xml_filter` units from `web.xml`): the filter chain that precedes action dispatch
- Security attributes: actions with `validate="true"` or explicit `input` page declarations

Struts 1 analysis hints warn about singleton actions (instance variables cause thread-safety bugs), the validation lifecycle (`validate()` before `execute()`), and JSTL/EL migration patterns. Struts 2 hints address OGNL injection risks, interceptor ordering, and dynamic method invocation (DMI) security concerns.

### FrameworkContext Injection

The serialized `FrameworkContext` dicts are injected into the LLM prompt that drives the chain analysis step. Rather than asking the LLM to infer what `@Transactional` means from its occurrence in source text, the prompt provides an explicit preamble: "This project uses Spring Boot. The following beans are registered in the DI container: [...]. The following AOP aspects are active: [...]. Transaction boundaries are at: [...]." The LLM can then focus on extracting business rules rather than reverse-engineering the framework.

---

## Impact Analysis

Impact analysis answers a specific migration question: if a team modifies or replaces a given code unit, how many other units are affected? This is the blast radius of a change.

### Edge Traversal for Blast Radius

The `UnderstandingEngine` computes impact by traversing the `code_edges` graph outward from a given code unit — following edges in reverse (looking for units that depend on the target rather than units that the target calls). All edge types are eligible: a unit that `calls` a function is affected if that function changes; a class that `inherits` from a base class is affected by changes to the base; a unit with a `type_dep` edge to a changed interface is affected because it consumes that interface's contract.

### Decay Scoring

Not all affected units carry equal risk. A direct caller is more likely to require changes than a transitive caller two hops away. The impact score decays with graph distance using a configurable multiplier (default 0.7 per hop, consistent with the retrieval engine's ASG expansion weights). A unit at depth 1 receives score 1.0; at depth 2 it receives 0.7; at depth 3, 0.49; and so on. Units with scores below a minimum threshold are excluded from the result set.

The output is an ordered list of affected code units with their impact scores, file paths, and the edge type that connects them to the changed unit. This list feeds directly into the migration engine's risk assessment: high-impact units — those with many dependents at shallow depth — are flagged as requiring careful validation during the Transform phase.

---

## Deep Analysis Narratives

Structured extraction data in a `DeepContextBundle` is useful for programmatic processing, but not for the humans and LLM prompts that need to understand what a call chain does. Each bundle also carries a `narrative` field: a plain-language description generated by the LLM after it has read the full call chain source.

### Business Rule Extraction

The chain analyzer prompt instructs the LLM to identify business rules embedded in the code. A business rule is any conditional or computational logic that encodes a domain constraint — validation logic, eligibility checks, pricing calculations, state transition guards, authorization conditions. Each extracted rule is stored with one or more `EvidenceRef` objects that link the rule back to the specific code unit and line range where it was found. This traceability allows reviewers to verify LLM-extracted rules against the source.

### Entity Identification

The LLM also identifies domain entities — the nouns of the business domain that the code operates on. An entity might be a JPA `@Entity` class, a form bean, a data transfer object, or an implicit grouping of related fields across several classes. The analysis records what CRUD operations each entity undergoes within the call chain, which external systems it flows to (database writes, API calls, message publishes), and what side effects its manipulation triggers.

### Narrative Injection into Chat

When a user asks the chat interface a question about a feature or flow, the retrieval engine may identify a `deep_analyses` row whose entry point matches the query context. The pre-rendered narrative string from that row is injected between the RAPTOR summary layer and the raw code evidence layer in the LLM context window. This means chat responses about analyzed features describe business behavior — "this endpoint validates the order total against the user's credit limit before calling the payment service" — rather than only returning code fragments.

---

## Integration with Migration

Deep understanding results are not stored in isolation. The migration engine reads them at two points in the pipeline.

### Framework Detection Informs Lane Selection

When a migration plan is created, the engine checks whether deep analysis has been run on the project. If framework contexts are available, the lane selector reads them to determine which migration lanes apply. A project with detected Struts action mappings automatically routes to the Struts-to-SpringBoot migration lane, which provides Struts-specific prompts and transformation strategies for each phase. Without deep understanding, the lane selector falls back to language-based defaults.

### Entry Points Drive MVP Clustering

The MVP clusterer in `core/migration/mvp_clusterer.py` groups code units into cohesive migration slices. When entry points are available, they serve as natural cluster seeds: the clusterer places each entry point's call chain as the core of a candidate MVP, then merges or splits based on size constraints (maximum cluster size is enforced). This produces MVPs that align with application features rather than arbitrary file boundaries.

### Call Chains Establish Dependency Order

Phase sequencing within a migration plan benefits from call chain data. If MVP A's entry point calls a shared utility that MVP B also calls, the migration engine can identify that utility as a cross-cutting dependency. The phasing logic uses this to recommend migrating the shared utility before either MVP that depends on it, reducing the risk of mid-migration breakage.

### Impact Scores Feed Risk Assessment

Before the Transform phase executes for a given MVP, the engine computes an aggregate impact score for the units in that MVP's call chain. Units with high blast radii — many dependents across other MVPs — are annotated as high-risk in the phase prompt. The LLM is instructed to preserve their interface contracts precisely and to flag any deviations in its transform output for human review.

---

## Background Worker

Deep analysis is computationally expensive: traversing large call graphs, calling an LLM with 100K+ token inputs, and storing and embedding the results can take several minutes per project. All of this happens off the request path.

### Job Model

The `deep_analysis_jobs` table records every analysis request. Each row tracks: `project_id`, `status` (`pending`, `running`, `complete`, `failed`), `created_at`, `started_at`, `completed_at`, `error_phase`, `error_message`, and `retry_count`. The worker queries for the oldest pending job, claims it by updating `status` to `running`, and proceeds with analysis. Only one job per project runs at a time; concurrent requests for the same project are detected and the second request returns the existing pending job ID.

### Retry with Backoff

If the worker encounters a transient failure — an LLM provider timeout, a database connection drop, a temporary embedding service outage — it increments `retry_count`, sets `status` back to `pending`, and schedules the next attempt after a delay. The delay follows an exponential backoff schedule: 30 seconds after the first failure, 2 minutes after the second, 8 minutes after the third. After a configurable maximum retry count (default 3), the job is marked `failed` permanently and the error details are preserved for operator inspection.

### Concurrency Control

The worker uses a semaphore to limit how many chain analysis LLM calls run simultaneously. Each call can consume tens of thousands of tokens; running too many in parallel exhausts token-per-minute rate limits and causes cascading failures. The default semaphore width is 3 concurrent LLM calls, configurable via `config/codeloom.yaml` under `understanding.max_concurrent_analyses`. The semaphore is scoped to the worker process, not the database, so it does not coordinate across multiple worker replicas. In multi-worker deployments, total concurrency is the semaphore width multiplied by the replica count.

### Status Tracking

The `code_files` table carries `ast_status` and `asg_status` columns that track per-file ingestion progress. The understanding engine adds analysis progress at the project level through the `deep_analysis_jobs` table. The frontend polls `GET /api/understanding/{project_id}/status` to display job progress. When the job transitions to `complete`, a subsequent call to `GET /api/understanding/{project_id}/results` returns the full list of `DeepContextBundle` summaries for display in the Understanding Dashboard.

---

## Module Layout

```
codeloom/core/understanding/
  __init__.py          Public API exports
  models.py            Data contracts: EntryPoint, CallTreeNode,
                       DeepContextBundle, AnalysisError, EvidenceRef
  chain_tracer.py      ChainTracer: detect_entry_points(), trace_call_tree()
  analyzer.py          ChainAnalyzer: tiered LLM chain analysis
  worker.py            Background job processor: retry, semaphore, status updates
  engine.py            UnderstandingEngine: public API, orchestration
  prompts.py           Prompt templates for chain analysis and narrative generation
  frameworks/
    __init__.py        detect_and_analyze() registry (ordered: Struts, Spring, ASP.NET)
    base.py            FrameworkAnalyzer ABC, FrameworkContext dataclass
    spring.py          Spring Boot / Spring MVC analyzer
    aspnet.py          ASP.NET Core analyzer
    struts.py          Apache Struts 1.x / 2.x analyzer
```

---

## Cross-References

This document covers the Deep Understanding Engine. The following documents describe the subsystems it depends on and those it enriches:

- **02 — Ingestion Pipeline**: `docs/architecture/02-ingestion-pipeline.md`
  AST parsing, ASG edge construction, and the `code_units` and `code_edges` tables that the understanding engine traverses.

- **03 — Query Engine**: `docs/architecture/03-query-engine.md`
  Hybrid retrieval, RAPTOR, and the context assembly process that injects deep analysis narratives into chat responses.

- **04 — Migration Engine**: `docs/architecture/04-migration-engine.md`
  The 6-phase pipeline, MVP clustering, lane detection, and transform phase execution that consume framework contexts, entry points, call chains, and impact scores produced by this engine.
