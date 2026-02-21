# Deep Understanding Engine — Architecture Document

> Transforming CodeLoom from a shallow migration tool into an enterprise-grade reverse-engineering and code intelligence platform.

*Created: February 19, 2026*
*Status: Architecture Design — Pending Approval*

---

## 1. The Problem

CodeLoom's current migration pipeline is **shallow**. Each of the 6 migration phases makes a single LLM call with token-budgeted context — roughly 15% of the source code per MVP. The LLM infers business rules from function names, never traces complete execution paths, and generates code it hasn't deeply read.

### What "Shallow" Looks Like Today

```
User uploads Java Spring codebase (200 files, 50K lines)
         |
         v
Phase 1 (Discovery): LLM sees ~7,500 tokens of code snippets
         |            Guesses business rules from function names
         |            Never follows a request from controller -> service -> repository
         v
Phase 5 (Transform): LLM gets ~7,500 tokens per MVP
                      Rewrites code it has never fully read
                      Misses data validation rules hidden in utility classes
                      Drops side effects (email, audit logging) it never saw
```

### What's Missing

| Capability | Current State | Enterprise-Grade |
|-----------|--------------|-----------------|
| **Entry Point Detection** | None — treats all units equally | Identifies API endpoints, CLI commands, scheduled jobs, event handlers |
| **Call Chain Tracing** | 1-hop ASG expansion at 0.7x decay | Full recursive call trees (depth 10+) with path reconstruction |
| **Business Rule Extraction** | LLM guesses from snippets | LLM reads complete execution paths, extracts structured rules |
| **Data Flow Analysis** | None | Tracks entities through CRUD operations across the call chain |
| **Integration Mapping** | None | Identifies external API calls, message queues, file I/O |
| **Side Effect Detection** | None | Maps DB writes, email sends, audit logging, cache invalidation |
| **Cross-Cutting Concerns** | None | Identifies auth, logging, error handling shared across flows |
| **Enriched Chat** | Returns code snippets only | Returns business-level narratives alongside code |
| **LLM Selection** | Single global LLM for all phases | Different LLMs for understanding vs. code generation |

---

## 2. The Vision

Transform CodeLoom into a **deep reverse-engineering and enrichment engine**. Instead of sending snippets to an LLM and hoping it guesses right, we:

1. **Trace every entry point** through the complete call graph
2. **Feed the full source code** of each call chain to the LLM
3. **Extract structured understanding**: business rules, data entities, integrations, side effects
4. **Embed that understanding** for RAG retrieval
5. **Enrich everything**: chat responses get functional narratives, migration phases get deep context
6. **Let users choose LLMs** per task type: one for understanding, another for code generation

### Architecture Overview

```
+-------------------------------------------------------------------------+
|                         React Frontend (:3000)                           |
|  +-----------+  +--------------+  +---------------+  +--------------+   |
|  | Code Chat |  | Project View |  | Migration     |  | Understanding|   |
|  | (Enriched)|  | (ASG Graph)  |  | Wizard        |  | Dashboard    |   |
|  +-----+-----+  +------+-------+  +-------+-------+  +------+-------+   |
+---------+---------------+------------------+-----------------+----------+
          |               |                  |                 |
          v               v                  v                 v
+-------------------------------------------------------------------------+
|                      FastAPI Backend (:9005)                              |
|                                                                          |
|  /api/projects/{id}/query  |  /api/migration  |  /api/understanding      |
|                            |                  |                          |
|  +-------------------------+------------------+--------------------+     |
|  |                  Enriched Context Layer                          |     |
|  |                                                                  |     |
|  |  Functional Narratives <-- Deep Analysis Results                |     |
|  |  (injected between RAPTOR summaries and code evidence)          |     |
|  +---------------------------+-------------------------------------+     |
|                              |                                           |
|  +---------------------------+-------------------------------------+     |
|  |              Deep Understanding Engine (NEW)                     |     |
|  |                                                                  |     |
|  |  +-------------+  +--------------+  +-----------------------+   |     |
|  |  | Chain       |  | Chain        |  | Understanding         |   |     |
|  |  | Tracer      |  | Analyzer     |  | Worker                |   |     |
|  |  |             |  |              |  | (Background Thread)   |   |     |
|  |  | Entry Point |  | LLM-powered  |  |                       |   |     |
|  |  | Detection   |  | Functional   |  | Job Queue -> Process  |   |     |
|  |  | + Recursive |  | Analysis per |  | -> Store -> Embed     |   |     |
|  |  | Call Trees  |  | Call Chain   |  |                       |   |     |
|  |  +------+------+  +------+-------+  +-------+---------------+   |     |
|  |         |                |                   |                   |     |
|  +---------+----------------+-------------------+-------------------+     |
|            |                |                   |                         |
|  +---------+----------------+-------------------+-------------------+     |
|  |                  Existing Infrastructure                          |     |
|  |                                                                  |     |
|  |  ASG Graph (code_edges)  |  pgvector Embeddings  |  LLM Providers|     |
|  |  _traverse() CTE         |  PGVectorStore        |  Settings.llm |     |
|  +-------------------------------------------------------------- --+     |
|                                                                          |
|  +------------------------------------------------------------------+    |
|  |              Per-Phase LLM Selection (NEW)                        |    |
|  |                                                                   |    |
|  |  Understanding LLM --> discovery, architecture, analyze, deep     |    |
|  |  Generation LLM    --> transform, test, design                    |    |
|  |  Falls back to Settings.llm when no override configured           |    |
|  +------------------------------------------------------------------+    |
+--------------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------------+
|                     PostgreSQL + pgvector                                  |
|                                                                           |
|  Existing:                         |  New:                                |
|  +---------------+ +------------+  |  +------------------+                |
|  |data_embeddings| |code_units  |  |  |deep_analysis_jobs|                |
|  |(code vectors) | |(AST nodes) |  |  |(batch job queue) |                |
|  +---------------+ +------------+  |  +------------------+                |
|  +---------------+ +------------+  |  +------------------+                |
|  |code_edges     | |migration   |  |  |deep_analyses     |                |
|  |(ASG graph)    | |_plans      |  |  |(per-entry-point  |                |
|  +---------------+ +------------+  |  | analysis results)|                |
|                                    |  +------------------+                |
|                                    |  +------------------+                |
|                                    |  |analysis_units    |                |
|                                    |  |(junction: analysis|               |
|                                    |  | <-> code_units)  |                |
|                                    |  +------------------+                |
+--------------------------------------------------------------------------+
```

---

## 3. Component Architecture

### 3.1 Chain Tracer — Entry Point Detection + Call Tree Reconstruction

**Location**: `codeloom/core/understanding/chain_tracer.py` (NEW)

**Purpose**: Detect all entry points in a codebase and trace their complete call trees using the existing ASG graph.

#### Entry Point Detection

Uses a **dual-pass strategy** to avoid missing framework-managed entry points that have incoming edges from framework infrastructure (component scanning, reflection-based routing, proxy chains):

**Pass 1 — Heuristic (zero incoming calls)**:
Scans `code_units` for units with zero incoming `calls` edges. This catches standalone entry points not invoked by other application code.

```sql
-- Pass 1: Units with zero incoming "calls" edges = potential entry points
SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
       u.signature, u.metadata, u.language
FROM code_units u
LEFT JOIN code_edges e
    ON e.target_unit_id = u.unit_id
    AND e.edge_type = 'calls'
    AND e.project_id = :pid
WHERE u.project_id = :pid
  AND u.unit_type IN ('function', 'method')
  AND e.id IS NULL
```

**Pass 2 — Annotation/metadata-based (regardless of incoming edges)**:
Scans all `code_units` for known entry point annotations and patterns in `signature`, `metadata`, and `modifiers`. This catches framework-invoked handlers that DO have incoming edges (from framework infrastructure, reflection, or proxy wiring).

```sql
-- Pass 2: Annotation-based detection (catches framework-managed entry points)
SELECT u.unit_id, u.name, u.qualified_name, u.unit_type,
       u.signature, u.metadata, u.language
FROM code_units u
WHERE u.project_id = :pid
  AND u.unit_type IN ('function', 'method')
  AND (
    -- Spring annotations in metadata or signature
    u.metadata->>'annotations' ILIKE ANY(ARRAY['%RequestMapping%', '%GetMapping%',
        '%PostMapping%', '%PutMapping%', '%DeleteMapping%', '%Scheduled%',
        '%EventListener%', '%MessageMapping%'])
    -- ASP.NET attributes
    OR u.metadata->>'annotations' ILIKE ANY(ARRAY['%HttpGet%', '%HttpPost%',
        '%HttpPut%', '%HttpDelete%', '%Route%'])
    -- Python decorators
    OR u.signature ILIKE ANY(ARRAY['%@app.route%', '%@router.%',
        '%@click.command%', '%@scheduled%'])
    -- JS/Express patterns
    OR u.signature ILIKE ANY(ARRAY['%app.get(%', '%app.post(%', '%router.get(%'])
    -- Main methods
    OR (u.name = 'main' AND u.language IN ('java', 'csharp', 'python'))
  )
```

**Merge and deduplicate**: Results from both passes are unioned by `unit_id`. Pass 2 catches what Pass 1 misses:

| Scenario Pass 1 Misses | Why | Pass 2 Catches It |
|------------------------|-----|-------------------|
| Spring `@Controller` method called by DispatcherServlet | Has incoming `calls` edge from framework proxy | `@RequestMapping` annotation in metadata |
| ASP.NET action method invoked by MVC pipeline | Has incoming edge from middleware chain | `[HttpGet]` attribute in metadata |
| Reflection-based route handlers | Called via `Method.invoke()` which ASG doesn't trace | Annotation/attribute pattern match |
| Config-driven endpoints (XML Spring, web.config) | No direct call edge, but not zero-incoming either | Signature pattern match on known patterns |

Then filter by signature/metadata pattern matching in Python to classify each entry type:

| Pattern | Entry Type | Languages |
|---------|-----------|-----------|
| `@app.route`, `@router.get/post/put/delete` | `api_endpoint` | Python (Flask, FastAPI) |
| `@GetMapping`, `@PostMapping`, `@RequestMapping` | `api_endpoint` | Java (Spring) |
| `@Controller`, `@RestController` methods | `api_endpoint` | Java (Spring) |
| `[HttpGet]`, `[HttpPost]`, `[Route]` | `api_endpoint` | C# (ASP.NET) |
| `app.get(`, `app.post(`, `router.get(` | `api_endpoint` | JavaScript (Express) |
| `def main(`, `if __name__` | `cli_command` | Python |
| `public static void main(` | `cli_command` | Java |
| `@click.command`, `@app.cli.command` | `cli_command` | Python |
| `@Scheduled`, `@scheduled_task` | `scheduled_job` | Java, Python |
| `@EventListener`, `on_event`, `handle_` | `event_handler` | Multiple |

#### Call Tree Reconstruction

Extends the existing `_traverse()` recursive CTE (`codeloom/core/asg_builder/queries.py:436`) with a PostgreSQL `ARRAY` path accumulator for **tree reconstruction with cycle prevention**:

```sql
WITH RECURSIVE chains AS (
    -- Seed: direct callees of the entry point
    SELECT
        target_unit_id AS current_id,
        ARRAY[source_unit_id::text, target_unit_id::text] AS path,
        1 AS depth
    FROM code_edges
    WHERE source_unit_id = :entry_id
      AND edge_type = 'calls'
      AND project_id = :pid

    UNION ALL

    -- Recurse: follow callees, prevent cycles via ARRAY membership check
    SELECT
        e.target_unit_id,
        c.path || e.target_unit_id::text,
        c.depth + 1
    FROM code_edges e
    JOIN chains c ON e.source_unit_id = c.current_id
    WHERE c.depth < :max_depth
      AND e.project_id = :pid
      AND e.edge_type = 'calls'
      AND NOT (e.target_unit_id::text = ANY(c.path))  -- cycle prevention
)
SELECT path, depth FROM chains ORDER BY depth DESC
```

**Key difference from existing `_traverse()`**: The current implementation returns flat lists with `MIN(depth)`. This new CTE returns **full paths as arrays**, allowing reconstruction of the complete tree structure as nested dicts.

**Methods**:
```python
class ChainTracer:
    def __init__(self, db: DatabaseManager): ...

    def detect_entry_points(self, project_id: str) -> List[Dict]
    """Returns [{unit_id, name, qualified_name, entry_type, entry_metadata, ...}]"""

    def trace_call_tree(self, project_id: str, entry_unit_id: str, max_depth: int = 10) -> Dict
    """Returns nested tree: {unit_id, name, ..., children: [{unit_id, name, ..., children: [...]}]}"""

    def trace_all_entry_points(self, project_id: str) -> Dict[str, Dict]
    """Returns {entry_unit_id: call_tree} for all detected entry points"""
```

### 3.2 Chain Analyzer — LLM-Powered Functional Analysis

**Location**: `codeloom/core/understanding/analyzer.py` (NEW)

**Purpose**: For each entry point's call tree, fetch the full source code of every unit in the tree, send it to the LLM with structural context, and extract structured functional understanding.

#### Analysis Flow

```
Entry Point Call Tree
    |
    v
Fetch Source Code (ordered by call depth)
    |  All units in tree: source, signature, docstring
    |  Preserve call relationship structure
    v
Build Analysis Prompt
    |  Source code + signatures + call graph structure
    |  Using chain_analysis_prompt() from prompts.py
    v
Call LLM (via _call_llm pattern from phases.py:130)
    |
    v
Parse Structured JSON Response
    |
    v
Store in DeepAnalysis row (JSONB columns)
```

#### LLM Output Schema

For each entry point, the LLM produces structured JSON:

```json
{
  "business_purpose": "Processes customer payments for orders",
  "business_rules": [
    {
      "id": "BR-1",
      "rule": "Orders over $10,000 require manager approval",
      "location": "validate_order",
      "behavior": "Throws InsufficientApprovalError if amount > 10000 and no manager flag",
      "evidence": {
        "unit_id": "uuid-of-validate-order",
        "file_path": "src/services/order_service.py",
        "line_hint": 142
      }
    }
  ],
  "data_entities": [
    {
      "name": "Order",
      "operations": ["read", "update"],
      "evidence": {
        "unit_id": "uuid-of-order-repo",
        "file_path": "src/repositories/order_repo.py",
        "line_hint": 35
      }
    },
    {
      "name": "Payment",
      "operations": ["create"],
      "evidence": {
        "unit_id": "uuid-of-payment-service",
        "file_path": "src/services/payment_service.py",
        "line_hint": 78
      }
    },
    {
      "name": "AuditLog",
      "operations": ["create"],
      "evidence": {
        "unit_id": "uuid-of-audit-logger",
        "file_path": "src/utils/audit.py",
        "line_hint": 15
      }
    }
  ],
  "integrations": [
    {
      "type": "external_api",
      "target": "Stripe",
      "description": "Charges customer card via stripe.create_charge()",
      "evidence": {
        "unit_id": "uuid-of-stripe-client",
        "file_path": "src/clients/stripe_client.py",
        "line_hint": 44
      }
    },
    {
      "type": "message_queue",
      "target": "RabbitMQ",
      "description": "Publishes payment_completed event",
      "evidence": {
        "unit_id": "uuid-of-event-publisher",
        "file_path": "src/events/publisher.py",
        "line_hint": 22
      }
    }
  ],
  "side_effects": [
    {
      "type": "db_write",
      "description": "Creates payment record in payments table",
      "evidence": {
        "unit_id": "uuid-of-payment-repo",
        "file_path": "src/repositories/payment_repo.py",
        "line_hint": 56
      }
    },
    {
      "type": "email",
      "description": "Sends order confirmation to customer",
      "evidence": {
        "unit_id": "uuid-of-email-service",
        "file_path": "src/services/email_service.py",
        "line_hint": 30
      }
    },
    {
      "type": "audit",
      "description": "Logs payment event to audit_log",
      "evidence": {
        "unit_id": "uuid-of-audit-logger",
        "file_path": "src/utils/audit.py",
        "line_hint": 15
      }
    }
  ],
  "cross_cutting": [
    {"concern": "authentication", "units": ["validate_token", "check_permissions"]},
    {"concern": "logging", "units": ["audit_log", "request_logger"]}
  ],
  "data_flow": {
    "input": "order_id, card_details",
    "transformations": ["validate_order", "charge_card", "record_payment"],
    "output": "payment_id, receipt_url"
  }
}
```

**Evidence contract**: Every extracted artifact (business rule, data entity, integration, side effect) includes an `evidence` object with `unit_id`, `file_path`, and `line_hint`. This enables:
- Migration outputs to cite the source code that backs each business rule
- Users to click through from extracted rules to actual code
- Validation that LLM extractions are grounded in real code, not hallucinated

#### Bounded Analysis Strategy

Enterprise codebases can produce call chains with 50+ units totaling hundreds of thousands of tokens — far exceeding any LLM context window. The analyzer uses a **tiered strategy** to bound the source code sent to the LLM:

**Tier 1 — Full Source (default, chain token count <= context budget)**:
When the total source code of all units in the call chain fits within the configured token budget (default: 100K tokens), send everything. This is the common case for most entry points.

**Tier 2 — Depth-prioritized Truncation (chain exceeds budget)**:
When the chain is too large, prioritize source code by depth from the entry point:
1. Always include full source for depth 0-2 (entry point + immediate callees + their callees)
2. For depth 3+, include only signatures + docstrings + first 20 lines of body
3. Fill remaining budget with the highest-connectivity units (most incoming/outgoing edges)

**Tier 3 — Summarization Fallback (chain exceeds 2x budget)**:
For extremely large chains (e.g., a Spring controller that transitively touches 200+ classes):
1. Depth 0-1: full source
2. Depth 2-3: signatures + docstrings only
3. Depth 4+: unit names and edge relationships only (graph structure without source)
4. Include a `"chain_truncated": true` flag in the analysis output so consumers know coverage is partial

**Configuration**:
```yaml
# config/codeloom.yaml
migration:
  deep_analysis:
    analysis_token_budget: 100000    # Tier 1 threshold
    truncation_budget: 200000        # Tier 2 -> Tier 3 threshold
    max_depth: 10                    # Hard limit on call tree depth
    priority_depth: 2                # Depth guaranteed full source
```

**Token counting**: Uses the existing `token_counter.py` from `core/code_chunker/` to count tokens consistently with the chunking pipeline.

#### Prompt Design

**Location**: `codeloom/core/understanding/prompts.py` (NEW)

Three prompt functions:

1. **`chain_analysis_prompt(chain_source_code, chain_signatures, chain_call_graph, language, framework_hints)`** — Per-entry-point analysis. Sends the full source of all units in the call tree with their relationships. The `framework_hints` parameter provides detected framework context (see below).

2. **`cross_cutting_prompt(chain_summaries, concern_type)`** — Aggregate detection of cross-cutting concerns across multiple entry points (auth patterns, logging patterns, error handling).

3. **`framework_detection_prompt(project_metadata, sample_units)`** — Fallback framework detection for projects where no code-level analyzer triggers. For projects detected by the code-level analyzers (Section 3.7), `framework_hints` are produced directly from `FrameworkContext` without an LLM call.

#### Framework-Aware Analysis (Two Layers)

Framework understanding operates at **two complementary layers**: code-level analyzers (Section 3.7) that augment the ASG graph with config-driven edges, and prompt-level instructions that guide the LLM to interpret framework semantics in source code.

**Layer 1 — Code-Level Analyzers (Spring + ASP.NET)**: See Section 3.7. These parse configuration files, DI registrations, middleware pipelines, and AOP pointcuts to inject additional `code_edges` into the graph *before* chain tracing. This makes framework-wired relationships visible to the tracer and produces structurally complete call trees.

**Layer 2 — Prompt-Level Semantic Instructions**: The `chain_analysis_prompt` includes framework-specific extraction instructions when `framework_hints` is present (produced by the code-level analyzers):

| Framework | Extra Extraction Instructions |
|-----------|-------------------------------|
| **Spring** | Identify `@Transactional` boundaries and propagation. Flag `@Aspect` cross-cutting. Note `@Qualifier`/`@Primary` DI ambiguity. Detect Spring Security filter chain participation. |
| **ASP.NET** | Identify `[Authorize]` attribute boundaries. Note DI scope (Singleton/Scoped/Transient). Detect Entity Framework navigation properties and lazy loading. Flag action filter pipeline participation. |
| **Django** | Note model signal connections. Identify middleware chain participation. Flag form validation logic. (Prompt-level only — no code-level analyzer in initial scope.) |
| **Express** | Map middleware stack order. Identify error handling chains. Note passport strategy usage. (Prompt-level only — no code-level analyzer in initial scope.) |

The extracted framework semantics are stored in the existing `cross_cutting` and `integrations` JSONB columns of `DeepAnalysis`.

**Remaining limitations** (after code-level + prompt-level analysis):
- **Reflection and dynamic dispatch**: `Method.invoke()`, `Activator.CreateInstance()`, and similar patterns produce call relationships invisible to both the ASG graph and config file parsing. These require runtime trace analysis (out of scope).
- **Convention-over-configuration magic**: Rails auto-routing, Django URL patterns referencing function names as strings, and similar convention-based patterns are not traceable through code structure or config parsing alone.
- **Frameworks without code-level analyzers**: Django and Express get prompt-level analysis only. Code-level analyzers for these frameworks are documented in Section 11 (Future Work).

For projects using unsupported frameworks or heavily reliant on reflection, `confidence` values on `DeepAnalysis` rows will reflect the reduced coverage. Consumers (migration phases, chat) should treat analyses with `confidence < 0.5` as supplementary rather than authoritative.

### 3.3 Understanding Worker — Background Batch Processing

**Location**: `codeloom/core/understanding/worker.py` (NEW)

**Purpose**: Process deep analysis jobs asynchronously in the background, following the exact pattern established by `RAPTORWorker` (`codeloom/core/raptor/worker.py`).

#### Architecture (mirrors RAPTORWorker)

```
+----------------------------------------------------------+
|                 UnderstandingWorker                        |
|                                                           |
|  Main Thread          Background Thread                   |
|  ----------          -----------------                    |
|  queue_job() ------> asyncio Event Loop                   |
|                        |                                  |
|                        +- _poll_pending_jobs()             |
|                        |   (DB query for status=pending)   |
|                        |                                  |
|                        +- asyncio.Queue                    |
|                        |   (job dispatch)                  |
|                        |                                  |
|                        +- asyncio.Semaphore(2)             |
|                            (concurrency control)           |
|                            |                              |
|                            +- _process_job(job)            |
|                            |   1. Run framework analyzers  |
|                            |      (augment code_edges)     |
|                            |   2. Detect entry points      |
|                            |   3. Trace call trees         |
|                            |   4. Analyze each chain       |
|                            |   5. Store DeepAnalysis rows  |
|                            |   6. Embed narratives         |
|                            |   7. Update progress          |
|                            |                              |
|                            +- _process_job(job)            |
|                                (concurrent job 2)          |
+----------------------------------------------------------+
```

**Key implementation details**:
- Daemon thread with its own `asyncio` event loop (same as `RAPTORWorker`)
- `asyncio.Queue` for job dispatch from the API layer
- `asyncio.Semaphore(max_concurrent=2)` for concurrency control
- DB polling for `status='pending'` jobs at configurable interval
- Progress tracking via `DeepAnalysisJob.progress` JSONB: `{total_chains, completed, current_entry}`
- Status updates: `pending -> running -> completed | failed`
- Job claim via `SELECT ... FOR UPDATE SKIP LOCKED` to prevent double-processing
- Heartbeat updates while running; stale jobs reclaimable after timeout

#### LLM Retry and Failure Semantics

Each chain analysis LLM call is wrapped with retry logic:

```python
MAX_RETRIES = 2
BASE_BACKOFF = 2.0  # seconds

async def _analyze_with_retry(self, chain_data: dict) -> dict:
    """Call LLM with exponential backoff retry for transient failures."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await self._analyzer.analyze_chain(chain_data)
        except (TimeoutError, ConnectionError, RateLimitError) as e:
            if attempt == MAX_RETRIES:
                raise  # Terminal failure after all retries
            backoff = BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES + 1, backoff, e
            )
            await asyncio.sleep(backoff)
        except (ValueError, json.JSONDecodeError) as e:
            # Non-transient: bad response format — don't retry
            raise AnalysisError(f"LLM returned unparseable response: {e}")
```

**Failure classification**:
- **Transient** (retry up to 2x with backoff): `TimeoutError`, `ConnectionError`, `RateLimitError`
- **Terminal** (immediate fail, no retry): JSON parse errors, validation failures, missing source code
- **Job-level**: If >50% of chains fail, the job status becomes `failed` with error details

**Idempotency**: Writing analyses uses upsert semantics on `(project_id, entry_unit_id, schema_version)` — enforced by the `uq_da_project_entry_schema` unique constraint on `deep_analyses`. Re-running a job for the same entry point overwrites the previous analysis rather than creating duplicates.

#### Distributed Worker Lease Protocol

For multi-instance deployments (e.g., 2+ backend processes behind a load balancer), the worker uses a database-level lease protocol to prevent double-processing:

**Job claim**:
```sql
-- Each worker instance has a unique worker_id (hostname + pid + random suffix)
UPDATE deep_analysis_jobs
SET status = 'running',
    claimed_at = NOW(),
    heartbeat_at = NOW(),
    params = jsonb_set(COALESCE(params, '{}'), '{worker_id}', to_jsonb(:worker_id::text))
WHERE job_id = (
    SELECT job_id FROM deep_analysis_jobs
    WHERE status = 'pending'
    ORDER BY created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *
```

**Heartbeat**: While processing, the worker updates `heartbeat_at` every 30 seconds:
```sql
UPDATE deep_analysis_jobs
SET heartbeat_at = NOW()
WHERE job_id = :job_id AND params->>'worker_id' = :worker_id
```

**Stale job reclaim**: A background sweep (every 2 minutes) detects stale jobs — those in `running` status whose `heartbeat_at` is older than the configured timeout:
```sql
-- Reclaim stale jobs (worker crashed or lost connectivity)
UPDATE deep_analysis_jobs
SET status = 'pending',
    claimed_at = NULL,
    heartbeat_at = NULL,
    retry_count = retry_count + 1,
    params = params - 'worker_id'
WHERE status = 'running'
  AND heartbeat_at < NOW() - INTERVAL '120 seconds'
  AND retry_count < 3
```

Jobs exceeding `retry_count >= 3` are moved to `failed` status with an error message indicating repeated worker failures.

**Embedding idempotency**: The `embedding_stored` boolean on `DeepAnalysis` ensures that re-processing a reclaimed job doesn't create duplicate vectors. Before embedding, the worker checks this flag and skips if already `true`.

**Configuration**:
```yaml
# config/codeloom.yaml
migration:
  deep_analysis:
    worker_heartbeat_interval: 30     # seconds between heartbeat updates
    worker_stale_timeout: 120         # seconds before a job is considered stale
    worker_max_retries: 3             # max reclaims before permanent failure
    worker_concurrency: 2             # max concurrent jobs per worker instance
```

### 3.4 Understanding Engine — Orchestrator

**Location**: `codeloom/core/understanding/engine.py` (NEW)

**Purpose**: High-level orchestrator (like `MigrationEngine`) that provides the public API for deep analysis operations.

```python
class UnderstandingEngine:
    """Orchestrates deep analysis of codebases.

    Coordinates chain tracing, LLM analysis, and result storage.
    Manages the background worker lifecycle.
    """

    def __init__(self, db_manager: DatabaseManager, pipeline=None):
        self._db = db_manager
        self._pipeline = pipeline
        self._tracer = ChainTracer(db_manager)
        self._worker = None  # Lazy-initialized

    # -- Public API --

    def start_analysis(
        self, project_id: str, user_id: str,
        job_type: str = "full_analysis", params: dict = None
    ) -> Dict:
        """Start a batch analysis job. Returns {job_id, status, ...}."""

    def get_job_status(self, job_id: str) -> Dict:
        """Returns {job_id, status, progress: {total, completed, current}, ...}."""

    def get_entry_points(self, project_id: str) -> List[Dict]:
        """Detect and return entry points without starting analysis."""

    def get_analysis_results(
        self, project_id: str, entry_unit_id: str = None
    ) -> List[Dict]:
        """List analysis results, optionally filtered by entry point."""

    def get_chain_detail(self, analysis_id: str) -> Dict:
        """Full detail for one analysis: call tree, business rules, etc."""
```

### 3.5 Enriched Chat — Functional Narratives in RAG Context

**Purpose**: When a user asks "how does the payment flow work?", the response includes business-level functional narratives alongside code snippets.

#### Current Chat Pipeline

```
code_chat.py:112-146

User Query
    |
    v
fast_retrieve()              <-- Hybrid BM25 + vector retrieval
    |                            Returns: code chunk NodeWithScore objects
    v
ASGExpander.expand()         <-- 1-hop neighbor expansion at 0.7x decay
    |                            Enriches results with callers/callees
    v
build_context_with_history() <-- Builds LLM context string
    |                            Sections: HISTORY -> RAPTOR -> EVIDENCE
    v
execute_query()              <-- LLM generates response
```

#### Enhanced Chat Pipeline (with Deep Understanding)

```
User Query
    |
    v
Detect Query Intent          <-- NEW: FLOW / DATA_LIFECYCLE intents
    |                            "how does X work?" -> FLOW
    |                            "where is entity X used?" -> DATA_LIFECYCLE
    v
fast_retrieve()              <-- Same hybrid retrieval
    |
    v
ASGExpander.expand()         <-- ENHANCED: intent-aware depth
    |                            FLOW intent -> depth=2, decay=0.5x
    |                            Default -> depth=1, decay=0.7x (unchanged)
    v
+------------------------------------------+
| Query DeepAnalysis (NEW)                  |
|                                           |
| For unit_ids in retrieval results:        |
| SELECT da.functional_summary,             |
|        da.business_rules,                 |
|        da.data_entities, da.side_effects  |
| FROM deep_analyses da                     |
| JOIN analysis_units au                    |
|   ON au.analysis_id = da.analysis_id      |
| WHERE au.project_id = :pid               |
|   AND au.unit_id = ANY(:unit_ids)         |
| GROUP BY da.analysis_id                   |
| ORDER BY COUNT(*) DESC  -- best overlap   |
| LIMIT 3                                   |
+----------------------+-------------------+
                       |
                       v
build_context_with_history()  <-- ENHANCED with functional_narratives param
    |
    |  Context structure (enhanced):
    |  +------------------------------------------+
    |  | ## CONVERSATION HISTORY                   |
    |  | User: ...  Assistant: ...                 |
    |  |                                           |
    |  | ## HIGH-LEVEL CONTEXT (RAPTOR Summaries)  |
    |  | ...                                       |
    |  |                                           |
    |  | ## FUNCTIONAL NARRATIVE (NEW)              |
    |  | [Flow: create_payment]                    |
    |  | Processes customer payments. Validates     |
    |  | card -> charges via Stripe -> records      |
    |  | transaction -> notifies user.             |
    |  | Data entities: Order (R,U), Payment (C).  |
    |  | Side effects: DB write, email.            |
    |  |                                           |
    |  | ## DETAILED EVIDENCE (Code Passages)       |
    |  | [Source: payments/service.py]              |
    |  | def process_payment(order_id, card):       |
    |  |     ...                                    |
    |  +------------------------------------------+
    v
execute_query()
```

**Backward compatible**: When no deep analysis exists for a project, the pipeline behaves identically to today.

### 3.6 Per-Phase LLM Selection

**Purpose**: Users can assign different LLMs for understanding tasks vs. code generation tasks.

#### Phase-to-Role Mapping

```python
_PHASE_LLM_ROLE = {
    # Understanding phases -- benefit from strong reasoning
    "discovery": "understanding",
    "architecture": "understanding",
    "analyze": "understanding",
    "deep_analysis": "understanding",

    # Generation phases -- benefit from strong code generation
    "design": "generation",
    "transform": "generation",
    "test": "generation",
}
```

#### Configuration

```yaml
# config/codeloom.yaml
migration:
  llm_overrides:
    understanding_llm: null    # e.g., "anthropic/claude-sonnet-4-5-20250929"
    generation_llm: null       # e.g., "groq/llama-3.3-70b-versatile"
```

When `null`, falls back to `Settings.llm` (the global LLM). This is fully backward compatible — existing behavior is unchanged unless overrides are configured.

#### Modified `_call_llm()` — `codeloom/core/migration/phases.py:130`

```python
# Current (line 130):
def _call_llm(prompt: str) -> str:
    llm = Settings.llm
    ...

# Enhanced (backward compatible):
def _call_llm(prompt: str, context_type: str = None) -> str:
    llm = _get_phase_llm(context_type)  # Resolves override or falls back
    ...

def _get_phase_llm(context_type: str = None) -> BaseLLM:
    """Resolve LLM for a given context type, falling back to Settings.llm."""
    if context_type:
        role = _PHASE_LLM_ROLE.get(context_type, "understanding")
        override_key = f"{role}_llm"
        # Check config for override
        override_spec = config.get(f"migration.llm_overrides.{override_key}")
        if override_spec:
            return _create_llm_from_spec(override_spec)  # provider/model string
    return Settings.llm
```

All existing callers pass only `prompt` -> `context_type=None` -> `Settings.llm`. Zero breaking changes.

### 3.7 Framework Analyzers — Code-Level (Spring + ASP.NET)

**Location**: `codeloom/core/understanding/frameworks/` (NEW directory)

**Purpose**: Parse framework-specific configuration files and code patterns to augment the ASG graph with edges invisible to generic tree-sitter analysis. Runs **before** chain tracing so the tracer follows framework-wired relationships naturally.

#### Why Code-Level Analysis Is Required

Prompt-level framework awareness (Section 3.2) instructs the LLM to interpret framework semantics in source code. But for enterprise Java/C# projects, critical wiring exists **outside source code** — in XML config, DI registrations, middleware pipelines, and AOP pointcuts. Without code-level analysis, the chain tracer never sees these relationships and produces structurally incomplete call trees.

| Gap | Example | Impact Without Code-Level Analysis |
|-----|---------|-----------------------------------|
| XML bean wiring | `<bean class="PaymentService"><property ref="stripeClient"/>` | Chain tracer never connects PaymentService to StripeClient |
| DI container registration | `services.AddScoped<IPaymentGateway, StripeGateway>()` | Chain tracer follows interface, misses implementation |
| Middleware pipeline | `app.UseAuthentication(); app.UseAuthorization();` | Middleware ordering invisible to call graph |
| Action filter inheritance | `[ServiceFilter(typeof(AuditFilter))]` on base controller | Filters on child controller actions missed |
| AOP pointcuts | `@Around("execution(* com.foo.service.*.*(..))")` | Cross-cutting behavior invisible to per-method analysis |
| Security filter chain | `http.addFilterBefore(jwtFilter, UsernamePasswordAuthenticationFilter.class)` | Auth flow missing from call trees |

#### Architecture

```
FrameworkAnalyzer (ABC)
|-- detect(project_id) -> bool
|-- analyze(project_id) -> FrameworkContext
|-- augment_edges(project_id, context) -> int  (edges added)
         |
    +----+----+
    |         |
SpringAnalyzer  AspNetAnalyzer
```

**Base class**: `codeloom/core/understanding/frameworks/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FrameworkContext:
    """Supplementary context produced by framework config analysis."""
    framework: str                          # "spring" | "aspnet"
    version_hint: Optional[str] = None      # "5.x", "6.0", etc.
    di_mappings: List[Dict] = field(default_factory=list)
                # [{interface: "IFoo", implementation: "FooImpl", scope: "scoped"}]
    middleware_chain: List[Dict] = field(default_factory=list)
                # [{name: "AuthMiddleware", order: 1, unit_id: "..."}]
    security_filters: List[Dict] = field(default_factory=list)
                # [{filter: "JwtFilter", position: "before", anchor: "..."}]
    aop_pointcuts: List[Dict] = field(default_factory=list)
                # [{advice: "around", target_pattern: "com.foo.service.*.*",
                #   unit_id: "..."}]
    config_sources: List[str] = field(default_factory=list)
                # File paths parsed: ["applicationContext.xml", "SecurityConfig.java"]


class FrameworkAnalyzer(ABC):
    """Base class for framework-specific code-level analyzers.

    Analyzers parse configuration files and framework-specific code patterns
    to produce additional code_edges that the chain tracer follows.
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    @abstractmethod
    def detect(self, project_id: str) -> bool:
        """Return True if this framework is detected in the project."""

    @abstractmethod
    def analyze(self, project_id: str) -> FrameworkContext:
        """Parse framework config and return structured context."""

    @abstractmethod
    def augment_edges(self, project_id: str, context: FrameworkContext) -> int:
        """Insert additional code_edges from framework analysis.

        Returns count of edges added. All edges are tagged with
        metadata: {"source": "framework_analyzer", "framework": "<name>"}.
        """
```

#### Spring Analyzer

**Location**: `codeloom/core/understanding/frameworks/spring.py`

**Capabilities**:

1. **XML Bean Wiring**: Parses `applicationContext.xml`, `beans.xml`, `spring-*.xml` from `code_files`. Extracts `<bean class="...">` definitions with `<constructor-arg ref="..."/>` and `<property ref="..."/>` dependencies. Resolves class names to `code_units` by `qualified_name`. Creates `calls` edges between bean units.

2. **@Configuration/@Bean DI Wiring**: Scans `code_units` with `@Configuration` annotation. For each `@Bean` method, resolves return type to find the concrete implementation unit. Creates `calls` edges from injection sites (constructor/field parameters) to `@Bean` provider methods.

3. **Spring Security Filter Chain**: Parses `SecurityConfig` classes (extends `WebSecurityConfigurerAdapter` or returns `SecurityFilterChain`). Extracts `addFilterBefore/After` calls to build ordered filter chain. Creates `calls` edges representing the security pipeline.

4. **@Transactional Boundaries**: Scans for `@Transactional` annotations. Stores transaction propagation settings as metadata on existing edges so the analyzer knows which calls cross transaction boundaries. Does NOT create new edges — enriches existing ones.

5. **AOP Pointcuts**: Parses `@Aspect` classes for `@Around`, `@Before`, `@After` advice with pointcut expressions. Resolves target patterns against `code_units` qualified names using glob-to-regex matching. Creates `calls` edges from advice methods to all matching target methods.

**Detection**:
```python
def detect(self, project_id: str) -> bool:
    with self._db.get_session() as session:
        # Check for Spring Boot main class or @Configuration
        spring_code = session.execute(text("""
            SELECT 1 FROM code_units
            WHERE project_id = :pid
              AND (metadata->>'annotations' ILIKE '%SpringBootApplication%'
                   OR metadata->>'annotations' ILIKE '%Configuration%')
            LIMIT 1
        """), {"pid": project_id})
        if spring_code.fetchone():
            return True

        # Check for Spring XML config files
        xml_config = session.execute(text("""
            SELECT 1 FROM code_files
            WHERE project_id = :pid
              AND (filename ILIKE '%applicationContext%.xml'
                   OR filename ILIKE '%spring-%.xml'
                   OR filename ILIKE '%beans.xml')
            LIMIT 1
        """), {"pid": project_id})
        return xml_config.fetchone() is not None
```

**XML Bean Parsing** (representative):
```python
def _parse_xml_beans(self, project_id: str, file_content: str,
                     file_id: str) -> List[Dict]:
    """Parse Spring XML bean definitions and generate edge descriptors."""
    import xml.etree.ElementTree as ET
    edges = []
    root = ET.fromstring(file_content)
    ns = {'beans': 'http://www.springframework.org/schema/beans'}

    for bean in root.findall('.//beans:bean', ns):
        bean_class = bean.get('class')
        if not bean_class:
            continue

        bean_unit = self._resolve_unit_by_qualified_name(project_id, bean_class)
        if not bean_unit:
            continue

        # Constructor injection
        for arg in bean.findall('beans:constructor-arg', ns):
            ref = arg.get('ref')
            if ref:
                ref_unit = self._resolve_bean_ref(project_id, ref, root, ns)
                if ref_unit:
                    edges.append({
                        "source_unit_id": bean_unit.unit_id,
                        "target_unit_id": ref_unit.unit_id,
                        "edge_type": "calls",
                        "metadata": {
                            "source": "framework_analyzer",
                            "framework": "spring",
                            "wiring": "xml_constructor_injection",
                            "config_file": str(file_id),
                        }
                    })

        # Property injection
        for prop in bean.findall('beans:property', ns):
            ref = prop.get('ref')
            if ref:
                ref_unit = self._resolve_bean_ref(project_id, ref, root, ns)
                if ref_unit:
                    edges.append({
                        "source_unit_id": bean_unit.unit_id,
                        "target_unit_id": ref_unit.unit_id,
                        "edge_type": "calls",
                        "metadata": {
                            "source": "framework_analyzer",
                            "framework": "spring",
                            "wiring": "xml_property_injection",
                            "config_file": str(file_id),
                        }
                    })

    return edges
```

**AOP Pointcut Resolution** (representative):
```python
def _resolve_aop_pointcuts(self, project_id: str) -> List[Dict]:
    """Find @Aspect classes and resolve pointcut target patterns."""
    edges = []
    with self._db.get_session() as session:
        aspects = session.execute(text("""
            SELECT unit_id, name, qualified_name, metadata, source
            FROM code_units
            WHERE project_id = :pid
              AND metadata->>'annotations' ILIKE '%Aspect%'
        """), {"pid": project_id}).fetchall()

        for aspect in aspects:
            # Extract pointcut expressions from source
            # e.g., @Around("execution(* com.foo.service.*.*(..))")
            pointcut_pattern = self._extract_pointcut_expression(aspect.source)
            if not pointcut_pattern:
                continue

            # Resolve pattern to matching code_units
            target_regex = self._pointcut_to_regex(pointcut_pattern)
            targets = session.execute(text("""
                SELECT unit_id FROM code_units
                WHERE project_id = :pid
                  AND qualified_name ~ :pattern
                  AND unit_type IN ('method', 'function')
            """), {"pid": project_id, "pattern": target_regex}).fetchall()

            for target in targets:
                edges.append({
                    "source_unit_id": aspect.unit_id,
                    "target_unit_id": target.unit_id,
                    "edge_type": "calls",
                    "metadata": {
                        "source": "framework_analyzer",
                        "framework": "spring",
                        "wiring": "aop_advice",
                        "pointcut": pointcut_pattern,
                    }
                })

    return edges
```

#### ASP.NET Analyzer

**Location**: `codeloom/core/understanding/frameworks/aspnet.py`

**Capabilities**:

1. **DI Container Registration**: Parses `Startup.cs` / `Program.cs` source code for `services.AddScoped<IFoo, FooImpl>()`, `AddTransient<>()`, `AddSingleton<>()` patterns. Resolves interface-to-implementation pairs via `code_units` qualified names. Creates `implements` edges for DI bindings not captured by tree-sitter (registration-based wiring that doesn't involve inheritance).

2. **Middleware Pipeline**: Parses `app.UseMiddleware<T>()` and `app.Use*()` calls to build ordered middleware chain. Creates ordered `calls` edges representing the request pipeline flow with `metadata.order` for position tracking.

3. **Action Filter Pipeline**: Scans for `[ServiceFilter(typeof(T))]`, `[TypeFilter(typeof(T))]`, `[ActionFilterAttribute]` on controllers and base controllers. Resolves filter inheritance — if `BaseController` has `[AuditFilter]`, identifies all derived controllers via existing `inherits` edges. Creates `calls` edges from controller action methods to filter `OnActionExecuting` / `OnActionExecuted` methods.

4. **Entity Framework DbContext**: Parses `DbSet<T>` properties in DbContext-derived classes to map entity types. Scans for navigation properties and `.Include()` chains to detect lazy/eager loading patterns. Stores entity relationship metadata in `FrameworkContext.di_mappings` for injection into analysis prompts (not as graph edges — EF relationships are data model, not call structure).

**Detection**:
```python
def detect(self, project_id: str) -> bool:
    with self._db.get_session() as session:
        aspnet = session.execute(text("""
            SELECT 1 FROM code_units
            WHERE project_id = :pid
              AND (metadata->>'annotations' ILIKE '%ApiController%'
                   OR metadata->>'annotations' ILIKE '%HttpGet%'
                   OR metadata->>'annotations' ILIKE '%HttpPost%'
                   OR qualified_name ILIKE '%Startup.Configure%'
                   OR qualified_name ILIKE '%.Program.Main%')
            LIMIT 1
        """), {"pid": project_id})
        return aspnet.fetchone() is not None
```

**DI Registration Parsing** (representative):
```python
def _parse_di_registrations(self, project_id: str) -> List[Dict]:
    """Parse Startup.cs/Program.cs for DI container registrations."""
    import re
    edges = []

    # Find Startup.ConfigureServices or Program.cs
    with self._db.get_session() as session:
        startup_units = session.execute(text("""
            SELECT unit_id, source, qualified_name FROM code_units
            WHERE project_id = :pid
              AND (qualified_name ILIKE '%Startup.ConfigureServices%'
                   OR qualified_name ILIKE '%Program.Main%'
                   OR qualified_name ILIKE '%Program.<Main>%')
        """), {"pid": project_id}).fetchall()

    for unit in startup_units:
        if not unit.source:
            continue

        # Match patterns: services.AddScoped<IFoo, FooImpl>()
        pattern = r'services\.Add(Scoped|Transient|Singleton)<(\w+),\s*(\w+)>'
        for match in re.finditer(pattern, unit.source):
            scope, interface_name, impl_name = match.groups()

            interface_unit = self._resolve_unit_by_name(project_id, interface_name)
            impl_unit = self._resolve_unit_by_name(project_id, impl_name)

            if interface_unit and impl_unit:
                edges.append({
                    "source_unit_id": impl_unit.unit_id,
                    "target_unit_id": interface_unit.unit_id,
                    "edge_type": "implements",
                    "metadata": {
                        "source": "framework_analyzer",
                        "framework": "aspnet",
                        "wiring": "di_registration",
                        "scope": scope.lower(),
                    }
                })

    return edges
```

#### Registry and Integration

**Location**: `codeloom/core/understanding/frameworks/__init__.py`

```python
from .spring import SpringAnalyzer
from .aspnet import AspNetAnalyzer

_ANALYZERS = [SpringAnalyzer, AspNetAnalyzer]

def detect_and_analyze(
    db: DatabaseManager, project_id: str
) -> List[FrameworkContext]:
    """Run all framework analyzers, return contexts for detected frameworks.

    Called by UnderstandingWorker._process_job() as step 1 before
    entry point detection and chain tracing.
    """
    contexts = []
    for analyzer_cls in _ANALYZERS:
        analyzer = analyzer_cls(db)
        if analyzer.detect(project_id):
            ctx = analyzer.analyze(project_id)
            edge_count = analyzer.augment_edges(project_id, ctx)
            logger.info(
                "Framework %s: added %d edges for project %s",
                ctx.framework, edge_count, project_id
            )
            contexts.append(ctx)
    return contexts
```

#### Edge Lifecycle

Framework-augmented edges are persisted in `code_edges` with `metadata->>'source' = 'framework_analyzer'`. They follow this lifecycle:

1. **Creation**: During `_process_job()` step 1, before chain tracing.
2. **Consumption**: Chain tracer follows them naturally (they're standard `code_edges` rows with `calls` or `implements` edge types).
3. **Cleanup on re-analysis**: Before augmenting, delete existing framework edges for the project:
   ```sql
   DELETE FROM code_edges
   WHERE project_id = :pid
     AND metadata->>'source' = 'framework_analyzer'
   ```
   This makes re-analysis idempotent — old framework edges are replaced, not accumulated.
4. **Distinguishability**: The `metadata.source` and `metadata.framework` fields allow queries to separate framework-injected edges from ASG-builder edges when needed (e.g., for debugging or edge provenance reporting).

#### What Each Layer Covers

| Concern | Code-Level Analyzer | Prompt-Level Instructions |
|---------|--------------------|-----------------------|
| XML bean wiring (Spring) | Parses XML, creates edges | N/A (edges exist, LLM sees relationships) |
| @Bean DI wiring (Spring) | Resolves return types, creates edges | Interprets `@Qualifier`/`@Primary` ambiguity |
| Security filter chain (Spring) | Builds ordered filter edges | Interprets auth flow semantics |
| @Transactional boundaries (Spring) | Enriches edge metadata | Interprets propagation semantics |
| AOP pointcuts (Spring) | Resolves patterns, creates edges | Interprets advice intent (logging, security, etc.) |
| DI registration (ASP.NET) | Resolves interface-to-impl, creates edges | Interprets scope semantics |
| Middleware pipeline (ASP.NET) | Builds ordered middleware edges | Interprets pipeline ordering impact |
| Action filter inheritance (ASP.NET) | Resolves inheritance, creates edges | Interprets filter behavior |
| EF DbContext (ASP.NET) | Stores entity metadata | Interprets navigation/loading semantics |

---

## 4. Data Models

### 4.0 Canonical Ownership and Transition

The canonical source of truth for deep understanding is the new `deep_analyses` rows (project/entry-point scoped). The existing `FunctionalMVP.analysis_output` JSONB column remains as a **derived cache only** during the transition period. Once deep analysis is complete for a project, migration phases should prefer `DeepContextBundle` from `deep_analyses` over `FunctionalMVP.analysis_output`. The old column is not removed — it continues to work for projects without deep analysis.

### 4.1 New Tables

#### `deep_analysis_jobs` — Batch Job Queue

```python
class DeepAnalysisJob(Base):
    __tablename__ = "deep_analysis_jobs"
    __table_args__ = (
        Index('idx_daj_project_status', 'project_id', 'status'),
        Index('idx_daj_status_created', 'status', 'created_at'),
    )

    job_id       = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id   = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"))
    user_id      = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"))
    status       = Column(String(20), default='pending')
                   # pending -> running -> completed | failed
    job_type     = Column(String(50))
                   # full_analysis | entry_points_only | single_chain
    params       = Column(JSONB, default=dict)
                   # {entry_unit_id: "...", max_depth: 10, ...}
    progress     = Column(JSONB, default=dict)
                   # {total_chains: 42, completed: 17, current: "process_payment"}
    retry_count  = Column(Integer, default=0)
                   # Number of times this job has been retried
    error        = Column(Text, nullable=True)
    claimed_at   = Column(TIMESTAMP, nullable=True)
                   # When a worker claimed this job (SELECT FOR UPDATE SKIP LOCKED)
    heartbeat_at = Column(TIMESTAMP, nullable=True)
                   # Last heartbeat from processing worker; stale = reclaimable
    created_at   = Column(TIMESTAMP, default=datetime.utcnow)
    completed_at = Column(TIMESTAMP, nullable=True)

    analyses = relationship("DeepAnalysis", back_populates="job",
                            cascade="all, delete-orphan")
```

#### `deep_analyses` — Per-Entry-Point Analysis Results

```python
class DeepAnalysis(Base):
    __tablename__ = "deep_analyses"
    __table_args__ = (
        UniqueConstraint('project_id', 'entry_unit_id', 'schema_version',
                         name='uq_da_project_entry_schema'),
        UniqueConstraint('analysis_id', 'project_id',
                         name='uq_da_analysis_project'),
        Index('idx_da_project_entry', 'project_id', 'entry_unit_id'),
        Index('idx_da_job', 'job_id'),
    )

    analysis_id      = Column(UUID(), primary_key=True, default=uuid.uuid4)
    job_id           = Column(UUID(), ForeignKey("deep_analysis_jobs.job_id",
                              ondelete="CASCADE"))
    project_id       = Column(UUID(), ForeignKey("projects.project_id",
                              ondelete="CASCADE"))
    entry_unit_id    = Column(UUID(), ForeignKey("code_units.unit_id",
                              ondelete="CASCADE"))
    entry_type       = Column(String(30))
                       # api_endpoint | cli_command | scheduled_job | event_handler
    entry_metadata   = Column(JSONB, default=dict)
                       # {http_method: "POST", path: "/api/payments", ...}
    call_tree        = Column(JSONB)
                       # Full nested tree structure from entry point
    chain_depth      = Column(Integer)
                       # Maximum depth reached in call tree
    unit_count       = Column(Integer)
                       # Total unique units in call tree
    functional_summary = Column(Text)
                       # LLM-generated business narrative (also embedded in pgvector)
    business_rules   = Column(JSONB, default=list)
                       # [{id, rule, location, behavior}]
    data_entities    = Column(JSONB, default=list)
                       # [{name, operations: ["create", "read", "update", "delete"]}]
    integrations     = Column(JSONB, default=list)
                       # [{type, target, description}]
    side_effects     = Column(JSONB, default=list)
                       # [{type, description}]
    cross_cutting    = Column(JSONB, default=list)
                       # [{concern, units}]
    schema_version   = Column(Integer, default=1)
                       # OUTPUT SCHEMA version — bumped when the JSON structure of
                       # business_rules/data_entities/etc. changes shape.
                       # Part of upsert key: (project_id, entry_unit_id, schema_version).
                       # Different schema_version -> new row (old shape preserved).
                       # Same schema_version -> upsert (overwrite with latest analysis).
    prompt_version   = Column(String(20), default="v1")
                       # PROMPT TEMPLATE version — tracks which prompt wording produced
                       # this analysis. NOT part of the upsert key. When prompt_version
                       # changes but schema_version stays the same, the existing row is
                       # overwritten (improved analysis, same output shape).
                       # Audit trail of all runs preserved in deep_analysis_jobs.
    confidence       = Column(Float, nullable=True)
                       # LLM self-reported confidence (0.0-1.0)
    coverage         = Column(Float, nullable=True)
                       # Fraction of call tree units with source code available
    embedding_stored = Column(Boolean, default=False)
    created_at       = Column(TIMESTAMP, default=datetime.utcnow)

    job = relationship("DeepAnalysisJob", back_populates="analyses")
    entry_unit = relationship("CodeUnit")
    analysis_units = relationship("AnalysisUnit", back_populates="analysis",
                                  cascade="all, delete-orphan")
```

#### `analysis_units` — Junction Table (Analysis <-> Code Units)

Replaces JSONB containment scans (`call_tree @> ANY(:unit_ids)`) with an indexed relational lookup. This is the **primary lookup path** for both chat enrichment and migration context injection.

```python
class AnalysisUnit(Base):
    __tablename__ = "analysis_units"
    __table_args__ = (
        PrimaryKeyConstraint('analysis_id', 'unit_id'),
        Index('idx_au_project_unit', 'project_id', 'unit_id'),
        Index('idx_au_analysis', 'analysis_id'),
        # Composite FK ensures analysis_units.project_id always matches
        # deep_analyses.project_id — prevents cross-project contamination.
        # Works because deep_analyses has UniqueConstraint on (analysis_id, project_id)
        # (implied by analysis_id being PK + uq_da_project_entry_schema).
        ForeignKeyConstraint(
            ['analysis_id', 'project_id'],
            ['deep_analyses.analysis_id', 'deep_analyses.project_id'],
            ondelete="CASCADE"
        ),
        ForeignKeyConstraint(
            ['unit_id'], ['code_units.unit_id'],
            ondelete="CASCADE"
        ),
    )

    analysis_id  = Column(UUID(), nullable=False)
    project_id   = Column(UUID(), nullable=False)
    unit_id      = Column(UUID(), nullable=False)
    min_depth    = Column(Integer, nullable=False)
                   # MINIMUM depth at which this unit appears in the call tree.
                   # If unit X is reachable via path A->B->X (depth 2) AND
                   # C->D->E->X (depth 3), min_depth = 2.
                   # Full path reconstruction uses call_tree JSONB on DeepAnalysis.
    path_count   = Column(Integer, default=1)
                   # Number of distinct paths from entry point to this unit.
                   # Higher path_count = more interconnected = higher impact.

    analysis = relationship("DeepAnalysis", back_populates="analysis_units")
```

**Design note**: The PK `(analysis_id, unit_id)` intentionally collapses multiple paths to a single row. `analysis_units` is a **lookup/index table** optimized for "which analyses touch unit X?" queries. It stores `min_depth` (shallowest reachable depth) and `path_count` (number of distinct paths) as summary statistics. Full path reconstruction — including all intermediate nodes and multi-path details — uses the `call_tree` JSONB column on `DeepAnalysis`, which preserves the complete nested tree structure.

**Note on the composite FK**: The `(analysis_id, project_id)` FK target requires that `deep_analyses` has a unique constraint covering these columns. Since `analysis_id` is the PK, the pair `(analysis_id, project_id)` is trivially unique. We add an explicit unique index to satisfy PostgreSQL's FK target requirement:

```python
# Add to deep_analyses __table_args__:
UniqueConstraint('analysis_id', 'project_id', name='uq_da_analysis_project'),
```

**Why a junction table instead of JSONB containment?**
- JSONB `@>` containment queries require sequential scan or GIN index on `call_tree`
- `analysis_units` with `(project_id, unit_id)` B-tree index gives O(log n) lookups
- Enables simple JOINs: "give me all analyses that touch unit X" is a single indexed query
- Scales to projects with thousands of code units without query degradation
- Composite FK on `(analysis_id, project_id)` prevents cross-project contamination at the DB level

### 4.2 Modified Tables

#### `projects` — Add analysis status column

```python
# Add to Project model:
deep_analysis_status = Column(String(20), default='none')
# none -> pending -> running -> completed | failed
```

### 4.3 Embedding Strategy

After LLM analysis of each entry point, the `functional_summary` is embedded as a new TextNode in pgvector:

```python
from llama_index.core.schema import TextNode

node = TextNode(
    text=analysis.functional_summary,
    metadata={
        "node_type": "flow_description",      # Distinguishes from code chunks
        "project_id": str(project_id),
        "entry_unit_id": str(entry_unit_id),
        "entry_name": entry_unit_name,
        "source_id": f"deep_analysis_{analysis_id}",
        "tree_level": 0,
    }
)
vector_store.add([node])  # PGVectorStore handles embedding + storage
```

**No changes to `PGVectorStore`** — it already supports arbitrary metadata on TextNode objects. The `node_type: "flow_description"` metadata allows retrieval to distinguish functional narratives from code chunks.

### 4.4 DeepContextBundle — Named Contract for Migration Integration

**Location**: `codeloom/core/understanding/models.py` (NEW)

The `DeepContextBundle` is the formal data contract between the understanding engine and migration phases. Migration context builders receive this typed object instead of raw dicts, ensuring consistent field access and clear ownership.

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EvidenceRef:
    """A traceable reference back to source code that produced an analysis artifact."""
    unit_id: str
    file_path: str
    line_hint: Optional[int] = None  # Best-effort start line

@dataclass
class DeepContextBundle:
    """Named contract for deep analysis context consumed by migration phases.

    Migration phases (analyze, transform, test) receive this object to enrich
    their LLM prompts with deep understanding of the codebase.
    """
    functional_summary: str
    business_rules: List[dict] = field(default_factory=list)
    data_entities: List[dict] = field(default_factory=list)
    side_effects: List[dict] = field(default_factory=list)
    integrations: List[dict] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    coverage: float = 0.0       # 0.0-1.0: fraction of MVP units covered by analysis
    confidence: float = 0.0     # 0.0-1.0: LLM self-reported confidence

    def format_for_prompt(self, budget: int = 2000) -> str:
        """Render this bundle as a text section for injection into LLM prompts.

        Args:
            budget: Maximum token budget for the rendered section.

        Returns:
            Formatted string with business rules, data entities, and side effects.
        """
        ...
```

### 4.5 Coverage Calculation and Gating

**Purpose**: Ensure migration phases have sufficient deep understanding before proceeding. Warn or block when analysis coverage is too low.

#### Coverage Calculation

Coverage is computed per MVP cluster as the fraction of the MVP's `unit_ids` that appear in the `analysis_units` junction table:

```python
def calculate_coverage(db: DatabaseManager, project_id: str, mvp_unit_ids: List[str]) -> float:
    """Calculate what fraction of MVP units are covered by deep analysis.

    Returns:
        Float 0.0-1.0. E.g., 0.85 means 85% of units in this MVP
        have been traced through at least one entry point's call tree.
    """
    with db.get_session() as session:
        covered = session.execute(text("""
            SELECT COUNT(DISTINCT au.unit_id)
            FROM analysis_units au
            WHERE au.project_id = :pid
              AND au.unit_id = ANY(:unit_ids)
        """), {"pid": project_id, "unit_ids": mvp_unit_ids})
        covered_count = covered.scalar() or 0

    return covered_count / max(len(mvp_unit_ids), 1)
```

#### Coverage Gating in Migration Phases

```yaml
# config/codeloom.yaml
migration:
  deep_analysis:
    coverage_warn_threshold: 0.70     # Warn if coverage < 70%
    coverage_fail_threshold: null     # Optional hard-fail (e.g., 0.50)
    coverage_fail_enabled: false      # Set true to enable hard-fail
```

Gating logic in migration context builder:

```python
coverage = calculate_coverage(db, project_id, mvp_unit_ids)
if coverage < config.coverage_warn_threshold:
    logger.warning(
        "Deep analysis coverage %.0f%% below threshold %.0f%% for MVP %s",
        coverage * 100, config.coverage_warn_threshold * 100, mvp_id
    )
if config.coverage_fail_enabled and coverage < config.coverage_fail_threshold:
    raise InsufficientCoverageError(
        f"Deep analysis coverage {coverage:.0%} below minimum {config.coverage_fail_threshold:.0%}"
    )
```

The coverage and confidence values are logged per phase and included in the `DeepContextBundle` so migration outputs can cite them.

### 4.6 Security and Multi-tenant Enforcement

Security is enforced at **two distinct boundaries**:

#### Boundary 1: API Layer (Authorization)

The security boundary for job creation and result access is the API layer. All authorization checks happen here:

1. **Endpoint enforcement**: Every `/api/understanding/*` endpoint verifies project ownership/access using the current user context (`project_id + user_id`) via the existing `get_current_user` + `verify_project_access` dependency chain in `api/deps.py`.
2. **Job creation gate**: `POST /api/understanding/{project_id}/analyze` only creates a `deep_analysis_jobs` row if the authenticated user has access to the project. The `user_id` is stored on the job for audit.
3. **Result access gate**: `GET /api/understanding/{project_id}/results` verifies project access before returning any `deep_analyses` rows.
4. **No cross-tenant leakage**: All `deep_analyses` and `analysis_units` queries in migration/chat paths include `project_id` in WHERE clauses. The `analysis_units` junction table includes `project_id` as a denormalized column specifically to enable project-scoped index lookups.

#### Boundary 2: Worker Layer (Trusted Backend)

The worker is a **trusted backend component** that processes all authorized jobs without re-checking user permissions:

1. **Worker claims any pending job**: The claim SQL (`SELECT ... FOR UPDATE SKIP LOCKED`) does NOT filter by project or user. This is intentional — the worker trusts that if a job exists in the `deep_analysis_jobs` table, it was authorized at creation time by Boundary 1.
2. **Why no worker-level filtering**: Adding project/user predicates to the claim SQL would require the worker to have a user session context, which it doesn't. The worker is a background process, not an API handler. Filtering at this level would also complicate multi-project batch processing.
3. **Audit trail**: Every job records `user_id` and `project_id`, so post-hoc audit can determine who initiated each analysis.
4. **Pool partitioning** (optional, future): For deployments requiring worker-level isolation (e.g., dedicated workers per tenant), add a `worker_pool` column to `deep_analysis_jobs` and filter claims by pool. This is not implemented in the initial release.

---

## 5. API Surface

### 5.1 New Endpoints — Understanding

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/understanding/{project_id}/analyze` | Start batch deep analysis job |
| `GET` | `/api/understanding/{project_id}/status/{job_id}` | Job progress (polling) |
| `GET` | `/api/understanding/{project_id}/entry-points` | Detected entry points (before analysis) |
| `GET` | `/api/understanding/{project_id}/results` | List all analysis results |
| `GET` | `/api/understanding/{project_id}/chain/{analysis_id}` | Full chain detail |

### 5.2 New Endpoints — LLM Overrides

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings/migration-llm` | Current LLM override configuration |
| `POST` | `/api/settings/migration-llm` | Set understanding_llm and/or generation_llm |

### 5.3 New Endpoints — Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/migration/{plan_id}/export?format=json` | Export plan + deep analysis + MVP docs |

### 5.4 Request/Response Examples

**Start Analysis**:
```
POST /api/understanding/{project_id}/analyze
Body: {"job_type": "full_analysis", "params": {"max_depth": 10}}

Response 202:
{
  "job_id": "abc-123",
  "status": "pending",
  "project_id": "def-456",
  "message": "Deep analysis job queued"
}
```

**Job Progress**:
```
GET /api/understanding/{project_id}/status/{job_id}

Response 200:
{
  "job_id": "abc-123",
  "status": "running",
  "progress": {
    "total_chains": 42,
    "completed": 17,
    "current": "process_payment"
  }
}
```

**Analysis Results**:
```
GET /api/understanding/{project_id}/results

Response 200:
[
  {
    "analysis_id": "...",
    "entry_unit_id": "...",
    "entry_type": "api_endpoint",
    "entry_metadata": {"http_method": "POST", "path": "/api/payments"},
    "chain_depth": 7,
    "unit_count": 23,
    "functional_summary": "Processes customer payments for orders...",
    "business_rules": [...],
    "data_entities": [...]
  }
]
```

---

## 6. Integration Points

Every modification is backward compatible. Existing behavior is preserved when deep analysis has not been run.

### 6.1 Chat Route Integration — `codeloom/api/routes/code_chat.py`

**Insertion point**: Between ASG expansion (line 139) and context building (line 142).

```python
# EXISTING (line 126-139): ASG expansion
if retrieval_results and project.get("asg_status") == "complete":
    expander = ASGExpander(db_manager)
    retrieval_results = expander.expand(...)

# NEW (~25 lines): Query deep analysis for functional narratives
functional_narratives = []
if retrieval_results:
    # Extract unit_ids from retrieval results
    unit_ids = {nws.node.metadata.get("unit_id") for nws in retrieval_results
                if nws.node.metadata.get("unit_id")}
    if unit_ids:
        # Query deep_analyses via analysis_units junction table
        # (indexed lookup, no JSONB containment scan)
        analyses = _get_matching_analyses(db_manager, project_id, unit_ids)
        for a in analyses[:3]:  # Limit to top 3 relevant narratives
            functional_narratives.append({
                "entry_name": a["entry_metadata"].get("path", a.get("name", "")),
                "summary": a["functional_summary"],
                "data_entities": a["data_entities"],
                "side_effects": a["side_effects"],
            })

# EXISTING (line 142): Build context -- now with narratives
context = build_context_with_history(
    retrieval_results=retrieval_results,
    conversation_history=conversation_history,
    max_chunks=data.max_sources,
    functional_narratives=functional_narratives,  # NEW param
)
```

### 6.2 Context Building — `codeloom/core/stateless/context.py`

**Modification**: Add `functional_narratives` parameter to `build_context_with_history()`.

```python
def build_context_with_history(
    retrieval_results: List[NodeWithScore],
    raptor_summaries: Optional[List[Tuple[TextNode, float]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    functional_narratives: Optional[List[Dict]] = None,  # NEW
    max_history: int = 10,
    max_summaries: int = 3,
    max_chunks: int = 6,
) -> str:
```

New section inserted between RAPTOR summaries and DETAILED EVIDENCE:

```python
# After RAPTOR summaries, before retrieved chunks:
if functional_narratives:
    narrative_texts = []
    for fn in functional_narratives:
        parts = [f"[Flow: {fn['entry_name']}]", fn['summary']]
        if fn.get('data_entities'):
            entities = ", ".join(f"{e['name']} ({','.join(e['operations'])})"
                                for e in fn['data_entities'])
            parts.append(f"Data entities: {entities}")
        if fn.get('side_effects'):
            effects = ", ".join(e['description'] for e in fn['side_effects'])
            parts.append(f"Side effects: {effects}")
        narrative_texts.append("\n".join(parts))

    context_parts.append(
        "## FUNCTIONAL NARRATIVE (Business Logic Flows)\n" +
        "\n\n".join(narrative_texts)
    )
```

### 6.3 Query Intent — `codeloom/core/engine/retriever.py`

**Modification**: Add FLOW and DATA_LIFECYCLE intents to the `QueryIntent` enum (line 47) and `INTENT_PATTERNS` dict (line 56).

```python
# Add to QueryIntent enum:
FLOW = "flow"
DATA_LIFECYCLE = "data"

# Add to INTENT_PATTERNS:
QueryIntent.FLOW: [
    r'\bhow\s+does\b.*\b(work|process|flow|handle)\b',
    r'\bwhat\s+happens\s+when\b',
    r'\bexplain\s+the\s+flow\b',
    r'\bcall\s+chain\b',
    r'\bexecution\s+path\b',
    r'\bbusiness\s+(logic|rules?|flow)\b',
],
QueryIntent.DATA_LIFECYCLE: [
    r'\bwhere\s+is\b.*\b(created|updated|deleted|used)\b',
    r'\blifecycle\s+of\b',
    r'\bentity\b.*\b(flow|lifecycle|usage)\b',
],
```

### 6.4 ASG Expander — `codeloom/core/asg_builder/expander.py`

**Modification**: Add optional `intent` parameter to `expand()` (line 31).

When intent is `FLOW`: use `depth=2` with `0.5x` decay (deeper exploration for flow understanding).
Default (`intent=None`): preserves current behavior (`depth=1`, `0.7x` decay).

### 6.5 Migration Context — `codeloom/core/migration/context_builder.py`

**New method**: `get_deep_analysis_context(unit_ids, budget) -> DeepContextBundle`:

1. Resolves MVP `unit_ids` via `analysis_units` junction table (indexed B-tree lookup)
2. Ranks matching `DeepAnalysis` rows by overlap count (most relevant first)
3. Calculates coverage: `covered_units / total_mvp_units`
4. Applies coverage gating (warn if `< 0.70`, optional hard-fail via config)
5. Builds a `DeepContextBundle` with evidence refs from each matching analysis
6. Formats `business_purpose + data_entities + side_effects + evidence_refs` into a bounded context section

```python
def get_deep_analysis_context(
    self, project_id: str, mvp_unit_ids: List[str], budget: int = 2000
) -> Optional[DeepContextBundle]:
    """Build deep analysis context for a migration MVP.

    Uses analysis_units junction table for O(log n) lookup instead
    of JSONB containment scans on call_tree.
    """
    # 1. Find analyses that overlap with this MVP's units
    analyses = self._find_overlapping_analyses(project_id, mvp_unit_ids)
    if not analyses:
        return None

    # 2. Calculate coverage
    coverage = calculate_coverage(self._db, project_id, mvp_unit_ids)

    # 3. Coverage gating
    if coverage < self._config.coverage_warn_threshold:
        logger.warning("Coverage %.0f%% below threshold for MVP", coverage * 100)

    # 4. Build bundle with evidence refs
    return DeepContextBundle(
        functional_summary=analyses[0].functional_summary,
        business_rules=analyses[0].business_rules,
        data_entities=analyses[0].data_entities,
        side_effects=analyses[0].side_effects,
        integrations=analyses[0].integrations,
        evidence_refs=self._extract_evidence_refs(analyses),
        coverage=coverage,
        confidence=analyses[0].confidence or 0.0,
    )
```

- Injected into `_build_phase_3_context` (analyze), `_build_phase_5_context` (transform), and `_build_phase_6_context` (test) to give the LLM deep understanding before generating code
- Phases `analyze`, `transform`, and `test` explicitly consume deep context
- Migration outputs must include references to extracted business rules/side effects used
- Coverage and confidence are logged per phase for observability

### 6.6 Wiring — Service Initialization

**`codeloom/api/deps.py`**: Add `get_understanding_engine()` dependency following the existing `get_migration_engine()` pattern.

**`codeloom/api/app.py`**: Register the understanding router: `app.include_router(understanding_router, prefix="/api")`.

**`codeloom/__main__.py`**: Initialize `UnderstandingEngine` and attach to `app.state`, following the existing pattern for `MigrationEngine`.

---

## 7. Sequence Diagram — Full Analysis Flow

```
User                  API                    Engine              Worker              ChainTracer         Analyzer           VectorStore
 |                     |                       |                    |                    |                   |                   |
 |  POST /analyze      |                       |                    |                    |                   |                   |
 |------------------->|                       |                    |                    |                   |                   |
 |                     |  start_analysis()     |                    |                    |                   |                   |
 |                     |--------------------->|                    |                    |                   |                   |
 |                     |                       |  Create Job (DB)   |                    |                   |                   |
 |                     |                       |-->                 |                    |                   |                   |
 |                     |                       |  queue_job()       |                    |                   |                   |
 |                     |                       |------------------>|                    |                   |                   |
 |  202 {job_id}       |                       |                    |                    |                   |                   |
 |<--------------------|                       |                    |                    |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | _process_job()     |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | framework_analyze()|                   |                   |
 |                     |                       |                    |  (augment edges)   |                   |                   |
 |                     |                       |                    |-->DB (code_edges)  |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | detect_entry_pts() |                   |                   |
 |                     |                       |                    |------------------>|                   |                   |
 |                     |                       |                    |  [{entry_points}]  |                   |                   |
 |                     |                       |                    |<------------------|                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | FOR EACH entry:    |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | trace_call_tree()  |                   |                   |
 |                     |                       |                    |------------------>|                   |                   |
 |                     |                       |                    |  {nested tree}     |                   |                   |
 |                     |                       |                    |<------------------|                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | analyze_chain()    |                   |                   |
 |                     |                       |                    |---------------------------------------->|                   |
 |                     |                       |                    |                    |   LLM call       |                   |
 |                     |                       |                    |  {analysis result} |                   |                   |
 |                     |                       |                    |<----------------------------------------|                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | Store DeepAnalysis |                   |                   |
 |                     |                       |                    |-->DB               |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | Embed narrative    |                   |                   |
 |                     |                       |                    |---------------------------------------------------------->|
 |                     |                       |                    |                    |                   |   vector_store.add|
 |                     |                       |                    |<----------------------------------------------------------|
 |                     |                       |                    |                    |                   |                   |
 |                     |                       |                    | Update progress    |                   |                   |
 |                     |                       |                    |-->DB               |                   |                   |
 |                     |                       |                    |                    |                   |                   |
 |  GET /status/{id}   |                       |                    |                    |                   |                   |
 |------------------->|  get_job_status()     |                    |                    |                   |                   |
 |  {progress: 17/42}  |                       |                    |                    |                   |                   |
 |<--------------------|                       |                    |                    |                   |                   |
```

---

## 8. Files Modified/Created

| File | Action | Phase | Est. Lines |
|------|--------|-------|-----------|
| `docs/deep-understanding-architecture.md` | **New** (this document) | 0 | ~1000 |
| `core/db/models.py` | Modify (add 3 models + 1 column) | A | ~90 |
| `core/understanding/__init__.py` | **New** | A | ~5 |
| `core/understanding/models.py` | **New** (DeepContextBundle, EvidenceRef) | A | ~50 |
| `core/understanding/chain_tracer.py` | **New** | A | ~150 |
| `core/understanding/worker.py` | **New** (+ lease protocol) | A | ~200 |
| `core/understanding/engine.py` | **New** | A | ~100 |
| `core/understanding/frameworks/__init__.py` | **New** (registry + detect_and_analyze) | B | ~30 |
| `core/understanding/frameworks/base.py` | **New** (FrameworkAnalyzer ABC, FrameworkContext) | B | ~60 |
| `core/understanding/frameworks/spring.py` | **New** (XML beans, DI, security, AOP) | B | ~250 |
| `core/understanding/frameworks/aspnet.py` | **New** (DI registration, middleware, filters, EF) | B | ~200 |
| `core/understanding/analyzer.py` | **New** (+ bounded analysis tiers) | B | ~200 |
| `core/understanding/prompts.py` | **New** (3 prompts + framework awareness) | B | ~120 |
| `api/routes/understanding.py` | **New** | A | ~80 |
| `api/deps.py` | Modify | A | ~10 |
| `api/app.py` | Modify | A | ~3 |
| `core/engine/retriever.py` | Modify (add intents) | C | ~30 |
| `core/stateless/context.py` | Modify (add narrative section) | C | ~25 |
| `core/asg_builder/expander.py` | Modify (intent-aware depth) | C | ~15 |
| `api/routes/code_chat.py` | Modify (wire narratives) | C | ~25 |
| `core/migration/phases.py` | Modify (`_call_llm` override) | D | ~40 |
| `core/migration/context_builder.py` | Modify (deep context + coverage) | D | ~60 |
| `config/codeloom.yaml` | Modify (llm_overrides + coverage thresholds) | D | ~15 |
| `api/routes/fastapi_settings.py` | Modify (LLM overrides API) | D | ~20 |
| `api/routes/migration.py` | Modify (export endpoint) | D | ~30 |
| Alembic migration | **New** | A | ~15 |

**Total: ~2,640 lines. 14 new files, 11 modified files. 0 new Python dependencies.**

---

## 9. Implementation Order

```
Step 0:  Write this architecture doc to docs/

Phase A (Foundation):
  A1. DB Models (DeepAnalysisJob, DeepAnalysis, AnalysisUnit, Project.deep_analysis_status)
  A1b. Data contracts (DeepContextBundle, EvidenceRef in core/understanding/models.py)
  A2. Alembic migration (autogenerate)
  A3. Chain Tracer (entry point detection + call tree tracing)
  A4. Understanding Worker (background thread)
  A5. Understanding Engine (orchestrator)
  A6. API Routes (/api/understanding/*)
  A7. Wiring (deps.py, app.py)

Phase B (Framework Analyzers + LLM Analysis):
  B1. Framework analyzer base class + registry (core/understanding/frameworks/)
  B2. Spring Analyzer (XML beans, @Configuration DI, Security filter chain, AOP pointcuts)
  B3. ASP.NET Analyzer (DI registration, middleware pipeline, action filters, EF DbContext)
  B4. Prompts (chain_analysis_prompt, cross_cutting_prompt, framework_detection_prompt)
  B5. Chain Analyzer (LLM-powered functional analysis + bounded tiers)
  B6. Wire framework analyzers + chain analyzer into worker (steps 1 + 4)

Phase C (Enriched Chat):
  C1. Extended Query Intents (FLOW, DATA_LIFECYCLE)
  C2. Embed analysis narratives in worker step 5
  C3. Enhanced context building (functional_narratives param)
  C4. Intent-aware ASG expansion
  C5. Wire into chat route

Phase D (Per-Phase LLM Selection):
  D1. Config (migration.llm_overrides in codeloom.yaml)
  D2. _call_llm() with context_type parameter
  D3. Deep context in migration phases
  D4. Settings API (GET/POST migration-llm)
  D5. Export endpoint
```

**Dependencies**: A -> B -> (C and D can be parallel)

---

## 10. Verification Plan

### 10.1 Unit Tests

| Test | What It Validates |
|------|------------------|
| Entry point detection | Given a project with `@GetMapping` units and zero incoming `calls` edges, `detect_entry_points()` returns them with correct `entry_type: api_endpoint` |
| Cycle prevention in call tree | Given a circular call graph (A -> B -> C -> A), `trace_call_tree()` terminates without infinite loop and path arrays contain no duplicates |
| Schema validation | `DeepAnalysis` model enforces `schema_version` default, `prompt_version` default, and JSONB column defaults |
| `analysis_units` mapping | After storing a `DeepAnalysis` with a call tree, `analysis_units` rows are created with correct `min_depth` and `path_count` values |
| `DeepContextBundle` contract | `format_for_prompt()` produces a string within the token budget and includes evidence refs |
| Coverage calculation | Given 10 MVP units where 7 appear in `analysis_units`, `calculate_coverage()` returns 0.7 |
| Access control | Querying analyses for a project the user doesn't have access to returns empty results |
| LLM retry logic | Transient errors (timeout, rate limit) trigger retry with backoff; terminal errors (JSON parse) fail immediately |
| Versioning semantics | Re-analyzing with same `schema_version` but different `prompt_version` upserts (overwrites) the existing row. Re-analyzing with a new `schema_version` creates a new row preserving the old one. Audit trail of all runs lives in `deep_analysis_jobs`. |
| Spring framework detection | Given a project with `@SpringBootApplication` unit, `SpringAnalyzer.detect()` returns `True`. Given a project with no Spring indicators, returns `False`. |
| Spring XML bean edge augmentation | Given a project with `applicationContext.xml` defining `<bean class="FooService"><property ref="barRepo"/>`, `augment_edges()` creates a `calls` edge from FooService to BarRepo with `metadata.source = "framework_analyzer"`. |
| Spring AOP pointcut resolution | Given an `@Aspect` class with `@Around("execution(* com.foo.service.*.*(..))")`, resolver matches 3 service methods and creates 3 `calls` edges from advice to targets. |
| ASP.NET DI registration parsing | Given `Startup.ConfigureServices` with `services.AddScoped<IPaymentGateway, StripeGateway>()`, creates `implements` edge from StripeGateway to IPaymentGateway with `metadata.scope = "scoped"`. |
| ASP.NET action filter inheritance | Given `[AuditFilter]` on `BaseController` and `OrderController : BaseController`, creates `calls` edges from OrderController action methods to `AuditFilter.OnActionExecuting`. |
| Framework edge cleanup idempotency | Running `augment_edges()` twice produces the same edge set (old framework edges deleted before re-augmentation). |

Versioning rule for operations: prompt-only changes should either (a) bump `schema_version` to preserve old analyses, or (b) keep `schema_version` unchanged and accept upsert overwrite behavior.

### 10.2 Integration Tests

| Test | What It Validates |
|------|------------------|
| Job lifecycle | `POST /analyze` -> job created with `status=pending` -> worker picks up -> `status=running` -> `status=completed` with analysis rows |
| Retry behavior | When LLM returns transient error, worker retries up to 2x then marks chain as failed; job continues with remaining chains |
| Migration context injection | Given completed deep analysis, `get_deep_analysis_context()` returns a `DeepContextBundle` with `coverage > 0` |
| No-analysis fallback | When no deep analysis exists, migration phases produce output identical to current behavior (no errors, no empty sections) |
| Chat enrichment | Query a project with completed analysis -> response context includes `## FUNCTIONAL NARRATIVE` section |
| Evidence traceability | Each `business_rule` in analysis results includes `evidence.unit_id` that maps to a real `code_units` row |
| Framework edge augmentation | For a Spring project with XML beans, `_process_job()` step 1 creates framework edges before chain tracing. Resulting call trees include bean-wired dependencies that pure tree-sitter ASG missed. |
| Framework edge cleanup | Re-running analysis deletes old `metadata.source = 'framework_analyzer'` edges before re-augmenting, producing identical edge sets. |

### 10.3 Performance Checks

| Test | What It Validates |
|------|------------------|
| `analysis_units` index lookup | Query `analysis_units` by `(project_id, unit_id)` with 10K rows completes in < 10ms |
| Coverage calculation | `calculate_coverage()` with 500 MVP units completes in < 50ms |
| Chat narrative injection | Adding functional narratives to context build adds < 20ms to query response time |

### 10.4 Acceptance Criteria

1. **Entry points**: Upload a Java Spring project -> `GET /api/understanding/{id}/entry-points` -> verify `@GetMapping`, `@PostMapping` endpoints detected with correct `entry_type` and `entry_metadata`
2. **Framework edges**: For a Spring project with XML bean config, verify `code_edges` includes edges with `metadata.source = 'framework_analyzer'` connecting bean-wired units. For an ASP.NET project, verify DI registration edges connect interfaces to implementations.
3. **Call chains**: `POST /api/understanding/{id}/analyze` -> poll status -> `GET /api/understanding/{id}/results` -> verify call trees have `chain_depth > 1` with proper tree structure (nested children, no cycles). For Spring projects, call trees must include XML-wired and AOP-advised paths.
4. **Functional analysis**: Each result has `functional_summary`, `business_rules`, `data_entities` populated with meaningful content. Every extracted artifact includes `evidence` with `unit_id`, `file_path`, `line_hint`.
5. **Enriched chat**: Ask "how does the payment flow work?" -> response includes FUNCTIONAL NARRATIVE section with business-level description, not just code snippets
6. **LLM selection**: Set `generation_llm: groq/llama-3.3-70b-versatile` in config -> run Transform phase -> verify Groq is used (check logs for LLM provider)
7. **Export**: `GET /api/migration/{plan_id}/export` -> verify JSON includes deep analysis + MVP docs
8. **Backward compatibility**: Existing chat without deep analysis works unchanged. Existing migration pipeline works unchanged.
9. **Coverage logging**: Migration output cites rule/side-effect evidence for covered MVPs. Coverage and confidence logged per phase.
10. **Build**: `npm run build` -- no frontend compilation errors

---

## 11. What This Does NOT Cover (Future Work)

The following features are **out of scope** for this implementation (Phases A-D). They are documented here as potential follow-up work:

- **Frontend UI for Understanding Results**: Dashboard to browse entry points, call trees, business rules. A new page at `/project/{id}/understanding` with `CallTreeViewer`, `EntryPointList`, and `BusinessRulesPanel` components.
- **Real-time SSE for Analysis Progress**: Currently polling-based; SSE would improve UX during long-running analysis jobs.
- **Incremental Re-analysis**: When code changes, detect which files changed, identify affected entry points via the ASG graph, and re-analyze only those chains instead of the full codebase.
- **Cross-Project Understanding**: Compare deep analysis results between source and target projects during migration — highlight business rules that exist in source but are missing from target, track data entity coverage.
- **Visual Call Tree Explorer**: Interactive force-directed graph visualization (D3.js) showing the full call graph with zoom, pan, filtering by entry type, and click-to-inspect nodes.
- **Framework Analyzers for Django and Express**: Code-level analyzers (matching the Spring/ASP.NET pattern in Section 3.7) for Django (signal dispatch graphs, URL resolver tracing, middleware chain parsing) and Express (middleware ordering from code, passport strategy resolution, error handler chain mapping). Currently these frameworks receive prompt-level analysis only.
- **Runtime Trace Analysis**: For reflection-based dispatch (`Method.invoke()`, `Activator.CreateInstance()`) and convention-over-configuration patterns that cannot be resolved statically, integrate runtime trace data (e.g., from instrumented test runs) to augment the call graph.

---

*This document is the authoritative reference for the Deep Understanding Engine implementation. All code should conform to the patterns and interfaces described here.*
