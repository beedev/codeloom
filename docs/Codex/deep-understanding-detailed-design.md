# Deep Understanding Engine Detailed Design

## 1. Purpose and Scope

This document defines an implementation-ready design for evolving CodeLoom into an enterprise-grade migration platform with deep code understanding across multiple technologies.

The design is grounded in the current codebase:

- Runtime/bootstrap: `codeloom/__main__.py`, `codeloom/api/app.py`
- Core orchestration: `codeloom/pipeline.py`
- Ingestion/AST/ASG: `codeloom/core/ingestion/code_ingestion.py`, `codeloom/core/ast_parser/*`, `codeloom/core/asg_builder/*`
- Migration flow: `codeloom/core/migration/engine.py`, `codeloom/core/migration/phases.py`, `codeloom/core/migration/context_builder.py`
- Chat/RAG flow: `codeloom/api/routes/code_chat.py`, `codeloom/core/stateless/*`, `codeloom/core/engine/retriever.py`
- Persistence: `codeloom/core/db/models.py`

Primary scope:

1. Add deep understanding as a first-class backend capability for migration accuracy.
2. Keep current chat/migration behavior backward compatible.
3. Support multi-tenant access control and multi-instance worker safety.

## 2. Current System Baseline

### 2.1 Request-serving Architecture

- FastAPI app stores shared services in `app.state` (`pipeline`, `db_manager`, `project_manager`, `code_ingestion`, `conversation_store`).
- API routes in `codeloom/api/routes/*` are session-authenticated via `get_current_user`.
- `MigrationEngine` is lazy-created via dependency injection (`get_migration_engine`).

### 2.2 Code Ingestion and Knowledge Construction

Current ingestion flow:

1. Source acquisition (`zip`, `git`, `local`) via `CodeIngestionService`.
2. Language detection + parse (`tree-sitter` or fallback parser).
3. Persist `CodeFile` and `CodeUnit` rows.
4. Build ASG edges (`contains`, `imports`, `calls`, `inherits`, `implements`, `overrides`, `calls_sp`) via `ASGBuilder`.
5. Chunk and embed code into pgvector-backed store.
6. Update project statuses (`ast_status`, `asg_status`).

### 2.3 Chat and Retrieval

Current chat route behavior:

1. Validate project status.
2. Load conversation history (if enabled).
3. Retrieve chunks (`fast_retrieve`).
4. Expand with 1-hop ASG neighbors (`ASGExpander.expand`, decay-based).
5. Build hierarchical context (`build_context_with_history`).
6. Execute LLM completion or stream.

### 2.4 Migration Engine

Current migration engine supports:

- V1 (6-phase) and V2 (4-phase) pipelines.
- MVP clustering plus agentic refinement loop.
- Phase approval gates and per-phase artifact persistence.
- Context generation from ASG + source for phase-specific prompts.
- V2 on-demand deep analyze output cached on `FunctionalMVP.analysis_output`.

## 3. Target Architecture: Deep Understanding Layer

Introduce a dedicated Understanding subsystem:

- `codeloom/core/understanding/engine.py`
- `codeloom/core/understanding/worker.py`
- `codeloom/core/understanding/chain_tracer.py`
- `codeloom/core/understanding/analyzer.py`
- `codeloom/core/understanding/models.py`
- `codeloom/core/understanding/frameworks/*` (Spring/ASP.NET baseline analyzers)

Goals:

1. Detect entry points robustly across framework-managed flows.
2. Reconstruct call trees with cycle protection and bounded depth.
3. Extract structured business artifacts with evidence references.
4. Persist and index analysis results for migration and chat retrieval.
5. Enforce versioned/idempotent analysis semantics.

## 4. Data Model Design

### 4.1 Existing Tables (unchanged semantics)

- `projects`, `code_files`, `code_units`, `code_edges`
- `migration_plans`, `migration_phases`, `functional_mvps`
- `conversations`, `query_logs`, RBAC tables

### 4.2 New Tables

#### `deep_analysis_jobs`

Tracks asynchronous analysis jobs.

Key fields:

- Identity/scope: `job_id`, `project_id`, `user_id`
- Execution state: `status`, `retry_count`, `error`
- Lease state: `claimed_at`, `heartbeat_at`, `params.worker_id`
- Progress: `progress` JSON
- Timestamps: `created_at`, `completed_at`

Indexes:

- `(project_id, status)`
- `(status, created_at)`

#### `deep_analyses`

Stores per-entry-point understanding outputs.

Key fields:

- Identity: `analysis_id`, `job_id`, `project_id`, `entry_unit_id`, `entry_type`
- Structural context: `entry_metadata`, `call_tree`, `chain_depth`, `unit_count`
- Extracted artifacts: `functional_summary`, `business_rules`, `data_entities`, `integrations`, `side_effects`, `cross_cutting`
- Versioning/confidence: `schema_version`, `prompt_version`, `confidence`, `coverage`
- Embedding lifecycle: `embedding_stored`

Constraints:

- `uq_da_project_entry_schema` on `(project_id, entry_unit_id, schema_version)`
- `uq_da_analysis_project` on `(analysis_id, project_id)` (for composite FK targeting)

#### `analysis_units`

Lookup index between analyses and impacted units.

Columns:

- `analysis_id`, `project_id`, `unit_id`
- `min_depth`: shallowest path depth from entry
- `path_count`: number of distinct paths

Constraints/indexes:

- PK `(analysis_id, unit_id)`
- FK `(analysis_id, project_id)` -> `deep_analyses(analysis_id, project_id)`
- FK `unit_id` -> `code_units(unit_id)`
- Index `(project_id, unit_id)` for overlap lookup
- Index `(analysis_id)`

### 4.3 Versioning Rule

- Same `(project_id, entry_unit_id, schema_version)` => upsert overwrite.
- New `schema_version` => new row, preserving historical shape.
- `prompt_version` alone is metadata; not part of upsert key.
- Full execution audit remains in `deep_analysis_jobs`.

## 5. Core Component Design

### 5.1 ChainTracer

Responsibilities:

1. Entry-point detection via dual pass:
- Pass A: zero incoming `calls` heuristic.
- Pass B: annotation/metadata/signature pattern detection.
2. Call-tree tracing using recursive SQL CTE with cycle checks.
3. Tree materialization for analyzer input and persistence.

Design details:

- Must include framework-managed handlers even with incoming framework edges.
- Must support bounded depth and deterministic traversal order.
- Must emit both tree and flattened unit membership for `analysis_units`.

### 5.2 Framework Analyzers (Spring + ASP.NET Baseline)

Responsibilities:

1. Detect framework presence.
2. Parse framework config/code patterns.
3. Augment `code_edges` with framework-derived relationships.

Rules:

- Framework edges are tagged in metadata (`source=framework_analyzer`).
- Analyzer runs before chain tracing.
- Re-runs clean previous framework-derived edges to remain idempotent.

### 5.3 Analyzer

Responsibilities:

1. Build bounded prompt context from call-tree units.
2. Execute LLM extraction with retry policy.
3. Validate structured JSON output.
4. Normalize and persist evidence references.

Bounded analysis tiers:

1. Tier 1: full source when within budget.
2. Tier 2: depth-prioritized partial source.
3. Tier 3: summarization fallback for extreme chains.

Outputs:

- Evidence-grounded artifacts for migration traceability.
- Coverage and confidence metrics.
- `chain_truncated` indicator when applicable.

### 5.4 Worker

Responsibilities:

1. Poll/claim pending jobs.
2. Execute end-to-end chain tracing + analysis + persistence.
3. Persist progress and heartbeats.
4. Reclaim stale leases safely.

Lease protocol:

- Claim with `FOR UPDATE SKIP LOCKED`.
- Set `worker_id` in JSON via `to_jsonb(:worker_id::text)`.
- Heartbeat at fixed interval.
- Reclaim stale jobs by timeout, bounded retry count.

### 5.5 UnderstandingEngine

Responsibilities:

1. API-facing orchestration surface.
2. Job start/status/result APIs.
3. Optional synchronous entry-point preview (`get_entry_points`).
4. Dependency boundary between routes and worker internals.

## 6. Integration with Existing Flows

### 6.1 Migration Integration

Extend `MigrationContextBuilder` with deep-context resolution:

1. Take MVP `unit_ids`.
2. Join `analysis_units` for overlap ranking.
3. Fetch top relevant `deep_analyses`.
4. Build `DeepContextBundle`.
5. Inject bundle into `analyze`, `transform`, and `test` phase contexts.

Coverage policy:

- Warn below configured threshold.
- Optional hard fail when enabled.
- Log coverage/confidence in phase metadata for auditability.

### 6.2 Chat Integration

Enhance chat route pipeline:

1. Detect intent (`FLOW`, `DATA_LIFECYCLE`).
2. For matching intents, resolve top analyses by overlap via `analysis_units`.
3. Inject `FUNCTIONAL NARRATIVE` section into built context before detailed evidence.
4. Preserve current behavior when no deep analysis exists.

### 6.3 LLM Override Integration

Add migration-specific overrides in config:

- `migration.llm_overrides.understanding_llm`
- `migration.llm_overrides.generation_llm`

Update migration phase executor:

- `_call_llm(prompt, context_type=None)`
- resolve model by phase role; fallback to `Settings.llm`.

## 7. API Design

### 7.1 New Understanding Endpoints

- `POST /api/understanding/{project_id}/analyze`
- `GET /api/understanding/{project_id}/status/{job_id}`
- `GET /api/understanding/{project_id}/entry-points`
- `GET /api/understanding/{project_id}/results`
- `GET /api/understanding/{project_id}/chain/{analysis_id}`

### 7.2 Existing Endpoint Extensions

- `GET/POST /api/settings/migration-llm`
- `GET /api/migration/{plan_id}/export?format=json` includes deep-analysis context

### 7.3 Security Requirements

1. Route-level access control remains authoritative.
2. All query paths include `project_id` scoping.
3. Job rows persist `user_id` for audit trail.
4. Worker is trusted backend; optional future pool partitioning for strict tenant isolation.

## 8. Non-Functional Design

### 8.1 Performance

Targets:

- Overlap lookup via `analysis_units(project_id, unit_id)` remains indexed.
- Coverage calculation remains cheap for typical MVP size.
- Chat narrative enrichment adds low overhead.

### 8.2 Reliability

- Retry transient LLM/network failures.
- Fail-fast on schema/parse errors.
- Stale lease reclaim for worker crashes.
- Idempotent writes for analysis rows and embeddings.

### 8.3 Observability

- Job lifecycle metrics: pending/running/completed/failed counts.
- Per-job progress and error payloads.
- Coverage/confidence emitted in migration phase logs.
- Query logs continue via existing `QueryLogger`.

## 9. Rollout Plan

### Phase A: Foundations

1. DB schema + Alembic.
2. Understanding package scaffolding.
3. Chain tracer + worker + engine.
4. Understanding APIs + app/deps wiring.

### Phase B: Analysis Quality

1. Prompt modules + schema validation.
2. Bounded analyzer tiers.
3. Framework analyzers (Spring, ASP.NET) and edge augmentation.

### Phase C: Consumption Paths

1. Chat narrative enrichment.
2. Query intent additions.
3. ASG expander intent-aware mode.

### Phase D: Migration Coupling

1. Deep context bundle injection in migration phases.
2. Coverage gates + metadata.
3. Per-phase LLM override APIs/config wiring.
4. Export endpoint enrichment.

## 10. Testing Strategy

### Unit

1. Entry-point detection dual-pass correctness.
2. Call-tree cycle prevention and depth bounds.
3. Framework analyzer detection and edge augmentation idempotency.
4. Analyzer JSON schema validation.
5. Versioning semantics (`schema_version`/`prompt_version`).
6. `analysis_units` min-depth/path-count generation.

### Integration

1. End-to-end job lifecycle.
2. Worker lease/heartbeat/reclaim behavior.
3. Migration phase context enrichment with coverage signals.
4. Chat flow intent -> narrative injection.
5. No-analysis fallback parity.

### Performance

1. Overlap lookup latency under realistic data volume.
2. Worker throughput at configured concurrency.
3. Chat enrichment overhead budget.

## 11. Key Risks and Mitigations

1. Framework complexity and false negatives.
- Mitigation: analyzer edge tagging, deterministic tests, confidence/coverage surfacing.

2. Large chain prompt overflow.
- Mitigation: bounded tier strategy and truncation flags.

3. Multi-instance race conditions.
- Mitigation: explicit lease protocol, stale reclaim, idempotent persistence.

4. Tenant data leakage.
- Mitigation: strict project-scoped query filters and composite FK protections.

## 12. Deliverables

1. New backend modules under `codeloom/core/understanding/`.
2. New understanding route module.
3. DB models and Alembic migration for deep-analysis tables.
4. Migration and chat integration updates.
5. Config loader extension for migration LLM overrides.
6. Test coverage for correctness, reliability, and performance baselines.

