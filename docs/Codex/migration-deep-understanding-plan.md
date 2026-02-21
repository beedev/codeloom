# Migration Deep-Understanding Engine (AST/ASG/Code Chat) Implementation Plan

## Summary
Implement a migration-first deep-understanding pipeline that converts AST/ASG structure into stable functional artifacts used by migration phases. Chat reuses the same artifacts, but migration correctness is the primary objective.

## 1. Scope and Non-goals
1. In scope: backend data model, analysis worker, APIs, migration phase integration, and minimal chat enrichment.
2. Out of scope: full dashboard UX, SSE streaming, advanced graph visualization.
3. Backward compatibility: existing migration/chat behavior remains when no deep analysis exists.

## 2. Canonical Data Model and Ownership
1. Canonical source of truth is new `deep_analyses` rows (project/entry-point scoped).
2. `FunctionalMVP.analysis_output` remains as a derived cache only during transition.
3. Add `deep_analysis_jobs` for asynchronous execution status.
4. Add `analysis_units` mapping table to join analysis to `code_units` without JSONB containment scans.
5. Add `projects.deep_analysis_status` (`none|pending|running|completed|failed`).

## 3. Exact Schema Decisions
1. `deep_analysis_jobs`:
   - Columns: `job_id`, `project_id`, `user_id`, `status`, `job_type`, `params`, `progress`, `retry_count`, `error`, `claimed_at`, `heartbeat_at`, `created_at`, `completed_at`.
   - Indexes: `(project_id, status)`, `(status, created_at)`.
2. `deep_analyses`:
   - Columns: `analysis_id`, `job_id`, `project_id`, `entry_unit_id`, `entry_type`, `entry_metadata`, `call_tree`, `chain_depth`, `unit_count`, `functional_summary`, `business_rules`, `data_entities`, `integrations`, `side_effects`, `cross_cutting`, `schema_version`, `confidence`, `coverage`, `created_at`.
   - Indexes: `(project_id, entry_unit_id)`, `(job_id)`.
3. `analysis_units`:
   - Columns: `analysis_id`, `project_id`, `unit_id`, `depth`, `path_hash`.
   - Constraints: PK `(analysis_id, unit_id)`, FK cascades to `deep_analyses` and `code_units`.
   - Indexes: `(project_id, unit_id)`, `(analysis_id)`.

## 4. Security and Multi-tenant Enforcement
1. Every new endpoint enforces project ownership/access using current user context (`project_id + user_id`).
2. Worker reads only jobs created for user-accessible projects.
3. Deep-analysis query joins in migration/chat are project-scoped and user-scoped.
4. No endpoint returns analysis rows for projects outside caller access.

## 5. Public APIs and Contracts
1. Understanding APIs:
   - `POST /api/understanding/{project_id}/analyze`
   - `GET /api/understanding/{project_id}/status/{job_id}`
   - `GET /api/understanding/{project_id}/entry-points`
   - `GET /api/understanding/{project_id}/results`
2. Migration deep context contract (`DeepContextBundle`):
   - `functional_summary`, `business_rules`, `data_entities`, `side_effects`, `integrations`, `evidence_refs`, `coverage`, `confidence`.
3. Evidence reference contract:
   - Each extracted rule includes `unit_id`, `file_path`, `line_hint`.

## 6. Worker Execution and Failure Semantics
1. Job claim protocol: `SELECT ... FOR UPDATE SKIP LOCKED` and transition `pending -> running`.
2. Heartbeat updates while running; stale jobs are reclaimable after timeout.
3. Retry policy: transient failures retry up to 2 times with backoff; terminal failures become `failed`.
4. Progress model tracks `total_chains`, `completed_chains`, `current_entry`.
5. Idempotency: writing analyses uses upsert semantics on `(project_id, entry_unit_id, schema_version)`.

## 7. Migration Integration
1. Context builder resolves MVP `unit_ids` via `analysis_units`, ranks analyses by overlap, and builds bounded deep context.
2. Phases `analyze`, `transform`, and `test` explicitly consume deep context.
3. Coverage gate:
   - Warn if `coverage < 0.70`.
   - Optional hard-fail controlled by config.
4. Migration outputs must include references to extracted business rules/side effects used.

## 8. Chat Integration (Minimal)
1. Add `QueryIntent.FLOW` and `QueryIntent.DATA_LIFECYCLE` with explicit pattern rules.
2. For these intents, fetch top matching `deep_analyses` and inject `FUNCTIONAL NARRATIVE` before code evidence.
3. Non-flow intents keep existing retrieval/ASG expansion unchanged.

## 9. LLM Override Wiring
1. Add config subtree in `config/codeloom.yaml`:
   - `migration.llm_overrides.understanding_llm`
   - `migration.llm_overrides.generation_llm`
2. Extend config loader with migration getter and runtime-safe fallback to `Settings.llm`.
3. Migration phase `_call_llm` resolves model by context role; no override means current behavior.

## 10. Validation and Acceptance
1. Unit tests:
   - entry-point detection, cycle prevention, schema validation, `analysis_units` mapping, access control checks.
2. Integration tests:
   - queued job lifecycle, retry behavior, migration context injection, no-analysis fallback.
3. Performance checks:
   - lookup by MVP unit IDs using `analysis_units` index under realistic project size.
4. Acceptance criteria:
   - migration output cites rule/side-effect evidence for covered MVPs.
   - coverage and confidence logged per phase.
   - no regressions in existing chat/migration flows when deep analysis is absent.
