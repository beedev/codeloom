# Deep Understanding Implementation Checklist

Source of truth:
- `docs/deep-understanding-implementation-spec.md`
- `docs/deep-understanding-detailed-design.md`

Status policy:
- Mark `[x]` only when code is merged in this repository and sanity-checked.
- Keep `[ ]` for planned/in-progress/not-verified work.

## Phase 0: Spec and Design Alignment

- [x] Align implementation spec with detailed design for:
  - worker lease ownership (`worker_id`)
  - retry backoff (`next_attempt_at`)
  - `analysis_units` semantics (`min_depth`, `path_count`)
  - evidence contract (`unit_id` required)
  - user-scoped API access checks

## Phase A: Schema and Models

- [x] Update `codeloom/core/db/models.py`:
  - [x] Add `Project.deep_analysis_status`
  - [x] Add `DeepAnalysisJob` with `user_id`, `worker_id`, `retry_count`, `next_attempt_at`
  - [x] Add `DeepAnalysis`
  - [x] Add `AnalysisUnit` with `min_depth`, `path_count`
  - [x] Add constraints/indexes matching spec
- [x] Update `codeloom/core/db/__init__.py` exports
- [x] Add Alembic migration in `alembic/versions/`
- [x] `alembic upgrade head` passes

## Phase B: Understanding Module

- [x] Add `codeloom/core/understanding/__init__.py`
- [x] Add `codeloom/core/understanding/models.py`
- [x] Add `codeloom/core/understanding/chain_tracer.py`
- [x] Add `codeloom/core/understanding/analyzer.py`
- [x] Add `codeloom/core/understanding/prompts.py`
- [x] Add `codeloom/core/understanding/worker.py`
- [x] Add `codeloom/core/understanding/engine.py`
- [x] Add `codeloom/core/understanding/frameworks/base.py`
- [x] Add `codeloom/core/understanding/frameworks/spring.py`
- [x] Add `codeloom/core/understanding/frameworks/aspnet.py`
- [x] Add `codeloom/core/understanding/frameworks/__init__.py`

## Phase C: Core Algorithm and Quality Gates

- [x] Dual-pass entry-point detection implemented
- [x] Recursive call-tree tracing with cycle prevention
- [x] Flattened membership outputs `min_depth` + `path_count`
- [x] Tiered analysis (T1/T2/T3) implemented
- [x] Enforce config quality gates:
  - [x] `max_entry_points` (enforced in `chain_tracer.py`)
  - [x] `require_evidence_refs` (enforced in `analyzer.py`)
  - [x] `min_narrative_length` (enforced in `analyzer.py`)
- [x] Persist full `DeepContextBundle` shape into `deep_analyses.result_json`
- [x] Persist `confidence_score` and `coverage_pct`

## Phase D: Worker Reliability

- [x] Claim SQL uses `FOR UPDATE SKIP LOCKED` with `worker_id`
- [x] Lease-scoped heartbeat (`WHERE worker_id = :worker_id`)
- [x] Lease-scoped complete/failure transitions
- [x] Stale reclaim + bounded retries (`max_retries`)
- [x] Exponential backoff scheduling (`next_attempt_at`)
- [x] Idempotent upsert semantics validated

## Phase E: API and App Wiring

- [x] Add `codeloom/api/routes/understanding.py`
  - [x] `POST /api/understanding/{project_id}/analyze`
  - [x] `GET /api/understanding/{project_id}/status/{job_id}`
  - [x] `GET /api/understanding/{project_id}/entry-points`
  - [x] `GET /api/understanding/{project_id}/results`
  - [x] `GET /api/understanding/{project_id}/chain/{analysis_id}`
- [x] Add `get_understanding_engine` in `codeloom/api/deps.py`
- [x] Register router in `codeloom/api/app.py`
- [x] All understanding routes use user-scoped project access checks

## Phase F: Migration and Chat Integration

- [x] Update `codeloom/core/migration/context_builder.py`
  - [x] Add deep-analysis overlap query and ranking
  - [x] Add deep-analysis context section composition
  - [x] Add coverage calculation and warn/fail policy hooks
- [x] Update `codeloom/core/migration/phases.py`
  - [x] Inject deep context in analyze/transform/test
  - [x] Add coverage/confidence metadata logging
  - [x] Add context-aware LLM override routing
- [x] Update `codeloom/core/engine/retriever.py`
  - [x] Add `QueryIntent.FLOW`
  - [x] Add `QueryIntent.DATA_LIFECYCLE`
- [x] Update `codeloom/core/asg_builder/expander.py`
  - [x] Add optional `intent` behavior for FLOW queries
- [x] Update `codeloom/api/routes/code_chat.py`
  - [x] Add narrative injection from deep analyses
  - [x] Preserve fallback behavior when no analyses exist

## Phase G: Config and Settings

- [x] Add `migration.deep_analysis.*` and `migration.llm_overrides.*` in `config/codeloom.yaml`
- [x] Add migration/deep-analysis getters in `codeloom/core/config/config_loader.py`
- [x] Add GET/POST `/api/settings/migration-llm` in `codeloom/api/routes/fastapi_settings.py`

## Phase H: Tests

- [x] Add unit tests for:
  - [x] entry-point detection (merge logic, classify type, max cap, annotation patterns)
  - [x] cycle prevention and max depth (flat membership walk, diamond pattern)
  - [x] membership aggregation (`min_depth`, `path_count`)
  - [x] analyzer validation/evidence requirements (tier selection, token counting, JSON parsing, quality gates, bundle building, source preparation)
  - [x] worker lease and retry/backoff paths (lifecycle, claim, heartbeat, failure handling, exponential backoff)
  - [x] call tree reconstruction (`_build_tree_from_paths` linear + branching)
- [x] Add integration tests for:
  - [x] full job lifecycle (start, status, entry points, results, chain detail)
  - [x] engine lazy initialization
  - [x] migration deep-context injection and fallback
  - [x] chat narrative enrichment and fallback
  - [x] framework detection registry
  - [x] data model defaults and enum values
- [x] All 96 tests pass (`pytest codeloom/tests/test_understanding/`)

## Final Readiness Gate

- [x] `alembic upgrade head` passes
- [x] `pytest codeloom/tests` passes (113 tests, 0 failures)
- [x] `cd frontend && npm run build` passes
- [ ] Manual smoke: ingest -> ASG -> analyze -> migration -> chat narrative
