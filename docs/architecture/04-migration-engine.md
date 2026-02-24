<!-- Architecture Doc: Migration Engine -->

# Migration Engine

> Part of the CodeLoom Architecture Series.
> Related: [01-platform-overview](./01-platform-overview.md) | [02-ingestion-pipeline](./02-ingestion-pipeline.md) | [05-deep-understanding](./05-deep-understanding.md)

---

## Overview

The CodeLoom Migration Engine transforms an analyzed codebase into a target architecture through a phased, human-supervised pipeline. Rather than attempting a whole-codebase transformation in a single pass, the engine breaks the migration into Functional MVPs — cohesive clusters of related code units that can be understood, transformed, and validated independently.

Three design principles govern the engine:

- **MVP-centric execution.** Each Functional MVP is a self-contained unit of migration work. Plan-level phases establish the architecture and scope; per-MVP phases do the transformation. This keeps individual work units small enough to reason about and test.
- **Human approval gates.** No phase advances automatically. A human reviewer approves or rejects each phase before the next begins. Rejection is a clean reset: the phase returns to an executable state, optionally carrying rejection feedback into the next run.
- **Deterministic first, LLM second.** Migration lanes supply rule-based transforms that produce target code without LLM involvement wherever the source pattern is unambiguous. The LLM handles residual complexity, guided by lane-injected prompt augmentation. This produces more predictable output and higher confidence scores.

---

## Pipeline Versions

CodeLoom supports two pipeline versions. New plans always use V2. V1 remains available for plans created before V2 was introduced and is preserved for backward compatibility.

```
V1 — 6-Phase Pipeline (legacy)
+--------+-----------------------------+---------+-----------+-----------+---------+
| Phase  | Name                        | Scope   | MVP-scoped| LLM       | Gate    |
+--------+-----------------------------+---------+-----------+-----------+---------+
|   1    | Discovery                   | Plan    | No        | Yes       | Approve |
|   2    | Architecture                | Plan    | No        | Yes       | Approve |
|   3    | Analyze                     | Per-MVP | Yes       | Yes       | Approve |
|   4    | Design                      | Per-MVP | Yes       | Yes       | Approve |
|   5    | Transform                   | Per-MVP | Yes       | Mixed     | Approve |
|   6    | Test                        | Per-MVP | Yes       | Yes       | Approve |
+--------+-----------------------------+---------+-----------+-----------+---------+

V2 — 4-Phase Pipeline (default)
+--------+-----------------------------+---------+-----------+-----------+---------+
| Phase  | Name                        | Scope   | MVP-scoped| LLM       | Gate    |
+--------+-----------------------------+---------+-----------+-----------+---------+
|   1    | Architecture                | Plan    | No        | Yes       | Approve |
|   2    | Discovery                   | Plan    | No        | Yes       | Approve |
|   3    | Transform                   | Per-MVP | Yes       | Mixed     | Approve |
|   4    | Test                        | Per-MVP | Yes       | Yes       | Approve |
+--------+-----------------------------+---------+-----------+-----------+---------+
```

V2 eliminates the separate Analyze and Design phases. Their intent is captured on-demand via `analyze_mvp()`, which stores a combined analysis and design output on the `FunctionalMVP` record. The Transform phase then reads this output as part of its context. For most framework migrations, the Architecture and Discovery phases supply sufficient context for the Transform LLM call without a dedicated per-MVP design step.

V2 runs Architecture before Discovery. This ordering matters: the Architecture phase produces a target-stack description and structural constraints that inform MVP clustering in Discovery. Cluster boundaries become more meaningful when the system already knows how the target application will be organized.

The active pipeline version is stored on `MigrationPlan.pipeline_version` and consulted at every stage. All engine methods are version-aware.

---

## Plan Lifecycle

A migration plan moves through three top-level states. Phase progression happens within the `in_progress` state. The `complete` state is only reached when every MVP's final phase has been approved.

```
                   create_plan()
                        |
                        v
              +--------------------+
              |       draft        |
              +--------------------+
                        |
              execute_phase() called
                        |
                        v
              +--------------------+
              |    in_progress     |<---------+
              |                    |          |
              |  [phase running]   |    reject_phase()
              |  [phase complete]  |    resets phase,
              |  [awaiting review] |    re-execution
              |  [phase approved]  |----------+
              +--------------------+
                        |
              All MVPs reach "migrated"
                        |
                        v
              +--------------------+
              |      complete      |
              +--------------------+
```

Plan-level phases (Architecture, Discovery) must both be approved before per-MVP phases can be created. Once created, per-MVP phases for different MVPs are independent and can be executed in any order — or in parallel.

---

## Phase Flow

### Plan-Level Phases

Plan-level phases have `mvp_id = NULL` in the database. They operate on the entire project.

**Architecture phase** — The LLM analyzes the target brief, target stack, and representative source patterns. It produces an architectural description of the target system: module boundaries, technology choices, cross-cutting concerns, and constraints. In V2 this runs first, before MVPs exist, establishing the structural context that Discovery will use.

**Discovery phase** — The MVP Clusterer groups source code units into Functional MVPs (see MVP Clustering below). The LLM then generates a discovery narrative that validates the cluster boundaries, identifies shared concerns, and summarizes what each MVP represents. Discovery metadata — including stored procedure analysis and shared concerns — is persisted on the plan for use in subsequent phases.

### Per-MVP Phases

Per-MVP phases are created in batch after both plan-level phases are approved, one set per discovered MVP.

**Transform phase** — Deterministic lane transforms run first. Any source units matched by transform rules receive generated target code immediately, without LLM involvement. The remaining units, plus all deterministic outputs, are assembled into a Transform prompt that is augmented with lane-specific guidance. The LLM produces target code for the non-deterministic residual. Quality gates run at the end of this phase.

**Test phase** — Scoped to the MVP's transformed output. The LLM generates test cases appropriate to the target framework, covering the MVP's exposed interfaces and critical paths. In V1, a separate Design phase preceded Transform, and the Analyze phase before that captured per-MVP dependency analysis.

### Human Approval Gate

Between each phase:

- `approve_phase(plan_id, phase_number, mvp_id)` — Marks the phase as approved and unlocks the next. Blocking gate failures prevent approval; the error message names the failing gates. When the final per-MVP phase is approved, the MVP status advances to `migrated`. When all MVPs reach `migrated`, the plan status advances to `complete`.
- `reject_phase(plan_id, phase_number, mvp_id, feedback)` — Resets the phase to a clean state. Output files are cleared. Retry state and checkpoints are removed. Optional feedback text is stored on the phase for the next execution to consume.

---

## MVP Clustering

### What a Functional MVP Is

A Functional MVP is a cluster of related source code units that form a coherent, independently-migratable unit of functionality. The goal is a cluster whose units share high internal cohesion (they collaborate closely with each other) and low external coupling (they have few dependencies on units outside the cluster).

A cluster that is too large becomes difficult to transform and validate. A cluster that is too small loses context — individual utility functions, for example, only make sense in relation to the callers that use them.

### RAPTOR-Driven Clustering

The primary clustering path leverages the RAPTOR hierarchical index. RAPTOR builds multi-level summaries of the codebase during ingestion. The clusterer reads these summaries to identify natural functional groupings, then uses ASG edges to refine boundaries.

This approach produces functionally coherent clusters because RAPTOR summaries capture semantic meaning, not just file proximity. A cluster built from hierarchical summaries reflects what the code does, not simply where the files live.

### Package-Based Fallback

When RAPTOR summaries are unavailable or insufficient, the clusterer falls back to package and namespace boundaries. Units within the same Java package, Python module, or C# namespace are grouped together, then merged or split based on size constraints.

The fallback produces structurally coherent clusters but may miss semantic relationships that cross package lines.

### Cohesion and Coupling Metrics

Each cluster is scored at creation time:

- **Cohesion** — The ratio of internal ASG edges (edges between units inside the cluster) to the total possible internal edges. A cohesion score of 1.0 means every unit connects to every other unit in the cluster. Higher is better.
- **Coupling** — The ratio of external ASG edges (edges from units inside the cluster to units outside it) to total edges touching the cluster. Lower is better.
- **Size** — Unit count. Clusters above the maximum size threshold are candidates for splitting.
- **Complexity** — Qualitative label (low, medium, high) derived from unit types and dependency depth.

These metrics are stored on the `FunctionalMVP` record and visible in the migration UI.

### MVP Lifecycle

```
   discovered        refined         in_progress        migrated
       |                |                 |                 |
  (after clustering) (after           (Transform         (final phase
                   merge/split)        phase starts)       approved)
```

The `refined` state indicates a human or agentic process has adjusted the cluster boundaries since initial discovery. Merging or splitting an MVP resets its analysis output and diagram cache, forcing fresh re-analysis.

### Agentic Refinement

After algorithmic clustering, the engine runs a multi-pass agentic refinement loop (up to three iterations):

1. Each cluster receives an LLM-generated functional description (name and purpose statement).
2. Inter-MVP edge density is analyzed across all clusters.
3. The LLM evaluates coherence and proposes merge or split suggestions.
4. Approved merges and splits are applied, subject to size guards.
5. The loop repeats until the cluster set is stable.

LLM calls during refinement use temperature 0 for deterministic output. The same codebase should produce the same clusters across runs.

### Merge and Split Operations

Users can manually adjust cluster boundaries before MVP phases are created:

- `merge_mvps(plan_id, mvp_ids, new_name)` — Combines multiple MVPs into one, unioning their unit and file sets, deduplicating stored procedure references, and triggering auto-naming via LLM.
- `split_mvp(plan_id, mvp_id, split_unit_ids, new_name)` — Moves a specified set of units from an existing MVP into a new one.

Both operations mark the affected MVPs as `refined` and invalidate cached diagrams.

---

## Migration Lanes

### Registry and Strategy Pattern

A migration lane is a self-contained module that owns the domain knowledge for a specific source-to-target migration path. The engine itself is orchestration only; lanes provide the intelligence.

All lanes are registered at import time through `LaneRegistry`. The registry is a class-level dictionary keyed by `lane_id`. There is no plugin discovery mechanism — all lanes ship as part of the application.

```
  LaneRegistry
      |
      +-- struts_to_springboot  (StrutsToSpringBootLane)
      +-- storedproc_to_orm     (StoredProcToORMLane)
      +-- vbnet_to_dotnetcore   (VbNetToDotNetCoreLane)
      +-- ...
```

Multiple lanes can be active simultaneously for a single plan. Asset strategies associate a `lane_id` with each language or sub-type; the engine collects all unique lane IDs from `MigrationPlan.asset_strategies` before each phase execution.

### What a Lane Provides

Every lane implements the `MigrationLane` abstract base class with four areas of responsibility:

| Capability | Method | Purpose |
|---|---|---|
| Deterministic transforms | `get_transform_rules()`, `apply_transforms()` | Pattern-matched source-to-target code generation |
| Prompt augmentation | `augment_prompt()` | Injects domain-specific context into LLM prompts |
| Quality gates | `get_gates()`, `run_gate()` | Validates migration completeness and correctness |
| Asset strategy overrides | `get_asset_strategy_overrides()` | Declares how specific file types should be handled |

### Built-In Lanes

| Lane ID | Display Name | Source | Target |
|---|---|---|---|
| `struts_to_springboot` | Struts 1.x/2.x to Spring Boot | Struts actions, form-beans, tiles | Spring MVC controllers, DTOs, Thymeleaf or REST |
| `storedproc_to_orm` | Stored Procedures to ORM | SQL stored procedures, functions, triggers | Spring Data JPA repositories and service classes |
| `vbnet_to_dotnetcore` | VB.NET to .NET Core | VB.NET modules, classes, forms | C# .NET Core equivalents |

The Struts lane supports three view-layer targets (`rest`, `thymeleaf`, `react`), selected from `target_stack.view_layer`. The StoredProc lane is polyglot by design — additional target generators (EF Core, Django) can be added as new generator methods without modifying rules, gates, or prompts.

### Lane Detection

When a user has not explicitly assigned a lane, `LaneRegistry.detect_lane(source_framework, target_stack)` scores every registered non-deprecated lane using `detect_applicability()`. Each lane returns a float from 0.0 (not applicable) to 1.0 (exact match). The highest-scoring lane above 0.0 is selected.

Explicit lane assignment via `asset_strategies` always takes precedence over auto-detection.

### Lane Versioning

Each lane declares a semantic version string via the `version` property. When a plan executes its first phase, the active lane versions are recorded in `MigrationPlan.lane_versions` (a JSONB dictionary keyed by `lane_id`). This version snapshot is included in the migration scorecard and provides a reproducibility record — if transform rules or gate logic change between releases, the stored versions identify which behavior the plan was executed with.

A lane with `deprecated = True` is excluded from auto-detection but remains accessible by explicit ID for backward compatibility with existing plans.

---

## Deterministic Transforms

A `TransformRule` describes a pattern-to-template mapping that the lane can execute without LLM assistance. Rules match on `CodeUnit` metadata fields (unit type, annotations, signatures) and produce target code from a named template.

Key properties of a transform rule:

- **`source_pattern`** — A dictionary of metadata criteria. The lane's `apply_transforms()` method evaluates each unit against this pattern.
- **`target_template`** — A template key that the lane's code generator expands into target-language source.
- **`confidence`** — A float from 0.0 to 1.0 indicating how certain the rule is that its output is correct. High confidence rules produce output that rarely needs human review. Low confidence rules produce a starting point that the LLM or a reviewer should refine.
- **`requires_review`** — When `True`, the transform result is flagged for mandatory human review regardless of confidence score. Used for rules that handle complex patterns with known edge cases.

When the Transform phase executes, deterministic transforms run before the LLM call. Results are injected into `plan_data._deterministic_transforms`, making the generated code available to the LLM as context. The LLM focuses on units that rules could not match.

This separation is significant: deterministic transforms are reproducible, testable, and produce confidence scores that feed the aggregation model. LLM output has implicit uncertainty; deterministic output does not.

---

## Asset Strategies

Asset strategies control how each language or file type in the source project is handled during migration. A strategy is assigned per language, with optional sub-type overrides for cases like XML configuration files versus Java source within the same language group.

The four strategy values are:

| Strategy | Meaning |
|---|---|
| `migrate` | Apply full transformation to the target stack. Deterministic transforms and LLM generate target code. |
| `keep` | Copy the file as-is. No transformation applied. Used for assets that are already compatible with the target. |
| `convert` | Change format without a framework migration. Used for configuration files that need structural changes but not a technology change (for example, `struts-config.xml` becoming `application.properties`). |
| `no_change` | Exclude from migration scope entirely. File is not copied or transformed. |

Strategies are initially proposed by a rule-based inventory analysis in `get_asset_inventory()`, then optionally refined by an LLM call in `refine_asset_strategies()`. The user reviews and confirms strategies before Discovery runs. Confirmed strategies are persisted on `MigrationPlan.asset_strategies`.

Each lane's `get_asset_strategy_overrides()` method can declare default strategies for file patterns specific to that lane. For example, the Struts lane specifies `convert` for `struts-config.xml` and `migrate` for Struts action classes.

Target stack derivation runs automatically when strategies are saved: language and framework names are extracted from strategy targets and lane metadata, populating `MigrationPlan.target_stack` without requiring manual input.

---

## Quality Gates

### Gate Taxonomy

Quality gates validate migration correctness after the Transform phase completes. Every gate belongs to one of six categories:

| Category | What It Checks |
|---|---|
| `PARITY` | Structural migration completeness. Example: every Struts action path has a corresponding `@RequestMapping` in the target controllers. |
| `COMPILE` | Target code compiles or builds without errors. |
| `UNIT_TEST` | Generated unit tests execute and pass. |
| `INTEGRATION` | Integration test validation across migrated components. |
| `CONTRACT` | API or interface contract verification. Ensures public interfaces are preserved. |
| `REGRESSION` | Regression detection against known baselines. |

Every lane must provide at least a `COMPILE` gate and a `UNIT_TEST` gate, even if those gates return a pass-through result when no external build system is available. This ensures gate execution never silently skips validation for lanes that lack a full test environment.

### Blocking vs. Non-Blocking Gates

A gate with `blocking = True` (mandatory) prevents phase approval if it fails. A gate with `blocking = False` (advisory) records a failure in the results but does not block approval. Advisory gates surface information without stopping the migration.

The default for `GateDefinition.blocking` is `True`. Lanes opt specific gates into advisory mode by setting `blocking = False`.

### Gate Execution Flow

```
  Transform phase completes
           |
           v
  +---------------------+
  |  _run_gates() called |
  |  for all active lanes|
  +---------------------+
           |
     for each lane:
       for each gate definition:
         run_gate() --> GateResult
           |
           v
  Gate results stored in
  phase_metadata["gate_results"]
           |
           v
  gates_all_passed =
    all blocking gates passed?
           |
      +----+----+
      |         |
     Yes        No
      |         |
      v         v
  approve_phase()  approve_phase()
  succeeds         raises ValueError
                   (names failing gates)
```

Gate execution errors are caught per-gate: if a gate raises an unexpected exception, it is recorded as a failure with an error detail rather than aborting the entire gate run. This prevents a misconfigured gate from silently blocking a valid migration.

---

## Confidence Model

### Aggregation

`aggregate_confidence(transform_results, weights)` computes a weighted average confidence across all deterministic transform results for a phase. Each result contributes its `confidence` float; an optional `weights` dictionary maps rule names to weight multipliers.

When no transforms are present (the phase was handled entirely by LLM), the confidence score is `None` for that phase.

### Tiers

| Tier | Score Range | Meaning |
|---|---|---|
| `high` | >= 0.90 | Transforms are well-understood patterns; output is likely correct. Suitable for auto-approval policies. |
| `standard` | >= 0.75 | Normal review recommended. Output is plausible but benefits from human inspection. |
| `low` | < 0.75 | Significant uncertainty. Human review is important before proceeding. |

The aggregate confidence score and tier are stored on `phase_metadata.phase_confidence` and `phase_metadata.confidence_tier` after Transform phase execution.

### Review Propagation

If any `TransformRule` in the active lanes has `requires_review = True`, and that rule matched at least one source unit during the Transform phase, `phase_metadata.requires_manual_review` is set to `True`. This flag propagates to the phase output and is surfaced in the UI.

`requires_review` is a rule-level flag, not a confidence-based calculation. It exists for rules that handle known-complex patterns where deterministic generation is best treated as a draft regardless of confidence.

High confidence alone does not suppress `requires_review`. A rule can have a confidence of 0.95 and still set `requires_review = True` if the pattern involves, for example, multi-database transaction semantics that a reviewer should confirm.

---

## Enterprise Reliability

### Run Tracking

Every call to `execute_phase()` generates a fresh `run_id` (UUID4) assigned to the phase row before execution begins. The run ID is stored in `phase_metadata.execution_metrics.run_id`. Repeated executions of the same phase — whether due to retries or rejection — produce distinct run IDs, making it possible to correlate logs and metrics with a specific attempt.

### Retry with Exponential Backoff

If a phase raises an exception, the engine increments a retry counter and schedules a next-retry timestamp using exponential backoff:

```
next_retry_delay = BASE_BACKOFF_SECONDS * (2 ^ retry_count)
                 = 5 * 2^1 = 10s  (first retry)
                 = 5 * 2^2 = 20s  (second retry)
                 = 5 * 2^3 = 40s  (third retry)

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5
```

After `MAX_RETRIES` consecutive failures, the phase is marked `terminal_failure = True` and its status is set to `error`. Further calls to `execute_phase()` raise immediately with an informative message. The only path forward is `reject_phase()`, which performs a clean reset.

### Checkpoints

For the Transform phase, the engine tracks which source units have already been processed. If a lane's `apply_transforms()` call raises mid-way through the unit list, the IDs of already-processed units are saved to `phase_metadata.checkpoint.processed_unit_ids`.

On the next execution attempt, the engine reads the checkpoint and skips any unit whose ID appears in `processed_unit_ids`. This prevents redundant work and avoids duplicate output files when a partial transform is retried.

```
  Transform starts
       |
       v
  Load checkpoint
  (processed_unit_ids)
       |
       v
  For each unit in MVP:
    skip if in checkpoint
    else: apply_transforms()
       |
    exception?
       |
    +--+--+
    |     |
   Yes    No
    |     |
    v     v
  Save    Continue
  checkpoint
  Raise exception
  --> retry tracking
```

On successful completion, the checkpoint is cleared from `phase_metadata`. The clean state ensures that a successfully completed phase, if re-run after rejection, processes all units fresh.

### Clean Rejection

`reject_phase()` performs a full state reset:

- `output_files` is cleared
- `checkpoint` is removed from `phase_metadata`
- `terminal_failure`, `retryable`, and `next_retry_after` are removed
- `retry_count` is reset to 0
- Optional `rejection_feedback` is stored for the next execution

This makes rejection idempotent. Calling `reject_phase()` twice on the same phase produces the same state as calling it once. The next `execute_phase()` call starts from scratch with a new run ID.

---

## Observability

### Execution Metrics

After each successful phase execution, the following metrics are stored in `phase_metadata.execution_metrics`:

- `run_id` — UUID for this execution attempt
- `started_at` — UTC ISO timestamp when execution began
- `completed_at` — UTC ISO timestamp when execution finished
- `duration_ms` — Wall-clock duration in milliseconds
- `llm_model` — LLM provider name from environment configuration

### Migration Scorecard

`get_migration_scorecard(plan_id)` aggregates metrics across all phases of a plan:

| Metric | Description |
|---|---|
| `completion_rate` | Phases complete as a fraction of total phases |
| `gate_pass_rate` | Gates passed as a fraction of total gates run |
| `avg_confidence` | Average phase confidence score across transform phases |
| `avg_confidence_tier` | Tier label for the average confidence |
| `rework_rate` | Rejected phases as a fraction of all completed or rejected phases |
| `total_duration_ms` | Cumulative wall-clock time across all phases |
| `time_per_phase_ms` | Average phase duration |
| `lane_versions` | Lane ID to version string map recorded at first execution |
| `pipeline_version` | V1 or V2 |

The scorecard provides a post-migration quality summary and supports comparison across plans or across pipeline versions.

### Lane Version Recording

Lane versions are recorded once, on the first `execute_phase()` call for a plan. At that point the active lanes are resolved from `asset_strategies` and each lane's `version` property is stored to `MigrationPlan.lane_versions`. Subsequent executions do not overwrite this record. If a plan spans multiple execution sessions, the version snapshot always reflects the lane state at the time the first phase ran.

---

## Batch Execution

For plans with many MVPs, per-MVP phases can be executed in parallel. Each MVP's phase set is independent — Transform for MVP 2 does not depend on Transform for MVP 1 being complete.

Auto-approval policies allow high-confidence phases to advance without a human reviewer when configured. A policy checks `confidence_tier == "high"` and `gates_all_passed == True`. Only phases meeting both criteria are eligible for auto-approval. Advisory gate failures do not block auto-approval; blocking gate failures do.

The `current_phase` field on `MigrationPlan` reflects the most recently executed phase number across all MVPs. In batch mode this value has limited meaning as a progress indicator; the `get_plan_status()` response provides per-MVP phase breakdowns for accurate progress tracking.

---

## Key Source Paths

| Path | Purpose |
|---|---|
| `codeloom/core/migration/engine.py` | `MigrationEngine` — orchestrator, plan lifecycle, phase execution, approval gates |
| `codeloom/core/migration/phases.py` | Phase executors — context building, prompt assembly, LLM calls, output parsing |
| `codeloom/core/migration/lanes/base.py` | `MigrationLane` ABC, `TransformRule`, `GateCategory`, `aggregate_confidence`, `confidence_tier` |
| `codeloom/core/migration/lanes/registry.py` | `LaneRegistry` — class-level dict, `detect_lane()`, `list_lanes()` |
| `codeloom/core/migration/lanes/struts_to_springboot.py` | Struts 1.x/2.x to Spring Boot lane |
| `codeloom/core/migration/lanes/storedproc_to_orm.py` | SQL Stored Procedures to ORM service layer lane |
| `codeloom/core/migration/lanes/vbnet_to_dotnetcore.py` | VB.NET to .NET Core lane |
| `codeloom/core/migration/mvp_clusterer.py` | RAPTOR-based and package-based MVP clustering |
| `codeloom/core/migration/context_builder.py` | `MigrationContextBuilder` — assembles ASG + code context for prompts |

---

## Cross-References

- **01-platform-overview** — System-level context: how the migration engine fits into the CodeLoom platform alongside Code RAG and Code Intelligence.
- **02-ingestion-pipeline** — How AST parsing and ASG edge construction produce the `code_units` and `code_edges` that the MVP Clusterer and context builder consume.
- **05-deep-understanding** — Framework detection and stored procedure analysis that informs Discovery phase outputs and lane selection.
