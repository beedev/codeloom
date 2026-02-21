# Background Analysis Architecture

> How CodeLoom handles long-running computation without blocking HTTP requests

*Created: 2026-02-22*
*Status: Reference documentation*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Understanding Worker](#2-understanding-worker)
3. [RAPTOR Worker](#3-raptor-worker)
4. [Migration Analysis — Request-Triggered Background](#4-migration-analysis--request-triggered-background)
5. [Auto-Triggered Post-Ingestion Analysis](#5-auto-triggered-post-ingestion-analysis)
6. [Job Lifecycle Diagrams](#6-job-lifecycle-diagrams)
7. [Database Schema](#7-database-schema)
8. [Configuration and Scaling](#8-configuration-and-scaling)

---

## 1. Overview

CodeLoom performs three categories of expensive computation: deep code understanding (LLM-driven call chain analysis), RAPTOR tree building (hierarchical summarization for retrieval), and migration MVP analysis (LLM inference per functional scope). All three are too slow to complete within a single HTTP request.

CodeLoom solves this with three distinct background processing patterns:

| Pattern | Used For | Lifecycle |
|---------|----------|-----------|
| **Daemon workers** | Understanding analysis, RAPTOR tree building | Long-running threads that poll the database for pending jobs |
| **Request-triggered background** | Migration MVP analysis | FastAPI `BackgroundTasks` launched when a client calls an analysis endpoint |
| **Auto-triggered** | Post-upload deep understanding | Upload endpoints automatically create a job row after ingestion completes; the daemon worker picks it up |

```
+-----------------------------------------------------------------------+
|                         Three Background Patterns                      |
+-----------------------------------------------------------------------+
|                                                                       |
|  1. DAEMON WORKERS (long-running threads)                             |
|                                                                       |
|   +-----------------+        +-----------------+                      |
|   | Understanding   |        | RAPTOR          |                      |
|   | Worker          |        | Worker          |                      |
|   | (asyncio loop)  |        | (asyncio loop)  |                      |
|   |                 |        |                 |                      |
|   | Polls DB every  |        | Polls DB every  |                      |
|   | N seconds       |        | N seconds       |                      |
|   +-------+---------+        +--------+--------+                      |
|           |                           |                               |
|           v                           v                               |
|   deep_analysis_jobs          raptor_jobs (conceptual)               |
|   (FOR UPDATE SKIP LOCKED)    (serial, no distributed lease needed)  |
|                                                                       |
+-----------------------------------------------------------------------+
|                                                                       |
|  2. REQUEST-TRIGGERED BACKGROUND (FastAPI BackgroundTasks)            |
|                                                                       |
|   Client                 FastAPI Route              Background Task   |
|      |                        |                           |           |
|      |-- POST /analyze -----> |                           |           |
|      |                        |-- set status="analyzing" |           |
|      |                        |-- add_task(analyze_mvp) ->|           |
|      |<-- 202 {analyzing} --- |                           |           |
|      |                        |               runs analyze_mvp()     |
|      |-- GET /status -------> |                           |           |
|      |<-- {analyzing} ------- |                           |           |
|      |-- GET /status -------> |                           |           |
|      |<-- {completed} ------- |               updates status         |
|                                                                       |
+-----------------------------------------------------------------------+
|                                                                       |
|  3. AUTO-TRIGGERED (upload endpoint -> job creation -> daemon pickup) |
|                                                                       |
|   Client          Upload Route       Background Task    Understanding |
|      |                 |                  |               Worker      |
|      |-- POST /upload->|                  |                  |        |
|      |                 |-- run ingestion  |                  |        |
|      |                 |-- add_task() --->|                  |        |
|      |<-- 200 {done} --| (returns early)  |                  |        |
|      |                 |            create job row            |        |
|      |                 |            (status=pending)          |        |
|      |                 |                  |         poll DB ->|        |
|      |                 |                  |         lock job  |        |
|      |                 |                  |         process ->|        |
|      |                 |                  |         complete  |        |
|                                                                       |
+-----------------------------------------------------------------------+
```

**Key principle**: The daemon workers are the only components that actually run analysis. `BackgroundTasks` and upload endpoints are thin triggers that either run fast work (MVP analysis) or create a job row and hand off to the daemon (deep understanding).

---

## 2. Understanding Worker

**Source**: `codeloom/core/understanding/worker.py`

### Purpose

The Understanding Worker performs deep LLM-driven analysis on a project's code graph. It traces entry points through call chains, feeds complete execution paths to an LLM, and extracts structured understanding: business rules, data entities, integrations, and side effects. This work takes minutes per project and cannot run in-process with a request handler.

### Threading Model

The worker runs as a **daemon thread** with its own asyncio event loop, started once at server startup from `codeloom/__main__.py`.

```python
# Conceptual startup in __main__.py
if not os.getenv("DISABLE_BACKGROUND_WORKERS"):
    understanding_worker = UnderstandingWorker(db_manager, engine)
    understanding_worker.start()   # daemon=True thread

    raptor_worker = RaptorWorker(pipeline)
    raptor_worker.start()          # daemon=True thread
```

Because the thread is a daemon, it exits automatically when the main process exits. No explicit shutdown coordination is required.

### Poll-and-Lock Loop

The worker continuously polls the `deep_analysis_jobs` table for rows with `status = 'pending'`. It uses PostgreSQL's `FOR UPDATE SKIP LOCKED` to acquire an exclusive row lock without blocking other workers:

```sql
SELECT *
FROM deep_analysis_jobs
WHERE status = 'pending'
ORDER BY created_at ASC
FOR UPDATE SKIP LOCKED
LIMIT 1
```

`SKIP LOCKED` means a second worker instance (if ever deployed) will skip any row already locked by the first worker and look for another pending row. This is a standard distributed lease pattern: no external lock manager or message queue is needed.

### Heartbeat

While processing a job, the worker updates the `heartbeat_at` column every 30 seconds:

```
status = 'processing'
heartbeat_at = now()   <-- updated every 30 seconds during processing
```

Any other process (or a future worker restart) can detect a stale job by checking:

```python
stale_threshold = now() - timedelta(seconds=120)
is_stale = job.heartbeat_at < stale_threshold
```

If a job's `heartbeat_at` falls more than 120 seconds in the past and its status is still `'processing'`, the worker reclaims it by resetting `status = 'pending'` and incrementing `retry_count`. This handles the case where a worker crashed mid-analysis without updating the job row to `'failed'`.

### Concurrency Control

A `threading.Semaphore(2)` limits how many analyses can run concurrently. Deep understanding analysis is LLM-intensive and involves many sequential sub-steps (entry point detection, call chain tracing, LLM inference, embedding). Running more than two at once degrades response quality and risks LLM provider rate limits.

### Failure Handling and Retry

If analysis raises an exception, the worker catches it and writes:

```
status    = 'failed'
error     = str(exception)
retry_count += 1
```

If `retry_count` is below the configured maximum, the worker resets `status = 'pending'` after an exponential backoff delay:

```
delay = base_delay * (2 ** retry_count)
```

This handles transient LLM provider failures (rate limits, timeouts) without requiring manual intervention.

### Job State Machine

```
              +----------+
              | pending  |  <-- initial state on job creation
              +----+-----+
                   |
          worker polls and locks
                   |
                   v
           +------------+
           | processing |  <-- heartbeat_at updated every 30s
           +----+-------+
                |
        +-------+--------+
        |                |
        v                v
  +-----------+     +--------+
  | completed |     | failed |
  +-----------+     +---+----+
                        |
              retry_count < max?
                        |
                        v (yes)
                   +----------+
                   | pending  |  <-- reset after backoff delay
                   +----------+
```

### Configuration Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| Poll interval | configurable | Seconds between database polls when idle |
| Heartbeat interval | 30 seconds | How often `heartbeat_at` is updated during processing |
| Stale reclaim threshold | 120 seconds | How long before a processing job is reclaimed |
| Max concurrent analyses | 2 | Semaphore limit |
| Retry backoff base | configurable | Base delay for exponential retry |

---

## 3. RAPTOR Worker

**Source**: `codeloom/core/raptor/worker.py`

### Purpose

The RAPTOR Worker builds hierarchical summary trees over a project's embedded code chunks. Higher-level nodes in the RAPTOR tree represent summaries of clusters of related chunks. Retrieval can then operate at multiple levels of abstraction — querying both raw code chunks and their summaries — improving recall for high-level questions.

Tree building requires loading all embeddings for a project, running clustering, and making many LLM calls for summarization. This takes minutes on large projects and is not compatible with HTTP request latency.

### Threading Model

The RAPTOR Worker follows the same daemon thread pattern as the Understanding Worker: a dedicated thread with its own asyncio event loop, started in `__main__.py` alongside the Understanding Worker.

### Processing Model: Serial

Unlike the Understanding Worker, the RAPTOR Worker processes one tree build at a time. There is no semaphore because RAPTOR tree construction is a sequential pipeline (cluster → summarize → cluster again at higher level). Parallelizing multiple projects' tree builds would compete for the same embedding and LLM resources without meaningful benefit.

### Key Difference from Understanding Worker

| Characteristic | Understanding Worker | RAPTOR Worker |
|---------------|---------------------|---------------|
| Concurrency | Semaphore(2) — up to 2 concurrent | Serial — 1 at a time |
| Lock strategy | FOR UPDATE SKIP LOCKED | Simple poll (no competing workers) |
| Work unit | Per project deep analysis job | Per project tree build |
| Output | `deep_analyses`, `analysis_units` tables | RAPTOR tree nodes in vector store |

### Startup Guard

Both workers check `DISABLE_BACKGROUND_WORKERS` at startup. When running under Gunicorn with multiple worker processes, each process would start its own worker threads, causing multiple threads to compete for the same jobs. Setting `DISABLE_BACKGROUND_WORKERS=true` prevents this. See [Section 8](#8-configuration-and-scaling) for the recommended Gunicorn configuration.

---

## 4. Migration Analysis — Request-Triggered Background

**Source**: `codeloom/api/routes/migration.py`

### The Problem

Migration analysis runs LLM inference over the code units that belong to a functional MVP (a bounded scope of migration work). A single MVP analysis (`analyze_mvp`) takes 30 seconds to 2 minutes. For a project with 10 or more MVPs, the `analyze-all` endpoint — which analyzes every MVP — could block for 10 to 20 minutes. This far exceeds any reasonable HTTP timeout.

### The Solution: FastAPI BackgroundTasks

FastAPI's `BackgroundTasks` mechanism runs a function after the HTTP response has been sent. The endpoint returns immediately with a status response, and the actual work happens in the background within the same process.

This is appropriate here because:
- The analysis work, while slow, is bounded (minutes, not hours)
- The client polls a lightweight status endpoint to check progress
- No persistent job queue or database polling is needed for this pattern

### Single MVP Analysis Flow

```
Client                     FastAPI Route                   Database
  |                              |                              |
  |-- POST /mvps/{id}/analyze -->|                              |
  |                              |-- UPDATE functional_mvps     |
  |                              |   SET analysis_status =      |
  |                              |   'analyzing'             -->|
  |                              |                              |
  |                              |-- background_tasks.add_task( |
  |                              |     engine.analyze_mvp,      |
  |                              |     mvp_id, project_id       |
  |                              |   )                          |
  |                              |                              |
  |<-- 202 {"status": "analyzing", "mvp_id": ...} ------------ |
  |                              |                              |
  |  (background task runs)      |                              |
  |                              |   engine.analyze_mvp()  ---->|
  |                              |   (30s - 2min)               |
  |                              |-- UPDATE analysis_status =   |
  |                              |   'completed' OR 'failed' -->|
  |                              |                              |
  |-- GET /mvps/{id}/analysis-status -------------------------> |
  |<-- {"status": "completed", "analysis": {...}} ------------ |
```

### Batch Analysis Flow (analyze-all)

When the client calls `POST /migration/{plan_id}/analyze-all`, the endpoint:

1. Queries all MVPs in the plan that are eligible for analysis
2. Marks every eligible MVP `analysis_status = 'analyzing'` in a single bulk update
3. Launches a single background task that processes each MVP sequentially
4. Returns immediately with a count of MVPs queued

The background task processes MVPs sequentially (not concurrently) to avoid overwhelming the LLM provider with parallel requests.

```
Client                     FastAPI Route               Background Task
  |                              |                            |
  |-- POST /analyze-all -------> |                            |
  |                              |-- bulk UPDATE              |
  |                              |   all MVPs -> 'analyzing'  |
  |                              |-- add_task(_analyze_all)-->|
  |<-- 202 {"queued": 12} ------ |                            |
  |                              |              analyze MVP 1 |
  |                              |              analyze MVP 2 |
  |                              |              ...           |
  |                              |              analyze MVP 12|
  |-- GET /mvps/{id}/status ---> |                            |
  |<-- {status: "analyzing"} --- |                            |
  |  (later)                     |                            |
  |-- GET /mvps/{id}/status ---> |                            |
  |<-- {status: "completed"} --- |                            |
```

### Status Tracking

Status is tracked on the `FunctionalMVP` row directly. No separate jobs table is needed because:
- Each MVP is its own unit of analysis
- The client already knows which MVPs it cares about
- The status transitions are simple and non-retryable at this layer

| `analysis_status` value | Meaning |
|------------------------|---------|
| `pending` | Not yet analyzed |
| `analyzing` | Background task is running |
| `completed` | Analysis available, result stored |
| `failed` | Analysis failed; see `analysis_error` |

### Error Handling

If `engine.analyze_mvp()` raises an exception, the background task catches it and writes:

```
functional_mvps.analysis_status = 'failed'
functional_mvps.analysis_error  = str(exception)
```

Failed MVPs can be retried by the user through the UI, which re-issues the `POST /analyze` request. There is no automatic retry in this pattern — the retry cost is low (one user click) and the failure modes are usually transient (LLM timeout, rate limit).

---

## 5. Auto-Triggered Post-Ingestion Analysis

**Source**: `codeloom/api/routes/projects.py`

### The Problem

After a user uploads a codebase, CodeLoom completes ingestion: AST parsing, code chunking, embedding, and vector store insertion. At that point the project is queryable via RAG chat. However, the deep understanding analysis — which produces business rule extractions, call chain narratives, and enriched context — had to be triggered manually. Users often did not realize this step existed.

### The Solution: Automatic Job Creation After Ingestion

The three upload endpoints automatically create a `DeepAnalysisJob` row after successful ingestion. The existing Understanding Worker daemon thread picks it up through its normal poll-and-lock loop. No new infrastructure is needed.

The important architectural point: the upload endpoint's `BackgroundTask` does not run the analysis itself. It only creates the job row. The daemon worker does the heavy lifting. This keeps the background task's work fast (a single database insert) and avoids the upload endpoint having any dependency on the understanding engine's long-running internals.

### Affected Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/projects/{id}/upload` | Zip file upload |
| `POST /api/projects/{id}/ingest-git` | Git repository ingestion |
| `POST /api/projects/{id}/ingest-local` | Local path ingestion |

All three follow the same pattern.

### Flow

```
Client                  Upload Route          Background Task     Understanding
  |                          |                      |               Worker
  |-- POST /upload --------> |                      |                 |
  |                          |-- extract zip        |                 |
  |                          |-- AST parse          |                 |
  |                          |-- chunk + embed      |                 |
  |                          |-- store vectors      |                 |
  |                          |                      |                 |
  |                          |-- ingestion result:  |                 |
  |                          |   files_processed=42 |                 |
  |                          |   errors=0           |                 |
  |                          |                      |                 |
  |                          |-- add_task(          |                 |
  |                          |   _auto_trigger_     |                 |
  |                          |   analysis, ...)  -->|                 |
  |                          |                      |                 |
  |<-- 200 {files: 42} ----- |   (returns to client)|                 |
  |                          |                      |                 |
  |                          |          INSERT INTO deep_analysis_jobs|
  |                          |          (status='pending')            |
  |                          |                      |                 |
  |                          |                      |   poll DB ----> |
  |                          |                      |   lock job      |
  |                          |                      |   process       |
  |                          |                      |   complete ---> |
```

### Trigger Conditions

The `_auto_trigger_analysis` background function only creates the job row if ingestion was genuinely successful:

```python
if result.files_processed > 0 and result.errors == 0:
    await understanding_engine.start_analysis(project_id, user_id)
    # start_analysis() creates the DeepAnalysisJob row with status='pending'
```

If ingestion partially failed (some files errored), the analysis is not triggered automatically. The user can trigger it manually after resolving the ingestion issues.

### Why Not Run Analysis Directly in the Background Task?

Running `understanding_engine.run_full_analysis()` directly inside the upload endpoint's `BackgroundTask` would work technically, but creates two problems:

1. **No heartbeat, no retry**: A long-running `BackgroundTask` has no mechanism for heartbeat updates. If the uvicorn worker is restarted mid-analysis, the work is lost silently.
2. **No distributed safety**: If multiple processes were ever deployed, each would run its own analysis simultaneously on the same project.

By creating a job row instead, the Understanding Worker's full state machine (heartbeat, stale reclaim, retry with backoff, concurrency semaphore) applies automatically to post-ingestion analyses without any code duplication.

---

## 6. Job Lifecycle Diagrams

### 6.1 Daemon Worker: Poll-Lock-Process Loop

```
+------------------+
|  Worker starts   |
|  (daemon thread) |
+--------+---------+
         |
         v
+------------------+
|  Sleep N seconds |
+--------+---------+
         |
         v
+---------------------------+
|  SELECT ... FOR UPDATE    |
|  SKIP LOCKED LIMIT 1      |
|  WHERE status='pending'   |
+--------+------------------+
         |
    +----+----+
    |         |
    |no rows  |row found
    |         |
    v         v
+------+  +----------------------------+
|Sleep |  |  UPDATE status='processing'|
|loop  |  |  heartbeat_at=now()        |
+------+  +------------+---------------+
               |
               |  (start processing)
               |
    +----------+----------+
    |                     |
    | heartbeat loop      |
    | (every 30s)         |
    |   UPDATE heartbeat  |
    |   WHERE job_id=...  |
    +---------------------+
               |
          +----+----+
          |         |
        success   failure
          |         |
          v         v
  +----------+  +--------+      +----------+
  |completed |  | failed |      | pending  |
  +----------+  +---+----+      | (retry)  |
                    |           +----------+
               retry < max? ------>  yes
                    |
                   no
                    |
                    v
               +--------+
               | failed |
               | final  |
               +--------+
```

### 6.2 Request-Triggered: Analyze MVP

```
Client               Route Handler              Background Task
  |                       |                           |
  |--POST /analyze------> |                           |
  |                       |--set status='analyzing'   |
  |                       |--add_task(analyze_mvp)--> |
  |<--202 {analyzing}---- |                           |
  |                       |               (runs async)|
  |                       |               LLM calls   |
  |                       |               30s-2min    |
  |--GET /status--------> |                           |
  |<--{analyzing}-------- |                           |
  |  (wait...)            |               done        |
  |                       |<--update status='completed'
  |--GET /status--------> |                           |
  |<--{completed, result}-|                           |
```

### 6.3 Auto-Triggered: Upload to Analysis

```
Client         Upload Route       BackgroundTask      Understanding Worker
  |                 |                  |                      |
  |--POST /upload-> |                  |                      |
  |                 |--run ingestion   |                      |
  |                 |  (sync, in-req)  |                      |
  |                 |--add_task()----> |                      |
  |<--200 {done}--- |                  |                      |
  |                 |    INSERT job row|                      |
  |                 |    status=pending|                      |
  |                 |                  |     poll: found job->|
  |                 |                  |     lock row         |
  |                 |                  |     run analysis     |
  |                 |                  |     heartbeat loop   |
  |                 |                  |     update completed |
  |--GET /understanding/status-------> (via understanding API)|
  |<--{status: completed, analyses: [...]}                    |
```

---

## 7. Database Schema

### `deep_analysis_jobs`

Tracks one analysis job per project execution. The Understanding Worker reads and writes this table exclusively.

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | UUID (PK) | Unique job identifier |
| `project_id` | UUID (FK) | Project being analyzed |
| `user_id` | UUID (FK) | User who triggered the analysis |
| `status` | TEXT | `pending` / `processing` / `completed` / `failed` |
| `created_at` | TIMESTAMP | When the job was created |
| `started_at` | TIMESTAMP | When the worker began processing |
| `completed_at` | TIMESTAMP | When processing finished |
| `heartbeat_at` | TIMESTAMP | Last heartbeat update from the worker |
| `error` | TEXT | Error message on failure (NULL on success) |
| `retry_count` | INTEGER | Number of times this job has been retried |

**Index**: `(status, created_at)` — supports efficient polling query.

**Index**: `(project_id, status)` — supports status lookups per project.

### `deep_analyses`

Stores the structured output of one completed analysis. A project may have multiple analyses over time (re-runs produce new rows).

| Column | Type | Description |
|--------|------|-------------|
| `analysis_id` | UUID (PK) | Unique analysis identifier |
| `project_id` | UUID (FK) | Project this analysis covers |
| `job_id` | UUID (FK) | The job that produced this analysis |
| `analysis_type` | TEXT | Classification of analysis output |
| `output` | JSONB | Structured analysis result |
| `created_at` | TIMESTAMP | When this analysis was stored |

### `analysis_units`

Stores per-code-unit understanding results. Each entry links an analysis to a specific `code_unit` and stores the extracted understanding (business rules, data entities, integrations, side effects) as JSONB.

| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | UUID (PK) | Unique entry identifier |
| `analysis_id` | UUID (FK) | Parent analysis |
| `code_unit_id` | UUID (FK) | The code unit being described |
| `understanding` | JSONB | Extracted understanding payload |
| `created_at` | TIMESTAMP | When this entry was stored |

### `functional_mvps` (relevant columns)

The `FunctionalMVP` table is owned by the migration engine. Two columns track analysis state for the request-triggered background pattern:

| Column | Type | Description |
|--------|------|-------------|
| `analysis_status` | TEXT | `pending` / `analyzing` / `completed` / `failed` |
| `analysis_error` | TEXT | Error message when `analysis_status = 'failed'`; NULL otherwise |

These columns are set by the background task running `engine.analyze_mvp()`. The Understanding Worker does not interact with `functional_mvps`.

---

## 8. Configuration and Scaling

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISABLE_BACKGROUND_WORKERS` | unset | Set to `true` to prevent daemon workers from starting. Required when running multiple Gunicorn worker processes. |

### Single-Worker Mode (Default — Development and Small Deployments)

With a single uvicorn process, both daemon workers start normally. `BackgroundTasks` and auto-triggered jobs all run within the same process. No additional infrastructure is needed.

```
uvicorn process (1 worker)
  |-- Understanding Worker thread (daemon)
  |-- RAPTOR Worker thread (daemon)
  |-- FastAPI request handlers
       |-- BackgroundTasks (migration analysis)
       |-- BackgroundTasks (job row creation for auto-trigger)
```

### Multi-Worker Mode (Gunicorn — Production)

Gunicorn spawns multiple worker processes to handle concurrent requests. Each process would independently start its own daemon threads, creating multiple workers competing for the same jobs. `FOR UPDATE SKIP LOCKED` handles competition correctly for the Understanding Worker, but the RAPTOR Worker (serial, no locking) is not safe in this mode.

**Recommended configuration**:

```
# Option A: One Gunicorn worker handles background work
gunicorn codeloom.api.app:app \
  --workers 1 \          # single worker process runs daemon threads
  --worker-class uvicorn.workers.UvicornWorker

# Option B: Separate processes — API workers + dedicated background worker
gunicorn ... --workers 4 --env DISABLE_BACKGROUND_WORKERS=true  &  # API only
python -m codeloom --background-only                              &  # workers only
```

Option B is the recommended production configuration for high-throughput deployments. It separates API latency concerns from background compute concerns and allows independent scaling.

### Timing Reference

| Threshold | Value | Controlled By |
|-----------|-------|---------------|
| Understanding Worker heartbeat | 30 seconds | Worker implementation |
| Stale job reclaim | 120 seconds | Worker implementation |
| RAPTOR Worker poll interval | configurable | RAPTOR worker config |
| MVP analysis timeout (per MVP) | LLM provider timeout | Provider config |
| Ingestion → job creation delay | Near-zero (one DB insert) | BackgroundTask overhead |

### Operational Considerations

**Checking job status**: Query `deep_analysis_jobs` where `project_id = ?` and `status IN ('pending', 'processing')` to see active work. A row with `status = 'processing'` and `heartbeat_at` more than 120 seconds in the past indicates a stale job that will be reclaimed on the next poll cycle.

**Manually retrying a failed job**: Reset `status = 'pending'` and `retry_count = 0` on the job row. The Understanding Worker will pick it up on its next poll.

**Monitoring**: The `heartbeat_at` column is the primary liveness signal. Alert if any job has `status = 'processing'` and `heartbeat_at < now() - interval '5 minutes'` — this indicates the worker itself has stopped.

**Token cost awareness**: Deep understanding analysis makes multiple LLM calls per entry point traced. For a large project (200+ files, 50K lines), a full analysis may consume significant token budget. Monitor LLM provider usage after the first few analyses on representative codebases to calibrate cost expectations.
