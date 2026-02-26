# CodeLoom Architecture Assessment

**Reviewer**: Claude (AI-assisted review)
**Date**: February 24, 2026
**Scope**: Full platform — 5 architecture documents covering Platform Overview, Ingestion Pipeline, Query Engine, Migration Engine, and Deep Understanding Engine
**Context**: Internal tool for code understanding and migration; security to be layered in per-customer later

---

## Executive Summary

CodeLoom is a well-architected code intelligence and migration platform. The architecture documents are thorough, internally consistent, and demonstrate strong engineering judgment across the ingestion, retrieval, and migration subsystems. The design is appropriate for an internal tool where the user base is trusted and the priority is functional depth over hardened multi-tenancy.

That said, several areas deserve attention before scaling beyond a small internal team or onboarding customer-specific deployments. The concerns below are organized by severity and grouped into structural, operational, and forward-looking categories.

---

## Strengths

### Thoughtful Retrieval Pipeline
The hybrid retrieval approach (BM25 + vector fusion + cross-encoder reranking + RAPTOR summaries + ASG graph expansion) is genuinely sophisticated. Each layer addresses a specific failure mode of the others, and the configurable weights and per-request QueryTimeSettings give operators real tuning control. The decision to keep RAPTOR summaries in the same pgvector table as raw chunks, distinguished by metadata, is elegant — it avoids a second retrieval path while still supporting multi-granularity answers.

### Migration Engine Design
The MVP-centric migration model is the standout architectural decision. Breaking migrations into clustered Functional MVPs with independent phase progression, human approval gates, and deterministic-first transforms is a mature pattern that addresses real-world migration complexity. The lane abstraction (with registry, versioning, deprecation, and auto-detection) creates a clean extension point for new source-to-target paths. The confidence model with tiered scoring and the separation of `requires_review` from confidence scores shows careful thought about human-in-the-loop workflows.

### Stateless Query Path
Maintaining both a legacy stateful mode and a production stateless mode — with clear documentation that API routes must only use the stateless path — is pragmatic. The stateless path's per-request retriever creation, thread-safe node cache with TTL and count-based staleness checks, and database-backed conversation history are sound choices for a multi-user deployment.

### ASG as a First-Class Artifact
Treating the Abstract Semantic Graph as a persistent, queryable structure (rather than a transient analysis artifact) pays dividends across the platform. The same edge data powers retrieval expansion, migration clustering, impact analysis, and call chain tracing. The domain-gated detector pattern in the ASG builder avoids unnecessary work for projects that don't need specialized edge detection.

### Documentation Quality
The five architecture documents are unusually well-written. They explain not just what the system does but why each design choice was made, with explicit tradeoff discussions. The cross-references between documents are consistent and useful.

---

## Areas of Concern

### 1. Single-Process Architecture (High Impact)

The platform runs as a single FastAPI/uvicorn process. The RAPTOR background worker, the deep analysis worker, and the HTTP server all share a single Python process with asyncio. This creates several constraints:

- **RAPTOR worker and `DISABLE_BACKGROUND_WORKERS`**: The docs note that asyncio doesn't fork safely, requiring workers to be disabled under Gunicorn. But no alternative worker deployment model (Celery, dedicated process, task queue) is described. For an internal tool this is manageable; for customer deployments it becomes a scaling bottleneck.
- **Deep analysis concurrency**: The semaphore-based concurrency control (default width 3) is process-scoped. In multi-worker deployments, total concurrency is `semaphore_width × replica_count` with no coordination. This could exceed LLM rate limits unpredictably.
- **No horizontal scaling story**: There's no mention of how to run multiple API server instances behind a load balancer. Session cookies are server-side (Starlette SessionMiddleware), which typically stores sessions in-process memory. This means sessions are not portable across instances without a shared session backend.

**Recommendation**: For internal use, this is fine as-is. Before customer deployment, consider extracting background workers into a dedicated task queue (Celery + Redis or a simple PostgreSQL-backed job queue) and switching to a database-backed or Redis-backed session store.

### 2. Node Cache Consistency (Medium Impact)

The in-memory node cache (`_node_cache` with 5-minute TTL) is guarded by a `threading.Lock()` but is process-local. In a multi-instance deployment:

- Two instances can serve stale data simultaneously after an upload completes on one instance.
- The staleness check compares cached node count against the database, which helps, but there's a race window between upload completion and the next cache check on peer instances.
- Cache invalidation (`invalidate_node_cache`) only affects the local process.

For a single-instance internal tool, this is a non-issue. For multi-instance deployments, consider Redis-backed caching or a pub/sub invalidation channel.

### 3. Ingestion Pipeline Limits (Low-Medium Impact)

The documented limits are conservative: 500 files per project, 50 MB per file, 300-second git clone timeout. These are reasonable for targeted migration analysis but may exclude:

- Large monorepos (thousands of files across multiple services)
- Generated code or vendored dependencies that inflate file counts
- Repositories with large binary assets mixed in

The 500-file limit (doc says 500 in the limits table, 1000 in the text above — a minor inconsistency worth correcting) could be a hard constraint for enterprise Java codebases with deep package hierarchies.

**Recommendation**: Make limits configurable per-project or per-deployment, and add a pre-flight scan endpoint that reports projected file counts before ingestion begins.

### 4. LLM Provider Dependency and Cost Exposure (Medium Impact)

The platform makes heavy LLM usage across multiple subsystems: query expansion (3 sub-queries per user query), RAPTOR tree building (one summary per cluster per tree level), deep analysis chain analysis (100K+ token inputs), migration phase execution (per-MVP, per-phase), and agentic MVP refinement (up to 3 iterations). For an internal tool, cost is tracked centrally and manageable. For customer-facing deployments, there's no documented:

- Token usage tracking or budgeting per project/user
- Cost estimation before triggering expensive operations (deep analysis, RAPTOR build)
- Rate limiting at the application level (beyond the semaphore for deep analysis)

The `migration.llm_overrides` config for routing different phases to different models is a smart cost optimization lever. Consider extending this pattern to the query engine (e.g., cheaper models for query expansion, premium models for final response generation).

### 5. Error Handling and Observability Gaps (Low-Medium Impact)

The migration engine has good observability: run IDs, execution metrics, scorecards, and retry tracking. The other subsystems are less instrumented:

- **Ingestion pipeline**: Status tracking (`ast_status`, `asg_status`) is present but coarse. There's no per-file progress reporting, no duration tracking, and no structured error collection beyond `ParseError` objects.
- **Query engine**: No mention of query latency tracking, retrieval quality metrics (e.g., reranker score distributions), or cache hit/miss rates.
- **Deep analysis**: Error phase and message are recorded on failure, but there's no structured telemetry for successful analyses (e.g., how many entry points detected, average chain depth, tier distribution).

**Recommendation**: Add structured logging or metrics emission at key pipeline boundaries. For an internal tool, even simple PostgreSQL-backed metric tables would suffice for debugging and tuning.

### 6. Data Model Coupling (Low Impact, Forward-Looking)

The schema is well-normalized and the foreign key relationships are clear. However, several design choices create tight coupling that could complicate future evolution:

- `deep_analyses` stores a full `DeepContextBundle` as JSONB. This is convenient but makes schema evolution harder — adding a field to the bundle requires understanding all consumers.
- `phase_metadata` on migration phases is also JSONB, accumulating execution metrics, gate results, confidence scores, checkpoints, and retry state. This is flexible but risks becoming a "junk drawer" where field semantics drift across pipeline versions.
- The `embedding_config` singleton row pattern works but doesn't support per-project embedding model selection (e.g., if different customers need different embedding dimensions).

For an internal tool, these are acceptable. Worth noting as technical debt items before a multi-tenant or multi-configuration deployment.

---

## Security Considerations (Deferred but Noted)

Per your guidance, security will be added later specific to customers. For the record, the current architecture has these security-relevant characteristics that the future security layer should address:

- **Default admin credentials** (`admin`/`admin123`) are seeded at startup. Acceptable for internal tooling; must be changed or disabled before any external deployment.
- **Session secret** comes from `FLASK_SECRET_KEY` env var. The naming is a legacy artifact but the mechanism is standard. Ensure this is a strong random value in production.
- **No input sanitization** is documented for uploaded zip files. Zip-slip attacks (path traversal via crafted archive entries) should be validated during extraction.
- **No rate limiting** on API endpoints. The `/api/projects/{id}/query/stream` endpoint triggers LLM calls — an unauthenticated or malicious internal user could drive up costs.
- **RBAC is in place** (admin/editor/viewer with project-level access grants), which is a solid foundation. The per-project `project_access` table supports the multi-customer model well.
- **CORS middleware** is mentioned but its configuration isn't detailed. For internal use, permissive CORS is fine; for customer deployments, lock it down to specific origins.

---

## Minor Issues and Inconsistencies

1. **File limit discrepancy**: The limits table says 500 files per project, but the preceding text says "if the total file count exceeds 1,000, ingestion is aborted." Clarify which is authoritative.

2. **Embedding dimension mismatch risk**: `PGVECTOR_EMBED_DIM` must match the embedding model's output dimension. If someone changes `EMBEDDING_MODEL` without updating `PGVECTOR_EMBED_DIM`, ingestion will silently produce misaligned vectors. Consider adding a startup validation check.

3. **RAPTOR clustering parameters**: The docs mention UMAP default of 10 components and GMM probability threshold of 0.3. These are sensible defaults but could produce poor clusters on very small projects (fewer than 20 chunks). Consider documenting minimum project size for RAPTOR to be useful, or auto-disabling RAPTOR below a threshold.

4. **V1/V2 pipeline coexistence**: Both pipeline versions are active and version-aware, which is good. But the docs don't describe what happens if a V1 plan is opened in a codebase that has since been re-ingested, or if the lane version changes between phases. The `lane_versions` snapshot at first execution is a good safeguard, but the behavior should be documented for edge cases.

5. **Bridge enrichment failure mode**: Java and C# bridges "silently skip" on failure. This is correct for graceful degradation but could mask configuration issues. Consider logging a warning (not error) when a bridge is expected but unavailable, so operators know they're running with reduced fidelity.

---

## Assessment for Intended Use Case

Given that CodeLoom is an internal tool for code understanding and migration:

| Dimension | Rating | Notes |
|---|---|---|
| **Functional completeness** | Strong | The ingestion → query → migration pipeline covers the core use case end-to-end. Deep understanding adds meaningful analytical depth. |
| **Architecture quality** | Strong | Clean separation of concerns, explicit dependency injection, well-defined extension points (plugins, lanes, framework analyzers). |
| **Scalability** | Adequate | Single-process architecture is fine for internal team use. Horizontal scaling requires work before customer deployments. |
| **Maintainability** | Strong | Codebase organization is logical, naming is consistent, and the architecture docs provide excellent onboarding material. |
| **Operability** | Adequate | Migration engine is well-instrumented. Other subsystems need more observability for production-grade operation. |
| **Extensibility** | Strong | Adding new languages (via BaseLanguageParser), migration lanes (via LaneRegistry), and framework analyzers (via FrameworkAnalyzer ABC) is clearly documented and low-friction. |
| **Documentation** | Excellent | Among the best internal architecture documentation I've reviewed. Clear, honest about tradeoffs, and internally consistent. |

---

## Recommended Priority Actions

1. **Fix the file limit discrepancy** in the ingestion pipeline docs (500 vs 1000).
2. **Add a startup validation** for embedding dimension alignment.
3. **Document the horizontal scaling path** (session backend, worker extraction, cache invalidation) even if implementation is deferred.
4. **Add basic query latency and cache metrics** to the query engine for tuning visibility.
5. **Implement zip-slip protection** in the ingestion pipeline's archive extraction before any non-internal deployment.
