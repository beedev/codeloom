<!-- Architecture Doc: Query Engine -->

# Query Engine

## Overview

When a user types a question about their codebase in CodeLoom, the query engine is what turns that
question into a precise, contextually grounded answer. It combines hybrid retrieval — BM25 keyword
matching fused with vector similarity search — with two complementary augmentation layers: RAPTOR
hierarchical summaries for document-level understanding, and ASG-aware graph expansion for
relationship-aware code context. The result is streamed back to the browser in real time over SSE.

This document describes how the query engine works, why it is designed the way it is, and the
trade-offs behind each major decision.

---

## Dual-Mode Architecture

`LocalRAGPipeline` exposes two distinct query patterns that exist for different operational
contexts.

### Stateful Mode (Legacy, Single-User)

```
pipeline.switch_project(project_id)   # mutates global state
pipeline.query(mode, message, ...)    # uses that global state
```

The stateful pattern maintains a single active project, a single engine instance, and an in-memory
conversation buffer. Switching projects resets that buffer. This mode is **not safe for concurrent
users** — if two requests arrive simultaneously, they race to mutate the same global state.

### Stateless Mode (Production, Multi-User)

```
pipeline.stateless_query(message, project_id, user_id, ...)
pipeline.stateless_query_streaming(message, project_id, user_id, ...)
```

The stateless pattern creates per-request retrievers, reads from a thread-safe node cache, and
loads conversation history from the database. No global state is mutated. This is the pattern all
API routes use.

### Comparison

| Dimension          | Stateful                        | Stateless                              |
|--------------------|---------------------------------|----------------------------------------|
| Thread safety      | Single-user only                | Concurrent users safe                  |
| State mutation     | Mutates global project/engine   | None — per-request only                |
| Performance        | Engine reuse across requests    | 4-8x faster (bypasses engine setup)    |
| Conversation       | In-memory buffer (session only) | Persisted to database, loaded per call |
| Retriever          | Shared engine retriever         | New retriever created per request      |
| API usage          | Never — legacy/test only        | Always — all `/api/` routes            |

**Rule**: API routes MUST call `stateless_query()` or `stateless_query_streaming()`. Calling
`pipeline.query()` directly from an API route is a thread-safety bug.

---

## Stateless Query Flow

The eight-step flow executed on every API request:

```
User Query
    |
    v
+-------------------+
| 1. Cache Check    |  _get_cached_nodes(project_id)
|                   |  Thread-safe Lock + 5-min TTL
+-------------------+
    |
    v
+-------------------+
| 2. Load History   |  load_conversation_history()
|                   |  Last N turns from DB via ConversationStore
+-------------------+
    |
    v
+-------------------+
| 3. Hybrid Retrieve|  fast_retrieve()
|                   |  BM25 + Vector → Fusion → Rerank
+-------------------+
    |
    v
+-------------------+
| 4. RAPTOR Augment |  get_raptor_summaries()
|                   |  Embedding similarity → top-k tree nodes
+-------------------+
    |
    v
+-------------------+
| 5. ASG Expand     |  ASGExpander.expand()
|                   |  callers + callees + deps → decayed scores
+-------------------+
    |
    v
+-------------------+
| 6. Assemble       |  build_context_with_history()
|                   |  History + Summaries + Chunks → prompt
+-------------------+
    |
    v
+-------------------+
| 7. LLM Call       |  execute_query() / execute_query_streaming()
|                   |  Single completion call with assembled context
+-------------------+
    |
    v
+-------------------+
| 8. Save + Stream  |  save_conversation_turn() to DB
|                   |  Yield SSE events to browser
+-------------------+
```

Steps 3, 4, and 5 run sequentially because each layer enriches the next. The result of step 6 is a
single string passed to the LLM — no retrieval happens inside the LLM call itself.

---

## Hybrid Retrieval

### BM25 + Vector Fusion

Two retrievers run in parallel against the same node index:

- **BM25 retriever** — token-based keyword matching using an inverted index. Strong on exact
  identifiers, class names, method names, and error strings. Produces results regardless of
  embedding dimension.
- **Vector retriever** — dense embedding similarity via pgvector. Strong on paraphrased or
  conceptual queries where keywords alone would miss relevant chunks.

Scores from both are merged using distribution-based score fusion (`dist_based_score` mode in
`QueryFusionRetriever`). Configurable weights default to `[0.5, 0.5]` for BM25 and vector
respectively, controlled by `retriever_weights` in `config/codeloom.yaml`. Operators can shift
the balance toward exact-match or semantic retrieval without code changes.

### Query Expansion

Before retrieval, the original query is expanded into sub-queries. The LLM generates `num_queries`
(default 3) alternative phrasings of the user's question. All phrasings are run through both BM25
and vector retrieval, and the union of results is fed into fusion. This broadens recall for
queries that may be expressed differently in code comments or documentation than how the user
phrased them.

### Cross-Encoder Reranking

After fusion, the `TwoStageRetriever` applies a cross-encoder reranker (`mixedbread-ai/mxbai-rerank-base-v1`).
Bi-encoders (used in the vector retrieval stage) produce approximate similarity by comparing
independent embeddings. A cross-encoder reads the query and each candidate chunk together in a
single forward pass, producing more precise relevance scores at higher computational cost. Running
the cross-encoder only on the fused candidate set (rather than the full index) keeps latency
acceptable.

The reranker is disabled via `DISABLE_RERANKER=true` or toggled per request. When disabled, the
top-k fusion results pass through directly.

### Intent Detection

Before building the retriever, the engine classifies the query into one of several intents using
regex pattern matching:

| Intent           | Example query                          | Effect                                    |
|------------------|----------------------------------------|-------------------------------------------|
| `SUMMARY`        | "Summarize this codebase"              | Prioritizes RAPTOR summary nodes          |
| `SEARCH`         | "Where is user auth implemented?"      | Standard hybrid retrieval (default)       |
| `FLOW`           | "How does a request get processed?"    | Deeper ASG expansion (depth 2)            |
| `DATA_LIFECYCLE` | "Where is the order entity persisted?" | Data-flow ASG expansion (depth 2)         |

MIGRATION and EXPLAIN intents are also recognized and route to appropriate retrieval strategies.
The default when no intent pattern matches is `SEARCH`.

---

## RAPTOR Integration

### What RAPTOR Is

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical tree
of LLM-generated summaries on top of the raw chunk index. Each level of the tree is a more
abstract summary of the level below. At query time, summaries from the upper levels of the tree
provide document-scale understanding that individual chunks cannot — answering questions like
"how does this service work overall?" or "what is the architecture of this module?" without
requiring the user to specify exactly which files are relevant.

### Tree Building

The tree is built bottom-up after ingestion completes:

```
Level 0: Raw chunks (output of code chunker, ~512 tokens each)
    |
    | Cluster: UMAP dimensionality reduction + GMM soft clustering
    v
Level 1: Cluster summaries (LLM summarizes each cluster → embed → store)
    |
    | Repeat until max_tree_depth or single cluster remains
    v
Level N: Root summary (one node summarizing the whole document)
```

Clustering uses UMAP to reduce embedding dimensionality (default 10 components) before applying
Gaussian Mixture Models. GMM produces soft cluster assignments, meaning a chunk can belong to
multiple clusters with different probabilities, which is appropriate for code units that serve
multiple purposes. The probability threshold (default 0.3) controls how aggressively chunks are
multi-assigned.

Each cluster summary is generated by the LLM (capped at `summary_max_tokens`, default 500),
embedded, and stored as a `TextNode` with `tree_level` metadata. The tree is stored in the same
pgvector table as regular chunks, distinguished by a `tree_level >= 1` metadata field.

### Query-Time Retrieval

At query time, `get_raptor_summaries()` embeds the user query and searches the vector store for
the most semantically similar summary nodes across all tree levels. Only nodes with a relevance
score above `min_raptor_score` (default 0.3) are returned. Up to `raptor_top_k` (default 5)
summaries are retrieved and added to the context ahead of the detailed chunk evidence.

This gives the LLM a high-level framing of the codebase before it reads the specific retrieved
chunks, which significantly improves answers to broad architectural questions.

### Background Worker Lifecycle

RAPTOR tree building is CPU- and LLM-intensive. It runs as a background worker
(`RAPTORWorker`) that polls for pending build jobs every 15 seconds. Jobs are queued
automatically after a codebase upload completes. The worker is disabled when
`DISABLE_BACKGROUND_WORKERS=true`, which is required in multi-worker deployments (Gunicorn)
because asyncio does not fork safely. In that mode, RAPTOR trees can be built externally or
on a dedicated worker process.

---

## ASG-Aware Expansion

After initial retrieval produces a ranked list of chunks, the `ASGExpander` enriches that list
using the Abstract Semantic Graph stored in the database.

### What It Does

Each retrieved chunk carries `unit_id` metadata linking it to a `CodeUnit` row in the database.
The expander queries the `CodeEdge` table to find graph neighbors of each retrieved unit: callers
(functions that call this function), callees (functions this function calls), and dependency
relationships (imports, type dependencies).

These neighbor nodes are fetched from the project's cached node list and added to the result set
with a decayed relevance score:

```
neighbor_score = parent_score * score_decay
```

The default `score_decay` is 0.7, meaning a direct neighbor gets 70% of the score of the chunk
that surfaced it. The merged list is sorted by score descending and trimmed to `max_expansion`
(default 12 results total).

### Intent-Adjusted Expansion

Expansion depth and decay are adjusted based on the detected intent:

| Intent           | Depth | Decay | Max Results | Purpose                          |
|------------------|-------|-------|-------------|----------------------------------|
| Default (SEARCH) | 1     | 0.7   | 12          | Immediate neighbors only         |
| FLOW             | 2     | 0.5   | 20          | Trace execution chains           |
| DATA_LIFECYCLE   | 2     | 0.6   | 12          | Follow data through layers       |
| Impact (special) | 3     | 0.4   | 24          | Full reverse dependency traversal|

### Why This Matters

Keyword and vector retrieval finds chunks based on textual similarity to the query. But code
understanding often requires context that is not textually similar — a caller function that is
essential to understanding the behavior of a retrieved callee, or a class that is not mentioned
by name but is a direct dependency. Graph expansion captures these structural relationships that
pure text search misses.

---

## Context Assembly

After retrieval and expansion, `build_context_with_history()` assembles the LLM prompt from three
layers, in this order:

```
## CONVERSATION HISTORY
[Last N turns from DB, user and assistant alternating]

## HIGH-LEVEL CONTEXT (Document Summaries)
[Up to 3 RAPTOR summary nodes, ordered by relevance]

## DETAILED EVIDENCE (Relevant Passages)
[Up to 6 retrieved chunks, each tagged with source filename]
[Source: src/auth/login.py]
<chunk text>
```

The system prompt instructs the LLM to cite file paths with line numbers, show function
signatures, format code blocks, and explain how different pieces of code connect. The full context
string — history, summaries, and chunks — is passed as a single user prompt to the LLM in one
completion call. No retrieval occurs inside the LLM call.

Token budget is managed by bounding the number of RAPTOR summaries (`max_summaries=3`),
the number of chunks (`max_chunks=6`), and the number of history turns (`max_history=10`).

---

## SSE Streaming

The streaming endpoint (`stateless_query_streaming`) yields three event types to the browser:

| Event type | Payload                                      | Timing                        |
|------------|----------------------------------------------|-------------------------------|
| `sources`  | `{"sources": [{filename, snippet, score}]}`  | Before LLM call starts        |
| `content`  | `{"content": "<chunk of text>"}`             | Once per LLM output token     |
| `done`     | `{}`                                         | After full response saved     |

The `sources` event fires first so the UI can display source citations immediately, before the
first token of the answer appears. The `content` events stream each text chunk as the LLM produces
it. The `done` event signals that the conversation turn has been saved to the database and the
connection can be closed.

### Why SSE Over WebSockets

SSE is unidirectional: the server pushes events, the client listens. Code chat follows exactly
this pattern — the user sends one request, the server streams the response. WebSockets add
bidirectional connection management overhead that provides no benefit here. SSE is supported
natively by browsers, works over standard HTTP/2, and requires no special connection lifecycle
handling on the client side.

---

## Node Cache

The node cache is the central performance optimization for the stateless query path. Loading all
vector nodes for a project from PostgreSQL on every request adds 100-300ms of database overhead.
The cache avoids this for the common case.

```
_node_cache: Dict[project_id, (nodes, timestamp, node_count)]
```

Key properties:

- **TTL**: 5 minutes. After expiry, nodes are reloaded from the database on the next request.
- **Thread safety**: All reads and writes are guarded by a `threading.Lock()`, making it safe
  under concurrent multi-user access.
- **Invalidation**: `pipeline.invalidate_node_cache(project_id)` must be called after any upload
  or deletion that changes the node set for a project. The code ingestion pipeline calls this
  automatically after completing ingestion.
- **Staleness check**: On cache hit, the code checks both the TTL and the stored node count.
  If the node count in the database has changed since the cache was populated, the entry is
  considered stale even within the TTL window.

---

## Configuration

Key settings in `config/codeloom.yaml` that govern query engine behavior:

| Setting                              | Default          | Description                                    |
|--------------------------------------|------------------|------------------------------------------------|
| `retrieval.similarity_top_k`         | 20               | Candidates retrieved by each retriever         |
| `retrieval.retriever_weights`        | `[0.5, 0.5]`     | BM25 and vector weights for fusion             |
| `retrieval.num_queries`              | 3                | Sub-queries generated for query expansion      |
| `retrieval.reranker.model`           | `base`           | Reranker model size (xsmall / base / large)    |
| `retrieval.reranker.top_k`           | 10               | Final results passed to LLM after reranking    |
| `retrieval.chat_v2.raptor_top_k`     | 5                | RAPTOR summaries retrieved per query           |
| `retrieval.chat_v2.min_raptor_score` | 0.3              | Minimum relevance score for RAPTOR inclusion   |
| `DISABLE_RERANKER`                   | (env var, false) | Disable cross-encoder reranking globally       |
| `DISABLE_BACKGROUND_WORKERS`        | (env var, false) | Disable RAPTOR worker (required for Gunicorn)  |

RAPTOR tree-building parameters live under the `raptor:` section of the config, including
`max_tree_depth` (default 4), `min_nodes_to_cluster` (default 5), and clustering parameters.

---

## Design Decisions

### Why Hybrid Retrieval

BM25 and vector retrieval have complementary failure modes. BM25 fails when the query uses
different terminology than the source (synonyms, paraphrases, conceptual descriptions). Vector
retrieval fails on rare or highly specific identifiers that are underrepresented in the embedding
space. Combining both — with configurable weights — covers both failure modes. The default equal
weighting is a reasonable starting point; teams working with heavily abbreviated codebases may
benefit from shifting weight toward BM25.

### Why Cross-Encoder Reranking

Bi-encoder retrieval (the vector retrieval stage) computes query and document embeddings
independently and compares them by dot product. This is fast but loses the direct query-document
interaction that matters for precision. The cross-encoder reads both together, capturing nuanced
relevance signals. Running it only on the top-k fusion candidates (rather than the full index)
keeps cost bounded. The quality improvement is significant for code questions where near-duplicate
chunks may receive very different scores under cross-encoder evaluation.

### Why RAPTOR

Code queries frequently operate at multiple levels of granularity simultaneously. A question like
"how does the authentication flow work?" requires both high-level flow understanding (across many
files) and specific implementation details (particular functions). Neither chunks alone nor a
single document summary addresses both. RAPTOR's tree structure makes both levels available and
lets the retriever select the appropriate level based on query intent.

### Why SSE Over WebSockets

The code chat interaction is inherently request-response: the user sends a query, the server
streams an answer. This maps directly to SSE's unidirectional push model. WebSockets solve a
different problem — two-way real-time communication — and add client-side connection state
management (reconnection, heartbeats, protocol upgrade) with no benefit for this use case.

---

## Key File Locations

| Purpose                | Path                                          |
|------------------------|-----------------------------------------------|
| Pipeline orchestrator  | `codeloom/pipeline.py`                        |
| Stateless query module | `codeloom/core/stateless/`                    |
| Retrieval engine       | `codeloom/core/engine/retriever.py`           |
| RAPTOR tree builder    | `codeloom/core/raptor/tree_builder.py`        |
| RAPTOR background worker | `codeloom/core/raptor/worker.py`            |
| ASG expander           | `codeloom/core/asg_builder/expander.py`       |
| Context assembly       | `codeloom/core/stateless/context.py`          |
| Chat API route         | `codeloom/api/routes/code_chat.py`            |
| Configuration          | `config/codeloom.yaml`                        |

---

## Cross-References

- `docs/architecture/01-platform-overview.md` — system-wide architecture, request flow, and
  startup sequence
- `docs/architecture/02-ingestion-pipeline.md` — how code is parsed, chunked, embedded, and
  stored before queries can be served
