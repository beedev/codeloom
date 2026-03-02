# CodeLoom MCP Server — Tool and Integration Reference

## MCP Tools (codeloom MCP server)

The CodeLoom MCP server exposes 11 tools covering project intelligence, MVP management, phase execution, source code access, RAG-based semantic search, ground truth validation, and migration lane detection. All tools communicate via JSON-encoded responses over the MCP stdio transport.

| Tool Name | Description | Required Params | Optional Params |
|-----------|-------------|-----------------|-----------------|
| `codeloom_list_projects` | List all CodeLoom projects. Returns project metadata including ID, name, creation date, and file count. | — | `page` (integer, default 1), `page_size` (integer, default 20, max 100) |
| `codeloom_get_project_intel` | Get comprehensive intelligence for a project: AST stats, ASG edge counts by type, understanding engine analysis status, and existing migration plan summaries. | `project_id` (string, UUID) | — |
| `codeloom_list_mvps` | List all MVPs for a migration plan with status, confidence score, cluster name, and per-phase status summary. | `plan_id` (string, UUID) | — |
| `codeloom_get_mvp_context` | Get rich context for an MVP: source unit summaries (up to 50), detected lane info, deep analysis narratives, and ground truth layer summary. | `plan_id` (string, UUID), `mvp_id` (integer) | — |
| `codeloom_execute_phase` | Trigger execution of a migration phase. Agentic runs execute in a background thread and persist results to the database; use `codeloom_get_phase_result` to poll for completion. | `plan_id` (string, UUID), `phase_number` (integer) | `mvp_id` (integer, required for per-MVP phases), `use_agent` (boolean, default true), `max_turns` (integer, default 10) |
| `codeloom_get_phase_result` | Poll the current status and output of a migration phase from the database. Returns status, truncated output, gate results, ground truth warnings, confidence tier, and execution metrics. | `plan_id` (string, UUID), `phase_number` (integer) | `mvp_id` (integer, for per-MVP phases) |
| `codeloom_approve_mvp` | Approve or reject a completed MVP phase. Writes the approval decision and optional feedback text to the phase record. | `plan_id` (string, UUID), `phase_number` (integer), `mvp_id` (integer), `approved` (boolean) | `feedback` (string) |
| `codeloom_get_source_unit` | Get full source code and metadata for a code unit by its UUID, including signature, docstring, qualified name, language, file path, and raw metadata. | `unit_id` (string, UUID) | — |
| `codeloom_search_codebase` | Semantic search over an ingested codebase using RAG retrieval. Requires the RAG pipeline to be initialized at server startup. Returns ranked text snippets with scores and metadata. | `project_id` (string, UUID), `query` (string) | `top_k` (integer, default 5, max 20) |
| `codeloom_validate_output` | Run ground truth advisory validation on a generated output text for a given phase type. Returns a list of issues with type, severity, and message. Does not block execution — advisory only. | `project_id` (string, UUID), `phase_type` (string: `discovery`, `architecture`, `analyze`, `design`, `transform`, `test`), `output_text` (string) | — |
| `codeloom_get_lane_info` | Detect the migration lane for a source framework and target stack pair. Returns lane details, transform rules (with patterns, templates, and per-rule confidence), and quality gates (with category and blocking flag). | `source_framework` (string, e.g. `struts`, `stored_proc`, `vbnet`), `target_stack_json` (string, JSON object e.g. `{"framework": "spring_boot"}`) | — |

### Notes on Phase Execution

- When `use_agent` is `true` (the default), `codeloom_execute_phase` returns immediately with `status: started` and runs the agentic loop in a background daemon thread. The engine persists results to the database on each turn cycle.
- If the same `(plan_id, phase_number, mvp_id)` combination is already running, the tool returns `status: already_running` without spawning a second thread.
- When `use_agent` is `false`, execution is synchronous and the tool blocks until the phase completes, returning `status: complete` with the full result.
- Plan-level phases (phases 1 and 2 in V2: Architecture and Discovery) do not require `mvp_id`. Per-MVP phases (Transform, Test) require it.

---

## Setup

### Registration

The MCP server is registered as a Claude Code integration via:

```bash
./dev.sh setup-mcp
```

This writes the server configuration into `~/.claude/claude_desktop_config.json` (or the equivalent Claude Code config) so that the `codeloom` server is available in Claude Code sessions.

### Manual Start

To start the MCP server directly without Claude Code:

```bash
source venv/bin/activate && python -m codeloom.mcp
```

All logging is directed to stderr. stdout is reserved exclusively for the MCP stdio transport channel. The server initializes a `DatabaseManager` and optionally a `LocalRAGPipeline` (for `codeloom_search_codebase`) on startup.

### Claude Code CLI Usage

Once registered, the `/migrate` skill in Claude Code invokes the MCP server tools to drive the migration workflow interactively.

---

## Environment Variables

The following environment variables are read by the MCP server entry point (`codeloom/mcp/__main__.py`) at startup.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | `postgresql://codeloom:codeloom@localhost:5432/codeloom_dev` | PostgreSQL connection string. Used by `DatabaseManager` for all DB queries. The MCP server connects directly to the database — the CodeLoom FastAPI server does not need to be running. |
| `LLM_PROVIDER` | No | `ollama` | LLM provider used by `LocalRAGPipeline` when semantic search is enabled (`codeloom_search_codebase`). Accepted values: `ollama`, `openai`, `anthropic`, `gemini`, `groq`. |
| `OLLAMA_HOST` | No | `localhost` | Hostname for the Ollama server. Passed to `LocalRAGPipeline` as the `host` argument. Only relevant when `LLM_PROVIDER=ollama`. |

Additional provider-specific variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`) are required when using the corresponding LLM provider. These are read by the LlamaIndex provider integrations rather than the MCP entry point directly.

---

## New Dependencies Added

| Package | Version Constraint | Purpose |
|---------|--------------------|---------|
| `mcp` | `>=1.0.0` | Model Context Protocol SDK. Provides `mcp.server.Server`, `mcp.server.stdio.stdio_server`, and `mcp.types.Tool` / `TextContent` used by the MCP server implementation. |

---

## External APIs and Services Integrated

| Service | Required at Runtime | Notes |
|---------|--------------------|-|
| PostgreSQL database | Yes | All 11 tools query the database directly via `DatabaseManager`. The `DATABASE_URL` environment variable must point to a running PostgreSQL instance with the CodeLoom schema applied (`alembic upgrade head`). |
| CodeLoom FastAPI server | No | The MCP server bypasses the HTTP API entirely and communicates with the database and migration engine directly. The FastAPI server (`./dev.sh local`) does not need to be running for MCP tools to function. |
| LLM provider | No (conditional) | Required only for `codeloom_search_codebase`. If the `LocalRAGPipeline` fails to initialize at startup (e.g. missing API key or unreachable Ollama host), all other tools continue to function; `codeloom_search_codebase` returns an error indicating the pipeline is unavailable. |
