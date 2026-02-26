"""Tool definitions for the migration agent loop.

Each tool wraps an existing CodeLoom service -- context_builder queries,
stateless RAG retrieval, Context7 framework docs, and tree-sitter validation.

Tool factories receive a bound ``MigrationContextBuilder`` (and optionally
a pipeline reference) and return ``ToolDefinition`` instances with closures
that capture the service references.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ToolDefinition dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
    """A single tool the migration agent can invoke.

    Attributes:
        name: Unique tool identifier (used in function-calling schemas).
        description: Human-readable purpose shown to the LLM.
        parameters: JSON Schema dict describing the tool's arguments.
        execute: Callable(args_dict) -> str  that runs the tool.
        category: Grouping label for UI display (e.g. "code", "search", "docs").
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[[Dict[str, Any]], str]
    category: str = "general"

    def to_openai_schema(self) -> Dict[str, Any]:
        """Serialize to the OpenAI function-calling tool format.

        Works with OpenAI, Anthropic, and Gemini via LlamaIndex gateway.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Tool #1: read_source_file
# ---------------------------------------------------------------------------

def _make_read_source_file(ctx) -> ToolDefinition:
    """Read source code for all units in a given file path."""

    def execute(args: Dict[str, Any]) -> str:
        file_path = args.get("file_path", "")
        if not file_path:
            return "Error: file_path is required."

        with ctx._db.get_session() as session:
            rows = session.execute(text("""
                SELECT cu.qualified_name, cu.unit_type, cu.source
                FROM code_units cu
                JOIN code_files cf ON cu.file_id = cf.file_id
                WHERE cf.project_id = :pid
                  AND cf.file_path = :path
                  AND cu.source IS NOT NULL
                ORDER BY cu.start_line
            """), {"pid": ctx._pid, "path": file_path}).fetchall()

        if not rows:
            return f"No source found for file '{file_path}'."

        parts = []
        for r in rows:
            parts.append(f"// {r.qualified_name} ({r.unit_type})")
            parts.append(r.source)
            parts.append("")
        return "\n".join(parts)

    return ToolDefinition(
        name="read_source_file",
        description=(
            "Read the full source code of a file from the ingested codebase. "
            "Returns all code units in the file ordered by line number."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative file path within the project (e.g. 'src/main/java/com/example/UserService.java').",
                },
            },
            "required": ["file_path"],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #2: get_unit_details
# ---------------------------------------------------------------------------

def _make_get_unit_details(ctx, unit_ids: List[str]) -> ToolDefinition:
    """Get unit signatures, params, return types, modifiers."""

    def execute(args: Dict[str, Any]) -> str:
        limit = args.get("limit", 40)
        rows = ctx._get_mvp_units_enriched(unit_ids, limit=limit)
        if not rows:
            return "No unit details found for the current MVP."

        lines = [f"## Unit Details ({len(rows)} units)"]
        for u in rows:
            meta = u.get("metadata") or {}
            lines.append(f"\n### {u['qualified_name']} ({u['unit_type']})")
            lines.append(f"File: {u['file_path']} | Language: {u['language']}")
            if u.get("signature"):
                lines.append(f"Signature: `{u['signature']}`")
            params = meta.get("parsed_params", [])
            if params:
                pstrs = []
                for p in params:
                    ps = p.get("name", "?")
                    if p.get("type"):
                        ps += f": {p['type']}"
                    pstrs.append(ps)
                lines.append(f"Parameters: ({', '.join(pstrs)})")
            if meta.get("return_type"):
                lines.append(f"Returns: `{meta['return_type']}`")
            if meta.get("modifiers"):
                lines.append(f"Modifiers: {', '.join(meta['modifiers'])}")
            lines.append(f"Connectivity: {u.get('connectivity', 0)}")
        return "\n".join(lines)

    return ToolDefinition(
        name="get_unit_details",
        description=(
            "Get enriched metadata for units in the current MVP: signatures, "
            "parameters with types, return types, modifiers, and connectivity scores."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of units to return (default 40).",
                },
            },
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #3: get_source_code
# ---------------------------------------------------------------------------

def _make_get_source_code(ctx, unit_ids: List[str]) -> ToolDefinition:
    """Full source code for MVP units ordered by connectivity."""

    def execute(args: Dict[str, Any]) -> str:
        budget = min(args.get("token_budget", 12000), 20000)
        results = ctx._get_mvp_source_code_by_connectivity(unit_ids, budget=budget)
        if not results:
            return "No source code found for the current MVP."
        return ctx._format_source_code_annotated(results)

    return ToolDefinition(
        name="get_source_code",
        description=(
            "Retrieve the full source code of units in the current MVP, "
            "ordered by architectural significance (connectivity). "
            "Use token_budget to control how much code is returned."
        ),
        parameters={
            "type": "object",
            "properties": {
                "token_budget": {
                    "type": "integer",
                    "description": "Approximate token budget for source code (default 12000, max 20000).",
                },
            },
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #4: get_functional_context
# ---------------------------------------------------------------------------

def _make_get_functional_context(ctx, unit_ids: List[str]) -> ToolDefinition:
    """Business domain: entities, rules, integrations, validations."""

    def execute(args: Dict[str, Any]) -> str:
        functional = ctx.get_mvp_functional_context(unit_ids)
        if not functional:
            return "No functional context extracted for the current MVP."
        return ctx.format_mvp_functional_context(functional)

    return ToolDefinition(
        name="get_functional_context",
        description=(
            "Extract business domain context for the current MVP: data entities, "
            "service-layer business methods, external integrations, and validation rules."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #5: get_dependencies
# ---------------------------------------------------------------------------

def _make_get_dependencies(ctx, unit_ids: List[str]) -> ToolDefinition:
    """Edges crossing the MVP boundary (blast radius)."""

    def execute(args: Dict[str, Any]) -> str:
        limit = args.get("limit", 50)
        edges = ctx._get_mvp_cross_edges(unit_ids, limit=limit)
        if not edges:
            return "No cross-boundary edges found for the current MVP."

        internal = [e for e in edges if e["direction"] == "internal"]
        outbound = [e for e in edges if e["direction"] == "outbound"]
        inbound = [e for e in edges if e["direction"] == "inbound"]

        lines = [f"## MVP Dependencies ({len(edges)} edges)"]
        for label, group in [("Internal", internal), ("Outbound (MVP calls external)", outbound),
                             ("Inbound (external calls MVP)", inbound)]:
            if group:
                lines.append(f"\n### {label} ({len(group)})")
                for e in group:
                    lines.append(f"- {e['source']} --[{e['edge_type']}]--> {e['target']}")
        return "\n".join(lines)[:8000]

    return ToolDefinition(
        name="get_dependencies",
        description=(
            "Get edges crossing the MVP boundary -- internal edges, outbound calls "
            "(MVP depends on external code), and inbound calls (external code depends on MVP). "
            "Essential for understanding blast radius of migration changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of edges to return (default 50).",
                },
            },
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #6: get_module_graph
# ---------------------------------------------------------------------------

def _make_get_module_graph(ctx) -> ToolDefinition:
    """File-level import/dependency graph."""

    def execute(args: Dict[str, Any]) -> str:
        graph = ctx._get_module_dependency_graph()
        if not graph:
            return "No file-level dependency graph available."

        lines = ["## Module Dependency Graph"]
        lines.append("| Source File | Target File | Edge Type | Count |")
        lines.append("|------------|-------------|-----------|-------|")
        for g in graph:
            lines.append(
                f"| {g['source_file']} | {g['target_file']} "
                f"| {g['edge_type']} | {g['edge_count']} |"
            )
        return "\n".join(lines)[:8000]

    return ToolDefinition(
        name="get_module_graph",
        description=(
            "Get the file-level dependency graph showing which files import or "
            "depend on which other files. Useful for understanding module structure."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #7: get_deep_analysis
# ---------------------------------------------------------------------------

def _make_get_deep_analysis(ctx, unit_ids: List[str]) -> ToolDefinition:
    """Deep understanding narratives + business rules."""

    def execute(args: Dict[str, Any]) -> str:
        max_narratives = args.get("max_narratives", 5)
        result = ctx.get_deep_analysis_context(unit_ids, max_narratives=max_narratives)
        if not result:
            return "No deep analysis results found for the current MVP units."
        return result[:10000]

    return ToolDefinition(
        name="get_deep_analysis",
        description=(
            "Retrieve deep understanding analysis overlapping with the current MVP. "
            "Returns AI-generated narratives explaining how code modules work together, "
            "business rules, and integration patterns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "max_narratives": {
                    "type": "integer",
                    "description": "Maximum number of analysis narratives to include (default 5).",
                },
            },
            "required": [],
        },
        execute=execute,
        category="code",
    )


# ---------------------------------------------------------------------------
# Tool #8: search_codebase
# ---------------------------------------------------------------------------

def _make_search_codebase(pipeline, project_id: str) -> ToolDefinition:
    """RAG semantic search across all project code."""

    def execute(args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        top_k = args.get("top_k", 6)
        if not query:
            return "Error: query is required."

        try:
            from ...stateless.retrieval import fast_retrieve

            nodes = pipeline._get_cached_nodes(project_id)
            if not nodes:
                return "No indexed code available for this project."

            results = fast_retrieve(
                nodes=nodes,
                query=query,
                project_id=project_id,
                vector_store=pipeline._vector_store,
                retriever_factory=pipeline._retriever,
                llm=pipeline.llm,
                top_k=top_k,
            )

            if not results:
                return f"No results found for query: '{query}'"

            lines = [f"## Search Results for '{query}' ({len(results)} hits)"]
            for i, r in enumerate(results, 1):
                meta = r.node.metadata or {}
                source = meta.get("file_name", meta.get("source", "unknown"))
                score = f"{r.score:.3f}" if r.score else "N/A"
                lines.append(f"\n### Result {i} (score: {score})")
                lines.append(f"Source: {source}")
                lines.append(f"```\n{r.node.text[:2000]}\n```")
            return "\n".join(lines)[:10000]
        except Exception as e:
            logger.warning("search_codebase tool error: %s", e)
            return f"Search failed: {e}"

    return ToolDefinition(
        name="search_codebase",
        description=(
            "Semantic search across the entire project codebase using RAG. "
            "Finds code chunks relevant to a natural language query. "
            "Useful for finding patterns, usages, or implementations across the project."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what you're looking for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 6).",
                },
            },
            "required": ["query"],
        },
        execute=execute,
        category="search",
    )


# ---------------------------------------------------------------------------
# Tool #9: lookup_framework_docs  (Context7 primary, Tavily fallback)
# ---------------------------------------------------------------------------

# Per-run cache to avoid duplicate lookups within the same agent execution.
_docs_cache: Dict[str, str] = {}


class _Context7DocTool:
    """Framework documentation tool: Context7 primary, Tavily fallback.

    Context7 provides curated, version-specific library documentation.
    Tavily serves as web search fallback when Context7 is unavailable or
    the library isn't in Context7's catalog.
    """

    C7_SEARCH_URL = "https://context7.com/api/v2/libs/search"
    C7_DOCS_URL = "https://context7.com/api/v2/context"

    def __init__(self, target_stack: Optional[Dict[str, Any]] = None):
        self._c7_key = os.environ.get("CONTEXT7_API_KEY", "")
        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")
        self._cache: Dict[str, str] = {}
        self._target_stack = target_stack or {}

    def _infer_framework(self) -> str:
        """Infer the primary target framework from the migration plan's target_stack."""
        # Try common keys in target_stack
        for key in ("framework", "name", "backend", "frontend", "runtime"):
            val = self._target_stack.get(key)
            if val and isinstance(val, str):
                return val
        # Fallback: join all string values as a search hint
        parts = [v for v in self._target_stack.values() if isinstance(v, str) and v]
        return parts[0] if parts else ""

    def execute(self, args: Dict[str, Any]) -> str:
        framework = args.get("framework", "")
        topic = args.get("topic", "")
        if not framework:
            framework = self._infer_framework()
        if not framework:
            return "Error: framework is required and could not be inferred from the migration plan."

        cache_key = f"{framework}::{topic}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = ""

        # Try Context7 first
        if self._c7_key:
            try:
                result = self._fetch_context7(framework, topic)
            except Exception as e:
                logger.warning("Context7 lookup failed: %s", e)

        # Fallback to Tavily
        if not result and self._tavily_key:
            try:
                result = self._fetch_tavily(framework, topic)
            except Exception as e:
                logger.warning("Tavily fallback failed: %s", e)

        if not result:
            result = (
                f"No documentation found for '{framework}'"
                + (f" on topic '{topic}'" if topic else "")
                + ". Check that CONTEXT7_API_KEY or TAVILY_API_KEY is set."
            )

        self._cache[cache_key] = result
        return result[:12000]

    def _fetch_context7(self, framework: str, topic: str) -> str:
        """Resolve library ID then fetch targeted docs from Context7."""
        from urllib.parse import quote as _urlquote

        headers = {"Authorization": f"Bearer {self._c7_key}"}

        # Step 1: Resolve library ID
        with httpx.Client(timeout=15) as client:
            search_resp = client.get(
                self.C7_SEARCH_URL,
                params={"libraryName": framework},
                headers=headers,
            )
            search_resp.raise_for_status()
            raw = search_resp.text.strip()
            if not raw:
                logger.debug("Context7 search returned empty body for '%s'", framework)
                return ""
            # Fix: guard against non-JSON responses (e.g. HTML error pages)
            try:
                search_data = search_resp.json()
            except Exception:
                logger.debug("Context7 search returned non-JSON for '%s'", framework)
                return ""

        results = search_data.get("results", [])
        if not results:
            return ""

        # Fix: skip /websites/ entries — they are rejected by the docs endpoint
        lib_id = ""
        first_result = None
        for r in results:
            rid = r.get("id", "")
            if rid and not rid.startswith("/websites/"):
                lib_id = rid
                first_result = r
                break
        if not lib_id:
            # Fallback: take first result even if it's a website entry
            lib_id = results[0].get("id", "")
            first_result = results[0]
        if not lib_id:
            return ""

        # Step 2: Fetch docs for topic
        # Fix: build URL manually so the slash in lib_id is NOT percent-encoded.
        # httpx.get(params={"libraryId": lib_id}) encodes "/" as "%2F" which
        # causes a 400 Bad Request from the Context7 docs endpoint.
        # Fix: add tokens=5000 so the API returns a useful amount of content.
        docs_url = f"{self.C7_DOCS_URL}?libraryId={lib_id}&tokens=5000"
        if topic:
            docs_url += f"&query={_urlquote(topic, safe='')}"

        with httpx.Client(timeout=20) as client:
            docs_resp = client.get(docs_url, headers=headers)
            docs_resp.raise_for_status()
            content = docs_resp.text.strip()
            if not content:
                logger.debug("Context7 docs returned empty body for '%s'", lib_id)
                return ""

        lib_name = (first_result or {}).get("title") or (first_result or {}).get("name") or framework
        header = f"## {lib_name} Documentation"
        if topic:
            header += f" — {topic}"
        return f"{header}\n\n{content}"

    def _fetch_tavily(self, framework: str, topic: str) -> str:
        """Fallback: web search via Tavily API."""
        query = f"{framework} documentation"
        if topic:
            query += f" {topic}"

        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self._tavily_key}"},
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 3,
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        parts = []
        answer = data.get("answer")
        if answer:
            parts.append(f"## {framework} — {topic or 'Overview'}\n\n{answer}")

        for r in data.get("results", [])[:3]:
            title = r.get("title", "")
            content = r.get("content", "")
            if content:
                parts.append(f"### {title}\n{content[:1500]}")

        return "\n\n".join(parts) if parts else ""


def _make_lookup_framework_docs(target_stack: Optional[Dict[str, Any]] = None) -> ToolDefinition:
    """Build the framework docs lookup tool."""
    tool_impl = _Context7DocTool(target_stack=target_stack)

    return ToolDefinition(
        name="lookup_framework_docs",
        description=(
            "Look up official documentation for a target framework or library. "
            "Use this to find correct API patterns, annotations, configuration syntax, "
            "and migration guides for the target technology stack. "
            "Examples: framework='Spring Boot', topic='dependency injection'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "framework": {
                    "type": "string",
                    "description": "Framework or library name (e.g. 'Spring Boot', 'ASP.NET Core', 'Django').",
                },
                "topic": {
                    "type": "string",
                    "description": "Specific topic to focus on (e.g. 'dependency injection', 'JPA repositories').",
                },
            },
            "required": ["framework"],
        },
        execute=tool_impl.execute,
        category="docs",
    )


# ---------------------------------------------------------------------------
# Tool #10: validate_syntax
# ---------------------------------------------------------------------------

def _make_validate_syntax() -> ToolDefinition:
    """Check if generated code parses without errors using tree-sitter."""

    def execute(args: Dict[str, Any]) -> str:
        code = args.get("code", "")
        language = args.get("language", "").lower()
        if not code:
            return "Error: code is required."
        if not language:
            return "Error: language is required."

        try:
            parser_mod = _get_tree_sitter_parser(language)
            if parser_mod is None:
                return f"No tree-sitter parser available for '{language}'. Skipping validation."

            import tree_sitter
            parser = tree_sitter.Parser(parser_mod)
            tree = parser.parse(code.encode("utf-8"))

            errors = []
            _collect_errors(tree.root_node, errors, max_errors=10)

            if not errors:
                return f"Syntax OK: code parses as valid {language} with no errors."

            lines = [f"Syntax errors found ({len(errors)}):"]
            for err in errors:
                lines.append(
                    f"  Line {err['line']+1}, Col {err['col']}: "
                    f"{err['type']} — '{err['text'][:80]}'"
                )
            return "\n".join(lines)
        except ImportError:
            return "tree-sitter is not installed. Syntax validation unavailable."
        except Exception as e:
            return f"Validation error: {e}"

    return ToolDefinition(
        name="validate_syntax",
        description=(
            "Validate that generated code has correct syntax using tree-sitter parsing. "
            "Returns either 'Syntax OK' or a list of parse errors with line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to validate.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (java, python, javascript, typescript, csharp).",
                },
            },
            "required": ["code", "language"],
        },
        execute=execute,
        category="validation",
    )


def _get_tree_sitter_parser(language: str):
    """Resolve a tree-sitter Language object for the given language name."""
    lang_map = {
        "java": "tree_sitter_java",
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "csharp": "tree_sitter_c_sharp",
        "c_sharp": "tree_sitter_c_sharp",
    }
    mod_name = lang_map.get(language)
    if not mod_name:
        return None
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        ts_lang = mod.language()
        import tree_sitter
        return tree_sitter.Language(ts_lang)
    except (ImportError, AttributeError):
        return None


def _collect_errors(node, errors: list, max_errors: int = 10):
    """Walk tree-sitter AST and collect ERROR / MISSING nodes."""
    if len(errors) >= max_errors:
        return
    if node.type == "ERROR" or node.is_missing:
        errors.append({
            "type": node.type,
            "line": node.start_point[0],
            "col": node.start_point[1],
            "text": node.text.decode("utf-8", errors="replace") if node.text else "",
        })
    for child in node.children:
        _collect_errors(child, errors, max_errors)


# ---------------------------------------------------------------------------
# Phase-specific tool assembly
# ---------------------------------------------------------------------------

# Which tools each phase context_type gets access to.
_PHASE_TOOL_MAP: Dict[str, List[str]] = {
    "architecture": [
        "read_source_file",       # read each file for exact method signatures
        "get_module_graph",       # enumerate all source files
        "get_functional_context", # understand business rules and data flows
        "search_codebase",        # find patterns across the project
        "lookup_framework_docs",  # target framework patterns (Context7)
        # No validate_syntax — spec produces markdown, not code
    ],
    "design": [
        "read_source_file",
        "get_source_code",
        "get_functional_context",
        "get_dependencies",
        "get_module_graph",
        "get_deep_analysis",
        "search_codebase",
        "lookup_framework_docs",  # Context7: look up target framework patterns
        # No validate_syntax — design produces a text spec, not code
    ],
    "transform": [
        "read_source_file",
        "get_unit_details",
        "get_source_code",
        "get_functional_context",
        "get_dependencies",
        "get_module_graph",
        "get_deep_analysis",
        "search_codebase",
        "lookup_framework_docs",
        "validate_syntax",
    ],
    "analyze": [
        "read_source_file",
        "get_unit_details",
        "get_source_code",
        "get_functional_context",
        "get_dependencies",
        "get_module_graph",
        "get_deep_analysis",
        "search_codebase",
    ],
    "test": [
        "read_source_file",
        "get_source_code",
        "get_functional_context",
        "get_dependencies",
        "search_codebase",
        "lookup_framework_docs",
        "validate_syntax",
    ],
    "discovery": [
        "get_unit_details",
        "get_module_graph",
        "search_codebase",
    ],
}


def build_tools_for_phase(
    context_type: str,
    ctx,
    unit_ids: Optional[List[str]] = None,
    pipeline=None,
    project_id: Optional[str] = None,
    target_stack: Optional[Dict[str, Any]] = None,
) -> List[ToolDefinition]:
    """Assemble the tool set for a given phase type.

    Args:
        context_type: Phase context type ("transform", "analyze", "test", "discovery").
        ctx: MigrationContextBuilder bound to the current project.
        unit_ids: MVP unit IDs for scoped tools (None for project-wide phases).
        pipeline: LocalRAGPipeline for search_codebase tool.
        project_id: Project UUID string for search_codebase tool.
        target_stack: Migration plan's target_stack dict for framework inference.

    Returns:
        List of ToolDefinition instances the agent can use.
    """
    allowed = _PHASE_TOOL_MAP.get(context_type, _PHASE_TOOL_MAP["transform"])
    uids = unit_ids or []

    # Build all available tools
    all_tools: Dict[str, ToolDefinition] = {}

    all_tools["read_source_file"] = _make_read_source_file(ctx)
    all_tools["get_module_graph"] = _make_get_module_graph(ctx)
    all_tools["lookup_framework_docs"] = _make_lookup_framework_docs(target_stack=target_stack)
    all_tools["validate_syntax"] = _make_validate_syntax()

    if uids:
        all_tools["get_unit_details"] = _make_get_unit_details(ctx, uids)
        all_tools["get_source_code"] = _make_get_source_code(ctx, uids)
        all_tools["get_functional_context"] = _make_get_functional_context(ctx, uids)
        all_tools["get_dependencies"] = _make_get_dependencies(ctx, uids)
        all_tools["get_deep_analysis"] = _make_get_deep_analysis(ctx, uids)

    if pipeline and project_id:
        all_tools["search_codebase"] = _make_search_codebase(pipeline, project_id)

    # Filter to allowed set for this phase
    return [all_tools[name] for name in allowed if name in all_tools]
