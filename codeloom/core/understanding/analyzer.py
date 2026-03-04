"""Tiered LLM analysis of traced call chains.

Token budget algorithm:
  Tier 1: Full source fits in ≤100K tokens → send everything
  Tier 2: 100K < total ≤ 200K → depth-prioritized truncation
  Tier 3: total > 200K → summarize deep branches, send summaries + shallow source

Reuses TokenCounter from core/code_chunker/token_counter.py.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llama_index.core import Settings

from ..code_chunker.token_counter import TokenCounter
from .models import (
    AnalysisTier, CallTreeNode, DeepContextBundle,
    EntryPoint, EvidenceRef,
)
from . import prompts

logger = logging.getLogger(__name__)

# ── JCL/COBOL cross-reference parsing (for context annotations) ──────
# Operate on CallTreeNode.source (JCL step text includes EXEC + DD lines)

_JCL_DD_LINE_RE = re.compile(
    r"^//([A-Z0-9@#$]{1,8})\s+DD\s+(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_JCL_DSN_RE = re.compile(
    r"\b(?:DSNAME|DSN)=(&&?[A-Z0-9@#$][A-Z0-9@#$.]{0,42}|[A-Z0-9@#$][A-Z0-9@#$.]{0,43})",
    re.IGNORECASE,
)
_JCL_DISP_RE = re.compile(r"\bDISP=(\([^)]+\)|\w+)", re.IGNORECASE)
_JCL_PARM_RE = re.compile(
    r"\bPARM=(?:'([^']*)'|(\([^)]*\))|([^,\s/]+))",
    re.IGNORECASE,
)
_COBOL_SELECT_RE = re.compile(
    r"\bSELECT\s+([\w-]+)\s+ASSIGN\s+(?:TO\s+)?(?:[\w]+-[A-Z]-)?([A-Z0-9@#$][\w-]*)",
    re.IGNORECASE,
)

# Tier thresholds (tokens) — defaults; overridden by config if present
TIER_1_MAX = 100_000
TIER_2_MAX = 200_000

# Quality gate defaults
DEFAULT_REQUIRE_EVIDENCE_REFS = True
DEFAULT_MIN_NARRATIVE_LENGTH = 100


class ChainAnalyzer:
    """Analyze a traced call chain to extract structured understanding.

    Args:
        token_counter: Optional TokenCounter instance (created if None)
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._tc = token_counter or TokenCounter()

    def analyze_chain(
        self,
        entry_point: EntryPoint,
        call_tree: CallTreeNode,
        framework_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> DeepContextBundle:
        """Run tiered analysis on a call chain.

        Steps:
        1. Count total tokens across all units in the tree
        2. Select tier based on total tokens
        3. Prepare source payload (full, truncated, or summarized)
        4. Build prompt with framework hints
        5. Call LLM and parse structured JSON output
        6. Validate evidence references

        Args:
            entry_point: The entry point being analyzed
            call_tree: Traced call tree from ChainTracer
            framework_contexts: Optional framework analysis results

        Returns:
            DeepContextBundle with extracted knowledge
        """
        # Step 1: Count tokens
        total_tokens = self._count_tree_tokens(call_tree)

        # Step 2: Select tier
        tier = self._select_tier(total_tokens)
        logger.info(
            f"Analyzing {entry_point.qualified_name}: "
            f"{total_tokens} tokens → {tier.value}"
        )

        # Step 3: Prepare source payload
        source_payload = self._prepare_source(call_tree, tier, total_tokens)

        # Step 4: Build prompt
        prompt = prompts.build_chain_analysis_prompt(
            entry_point=entry_point,
            source_payload=source_payload,
            tier=tier,
            framework_contexts=framework_contexts or [],
        )

        # Step 5: Call LLM
        llm = Settings.llm
        if llm is None:
            raise RuntimeError("No LLM configured")

        response = llm.complete(prompt)
        raw_output = response.text.strip()

        # Step 6: Parse and validate
        parsed = self._parse_json_output(raw_output)
        bundle = self._build_bundle(
            entry_point=entry_point,
            tier=tier,
            total_tokens=total_tokens,
            call_tree=call_tree,
            parsed=parsed,
        )

        return bundle

    # ── Token Counting ──────────────────────────────────────────────────

    def _count_tree_tokens(self, node: CallTreeNode) -> int:
        """Recursively count tokens across all source in the tree."""
        count = 0
        if node.source:
            node.token_count = self._tc.count(node.source)
            count += node.token_count
        for child in node.children:
            count += self._count_tree_tokens(child)
        return count

    def _select_tier(self, total_tokens: int) -> AnalysisTier:
        """Select analysis tier based on total source tokens."""
        if total_tokens <= TIER_1_MAX:
            return AnalysisTier.TIER_1
        elif total_tokens <= TIER_2_MAX:
            return AnalysisTier.TIER_2
        else:
            return AnalysisTier.TIER_3

    # ── Source Preparation ──────────────────────────────────────────────

    def _prepare_source(
        self,
        tree: CallTreeNode,
        tier: AnalysisTier,
        total_tokens: int,
    ) -> str:
        """Prepare the source code payload for the LLM prompt.

        Tier 1: Full source from all units, formatted with file paths
        Tier 2: Depth-prioritized — full source for depth 0-2,
                 signatures only for depth 3+, fill remaining budget
                 with highest-connectivity deeper units
        Tier 3: Summarize branches at depth 3+ via separate LLM call,
                 include depth 0-1 full source + summaries
        """
        if tier == AnalysisTier.TIER_1:
            return self._format_full_source(tree)
        elif tier == AnalysisTier.TIER_2:
            return self._format_depth_prioritized(tree, budget=TIER_2_MAX)
        else:
            return self._format_with_summaries(tree, budget=TIER_2_MAX)

    def _format_full_source(
        self,
        node: CallTreeNode,
        indent: int = 0,
        parent_node: Optional[CallTreeNode] = None,
    ) -> str:
        """Format full source tree with file path headers."""
        parts = []
        prefix = "  " * indent

        header = f"{prefix}## {node.qualified_name}"
        header += f" [{node.file_path}:{node.start_line}-{node.end_line}]"
        header += f" (depth={node.depth}, type={node.unit_type})"
        parts.append(header)

        if node.source:
            annotation = self._build_jcl_context_annotation(node, parent_node)
            if annotation:
                parts.append(annotation)
            parts.append(f"{prefix}```{node.language}")
            parts.append(node.source)
            parts.append(f"{prefix}```")

        for child in node.children:
            parts.append(self._format_full_source(child, indent + 1, parent_node=node))

        return "\n".join(parts)

    def _format_depth_prioritized(
        self,
        tree: CallTreeNode,
        budget: int,
    ) -> str:
        """Depth-prioritized truncation for Tier 2.

        Algorithm:
        1. Include full source for depth 0-2
        2. For depth 3+, include only signatures
        3. If budget remains, fill with highest-connectivity deep units
        """
        parts = []
        remaining_budget = budget
        deep_candidates = []

        def _walk(
            node: CallTreeNode,
            parent_node: Optional[CallTreeNode] = None,
        ):
            nonlocal remaining_budget

            if node.depth <= 2 and node.source:
                source_tokens = node.token_count or self._tc.count(node.source)
                if source_tokens <= remaining_budget:
                    parts.append(self._format_unit_full(node, parent_node))
                    remaining_budget -= source_tokens
                else:
                    parts.append(self._format_unit_signature(node))
                    remaining_budget -= 50  # Estimate for signature
            elif node.depth > 2:
                deep_candidates.append(node)
                parts.append(self._format_unit_signature(node))
                remaining_budget -= 50
            else:
                parts.append(self._format_unit_signature(node))
                remaining_budget -= 50

            for child in node.children:
                _walk(child, parent_node=node)

        _walk(tree)

        # Fill remaining budget with deep units by connectivity
        deep_candidates.sort(
            key=lambda n: len(n.children), reverse=True
        )
        for node in deep_candidates:
            if remaining_budget <= 0:
                break
            if node.source and node.token_count <= remaining_budget:
                parts.append(f"\n### [Deep unit - full source] {node.qualified_name}")
                parts.append(f"```{node.language}\n{node.source}\n```")
                remaining_budget -= node.token_count

        return "\n".join(parts)

    def _format_with_summaries(
        self,
        tree: CallTreeNode,
        budget: int,
    ) -> str:
        """Tier 3: Summarize deep branches, include shallow full source.

        Algorithm:
        1. Full source for depth 0-1
        2. Group depth 2+ branches by their depth-1 parent
        3. Summarize each branch group via LLM
        4. Include summaries in payload
        """
        parts = []
        branch_sources = {}

        def _collect_branches(node: CallTreeNode, branch_key: str = "root"):
            if node.depth <= 1:
                if node.source:
                    parts.append(self._format_unit_full(node))
                # Each depth-1 child starts a new branch
                for child in node.children:
                    _collect_branches(child, branch_key=child.qualified_name)
            else:
                if branch_key not in branch_sources:
                    branch_sources[branch_key] = []
                if node.source:
                    branch_sources[branch_key].append(
                        f"// {node.qualified_name} [{node.file_path}:{node.start_line}]\n{node.source}"
                    )
                for child in node.children:
                    _collect_branches(child, branch_key)

        _collect_branches(tree)

        # Summarize each branch
        for branch_name, sources in branch_sources.items():
            if not sources:
                continue
            combined = "\n\n".join(sources)
            summary = self._summarize_branch(branch_name, combined)
            parts.append(f"\n### [Branch summary] {branch_name}\n{summary}")

        return "\n".join(parts)

    def _summarize_branch(self, branch_name: str, source: str) -> str:
        """Summarize a deep branch via LLM call."""
        prompt = prompts.build_branch_summary_prompt(branch_name, source)
        llm = Settings.llm
        if not llm:
            return f"[Summary unavailable for {branch_name}]"
        response = llm.complete(prompt)
        return response.text.strip()

    # ── Output Parsing ──────────────────────────────────────────────────

    def _parse_json_output(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences."""
        # Strip markdown code fences
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Attempting repair.")
            # Try to find the outermost { }
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"parse_error": str(e), "raw_output": raw[:2000]}

    def _build_bundle(
        self,
        entry_point: EntryPoint,
        tier: AnalysisTier,
        total_tokens: int,
        call_tree: CallTreeNode,
        parsed: Dict[str, Any],
    ) -> DeepContextBundle:
        """Build DeepContextBundle from parsed LLM output.

        Applies quality gates:
        - require_evidence_refs: warns if business_rules lack evidence references
        - min_narrative_length: warns if narrative is too short
        """
        from datetime import datetime

        total_units = len(self._flatten_units(call_tree))

        # Load quality gate config
        require_evidence = DEFAULT_REQUIRE_EVIDENCE_REFS
        min_narrative_len = DEFAULT_MIN_NARRATIVE_LENGTH
        try:
            from ..config.config_loader import get_config_value
            require_evidence = get_config_value(
                "codeloom", "migration", "deep_analysis", "require_evidence_refs",
                default=DEFAULT_REQUIRE_EVIDENCE_REFS,
            )
            min_narrative_len = get_config_value(
                "codeloom", "migration", "deep_analysis", "min_narrative_length",
                default=DEFAULT_MIN_NARRATIVE_LENGTH,
            )
        except Exception:
            pass  # Use defaults if config is unavailable

        # Quality gate: evidence references
        business_rules = parsed.get("business_rules", [])
        if require_evidence and business_rules:
            rules_without_evidence = []
            for rule in business_rules:
                refs = rule.get("evidence_refs") or rule.get("evidence") or []
                if not refs:
                    rules_without_evidence.append(
                        rule.get("id") or rule.get("name") or "unknown"
                    )
            if rules_without_evidence:
                logger.warning(
                    "Quality gate: %d/%d business rules lack evidence refs for %s: %s",
                    len(rules_without_evidence),
                    len(business_rules),
                    entry_point.qualified_name,
                    rules_without_evidence[:5],
                )

        # Quality gate: narrative length
        narrative = parsed.get("narrative", "")
        if min_narrative_len and len(narrative) < min_narrative_len:
            logger.warning(
                "Quality gate: narrative too short for %s (%d chars < %d minimum)",
                entry_point.qualified_name,
                len(narrative),
                min_narrative_len,
            )

        return DeepContextBundle(
            entry_point=entry_point,
            tier=tier,
            total_units=total_units,
            total_tokens=total_tokens,
            business_rules=business_rules,
            data_entities=parsed.get("data_entities", []),
            integrations=parsed.get("integrations", []),
            side_effects=parsed.get("side_effects", []),
            cross_cutting_concerns=parsed.get("cross_cutting_concerns", []),
            narrative=narrative,
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            coverage=float(parsed.get("coverage", 0.0) or 0.0),
            chain_truncated=bool(parsed.get("chain_truncated", tier != AnalysisTier.TIER_1)),
            schema_version=1,
            prompt_version="v1.0",
            analyzed_at=datetime.utcnow().isoformat(),
        )

    def _flatten_units(self, node: CallTreeNode) -> List[str]:
        """Collect all unit_ids in the tree."""
        ids = [node.unit_id]
        for child in node.children:
            ids.extend(self._flatten_units(child))
        return ids

    # ── Formatting Helpers ──────────────────────────────────────────────

    def _format_unit_full(
        self,
        node: CallTreeNode,
        parent_node: Optional[CallTreeNode] = None,
    ) -> str:
        header = f"## {node.qualified_name} [{node.file_path}:{node.start_line}-{node.end_line}]"
        annotation = self._build_jcl_context_annotation(node, parent_node)
        if annotation:
            return f"{header}\n{annotation}```{node.language}\n{node.source}\n```"
        return f"{header}\n```{node.language}\n{node.source}\n```"

    def _format_unit_signature(self, node: CallTreeNode) -> str:
        sig_line = node.source.split("\n")[0] if node.source else node.name
        return f"- `{node.qualified_name}` ({node.unit_type}, depth={node.depth}): `{sig_line}`"

    # ── JCL/COBOL Context Annotation ────────────────────────────────────

    def _build_jcl_context_annotation(
        self,
        node: CallTreeNode,
        parent_node: Optional[CallTreeNode],
    ) -> str:
        """Build a JCL context annotation table for a COBOL program node.

        Inserted between the unit header and its code fence when a COBOL
        program is invoked by a JCL step. The table cross-references:
          - COBOL internal file names (from SELECT...ASSIGN clauses)
          - JCL DDNAMEs
          - Actual dataset names (DSN= values) and disposition (read/write)

        Returns empty string if context is unavailable or not applicable.
        """
        if node.unit_type != "program" or node.language != "cobol":
            return ""
        if (
            parent_node is None
            or parent_node.unit_type != "step"
            or parent_node.language != "jcl"
        ):
            return ""

        dd_allocs = self._extract_jcl_dd_allocs(parent_node.source or "")
        if not dd_allocs:
            return ""

        # Reverse map: ddname_upper → cobol_name from SELECT...ASSIGN
        file_assigns = self._extract_cobol_file_assigns(node.source or "")
        ddname_to_cobol = {v: k for k, v in file_assigns.items()}

        lines = [
            f"> **JCL Context** (allocated by `{parent_node.name}`):",
            "> | COBOL File Name | DDNAME | Dataset (DSN) | Access |",
            "> |----------------|--------|---------------|--------|",
        ]
        for ddname in sorted(dd_allocs):
            alloc = dd_allocs[ddname]
            cobol_name = ddname_to_cobol.get(ddname, "—")
            dsn = alloc.get("dsn") or "—"
            disp = alloc.get("disp") or "—"
            lines.append(f"> | {cobol_name} | {ddname} | {dsn} | {disp} |")

        parm = self._extract_jcl_parm(parent_node.source or "")
        if parm:
            lines.append(">")
            lines.append(f"> **PARM**: `{parm}`")

        return "\n".join(lines) + "\n\n"

    def _extract_jcl_dd_allocs(
        self, source: str
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Extract {DDNAME: {dsn, disp}} from a JCL step source text."""
        allocs: Dict[str, Dict[str, Optional[str]]] = {}
        for m in _JCL_DD_LINE_RE.finditer(source):
            ddname = m.group(1).strip().upper()
            operands = m.group(2) or ""
            dsn_m = _JCL_DSN_RE.search(operands)
            disp_m = _JCL_DISP_RE.search(operands)
            dsn_val: Optional[str] = (
                dsn_m.group(1).rstrip(".").lstrip("&") if dsn_m else None
            )
            disp_val: Optional[str] = (
                disp_m.group(1).strip("()").strip() if disp_m else None
            )
            if ddname:
                allocs[ddname] = {"dsn": dsn_val, "disp": disp_val}
        return allocs

    def _extract_cobol_file_assigns(self, source: str) -> Dict[str, str]:
        """Extract {cobol_name_upper: ddname_upper} from COBOL SELECT...ASSIGN."""
        assigns: Dict[str, str] = {}
        for m in _COBOL_SELECT_RE.finditer(source):
            cobol_name = m.group(1).strip().upper()
            ddname = m.group(2).strip().upper()
            if cobol_name not in assigns:
                assigns[cobol_name] = ddname
        return assigns

    def _extract_jcl_parm(self, source: str) -> Optional[str]:
        """Extract PARM= value from JCL source text."""
        m = _JCL_PARM_RE.search(source)
        if not m:
            return None
        return (m.group(1) or m.group(2) or m.group(3) or "").strip() or None
