"""Prompt templates for the Deep Understanding Engine.

Three main templates:
1. chain_analysis_prompt - Per-entry-point analysis with framework hints
2. branch_summary_prompt - Summarize deep branches for Tier 3
3. cross_cutting_prompt - Aggregate concern detection across entry points
"""

from typing import Any, Dict, List, Optional

from .models import AnalysisTier, EntryPoint


def build_chain_analysis_prompt(
    entry_point: EntryPoint,
    source_payload: str,
    tier: AnalysisTier,
    framework_contexts: List[Dict[str, Any]],
) -> str:
    """Build the main chain analysis prompt.

    Args:
        entry_point: The entry point being analyzed
        source_payload: Formatted source code (full, truncated, or summarized)
        tier: Analysis tier for tier-aware instructions
        framework_contexts: Detected framework contexts with analysis hints
    """
    framework_section = ""
    if framework_contexts:
        hints = []
        for ctx in framework_contexts:
            hints.append(f"Framework: {ctx.get('framework_name', 'Unknown')}")
            for hint in ctx.get("analysis_hints", []):
                hints.append(f"  - {hint}")
            if ctx.get("transaction_boundaries"):
                hints.append(f"  Transaction boundaries: {', '.join(ctx['transaction_boundaries'][:5])}")
            if ctx.get("security_config"):
                hints.append(f"  Security config: {list(ctx['security_config'].keys())}")
        framework_section = "\n## FRAMEWORK CONTEXT\n" + "\n".join(hints)

    tier_notice = ""
    if tier == AnalysisTier.TIER_2:
        tier_notice = (
            "\nNOTE: Source code has been depth-prioritized. Shallow code (depth 0-2) "
            "is shown in full. Deeper code may show signatures only. Focus your analysis "
            "on the available source and note where deeper analysis would improve confidence."
        )
    elif tier == AnalysisTier.TIER_3:
        tier_notice = (
            "\nNOTE: Source code has been partially summarized due to size. Shallow code "
            "(depth 0-1) is shown in full. Deeper branches are summarized. Adjust your "
            "confidence scores accordingly and note where full source would help."
        )

    return f"""You are a senior software architect performing deep functional analysis of a codebase entry point.

## ENTRY POINT
- Name: {entry_point.qualified_name}
- Type: {entry_point.entry_type.value}
- File: {entry_point.file_path}
- Language: {entry_point.language}
- Detection: {entry_point.detected_by}
{framework_section}
{tier_notice}

## CALL CHAIN SOURCE CODE
The following source code shows the complete (or partial) call chain starting from the entry point above.
Each unit is annotated with its file path, line numbers, and call depth.

{source_payload}

## YOUR TASK
Analyze this call chain and extract structured functional understanding.
Return your analysis as a JSON object with exactly this schema:

```json
{{
  "business_rules": [
    {{
      "description": "Clear description of the business rule",
      "severity": "critical|important|minor",
      "evidence": [
        {{
          "unit_id": "UUID of the code unit where this rule is implemented",
          "qualified_name": "fully.qualified.name",
          "file_path": "relative/file/path.ext",
          "start_line": 42,
          "end_line": 48,
          "snippet": "relevant code excerpt (max 10 lines)"
        }}
      ]
    }}
  ],
  "data_entities": [
    {{
      "name": "Entity name (e.g., Order, User)",
      "operations": ["create", "read", "update", "delete"],
      "evidence": [{{ "unit_id": "...", "qualified_name": "...", "file_path": "...", "start_line": 0, "end_line": 0, "snippet": "..." }}]
    }}
  ],
  "integrations": [
    {{
      "type": "database|api|message_queue|file_system|email|cache",
      "description": "What this integration does",
      "evidence": [{{ "unit_id": "...", "qualified_name": "...", "file_path": "...", "start_line": 0, "end_line": 0, "snippet": "..." }}]
    }}
  ],
  "side_effects": [
    {{
      "type": "db_write|email|audit_log|cache_invalidation|notification|file_write",
      "description": "What side effect occurs",
      "evidence": [{{ "unit_id": "...", "qualified_name": "...", "file_path": "...", "start_line": 0, "end_line": 0, "snippet": "..." }}]
    }}
  ],
  "cross_cutting_concerns": ["authentication", "logging", "error_handling", ...],
  "narrative": "A 2-4 paragraph human-readable summary of what this entry point does, suitable for inclusion in a chat response. Write as if explaining to a developer who hasn't read the code.",
  "confidence": 0.85,
  "coverage": 0.90,
  "chain_truncated": false
}}
```

## RULES
1. Every business_rule, data_entity, integration, and side_effect MUST include at least one evidence reference with a valid unit_id.
2. The narrative should be informative but concise (200-500 words).
3. Confidence (0-1): How confident are you in the extracted facts? Lower if source was truncated.
4. Coverage (0-1): What fraction of the call chain's behavior did you analyze? Lower if deep branches were summarized.
5. Only report facts you can point to in the source code. Do not speculate.
6. Return ONLY the JSON object, no markdown fences or additional text.
"""


def build_branch_summary_prompt(branch_name: str, source: str) -> str:
    """Build a prompt to summarize a deep branch for Tier 3 analysis."""
    return f"""Summarize the following code branch in 2-3 sentences.
Focus on: what it does, what data it touches, and any side effects.

Branch: {branch_name}

```
{source}
```

Summary:"""


def build_cross_cutting_prompt(
    entry_point_summaries: List[Dict[str, Any]],
) -> str:
    """Build a prompt for cross-cutting concern detection across entry points."""
    summaries_text = "\n\n".join(
        f"### {s.get('qualified_name', 'Unknown')}\n"
        f"Type: {s.get('entry_type', 'unknown')}\n"
        f"Cross-cutting: {', '.join(s.get('cross_cutting_concerns', []))}\n"
        f"Narrative: {s.get('narrative', 'N/A')[:200]}"
        for s in entry_point_summaries
    )

    return f"""You are analyzing cross-cutting concerns across multiple entry points in a codebase.

## ENTRY POINT SUMMARIES
{summaries_text}

## YOUR TASK
Identify cross-cutting concerns that appear across multiple entry points.
Return a JSON object:

```json
{{
  "shared_concerns": [
    {{
      "concern": "Name of the cross-cutting concern",
      "description": "What it does and how it manifests",
      "affected_entry_points": ["qualified.name.1", "qualified.name.2"],
      "pattern": "middleware|decorator|aspect|base_class|utility"
    }}
  ],
  "summary": "Brief overview of the cross-cutting architecture"
}}
```

Return ONLY the JSON object.
"""
