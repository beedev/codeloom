"""LLM prompt templates for the MVP-centric migration pipeline.

Plan-level prompts:
  Phase 1: Discovery — codebase analysis + strategy recommendation
  Phase 2: Architecture — system-wide target design

Per-MVP prompts:
  Phase 3: Analyze — scoped deep analysis
  Phase 4: Design — detailed module design (+ SP stubs when SPs exist)
  Phase 5: Transform — code generation (+ SP stubs when SPs exist)
  Phase 6: Test — scoped test generation
"""

from typing import Any, Dict, List, Optional

MIGRATION_ROLE = """You are an expert software architect specializing in code migration and modernization.
You have deep expertise in analyzing codebases, designing target architectures, and planning
safe, incremental migration strategies. You use evidence from the codebase's actual structure
(AST analysis, dependency graphs, call relationships) to make informed recommendations.

You work with Functional MVPs — vertical slices of related code units that can be migrated
independently. Each MVP represents a cohesive functional area (e.g., user authentication,
order processing) with measured coupling and cohesion metrics."""


def _has_sp_refs(mvp_context: Optional[Dict[str, Any]]) -> bool:
    """Check whether an MVP context has any stored procedure references."""
    if not mvp_context:
        return False
    return len(mvp_context.get("sp_references", []) or []) > 0


def _format_mvp_section(mvp_context: Optional[Dict[str, Any]]) -> str:
    """Format MVP context into a prompt section."""
    if not mvp_context:
        return ""

    sp_refs = mvp_context.get("sp_references", [])
    sp_section = ""
    if sp_refs:
        sp_lines = []
        for ref in sp_refs:
            sites = ref.get("call_sites", [])
            sp_lines.append(f"  - **{ref.get('sp_name', '?')}** ({len(sites)} call site(s))")
        sp_section = "\n**Stored Procedure References**:\n" + "\n".join(sp_lines)

    metrics = mvp_context.get("metrics", {})
    metrics_str = ""
    if metrics:
        metrics_str = (
            f"\n**Metrics**: Cohesion={metrics.get('cohesion', '?')}, "
            f"Coupling={metrics.get('coupling', '?')}, "
            f"Size={metrics.get('size', '?')} units, "
            f"Readiness={metrics.get('readiness', '?')}"
        )

    return f"""
### Current MVP: {mvp_context.get('name', 'Unnamed')}
- **Status**: {mvp_context.get('status', 'unknown')}
- **Priority**: {mvp_context.get('priority', '?')}
- **Units**: {len(mvp_context.get('unit_ids', []))} code units across {len(mvp_context.get('file_ids', []))} files{metrics_str}{sp_section}
"""


# ── Agentic MVP Refinement Prompts ──────────────────────────────────


def mvp_functional_description(
    source_code: str,
    unit_signatures: str,
    functional_context: str,
    cluster_metrics: Dict,
) -> str:
    """Build prompt to generate a functional description for one MVP cluster."""
    metrics_str = ""
    if cluster_metrics:
        metrics_str = (
            f"- Cohesion: {cluster_metrics.get('cohesion', '?')}\n"
            f"- Coupling: {cluster_metrics.get('coupling', '?')}\n"
            f"- Size: {cluster_metrics.get('size', '?')} units\n"
            f"- Internal edges: {cluster_metrics.get('internal_edges', '?')}\n"
            f"- External edges: {cluster_metrics.get('external_edges', '?')}"
        )

    return f"""{MIGRATION_ROLE}

## Task: Functional Description for MVP Cluster

Analyze this MVP cluster and produce a functional description that captures its
business purpose, not its structural location.

### Cluster Metrics
{metrics_str}

### Unit Signatures
{unit_signatures}

### Source Code (Excerpts)
{source_code}

### Functional Context (Auto-Extracted)
{functional_context}

### Instructions
1. What is the business purpose of this code? (1-2 sentences)
2. List key behaviors (bullet list of what this code DOES)
3. Identify data ownership (entities, tables, fields managed by this code)
4. List external dependencies (APIs, services, shared modules this code calls)
5. Suggest a descriptive functional name (e.g., "User Authentication", "Order Processing")
   — NOT a package name like "auth" or "controllers"

### Output Format
Return ONLY a JSON object (no markdown fences):
{{"name": "Descriptive Functional Name", "description": "1-2 sentence business purpose", "business_purpose": "What problem this code solves", "key_behaviors": ["behavior1", "behavior2"], "data_ownership": ["Entity1", "Entity2"], "external_dependencies": ["ServiceA", "ModuleB"]}}
"""


def mvp_coherence_evaluation(
    mvp_summaries: List[Dict],
    inter_mvp_edges: List[Dict],
) -> str:
    """Build prompt to evaluate all MVPs for functional coherence and suggest merges/splits."""
    summaries_str = ""
    for s in mvp_summaries:
        summaries_str += (
            f"\n- **MVP {s.get('idx', '?')}**: {s.get('name', 'Unknown')}\n"
            f"  Description: {s.get('description', 'No description')}\n"
            f"  Units: {s.get('unit_count', '?')} | "
            f"Cohesion: {s.get('metrics', {}).get('cohesion', '?')} | "
            f"Coupling: {s.get('metrics', {}).get('coupling', '?')}\n"
        )

    edges_str = ""
    if inter_mvp_edges:
        edges_str = "\n### Inter-MVP Edge Counts\n"
        for e in inter_mvp_edges[:30]:
            edges_str += (
                f"- MVP {e['source_mvp']} → MVP {e['target_mvp']}: "
                f"{e['edge_count']} edges ({e.get('edge_types', {})})\n"
            )
    else:
        edges_str = "\n### Inter-MVP Edge Counts\nNo cross-MVP edges detected.\n"

    return f"""{MIGRATION_ROLE}

## Task: MVP Coherence Evaluation

Evaluate these MVP clusters for functional coherence and suggest merges or splits
to improve the migration plan quality.

### MVP Summaries
{summaries_str}
{edges_str}

### Instructions
1. Is each MVP a coherent functional unit? Does it represent a single business capability?
2. Look for **merge signals**:
   - Overlapping data ownership (two MVPs managing the same entities)
   - Utility MVPs that are too small to migrate independently
   - High coupling between two MVPs (many cross-edges)
3. Look for **split signals**:
   - Multiple unrelated concerns in one MVP (e.g., auth + reporting)
   - Mixed layers (API controllers + database entities in one MVP)
   - An MVP that is too large to reason about (>50 units with low cohesion)
4. **Conservative bias**: suggest NO changes if unsure. Empty suggestions are fine.
   Only suggest merges/splits with clear functional justification.

### Output Format
Return ONLY a JSON object (no markdown fences):
{{"merge_suggestions": [{{"ids": [0, 2], "reason": "Both manage order data with 15 cross-edges"}}], "split_suggestions": [{{"id": 1, "reason": "Contains both auth and billing concerns", "split_hint": "Separate auth units from billing units"}}], "assessment": "Brief overall assessment of MVP coherence quality"}}

If no changes are needed, return:
{{"merge_suggestions": [], "split_suggestions": [], "assessment": "MVPs are well-bounded functional units"}}
"""


# ── Phase 1: Discovery ──────────────────────────────────────────────

def phase_1_discovery(
    target_brief: str,
    target_stack: dict,
    constraints: dict,
    codebase_context: str,
    has_sps: bool = False,
    architecture_output: str = "",
    mvp_summaries: str = "",
) -> str:
    """Build prompt for Discovery Analysis.

    Args:
        architecture_output: In V2 pipeline, Architecture runs first and its
            output is available here for context-informed discovery.
        mvp_summaries: Formatted MVP functional summaries from agentic refinement.
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    langs_str = ", ".join(target_stack.get("languages", []))
    constraint_lines = []
    if constraints:
        if constraints.get("timeline"):
            constraint_lines.append(f"- Timeline: {constraints['timeline']}")
        if constraints.get("team_size"):
            constraint_lines.append(f"- Team size: {constraints['team_size']}")
        if constraints.get("risk_tolerance"):
            constraint_lines.append(f"- Risk tolerance: {constraints['risk_tolerance']}")
    constraints_str = "\n".join(constraint_lines) if constraint_lines else "No specific constraints."

    # Conditional SP instructions and output sections
    sp_instruction = ""
    sp_output_section = ""
    if has_sps:
        sp_instruction = "\n3. Identify stored procedure dependencies that complicate migration."
        sp_output_section = """
## Stored Procedure Impact
[Which SPs are critical, which are orphaned, decoupling strategy]
"""

    # V2: architecture output is available for informed discovery
    architecture_section = ""
    if architecture_output:
        architecture_section = f"""
### Architecture Output (Approved)
{architecture_output}

Use the architecture blueprint above to inform your MVP assessment and migration ordering.
"""

    # MVP functional summaries from agentic refinement
    mvp_summaries_section = ""
    if mvp_summaries:
        mvp_summaries_section = f"""
### MVP Functional Summaries
{mvp_summaries}
"""

    return f"""{MIGRATION_ROLE}

## Task: Discovery Analysis

Analyze the source codebase structure and recommend a migration strategy.
The system has already identified Functional MVPs (vertical slices) using ASG-based clustering.
Your job is to validate the clustering, assess the overall migration complexity, and recommend strategy.

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}

### Constraints
{constraints_str}

### Source Codebase Analysis
{codebase_context}
{architecture_section}{mvp_summaries_section}
### Instructions
1. Recommend ONE migration strategy:
   - **Strangler Fig**: Build new alongside old, migrate MVP by MVP
   - **Incremental Refactor**: Gradually modify existing code
   - **Full Rewrite**: Start fresh with the target architecture

2. Assess the MVP clustering quality — are the identified MVPs well-bounded?{sp_instruction}
3. Rank the top risks and mitigation strategies.

### Output Format (Markdown)

## Strategy Recommendation
[Strategy with justification from codebase metrics]

## MVP Assessment
[Validate whether the auto-detected MVPs are well-bounded, suggest merges/splits]
{sp_output_section}
## Risk Assessment
| Risk | Severity | Mitigation |
|------|----------|------------|
| ... | High/Medium/Low | ... |

## Migration Order Recommendation
[Suggested order for migrating MVPs, with rationale]

## Timeline Estimate
[Based on codebase size, complexity, and constraints]
"""


# ── Phase 2: Architecture ────────────────────────────────────────────

def phase_2_architecture(
    target_brief: str,
    target_stack: dict,
    phase_1_output: str,
    codebase_context: str,
    has_sps: bool = False,
    framework_docs: str = "",
    source_patterns: str = "",
    migration_type: str = "framework_migration",
    asset_strategies: dict | None = None,
    layer_summary: str = "",
) -> str:
    """Build prompt for Phase 2: Technical Migration Blueprint.

    Args:
        framework_docs: Target framework best practices from DocEnricher.
        source_patterns: Formatted source pattern analysis from context_builder.
        migration_type: One of "version_upgrade", "framework_migration", "rewrite".
        asset_strategies: Per-language migration strategies from the plan.
            Keys are language names, values are dicts with "strategy" and optional "target".
        layer_summary: Verified infrastructure layer summary from CodebaseGroundTruth.
            Constrains tech recommendations to layers that exist in the source.
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    langs_str = ", ".join(target_stack.get("languages", []))

    # Per-language strategy overrides (e.g. "TypeScript: keep_as_is")
    asset_strategy_section = ""
    if asset_strategies:
        strategy_lines = []
        for lang, spec in asset_strategies.items():
            strat = spec.get("strategy", "convert")
            target_lang = spec.get("target", "")
            if strat == "keep_as_is":
                strategy_lines.append(
                    f"- **{lang}**: KEEP AS-IS — Do NOT convert {lang} files. "
                    f"Preserve original language, update imports/references only."
                )
            elif strat == "no_change":
                strategy_lines.append(
                    f"- **{lang}**: NO CHANGE — Exclude {lang} files from migration entirely."
                )
            elif strat == "convert" and target_lang:
                strategy_lines.append(
                    f"- **{lang}**: CONVERT to {target_lang} — Rewrite {lang} source files in {target_lang}."
                )
            elif strat in ("framework_migration", "rewrite") and target_lang:
                strategy_lines.append(
                    f"- **{lang}**: {strat.upper().replace('_', ' ')} to {target_lang} — "
                    f"Migrate {lang} code to {target_lang} using target framework best practices."
                )
            elif strat == "version_upgrade":
                strategy_lines.append(
                    f"- **{lang}**: VERSION UPGRADE — Keep {lang}, upgrade framework/library versions."
                )
            else:
                strategy_lines.append(f"- **{lang}**: {strat}")

        if strategy_lines:
            asset_strategy_section = (
                "\n### Per-Language Migration Strategies (User-Configured)\n"
                "CRITICAL — these override the default source code conversion rules. "
                "Only convert languages explicitly marked for conversion:\n"
                + "\n".join(strategy_lines)
                + "\n"
            )

    # Conditional SP instructions and output sections
    sp_instruction = ""
    sp_output_section = ""
    if has_sps:
        sp_instruction = "\n6. Define the SP decoupling strategy (API stubs, data access layer)."
        sp_output_section = """
## SP Decoupling Strategy
[How stored procedures will be replaced -- API stubs, repository pattern, etc.]
"""

    # Optional enrichment sections
    source_patterns_section = ""
    if source_patterns:
        source_patterns_section = f"""
### Source Framework Patterns (Auto-Detected)
{source_patterns}
"""

    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Best Practices
{framework_docs}
"""

    # Ground truth layer summary — constrains tech recommendations to real layers
    layer_summary_section = ""
    if layer_summary:
        layer_summary_section = f"\n{layer_summary}\n"

    # In V2, architecture runs first (no prior discovery). In V1, discovery output is available.
    discovery_section = ""
    if phase_1_output:
        discovery_section = f"""
### Discovery Output (Approved)
{phase_1_output}
"""

    # ── Version Upgrade path ──────────────────────────────────────────
    if migration_type == "version_upgrade":
        return f"""{MIGRATION_ROLE}

## Task: Version Upgrade Blueprint

Analyze this codebase for a **version upgrade** — the existing frameworks and architecture
stay the same. The goal is to upgrade to the target version(s) while preserving all current
functionality and patterns.

CRITICAL: This is NOT a framework migration. Do NOT introduce new frameworks or redesign
the architecture. Focus on:
- Deprecated APIs that must be replaced
- New language/framework features to adopt
- Build system and dependency version bumps
- Runtime and bytecode compatibility
- Breaking changes in the target version

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}
{discovery_section}
### Source Codebase Structure
{codebase_context}
{source_patterns_section}{framework_docs_section}{layer_summary_section}
### Instructions
1. Identify ALL deprecated APIs used in the source code that are removed or changed
   in the target version. For each, specify the replacement API.
2. Identify new language features available in the target version that the codebase
   should adopt (e.g., records, sealed classes, pattern matching, virtual threads).
3. List dependency version bumps required for compatibility (e.g., Spring Boot 2.x → 3.x
   if upgrading Java 11 → 21, or library X requires version Y for the new runtime).
4. Identify build system changes (Maven/Gradle plugin versions, compiler flags,
   source/target compatibility settings).
5. Flag breaking changes that require code modifications (removed classes, changed method
   signatures, behavioral changes).
6. Identify runtime behavior changes (GC defaults, threading model, classloading).{sp_instruction}

### Output Format (Markdown)

## Deprecated API Replacements
| Deprecated API | Location | Replacement | Migration Effort |
|---------------|----------|-------------|-----------------|
| [old API] | [Class.method():line] | [new API] | Low/Medium/High |

## New Features to Adopt
| Feature | Benefit | Applicable Locations | Priority |
|---------|---------|---------------------|----------|
| [feature name] | [why adopt] | [where in codebase] | High/Medium/Low |

## Dependency Version Bumps
| Dependency | Current Version | Target Version | Breaking Changes |
|-----------|----------------|---------------|-----------------|
| [library] | [current] | [required] | [yes/no + details] |

## Build System Changes
| Change | Current | Target | Notes |
|--------|---------|--------|-------|
| [setting] | [old value] | [new value] | [why] |

## Breaking Changes
| Change | Impact | Files Affected | Migration Steps |
|--------|--------|---------------|----------------|
| [description] | High/Medium/Low | [file list] | [step-by-step fix] |

## Runtime Behavior Changes
| Behavior | Old Default | New Default | Action Required |
|----------|-----------|-----------|----------------|
| [behavior] | [old] | [new] | [what to do] |
{sp_output_section}
## Architecture Diagram (PlantUML)

Produce a PlantUML component diagram inside a fenced code block (```plantuml ... ```):

```plantuml
@startuml upgrade-impact
' Show which components are affected by the version upgrade.
' Use stereotypes or colors to distinguish: no-change, minor-update, breaking-change.
' Group by module/layer. Label with actual class/module names from the codebase.
@enduml
```

## Risk Register
| Risk | Source Pattern | Impact | Mitigation |"""

    # ── Framework Migration / Rewrite path (existing behavior) ────────
    return f"""{MIGRATION_ROLE}

## Task: Technical Migration Blueprint

Create a concrete technical migration blueprint mapping source framework patterns
to target BEST PRACTICES.

CRITICAL: Source patterns tell you WHAT exists, not HOW target should work.
Target ALWAYS follows framework best practices. If source has anti-patterns
(field injection, god classes, raw SQL), the target MUST correct them.

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}
{discovery_section}
### Source Codebase Structure
{codebase_context}
{source_patterns_section}{framework_docs_section}{layer_summary_section}
### File Type Conversion Rules
CRITICAL — only convert files whose source language matches a convertible type:
- **Source code files** (.py, .js, .ts, .java, .cs, .go, .rb, etc.) → Convert to target language
  UNLESS per-language strategies below specify otherwise (keep_as_is, no_change).
- **SQL files** (.sql — schemas, migrations, stored procedures, queries) → Keep as SQL.
  Map to target data access patterns (e.g., repository classes that CALL the SQL), but do NOT
  rewrite the SQL itself into Java/Python/Go classes. SQL migrations stay as SQL migrations.
- **Config files** (.yaml, .json, .xml, .properties, .toml, .env) → Map to target config format
  (e.g., application.properties → application.yml), do NOT convert to source code.
- **Build files** (pom.xml, build.gradle, package.json, Makefile, Dockerfile) → Map to target
  build system equivalent, do NOT convert to source code.
- **Documentation** (.md, .txt, .rst) → Preserve as-is, update references only.
{asset_strategy_section}
### Instructions
1. For EVERY source pattern detected, specify the target BEST PRACTICE equivalent.
   If source uses an anti-pattern, specify the correct target pattern and explain why.
2. Map each source module/class to target module/class with file paths following target conventions.
   Respect the File Type Conversion Rules above — do NOT map SQL files to Java/Python classes.
3. Map DI: source injection style to target framework's recommended DI approach.
4. Map data access: source ORM/queries to target data access best practices.
5. Map web layer: source controllers to target framework conventions.{sp_instruction}
6. Map configuration: source config to target config approach.
7. Map cross-cutting concerns: logging, transactions, error handling, security.
8. List new dependencies required and what they replace.

### Output Format (Markdown)

## Pattern Migration (Anti-Pattern Correction)
| Source Pattern | Assessment | Target Best Practice | Why This Target |
|--------------|-----------|---------------------|-----------------|
| [pattern] | OK / Anti-pattern | [framework best practice] | [from framework docs or conventions] |

## Module Structure Mapping
| Source Path | Source Language | Target Path | Target Language | Target Class | Changes |
|------------|---------------|------------|----------------|-------------|---------|
| ... | [py/java/sql/...] | [target conventions] | [target lang] | ... | [split/merge/rename/keep-as-is] |

NOTE: For SQL files, the Target Language should be "sql" and Changes should be "keep-as-is" or
"adapt schema". For config files, map to target config format. Only source code converts to target language.
{sp_output_section}
## DI Mapping
[Source injection style to target DI best practice, with code examples]

## Data Layer Mapping
[Source ORM/queries to target data access approach]

## Configuration Mapping
[Source config to target configuration approach]

## Cross-Cutting Concerns
| Concern | Source Approach | Target Best Practice |
|---------|---------------|---------------------|
| Logging | ... | ... |
| Error Handling | ... | ... |
| Transactions | ... | ... |
| Security | ... | ... |

## New Dependencies
| Package | Purpose | Replaces |
|---------|---------|----------|
| ... | ... | ... |

## Architecture Diagrams (PlantUML)

Produce TWO PlantUML diagrams inside fenced code blocks (```plantuml ... ```):

### 1. Component Diagram — target architecture layers and dependencies
```plantuml
@startuml target-components
' Use packages for layers, components for services, arrows for dependencies.
' Show: API layer, service/business layer, data layer (if applicable),
' external integrations, and cross-cutting concerns.
' Label each component with the target class/module name from Module Structure Mapping.
@enduml
```

### 2. Migration Sequence Diagram — MVP migration flow with parallel paths
```plantuml
@startuml migration-sequence
' Show the order in which MVPs are migrated.
' Use 'par' / 'end' blocks for MVPs that can be migrated in PARALLEL (no dependency between them).
' Use 'group' for sequential phases within each MVP.
' Mark external integration points with boundary actors.
' Do NOT produce a flat numbered list — show actual parallelism and dependencies.
@enduml
```

CRITICAL: The sequence diagram MUST use PlantUML parallel (`par`/`end`) and grouping
constructs to reflect the actual MVP dependency graph. Independent MVPs run in parallel;
dependent MVPs show sequential ordering. A flat list of numbered steps is NOT acceptable.

## Risk Register
| Risk | Source Pattern | Impact | Mitigation |
|------|--------------|--------|------------|
| ... | ... | ... | ... |
"""


# ── Phase 3: Analyze (Per-MVP) ──────────────────────────────────────

def phase_3_analyze(
    target_brief: str,
    phase_2_output: str,
    codebase_context: str,
    mvp_context: Optional[Dict] = None,
    functional_context: str = "",
    framework_docs: str = "",
) -> str:
    """Build prompt for Phase 3: MVP-Scoped Analysis + Functional Requirements Register.

    Args:
        functional_context: Formatted MVP functional context from context_builder.
        framework_docs: Phase-filtered framework docs from DocEnricher.
    """
    mvp_section = _format_mvp_section(mvp_context)
    has_sps = _has_sp_refs(mvp_context)

    # Conditional SP instructions and output sections
    sp_instruction = ""
    sp_output_section = ""
    if has_sps:
        sp_instruction = "\n3. Map stored procedure dependencies and propose decoupling approach."
        sp_output_section = """
## SP Dependencies
[For each SP referenced by this MVP: current usage, proposed replacement]
"""

    # Optional enrichment sections
    functional_section = ""
    if functional_context:
        functional_section = f"""
### MVP Functional Context (Auto-Extracted)
{functional_context}
"""

    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Reference
{framework_docs}
"""

    mvp_name = mvp_context.get('name', 'Unnamed') if mvp_context else 'Unknown'

    return f"""{MIGRATION_ROLE}

## Task: MVP Analysis + Functional Requirements Register (Per-MVP)

Perform a deep analysis of this specific MVP's code units, their dependencies,
and migration complexity. Produce a FUNCTIONAL REQUIREMENTS REGISTER that inventories
every business rule, data entity, integration, and validation that must be preserved.
{mvp_section}

### Target
{target_brief}

### Phase 2 Output (Approved Architecture)
{phase_2_output}

### MVP Code Analysis
{codebase_context}
{functional_section}{framework_docs_section}
### Instructions
1. Analyze each code unit in this MVP: what it does, how it connects to others.
   If source code is provided, describe the exact algorithms and data flows —
   do not guess from signatures alone.
2. Identify migration challenges specific to this MVP.{sp_instruction}
3. Assess blast radius — what breaks if this MVP is migrated incorrectly.
4. Create a FUNCTIONAL REQUIREMENTS REGISTER with unique IDs for every item:
   - BR-N: Business rules — logic that MUST be preserved exactly
   - DE-N: Data entities — classes/tables with fields and relationships
   - INT-N: External integrations — APIs, services, message queues
   - VAL-N: Validation rules — constraints, validators, business checks
5. Every code unit in this MVP must map to at least one register item.
6. Flag any business logic that is implicit (not documented) but critical.
7. For each register item, document input/output contracts and side effects.

### Output Format (Markdown)

## MVP Summary: {mvp_name}

## Unit Analysis
| Unit | Type | Complexity | Dependencies | Notes |
|------|------|-----------|--------------|-------|
| ... | ... | High/Med/Low | ... | ... |
{sp_output_section}
## Functional Requirements Register

### Business Rules
| ID | Rule | Location | Behavior | Input/Output Contract | Side Effects | Criticality |
|----|------|----------|----------|-----------------------|-------------|-------------|
| BR-1 | [rule name] | [Class.method():line] | [exact behavior] | [params → return type] | [DB writes, events, etc.] | Critical/High/Medium |

### Data Entities
| ID | Entity | Key Fields | Relationships | Constraints | Error Conditions |
|----|--------|-----------|---------------|-------------|-----------------|
| DE-1 | [entity name] | [field list] | [relationships] | [constraints] | [invalid states] |

### External Integrations
| ID | Integration | Implementation | Protocol | SLA/Contract | Error Conditions |
|----|------------|---------------|----------|-------------|-----------------|
| INT-1 | [service name] | [class/client] | [REST/gRPC/queue] | [timeout/retry] | [failure modes] |

### Validation Rules
| ID | Validator | Applied To | Rule | Error Behavior | Side Effects |
|----|-----------|-----------|------|---------------|-------------|
| VAL-1 | [validator] | [target fields/objects] | [constraint] | [error response] | [logging, alerts] |

## Register Summary
- Business Rules: N items (N critical)
- Data Entities: N items
- External Integrations: N items
- Validation Rules: N items
- Total tracked requirements: N

## Migration Challenges
[Specific risks for this MVP]

## Blast Radius
[What other MVPs or shared code would be affected]

## Recommended Approach
[Specific migration steps for this MVP]
"""


# ── Phase 4: Design (Per-MVP) ───────────────────────────────────────

def phase_4_design(
    target_brief: str,
    target_stack: dict,
    phase_3_output: str,
    codebase_context: str,
    mvp_context: Optional[Dict] = None,
    framework_docs: str = "",
    functional_context: str = "",
) -> str:
    """Build prompt for Phase 4: MVP-Scoped Design with register traceability.

    Args:
        framework_docs: Phase-filtered framework docs from DocEnricher.
        functional_context: Formatted MVP functional context from context_builder.
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    mvp_section = _format_mvp_section(mvp_context)
    has_sps = _has_sp_refs(mvp_context)

    # Conditional SP content
    if has_sps:
        task_title = "Detailed Design + SP Stubs (Per-MVP)"
        task_desc = "Create the detailed migration design for this MVP, including SP stub interfaces."
        sp_instruction = "\n4. For each referenced SP, generate an API stub interface."
        sp_output_section = """
### SP Stub Interfaces
For each stored procedure referenced by this MVP:
```
[API stub interface that replaces the SP call]
```
"""
    else:
        task_title = "Detailed Design (Per-MVP)"
        task_desc = "Create the detailed migration design for this MVP."
        sp_instruction = ""
        sp_output_section = ""

    # Optional enrichment sections
    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Conventions
{framework_docs}
Follow these conventions for interface design and type mappings.
Reference the Phase 2 Pattern Migration Map for source-to-target patterns.
"""

    functional_section = ""
    if functional_context:
        functional_section = f"""
### MVP Functional Context
{functional_context}
"""

    mvp_name = mvp_context.get('name', 'Unnamed') if mvp_context else 'Unknown'

    return f"""{MIGRATION_ROLE}

## Task: {task_title}

{task_desc}
{mvp_section}

### Target
{target_brief}

### Target Frameworks
{stack_str}

### Phase 3 Output (Approved Analysis)
{phase_3_output}

### Source Code Details
{codebase_context}
{framework_docs_section}{functional_section}
### Instructions
For each unit in this MVP:
1. Show the current interface (signatures, types).
2. Design the new interface in the target architecture.
3. Document type mappings and breaking changes.{sp_instruction}
4. For EVERY item in the Phase 3 Functional Requirements Register, provide a design mapping.
   No register item may be left unmapped. If an item cannot be mapped, explain why.

### Output Format (Markdown)

## Module: {mvp_name}

### Current Interface
```
[Current function signatures and types]
```

### Target Interface
```
[New function signatures in target architecture]
```
{sp_output_section}
### Type Mapping
| Source Type | Target Type | Notes |
|------------|-------------|-------|
| ... | ... | ... |

### Design Traceability Matrix
| Register ID | Source | Target Design | Notes |
|------------|--------|--------------|-------|
| BR-1 | [source location] | [target interface/method] | [changes] |
| DE-1 | [source entity] | [target entity] | [changes] |
| INT-1 | [source client] | [target client] | [changes] |
| VAL-1 | [source validator] | [target validator] | [changes] |

### Gap Check
- Unmapped items: [list any register items not covered, with reason]
- New items discovered: [any requirements found during design not in register]

### Breaking Changes
[Breaking changes and how to handle them]

### Migration Steps
1. [Step-by-step migration plan]
"""


# ── On-Demand MVP Analysis (V2 Pipeline) ────────────────────────────

def mvp_analysis(
    target_brief: str,
    target_stack: dict,
    architecture_output: str,
    codebase_context: str,
    mvp_context: Optional[Dict] = None,
    functional_context: str = "",
    framework_docs: str = "",
) -> str:
    """Build prompt for on-demand Deep Analysis (merges old Analyze + Design).

    Used by V2 pipeline. Result is stored on FunctionalMVP, not as a phase.
    Produces:
    1. Functional Requirements Register (BR, DE, INT, VAL)
    2. Design Traceability Matrix (interface/type mappings)
    3. Migration Steps for this MVP
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    langs_str = ", ".join(target_stack.get("languages", []))
    mvp_section = _format_mvp_section(mvp_context)
    has_sps = _has_sp_refs(mvp_context)
    mvp_name = mvp_context.get('name', 'Unnamed') if mvp_context else 'Unknown'

    sp_instruction = ""
    sp_output_section = ""
    if has_sps:
        sp_instruction = "\n5. Map stored procedure dependencies and propose decoupling approach with API stubs."
        sp_output_section = """
## SP Dependencies & Stubs
[For each SP: current usage, proposed API stub interface, decoupling approach]
"""

    functional_section = ""
    if functional_context:
        functional_section = f"""
### MVP Functional Context (Auto-Extracted)
{functional_context}
"""

    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Reference
{framework_docs}
"""

    return f"""{MIGRATION_ROLE}

## Task: Deep MVP Analysis (Analysis + Design)

Perform a comprehensive analysis of this MVP combining functional requirements discovery
and detailed migration design. This produces the complete analysis needed before code generation.
{mvp_section}

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}

### Architecture Output (Approved)
{architecture_output}

### MVP Code Analysis
{codebase_context}
{functional_section}{framework_docs_section}
### Instructions

**Part 1 — Functional Requirements Register**:
1. Analyze each code unit in this MVP: what it does, how it connects to others.
   If source code is provided, describe the exact algorithms and data flows —
   do not guess from signatures alone.
2. Create a FUNCTIONAL REQUIREMENTS REGISTER with unique IDs:
   - BR-N: Business rules — logic that MUST be preserved exactly
   - DE-N: Data entities — classes/tables with fields and relationships
   - INT-N: External integrations — APIs, services, message queues
   - VAL-N: Validation rules — constraints, validators, business checks
3. Every code unit in this MVP must map to at least one register item.
4. Flag any business logic that is implicit (not documented) but critical.
5. For each register item, document input/output contracts and side effects.{sp_instruction}

**Part 2 — Design Traceability Matrix**:
6. For each unit, show current interface (signatures, types) and target interface.
7. Map type conversions and breaking changes.
8. For EVERY register item, provide a target design mapping. None may be left unmapped.
9. Define step-by-step migration plan for this MVP.

### Output Format (Markdown)

## MVP Summary: {mvp_name}

## Unit Analysis (Grouped by Class)

Organize the unit analysis by file and class. For each file, group methods under
their parent class. This makes class-level concerns visible.

### File: [path/to/File.java]

#### ClassName (class, Layer)
| Method | Complexity | Dependencies | Notes |
|--------|-----------|-------------|-------|
| methodA | High/Med/Low | [deps] | [notes] |
| methodB | ... | ... | ... |

**Class-Level Concerns:**
- [Thread safety, error handling, transaction boundaries, etc.]

Standalone functions (not belonging to a class) get their own row at file level.
If the context provides a grouped hierarchy, follow that structure exactly.

## Functional Requirements Register

### Business Rules
| ID | Rule | Location | Behavior | Input/Output Contract | Side Effects | Criticality |
|----|------|----------|----------|-----------------------|-------------|-------------|
| BR-1 | [rule name] | [Class.method():line] | [exact behavior] | [params → return type] | [DB writes, events, etc.] | Critical/High/Medium |

### Data Entities
| ID | Entity | Key Fields | Relationships | Constraints | Error Conditions |
|----|--------|-----------|---------------|-------------|-----------------|
| DE-1 | [entity name] | [field list] | [relationships] | [constraints] | [invalid states] |

### External Integrations
| ID | Integration | Implementation | Protocol | SLA/Contract | Error Conditions |
|----|------------|---------------|----------|-------------|-----------------|
| INT-1 | [service name] | [class/client] | [REST/gRPC/queue] | [timeout/retry] | [failure modes] |

### Validation Rules
| ID | Validator | Applied To | Rule | Error Behavior | Side Effects |
|----|-----------|-----------|------|---------------|-------------|
| VAL-1 | [validator] | [target fields/objects] | [constraint] | [error response] | [logging, alerts] |

## Register Summary
- Business Rules: N items (N critical)
- Data Entities: N items
- External Integrations: N items
- Validation Rules: N items
- Total tracked requirements: N
{sp_output_section}
## Design Traceability Matrix
| Register ID | Source | Target Design | Type Mapping | Notes |
|------------|--------|--------------|-------------|-------|
| BR-1 | [source location] | [target interface/method] | [type changes] | [changes] |
| DE-1 | [source entity] | [target entity] | [field mappings] | [changes] |

## Gap Check
- Unmapped items: [list any register items not covered, with reason]
- New items discovered: [any requirements found during design not in register]

## Migration Steps
1. [Step-by-step migration plan for this MVP]

## Blast Radius
[What other MVPs or shared code would be affected]
"""


# ── Foundation MVP Analysis ────────────────────────────────────────

def mvp_foundation_analysis(
    target_brief: str,
    target_stack: dict,
    architecture_output: str,
    mvp_context: Optional[Dict] = None,
    discovery_metadata: Optional[Dict] = None,
) -> str:
    """Build prompt for Foundation MVP analysis (priority 0, no code units).

    Foundation MVPs contain prerequisite activities: build system setup,
    CI/CD, shared infrastructure, cross-cutting concerns. This prompt
    produces a prerequisites-focused deliverable — NOT a unit analysis
    or functional requirements register.
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    langs_str = ", ".join(target_stack.get("languages", []))
    mvp_name = mvp_context.get("name", "Foundation & Prerequisites") if mvp_context else "Foundation & Prerequisites"
    mvp_desc = mvp_context.get("description", "") if mvp_context else ""

    discovery_section = ""
    if discovery_metadata:
        total_units = discovery_metadata.get("total_units", "?")
        total_files = discovery_metadata.get("total_files", "?")
        languages = discovery_metadata.get("languages", [])
        discovery_section = f"""
### Source Codebase Overview
- Total code units: {total_units}
- Total files: {total_files}
- Languages detected: {', '.join(languages) if languages else 'N/A'}
"""

    return f"""{MIGRATION_ROLE}

## Task: Foundation MVP Analysis — Prerequisites & Infrastructure

This is the **Foundation MVP** (MVP 0). It contains NO source code units to migrate.
Instead, it covers all prerequisite activities that must be completed BEFORE any
code migration begins.

Do NOT produce a Unit Analysis table or Functional Requirements Register.

### MVP: {mvp_name}
{mvp_desc}

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}

### Architecture Output (Approved)
{architecture_output}
{discovery_section}
### Instructions

Analyze the migration target and architecture to produce a comprehensive
Foundation checklist. Cover every prerequisite activity needed before the
first code-bearing MVP can begin.

### Output Format (Markdown)

## MVP Summary: {mvp_name}

## 1. Prerequisites Checklist
| # | Activity | Category | Description | Acceptance Criteria | Est. Effort |
|---|----------|----------|-------------|---------------------|-------------|
| 1 | [activity] | Build/CI/Env/Deps | [what needs to happen] | [how to verify done] | S/M/L |

Categories: Build System, CI/CD Pipeline, Development Environment, Dependencies,
Repository Structure, Code Standards, Security Baseline.

## 2. Shared Infrastructure Setup
For each shared concern, specify what to set up and why:
- **Logging & Observability**: framework, format, levels
- **Configuration Management**: env vars, config files, secrets
- **Dependency Injection**: container/framework choice
- **Error Handling**: global strategy, error types, reporting
- **Monitoring & Health Checks**: endpoints, metrics

## 3. Cross-Cutting Concerns
| Concern | Current Approach | Target Approach | Migration Notes |
|---------|-----------------|-----------------|-----------------|
| Authentication | [current] | [target] | [notes] |
| Error Handling | [current] | [target] | [notes] |
| Internationalization | [current] | [target] | [notes] |
| Logging | [current] | [target] | [notes] |

## 4. Dependency Audit
| Dependency | Current Version | Target Equivalent | Status | Risk |
|-----------|----------------|-------------------|--------|------|
| [lib/framework] | [version] | [replacement] | Keep/Replace/Remove | Low/Med/High |

## 5. Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| [risk] | Low/Med/High | Low/Med/High | [strategy] |

## 6. Recommended Sequence
Ordered list of Foundation activities showing dependencies:
1. [First activity] — no dependencies
2. [Second activity] — depends on #1
...

## Foundation Summary
- Total prerequisite activities: N
- Estimated total effort: [S/M/L/XL]
- Critical path items: [list]
- Blockers for subsequent MVPs: [what must be done before MVP 1 can start]
"""


# ── Phase 5: Transform (Per-MVP) ────────────────────────────────────

def phase_5_transform(
    target_brief: str,
    target_stack: dict,
    phase_4_output: str,
    codebase_context: str,
    mvp_context: Optional[Dict] = None,
    framework_docs: str = "",
    analysis_output: str = "",
    mvp_description: str = "",
    prior_mvp_manifest: str = "",
) -> str:
    """Build prompt for Code Transform with register traceability.

    Args:
        phase_4_output: Prior phase output (Design in V1, Architecture in V2).
        framework_docs: Phase-filtered framework docs from DocEnricher.
        analysis_output: On-demand MVP analysis output (V2 pipeline).
        mvp_description: Functional description from agentic MVP refinement.
        prior_mvp_manifest: File paths already generated by prior MVP transforms.
    """
    stack_str = ", ".join(target_stack.get("frameworks", []))
    langs_str = ", ".join(target_stack.get("languages", []))
    mvp_section = _format_mvp_section(mvp_context)
    has_sps = _has_sp_refs(mvp_context)

    # Conditional SP content
    if has_sps:
        task_desc = "Generate the migrated code for this MVP, including completed SP stub implementations."
        sp_instruction = "\n2. Complete SP stub implementations (interfaces from design to concrete implementations)."
        sp_note = '\nInclude SP stub files with `"is_sp_stub": true`.'
    else:
        task_desc = "Generate the migrated code for this MVP."
        sp_instruction = ""
        sp_note = ""

    # Optional enrichment
    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Patterns
{framework_docs}
Use these idiomatic patterns. Follow the architecture blueprint and design.
"""

    # V2: on-demand analysis output provides register + traceability
    analysis_section = ""
    if analysis_output:
        analysis_section = f"""
### MVP Deep Analysis (Register + Traceability)
{analysis_output}

Use the Functional Requirements Register and Design Traceability Matrix above
to ensure complete coverage in the generated code.
"""

    # MVP functional description from agentic refinement
    mvp_desc_section = ""
    if mvp_description:
        mvp_desc_section = f"""
### MVP Functional Description
{mvp_description}
"""

    # Fix 1: Cross-MVP file manifest — prevents duplicate file creation
    manifest_section = ""
    if prior_mvp_manifest:
        manifest_section = f"""
### Files Already Generated by Prior MVPs
These files ALREADY EXIST in the target codebase. Do NOT recreate them — import from them:

{prior_mvp_manifest}

CRITICAL RULES:
- If a client, service, or utility you need is listed above → import it using the exact path shown.
- Do NOT generate a new file with the same or similar name as any file listed above.
- Match the naming convention of existing files (e.g. if prior files use *.util.ts, use *.util.ts).
"""

    return f"""{MIGRATION_ROLE}

## Task: Code Migration (Per-MVP)

{task_desc}
{mvp_section}
{mvp_desc_section}

### Target
{target_brief}

### Target Stack
- Languages: {langs_str}
- Frameworks: {stack_str}

### Prior Phase Output (Approved)
{phase_4_output}

### Source Code
{codebase_context}
{analysis_section}
{manifest_section}
{framework_docs_section}
### File Type Conversion Rules
CRITICAL — only convert files whose type is appropriate:
- **Source code** (.py, .js, .ts, .java, .cs, etc.) → Convert to target language ({langs_str})
- **SQL files** (.sql) → Keep as SQL. Generate target data access classes (repositories,
  DAOs) that USE the SQL, but do NOT rewrite .sql files as {langs_str} code.
  Output SQL files with `"language": "sql"`.
- **Config files** → Map to target format. Output with appropriate language (yaml, json, xml, properties).
- **Build/infra files** → Map to target build system. Output with appropriate language.
- If a source file is SQL, the output MUST be SQL (or a target-framework migration file),
  never a {langs_str} class that inlines the SQL logic.

### Instructions
1. Generate the migrated code following the Phase 4 design.{sp_instruction}
2. Use target framework idioms and best practices.
3. Include necessary imports and configuration.
4. Add comments where migration required non-obvious changes.
5. For EVERY item in the Phase 4 Design Traceability Matrix, generate corresponding code.
   Include a traceability comment in generated code: // Implements BR-1, DE-1, etc.
6. Respect the File Type Conversion Rules above — SQL stays SQL, configs stay configs.
7. **File naming conventions** — use this canonical structure:
   - API clients: `src/clients/*.client.ts`
   - Business services: `src/services/*.service.ts`
   - Utilities/helpers: `src/utils/*.util.ts`
   - Express controllers: `src/api/controllers/*.controller.ts`
   - DI container: `src/container.ts` (ONE file total — never create a second one)
   - App entry point: `src/index.ts` (ONE entry point only — do not also create src/app.ts or src/server.ts)
   - Shared types: `src/types/*.types.ts`
8. **No duplicate infrastructure** — if container.ts, src/index.ts, or package.json already
   appear in the "Files Already Generated by Prior MVPs" list above, do NOT regenerate them.

### Quality Requirements (MANDATORY)

1. **NO STUBS OR PLACEHOLDERS**: Every method MUST contain actual migrated business logic.
   Do NOT generate methods returning null/None, throwing NotImplementedError, or containing
   only TODO comments. If a source method has 50 lines of logic, the migrated version must
   have equivalent logic — not a 1-line stub.

2. **MIGRATION-TODO MARKER**: For genuinely unmigrateable code only — use
   `// MIGRATION-TODO: [specific reason]` inside an otherwise complete method body that
   contains your best-effort migration. Never leave a method empty.

3. **SINGLE ENTRY POINT**: Exactly ONE bootstrap/entry point per MVP. Do NOT generate
   app.listen(), main(), or server startup unless this MVP owns the application entry point.

4. **DEPENDENCY MANIFEST**: Include a package.json (JS/TS), pom.xml (Java), build.gradle
   (Kotlin), .csproj (C#), or requirements.txt (Python) that declares ALL third-party
   packages referenced in your generated code. Every import must have a corresponding
   dependency declaration.

5. **DI CONTAINER COMPLETENESS**: If the target framework uses dependency injection,
   register ALL injectable classes with class-based tokens, not string literals.
   Example: `container.bind(UserService).toSelf()` not `container.bind("UserService")`.

6. **CORRECT IMPORTS**: Every import statement must reference an actual file path that
   exists in your generated output or a package in the dependency manifest. No imports
   to files you did not generate.

### Output Format

Return a JSON array of migrated files:
```json
[
  {{
    "file_path": "package.json",
    "language": "json",
    "content": "{{ \\"name\\": \\"mvp-name\\", \\"dependencies\\": {{...}} }}"
  }},
  {{
    "file_path": "src/container.ts",
    "language": "typescript",
    "content": "// DI container with class-based bindings..."
  }},
  {{
    "file_path": "src/target/path/Module.java",
    "language": "java",
    "content": "// Full migrated source code...",
    "is_sp_stub": false
  }}
]
```
{sp_note}
After the JSON array, include an implementation traceability table as a JSON comment block:
```
// TRACEABILITY:
// | Register ID | Target File | Target Location | Status |
// | BR-1 | File.cs | Method():line | Implemented |
// | DE-1 | Entity.cs | Entity class | Implemented |
```

List any unimplemented register items with reason.

IMPORTANT: Return ONLY the JSON array, no markdown wrapping. The traceability
table should be embedded as a comment in the first file's content.
"""


# ── Integration Analysis (MVP 99) ──────────────────────────────────


def integration_analysis_prompt(
    architecture_output: str,
    target_stack: dict,
    merged_register: str,
    cross_mvp_dependencies: str,
    mvp_summary_table: str,
) -> str:
    """Build the prompt for MVP 99 integration analysis.

    Unlike regular MVP analysis which examines source code, this prompt
    receives pre-aggregated functional requirements from ALL other MVPs
    and asks the LLM to produce an integration-focused analysis.
    """
    stack_str = ", ".join(f"{k}: {v}" for k, v in target_stack.items()) if target_stack else "not specified"

    return f"""\
You are a senior software architect performing integration analysis for a
migration project. All functional MVPs have been individually analysed.
Your job is to produce the **Integration & Consolidation analysis** — the
blueprint that ties every MVP together into a working application.

## Target Stack
{stack_str}

## Architecture Overview
{architecture_output[:6000] if architecture_output else "(no architecture output available)"}

## MVP Summary
{mvp_summary_table}

## Aggregated Functional Requirements (from all MVPs)
{merged_register}

## Cross-MVP Dependencies (pre-computed)
{cross_mvp_dependencies}

---

## Your Task

Produce a **complete integration analysis** in markdown with these sections:

### 1. Integration Summary
One paragraph describing the overall integration challenge and strategy.

### 2. Unified Functional Requirements Register

Generate NEW integration-specific requirements (use prefix `IBR-`, `IDE-`,
`IINT-`, `IVAL-` to distinguish from per-MVP items):

#### Integration Business Rules
| ID | Rule | Consuming MVPs | Behavior | Criticality |
|----|------|---------------|----------|-------------|
(e.g., IBR-1: Unified application bootstrap, IBR-2: Shared error handling middleware)

#### Integration Data Entities
| ID | Entity | Shared By MVPs | Key Fields | Constraints |
|----|--------|---------------|-----------|-------------|
(e.g., IDE-1: Shared configuration schema)

#### Integration External Integrations
| ID | Integration | Shared By MVPs | Protocol | Notes |
|----|------------|---------------|----------|-------|
(e.g., IINT-1: Consolidated API client for Twitter — used by 3 MVPs)

#### Integration Validation Rules
| ID | Validator | Applied To | Rule | Error Behavior |
|----|-----------|-----------|------|---------------|

### 3. Cross-MVP Dependency Matrix
| MVP (depends on) | Depends On MVP | Shared Entity/Integration | Direction |
|-------------------|---------------|--------------------------|-----------|

### 4. Integration Touchpoints
For each cross-MVP boundary, describe:
- Which MVPs are involved
- What needs to be shared (types, interfaces, services, config)
- Recommended pattern (shared module, re-export, DI registration, event)

### 5. Integration Artifacts Required
List the specific files MVP 99 must generate:
- Unified entry point / bootstrap
- Consolidated DI container
- Merged package manifest (package.json / pom.xml / .csproj)
- Shared type definitions
- Bridge modules for cross-MVP imports
- Shared middleware / error handling

### 6. Integration Risk Assessment
| Risk | Severity | Affected MVPs | Mitigation |
|------|----------|--------------|------------|

### 7. Migration Steps
Ordered list of integration tasks, each referencing specific MVPs and artifacts.

## Rules
- Reference actual MVP names, not placeholders like "MVP X".
- Every integration requirement must trace back to specific per-MVP requirements.
- Be specific about file paths and class names where possible.
- If two MVPs have overlapping integrations (e.g., both use Twitter API), call it out explicitly.
"""


# ── Integration Review (MVP 99) ────────────────────────────────────

def integration_review_prompt(
    architecture_output: str,
    cross_ref_matrix: dict,
    requirements_coverage: dict,
    target_stack: dict,
    mvp_file_listing: str,
) -> str:
    """Build prompt for MVP 99 Integration Transform.

    The LLM receives pre-computed gap analysis and resolves specific issues.
    It does NOT need to discover gaps — only generate code to fix them.
    """
    import json as _json

    # ── Format pre-computed gaps as a numbered punch list ──
    unresolved = cross_ref_matrix.get("unresolved_imports", [])
    duplicates = cross_ref_matrix.get("duplicate_exports", [])
    entry_count = cross_ref_matrix.get("entry_point_count", 0)
    missing_manifest = cross_ref_matrix.get("missing_manifest", False)
    partial_di = cross_ref_matrix.get("partial_di", [])
    uncovered_reqs = requirements_coverage.get("uncovered_requirements", [])

    # Unresolved imports section
    unresolved_section = ""
    if unresolved:
        items = []
        for i, u in enumerate(unresolved[:30], 1):  # Cap at 30 items
            items.append(
                f"  {i}. {u['consumer_mvp']} imports `{u['import_ref']}` "
                f"— no MVP exports it."
            )
        unresolved_section = (
            f"### Unresolved Imports ({len(unresolved)})\n"
            "For each: generate a bridge module, re-export, or shared interface.\n\n"
            + "\n".join(items)
        )

    # Duplicate exports section
    duplicates_section = ""
    if duplicates:
        items = []
        for i, d in enumerate(duplicates[:15], 1):
            items.append(
                f"  {i}. `{d['symbol']}` exported by: {', '.join(d['mvps'])}"
            )
        duplicates_section = (
            f"### Duplicate Exports ({len(duplicates)})\n"
            "For each: consolidate into one canonical location or namespace.\n\n"
            + "\n".join(items)
        )

    # Missing infrastructure section
    infra_items = []
    if entry_count != 1:
        infra_items.append(
            f"- **Entry Points**: {entry_count} found (need exactly 1). "
            f"Generate a unified bootstrap that imports all MVP modules."
        )
    if missing_manifest:
        infra_items.append(
            "- **Package Manifest**: MISSING. Generate a consolidated "
            "package.json / pom.xml / .csproj with ALL dependencies from all MVPs."
        )
    if partial_di:
        infra_items.append(
            f"- **DI Container**: {len(partial_di)} injectable classes not registered. "
            f"Generate a complete DI container. Unregistered: {', '.join(partial_di[:20])}"
        )
    infra_section = ""
    if infra_items:
        infra_section = (
            "### Missing Infrastructure\n\n" + "\n".join(infra_items)
        )

    # Uncovered requirements section
    uncovered_section = ""
    if uncovered_reqs:
        items = []
        for i, r in enumerate(uncovered_reqs[:20], 1):
            items.append(f"  {i}. `{r['text']}` (from {r['source_mvp']})")
        uncovered_section = (
            f"### Uncovered Requirements ({len(uncovered_reqs)})\n"
            "These services/components appear in functional requirements but "
            "no MVP generated them. For each: generate an interface with a "
            "MIGRATION-TODO body, or a full implementation if the purpose is clear.\n\n"
            + "\n".join(items)
        )

    # All external packages section (Fix 4)
    all_packages = cross_ref_matrix.get("all_external_packages", [])
    packages_section = ""
    if all_packages:
        pkg_list = "\n".join(f"  - {p}" for p in all_packages[:60])
        packages_section = (
            f"### All External Packages Detected ({len(all_packages)})\n"
            "Every package below MUST appear in the generated package.json dependencies:\n\n"
            + pkg_list + "\n"
        )

    mandatory_section = """\
### MANDATORY OUTPUT FILES
You MUST generate ALL THREE of these files:
1. `src/container.ts` — Complete DI container with EVERY injectable class bound via
   `container.registerSingleton(ClassName, ClassName)`. Import each class at the top.
   Include ALL classes listed in the "DI Container" gap item above.
2. `package.json` — Include ALL packages from "All External Packages" above in dependencies.
   Use a real version (e.g. "^1.0.0") or "*" if unknown. Include scripts: build, start, dev.
3. `src/index.ts` — Single unified entry point. Import and mount ALL MVP route modules.
   Call app.listen(). This is the ONLY entry point — do not also output src/app.ts or src/server.ts.
"""

    target_stack_str = _json.dumps(target_stack, indent=2) if target_stack else "{}"

    return f"""{MIGRATION_ROLE}

## Task: Integration & Consolidation (MVP 99)

All functional MVPs have been transformed independently. Your job is to produce
integration files that consolidate them into a single working application.

You are given a **pre-computed gap analysis** below. You MUST address EVERY item
in the gap list. Do NOT skip any item.

### Architecture Blueprint (Approved)
{architecture_output[:6000] if architecture_output else "No architecture output available."}

### Target Stack
```json
{target_stack_str}
```

### Generated Files — Per-MVP Listing
{mvp_file_listing[:5000] if mvp_file_listing else "No file listing available."}

---

## Pre-Computed Integration Gaps (address EACH item)

{unresolved_section}

{duplicates_section}

{infra_section}

{packages_section}

{mandatory_section}

{uncovered_section}

---

### Quality Requirements
- Every generated file must contain real code, not stubs.
- Use class-based DI tokens, not string literals.
- Every import must reference an actual file from the MVP listings above
  or a package in the consolidated manifest.
- Generate ONLY integration/consolidation files. Do NOT regenerate files
  that already exist in the MVPs.

### Output Format
Return a JSON array of integration files:
```json
[
  {{"file_path": "src/main.ts", "language": "typescript", "content": "// Unified bootstrap..."}},
  {{"file_path": "package.json", "language": "json", "content": "..."}},
  {{"file_path": "src/container.ts", "language": "typescript", "content": "// Full DI container..."}}
]
```

IMPORTANT: Return ONLY the JSON array, no markdown wrapping.
"""


# ── Phase 6: Test (Per-MVP) ─────────────────────────────────────────

def phase_6_test(
    target_brief: str,
    target_stack: dict,
    phase_5_output: str,
    codebase_context: str,
    mvp_context: Optional[Dict] = None,
    framework_docs: str = "",
    analysis_output: str = "",
    mvp_description: str = "",
) -> str:
    """Build prompt for Phase 6: Scoped Test Generation with register traceability.

    Args:
        framework_docs: Phase-filtered framework docs from DocEnricher.
        analysis_output: Functional Requirements Register from analysis (V2 pipeline).
        mvp_description: Functional description from agentic MVP refinement.
    """
    langs_str = ", ".join(target_stack.get("languages", []))
    mvp_section = _format_mvp_section(mvp_context)
    has_sps = _has_sp_refs(mvp_context)

    # Conditional SP content
    if has_sps:
        sp_instruction = "\n3. Generate SP stub verification tests (ensure stubs return expected data shapes)."
        sp_test_type = '\ntest_type is one of: "unit", "integration", "equivalence", "sp_stub"'
    else:
        sp_instruction = ""
        sp_test_type = '\ntest_type is one of: "unit", "integration", "equivalence"'

    # Optional enrichment
    framework_docs_section = ""
    if framework_docs:
        framework_docs_section = f"""
### Target Framework Test Patterns
{framework_docs}
"""

    analysis_section = ""
    if analysis_output:
        analysis_section = f"""
### Functional Requirements Register (from Analysis)
{analysis_output}
"""

    mvp_desc_section = ""
    if mvp_description:
        mvp_desc_section = f"""
### MVP Functional Description
{mvp_description}
"""

    return f"""{MIGRATION_ROLE}

## Task: Test Generation (Per-MVP)

Generate a test suite scoped to this MVP's migrated code.
{mvp_section}
{mvp_desc_section}

### Target
{target_brief}

### Target Languages
{langs_str}

### Phase 5 Output (Migrated Code)
{phase_5_output}

### Source Codebase (Call Paths & Integration Points)
{codebase_context}
{framework_docs_section}{analysis_section}
### Instructions
1. Generate unit tests for each migrated module in this MVP.
2. Generate integration tests covering the MVP's key call paths.{sp_instruction}
3. Generate equivalence tests that verify migrated behavior matches source.
4. EVERY register item (BR-N, DE-N, INT-N, VAL-N) from the Functional Requirements
   Register MUST have at least one test. If the register is provided above, use it
   as the authoritative list. No register item may be left untested.
5. Generate integration tests for every external integration (INT-N).

### Output Format

Return a JSON array of test files:
```json
[
  {{
    "file_path": "tests/test_module.py",
    "language": "python",
    "content": "// Full test source code...",
    "test_type": "unit"
  }}
]
```
{sp_test_type}

After the JSON array, include a test coverage traceability as a JSON comment block:
```
// TEST_TRACEABILITY:
// | Register ID | Test File | Test Method | Type |
// | BR-1 | test_orders.py | test_calculate_total | unit |
// | VAL-1 | test_validation.py | test_required_fields | unit |
// | INT-1 | test_payment.py | test_charge_payment | integration |
```

List any untested register items with reason.
Coverage: N of M register items have tests.

IMPORTANT: Return ONLY the JSON array, no markdown wrapping. The traceability
table should be embedded as a comment in the first file's content.
"""


# ── Asset Inventory Refinement ──────────────────────────────────────

ASSET_REFINEMENT_PROMPT = """\
You are a migration architect. Given a project's asset inventory and migration context,
refine the suggested per-file-type migration strategy. Analyze sample file paths to
understand what each file type actually contains (e.g., Spring XML configs vs Maven POMs,
Flyway migrations vs ad-hoc SQL scripts) and override the rule-based defaults when
your understanding is more nuanced.

Migration type: {migration_type}
Target brief: {target_brief}
Target stack: {target_stack}

Asset Inventory:
{inventory_text}

Current suggestions (rule-based):
{rule_based_json}

Valid strategies: version_upgrade, framework_migration, rewrite, convert, keep_as_is, no_change

For each asset type, confirm or override the strategy. If overriding, provide a one-line reason.
Return ONLY a JSON object mapping language to {{"strategy": "...", "target": "..." or null, "reason": "..."}}.
No markdown fences, no extra text.
"""
