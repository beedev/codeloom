"""Chapter generators for reverse engineering documentation.

Each chapter generator takes (db_manager, project_id, pipeline=None) and returns
a markdown string. Most chapters are pure data (no LLM). Chapters 2, 5, and 14
use LLM synthesis via the gateway.

Chapter list:
    1.  System Overview & Processing Patterns
    2.  Component Architecture & Module Dependencies (LLM)
    3.  Entry Points & API Surface
    4.  Business Rules & Functional Specifications
    5.  Data Architecture & Entity Model (LLM)
    6.  Database Architecture & SQL Patterns
    7.  Processing Flow & Call Hierarchies
    8.  Integration Architecture & External Systems
    9.  Error Handling & Recovery Patterns
    10. Technology Stack & Platform Profile
    11. Performance & Scalability Characteristics
    12. Technical Debt & Complexity Assessment
    13. Non-Functional Requirements & Compliance
    14. Architecture Risks & Migration Gaps (LLM)
    15. Cross-Reference Findings & Architectural Issues (deterministic)"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)

# Canonical chapter titles indexed from 1
# Comprehensive coverage matching enterprise architecture documentation standards
CHAPTER_TITLES = [
    "System Overview & Processing Patterns",           # 1 - system arch, pattern classification
    "Component Architecture & Module Dependencies",     # 2 - PERFORM hierarchy, coupling, call graph
    "Entry Points & API Surface",                       # 3 - HTTP/batch/CLI/scheduled entry points
    "Business Rules & Functional Specifications",       # 4 - extracted business rules per entry point
    "Data Architecture & Entity Model",                 # 5 - FD layouts, type mappings, entity relationships
    "Database Architecture & SQL Patterns",             # 6 - CRUD matrix, cursor patterns, SQL access
    "Processing Flow & Call Hierarchies",               # 7 - batch flows, CICS flows, call trees
    "Integration Architecture & External Systems",      # 8 - COMMAREA, MQ, file sharing, APIs
    "Error Handling & Recovery Patterns",               # 9 - abend, IO-status, RESP codes, recovery
    "Technology Stack & Platform Profile",              # 10 - languages, frameworks, middleware
    "Performance & Scalability Characteristics",        # 11 - throughput, batch window, concurrency
    "Technical Debt & Complexity Assessment",           # 12 - complexity hotspots, dead code, coupling risks
    "Non-Functional Requirements & Compliance",         # 13 - security, audit, availability, data integrity
    "Architecture Risks & Migration Gaps",              # 14 - risks, unknowns, further analysis needed
    "Cross-Reference Findings & Architectural Issues",  # 15 - deterministic findings engine
]


def _pid(project_id: str) -> UUID:
    """Normalize project_id to UUID."""
    return UUID(project_id) if isinstance(project_id, str) else project_id


# =============================================================================
# Chapter 1: System Overview & Processing Patterns (NO LLM)
# =============================================================================


def generate_executive_summary(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate a data-driven executive summary of the codebase."""
    pid = _pid(project_id)

    with db.get_session() as session:
        # Project info
        proj = session.execute(
            text("SELECT name, primary_language, languages, file_count, total_lines "
                 "FROM projects WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchone()
        if not proj:
            return f"# 1. {CHAPTER_TITLES[0]}\n\nProject not found."

        # Language breakdown from code_files
        lang_rows = session.execute(
            text("SELECT language, COUNT(*) AS cnt FROM code_files "
                 "WHERE project_id = :pid GROUP BY language ORDER BY cnt DESC"),
            {"pid": pid},
        ).fetchall()

        # Unit counts by type
        unit_rows = session.execute(
            text("SELECT unit_type, COUNT(*) AS cnt FROM code_units "
                 "WHERE project_id = :pid GROUP BY unit_type ORDER BY cnt DESC"),
            {"pid": pid},
        ).fetchall()

        # ASG edge counts
        edge_rows = session.execute(
            text("SELECT edge_type, COUNT(*) AS cnt FROM code_edges "
                 "WHERE project_id = :pid GROUP BY edge_type ORDER BY cnt DESC"),
            {"pid": pid},
        ).fetchall()

        # Entry point count
        ep_count = session.execute(
            text("SELECT COUNT(*) AS cnt FROM deep_analyses WHERE project_id = :pid"),
            {"pid": pid},
        ).scalar() or 0

        # Deep analysis count
        analysis_count = session.execute(
            text("SELECT COUNT(*) AS cnt FROM deep_analysis_jobs "
                 "WHERE project_id = :pid AND status = 'completed'"),
            {"pid": pid},
        ).scalar() or 0

    # Format
    lines = [
        f"# 1. {CHAPTER_TITLES[0]}",
        "",
        f"**Project**: {proj.name}",
        f"**Primary Language**: {proj.primary_language or 'N/A'}",
        f"**Total Files**: {proj.file_count or 0}",
        f"**Total Lines**: {proj.total_lines or 0}",
        f"**Deep Analyses Completed**: {analysis_count}",
        f"**Entry Points Analyzed**: {ep_count}",
        "",
        "## Language Breakdown",
        "",
        "| Language | Files |",
        "|----------|-------|",
    ]
    for row in lang_rows:
        lines.append(f"| {row.language or 'unknown'} | {row.cnt} |")

    lines += [
        "",
        "## Code Unit Summary",
        "",
        "| Unit Type | Count |",
        "|-----------|-------|",
    ]
    total_units = 0
    for row in unit_rows:
        lines.append(f"| {row.unit_type} | {row.cnt} |")
        total_units += row.cnt
    lines.append(f"| **Total** | **{total_units}** |")

    lines += [
        "",
        "## ASG Relationship Summary",
        "",
        "| Edge Type | Count |",
        "|-----------|-------|",
    ]
    total_edges = 0
    for row in edge_rows:
        lines.append(f"| {row.edge_type} | {row.cnt} |")
        total_edges += row.cnt
    lines.append(f"| **Total** | **{total_edges}** |")

    return "\n".join(lines)


# =============================================================================
# Chapter 2: Component Architecture & Module Dependencies (LLM synthesis)
# =============================================================================


def generate_architecture_overview(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate architecture overview using LLM synthesis of module deps and class hierarchy."""
    pid = _pid(project_id)

    # Gather module dependency graph (directory level)
    from ..asg_builder.queries import get_module_dependency_graph, get_class_hierarchy
    mod_graph = get_module_dependency_graph(db, project_id, level="directory", dir_depth=2)
    class_hier = get_class_hierarchy(db, project_id)

    # Gather project basics
    with db.get_session() as session:
        proj = session.execute(
            text("SELECT name, primary_language, languages FROM projects WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchone()

        # Get ground truth patterns from deep analyses
        gt_rows = session.execute(
            text("SELECT result_json FROM deep_analyses WHERE project_id = :pid LIMIT 20"),
            {"pid": pid},
        ).fetchall()

    # Extract patterns from ground truth
    patterns = set()
    frameworks = set()
    for row in gt_rows:
        rj = row.result_json if isinstance(row.result_json, dict) else {}
        for item in rj.get("integrations", []):
            if isinstance(item, dict):
                itype = item.get("type", "")
                if itype:
                    patterns.add(itype)
        # Some analyses have metadata about patterns
        for br in rj.get("business_rules", []):
            if isinstance(br, dict) and br.get("category"):
                patterns.add(br["category"])

    # Build context for LLM
    mod_nodes = mod_graph.get("nodes", [])[:30]
    mod_links = mod_graph.get("links", [])[:50]
    class_nodes = class_hier.get("nodes", [])[:30] if isinstance(class_hier, dict) else []
    class_edges = class_hier.get("edges", [])[:30] if isinstance(class_hier, dict) else []

    context_parts = [
        f"Project: {proj.name if proj else 'Unknown'}",
        f"Primary language: {proj.primary_language if proj else 'Unknown'}",
        f"Languages: {proj.languages if proj else []}",
        "",
        f"Module dependency graph ({mod_graph.get('node_count', 0)} modules, {mod_graph.get('link_count', 0)} dependencies):",
    ]
    for link in mod_links:
        context_parts.append(f"  {link.get('source', '?')} -> {link.get('target', '?')} (weight: {link.get('weight', 1)})")

    if class_nodes:
        context_parts.append(f"\nClass hierarchy ({len(class_nodes)} classes):")
        for edge in class_edges:
            context_parts.append(f"  {edge.get('child_name', '?')} --{edge.get('edge_type', 'inherits')}--> {edge.get('parent_name', '?')}")

    if patterns:
        context_parts.append(f"\nDetected patterns: {', '.join(sorted(patterns))}")

    context = "\n".join(context_parts)

    prompt = f"""You are a software architect writing a reverse engineering document.
Based on the following codebase intelligence, write Chapter 2: {CHAPTER_TITLES[1]}.

Include:
1. A high-level description of the system architecture (layered, microservices, monolith, etc.)
2. Key architectural patterns detected
3. Module structure and their responsibilities
4. Inter-module dependencies and data flow
5. Class hierarchy highlights (if any)

Write in markdown. Start with "# 2. {CHAPTER_TITLES[1]}". Be concise but thorough.
Use bullet points and sub-sections. Do not invent information not present in the data.

Codebase Intelligence:
{context}
"""

    try:
        from llama_index.core import Settings as LISettings
        response = LISettings.llm.complete(prompt)
        return str(response)
    except Exception as e:
        logger.error("LLM call failed for architecture overview: %s", e)
        # Fallback: data-only
        lines = [
            f"# 2. {CHAPTER_TITLES[1]}",
            "",
            f"**Project**: {proj.name if proj else 'Unknown'}",
            f"**Primary Language**: {proj.primary_language if proj else 'Unknown'}",
            "",
            "## Module Dependencies",
            "",
            f"Total modules: {mod_graph.get('node_count', 0)}",
            f"Total dependencies: {mod_graph.get('link_count', 0)}",
            "",
        ]
        for link in mod_links[:20]:
            lines.append(f"- {link.get('source', '?')} -> {link.get('target', '?')}")

        if class_nodes:
            lines += ["", "## Class Hierarchy", ""]
            for edge in class_edges[:20]:
                lines.append(f"- {edge.get('child_name', '?')} extends {edge.get('parent_name', '?')}")

        return "\n".join(lines)


# =============================================================================
# Chapter 3: Entry Points & API Surface (NO LLM)
# =============================================================================


def generate_entry_points(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate entry points chapter from deep_analyses table."""
    pid = _pid(project_id)

    with db.get_session() as session:
        rows = session.execute(
            text("""
                SELECT a.entry_type, u.name, u.qualified_name, u.language,
                       f.file_path, u.start_line
                FROM deep_analyses a
                JOIN code_units u ON a.entry_unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE a.project_id = :pid
                ORDER BY a.entry_type, f.file_path, u.name
            """),
            {"pid": pid},
        ).fetchall()

    # Group by entry type
    grouped: Dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.entry_type].append(row)

    lines = [
        f"# 3. {CHAPTER_TITLES[2]}",
        "",
        f"**Total Entry Points**: {len(rows)}",
        "",
    ]

    if not rows:
        lines.append("No entry points found. Run deep analysis first.")
        return "\n".join(lines)

    for entry_type in sorted(grouped.keys()):
        items = grouped[entry_type]
        lines += [
            f"## {entry_type} ({len(items)})",
            "",
            "| Name | Qualified Name | File | Line | Language |",
            "|------|---------------|------|------|----------|",
        ]
        for item in items:
            lines.append(
                f"| {item.name} | {item.qualified_name or ''} | "
                f"{item.file_path} | {item.start_line or ''} | {item.language or ''} |"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Chapter 4: Business Rules & Functional Specifications (NO LLM)
# =============================================================================


def generate_functional_requirements(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate functional requirements from business rules in deep analyses."""
    pid = _pid(project_id)

    with db.get_session() as session:
        rows = session.execute(
            text("""
                SELECT a.analysis_id, a.result_json, a.entry_type,
                       u.name AS entry_name, u.qualified_name AS entry_qualified,
                       f.file_path AS entry_file
                FROM deep_analyses a
                JOIN code_units u ON a.entry_unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE a.project_id = :pid
                ORDER BY f.file_path, u.name
            """),
            {"pid": pid},
        ).fetchall()

    lines = [
        f"# 4. {CHAPTER_TITLES[3]}",
        "",
    ]

    if not rows:
        lines.append("No deep analysis results found. Run deep analysis first.")
        return "\n".join(lines)

    br_count = 0
    all_rules = []

    for row in rows:
        rj = row.result_json if isinstance(row.result_json, dict) else {}
        rules = rj.get("business_rules", [])
        if not isinstance(rules, list):
            continue

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            br_count += 1
            evidence = rule.get("evidence", rule.get("evidence_refs", []))
            evidence_str = ""
            if isinstance(evidence, list) and evidence:
                evidence_parts = []
                for ev in evidence[:3]:
                    if isinstance(ev, dict):
                        evidence_parts.append(f"{ev.get('file', '')}:{ev.get('line', '')}")
                    elif isinstance(ev, str):
                        evidence_parts.append(ev)
                evidence_str = "; ".join(evidence_parts)

            all_rules.append({
                "id": f"BR-{br_count:04d}",
                "name": rule.get("name", rule.get("rule", "Unnamed")),
                "description": rule.get("description", rule.get("detail", "")),
                "entry_point": row.entry_name,
                "evidence": evidence_str,
            })

    lines.append(f"**Total Business Rules**: {br_count}")
    lines.append("")

    if all_rules:
        lines += [
            "| ID | Name | Description | Entry Point | Evidence |",
            "|----|------|-------------|-------------|----------|",
        ]
        for rule in all_rules:
            desc = (rule["description"] or "")[:100]
            if len(rule.get("description", "") or "") > 100:
                desc += "..."
            lines.append(
                f"| {rule['id']} | {rule['name']} | {desc} | "
                f"{rule['entry_point']} | {rule['evidence']} |"
            )
    else:
        lines.append("No business rules extracted from deep analysis.")

    return "\n".join(lines)


# =============================================================================
# Chapter 5: Data Architecture & Entity Model (LLM synthesis)
# =============================================================================


def generate_data_model(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate data model chapter using LLM synthesis of data entities and type_dep edges."""
    pid = _pid(project_id)

    with db.get_session() as session:
        # Get data entities from deep analyses
        rows = session.execute(
            text("SELECT result_json FROM deep_analyses WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchall()

        # Get type_dep edges for entity relationships
        type_deps = session.execute(
            text("""
                SELECT su.name AS source_name, su.qualified_name AS source_qualified,
                       su.unit_type AS source_type,
                       tu.name AS target_name, tu.qualified_name AS target_qualified,
                       tu.unit_type AS target_type
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'type_dep'
                ORDER BY su.name, tu.name
                LIMIT 100
            """),
            {"pid": pid},
        ).fetchall()

    # Collect data entities from all analyses
    all_entities = []
    for row in rows:
        rj = row.result_json if isinstance(row.result_json, dict) else {}
        entities = rj.get("data_entities", [])
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    all_entities.append(ent)

    # Build context for LLM
    context_parts = ["Data entities extracted from deep analysis:"]
    seen = set()
    for ent in all_entities[:50]:
        name = ent.get("name", "Unknown")
        if name in seen:
            continue
        seen.add(name)
        desc = ent.get("description", "")
        etype = ent.get("type", ent.get("entity_type", ""))
        fields_list = ent.get("fields", [])
        fields_str = ""
        if isinstance(fields_list, list) and fields_list:
            fields_str = ", ".join(
                f.get("name", str(f)) if isinstance(f, dict) else str(f)
                for f in fields_list[:10]
            )
        context_parts.append(f"  - {name} ({etype}): {desc} [fields: {fields_str}]")

    context_parts.append(f"\nType dependency edges ({len(type_deps)} total):")
    for dep in type_deps[:40]:
        context_parts.append(
            f"  {dep.source_name} ({dep.source_type}) -> {dep.target_name} ({dep.target_type})"
        )

    context = "\n".join(context_parts)

    prompt = f"""You are a software architect writing a reverse engineering document.
Based on the following data entities and type dependencies, write Chapter 5: {CHAPTER_TITLES[4]}.

Include:
1. Summary of identified data entities and their roles
2. Entity relationships based on type dependencies
3. Key data structures and their fields
4. Data flow patterns (if apparent from the dependencies)

Write in markdown. Start with "# 5. {CHAPTER_TITLES[4]}". Be concise but thorough.
Do not invent information not present in the data.

Codebase Intelligence:
{context}
"""

    try:
        from llama_index.core import Settings as LISettings
        response = LISettings.llm.complete(prompt)
        return str(response)
    except Exception as e:
        logger.error("LLM call failed for data model: %s", e)
        # Fallback: data-only
        lines = [
            f"# 5. {CHAPTER_TITLES[4]}",
            "",
            f"**Data Entities Found**: {len(seen)}",
            "",
            "## Entities",
            "",
            "| Name | Type | Description |",
            "|------|------|-------------|",
        ]
        for ent in all_entities[:50]:
            name = ent.get("name", "Unknown")
            etype = ent.get("type", ent.get("entity_type", ""))
            desc = (ent.get("description", "") or "")[:80]
            lines.append(f"| {name} | {etype} | {desc} |")

        lines += [
            "",
            "## Type Dependencies",
            "",
            "| Source | Source Type | Target | Target Type |",
            "|--------|------------|--------|-------------|",
        ]
        for dep in type_deps[:40]:
            lines.append(
                f"| {dep.source_name} | {dep.source_type} | "
                f"{dep.target_name} | {dep.target_type} |"
            )

        return "\n".join(lines)


# =============================================================================
# Chapter 6: Database Architecture & SQL Patterns (NO LLM)
# =============================================================================


def generate_database_architecture(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate database architecture chapter from SQL patterns, CRUD operations, and SP calls."""
    pid = _pid(project_id)

    lines = [
        f"# 6. {CHAPTER_TITLES[5]}",
        "",
    ]

    # --- SQL-related code units (EXEC SQL, cursors, stored procs) ---
    try:
        with db.get_session() as session:
            sql_units = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type, u.signature,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Universal SQL patterns
                        u.unit_type ILIKE '%sql%'
                        OR u.unit_type ILIKE '%cursor%'
                        OR u.unit_type ILIKE '%stored_proc%'
                        OR u.name ILIKE '%SQL%'
                        OR u.signature ILIKE '%EXEC SQL%'
                        OR u.signature ILIKE '%SELECT %'
                        OR u.signature ILIKE '%INSERT %'
                        OR u.signature ILIKE '%UPDATE %'
                        OR u.signature ILIKE '%DELETE %'
                        -- ORM / data access patterns
                        OR u.name ILIKE '%Repository%'
                        OR u.name ILIKE '%EntityManager%'
                        OR u.name ILIKE '%Session%'
                        OR u.name ILIKE '%Model%'
                        OR u.name ILIKE '%Schema%'
                        OR u.name ILIKE '%DAO%'
                        OR u.name ILIKE '%Mapper%'
                        OR u.source ILIKE '%@Entity%'
                        OR u.source ILIKE '%@Table%'
                        OR u.source ILIKE '%@Column%'
                        OR u.source ILIKE '%@Repository%'
                        OR u.source ILIKE '%Query(%'
                        OR u.source ILIKE '%QuerySet%'
                        OR u.source ILIKE '%objects.filter%'
                        OR u.source ILIKE '%objects.get%'
                        OR u.source ILIKE '%findAll%'
                        OR u.source ILIKE '%findBy%'
                        -- Connection / data source patterns
                        OR u.name ILIKE '%DataSource%'
                        OR u.name ILIKE '%Connection%'
                        OR u.source ILIKE '%DriverManager%'
                        OR u.source ILIKE '%connection_string%'
                        OR u.source ILIKE '%connectionString%'
                        OR u.source ILIKE '%db_url%'
                        OR u.source ILIKE '%DATABASE_URL%'
                        -- Migration patterns
                        OR u.name ILIKE '%Migration%'
                        OR u.source ILIKE '%alembic%'
                        OR u.source ILIKE '%flyway%'
                        OR u.source ILIKE '%liquibase%'
                        OR u.source ILIKE '%knex.migrate%'
                        OR u.source ILIKE '%sequelize%'
                        -- NoSQL patterns
                        OR u.source ILIKE '%MongoClient%'
                        OR u.source ILIKE '%mongoose%'
                        OR u.source ILIKE '%redis%'
                        OR u.source ILIKE '%dynamodb%'
                        OR u.source ILIKE '%collection%'
                        OR u.source ILIKE '%MongoRepository%'
                        -- COBOL-specific cursor patterns
                        OR u.name ILIKE '%CURSOR%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 200
                """),
                {"pid": pid},
            ).fetchall()

        if sql_units:
            lines += [
                f"## Database & Data Access Code Units ({len(sql_units)})",
                "",
                "| Name | Type | File | Line | Signature |",
                "|------|------|------|------|-----------|",
            ]
            for u in sql_units:
                sig = (u.signature or "")[:80]
                if len(u.signature or "") > 80:
                    sig += "..."
                lines.append(
                    f"| {u.name} | {u.unit_type} | {u.file_path} | "
                    f"{u.start_line or ''} | {sig} |"
                )
            lines.append("")
        else:
            lines.append("No database or data access code units detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query SQL units: %s", e)
        lines.append(f"SQL unit query unavailable: {e}\n")

    # --- CRUD matrix: search source for SQL DML keywords ---
    try:
        with db.get_session() as session:
            crud_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, f.file_path, u.source
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND u.source IS NOT NULL
                      AND (
                        -- Universal SQL DML patterns
                        u.source ILIKE '%SELECT %FROM%'
                        OR u.source ILIKE '%INSERT %INTO%'
                        OR u.source ILIKE '%UPDATE %SET%'
                        OR u.source ILIKE '%DELETE %FROM%'
                        -- ORM read patterns
                        OR u.source ILIKE '%findAll%'
                        OR u.source ILIKE '%findBy%'
                        OR u.source ILIKE '%findOne%'
                        OR u.source ILIKE '%objects.filter%'
                        OR u.source ILIKE '%objects.get%'
                        OR u.source ILIKE '%.query(%'
                        OR u.source ILIKE '%.fetch(%'
                        -- ORM write patterns
                        OR u.source ILIKE '%.save(%'
                        OR u.source ILIKE '%.create(%'
                        OR u.source ILIKE '%.update(%'
                        OR u.source ILIKE '%.delete(%'
                        OR u.source ILIKE '%.remove(%'
                        OR u.source ILIKE '%.persist(%'
                        OR u.source ILIKE '%.merge(%'
                        -- COBOL-specific SQL patterns
                        OR u.source ILIKE '%EXEC SQL%'
                      )
                    LIMIT 200
                """),
                {"pid": pid},
            ).fetchall()

        if crud_rows:
            # Build a simple CRUD matrix by unit
            crud_matrix = []
            for row in crud_rows:
                src = (row.source or "").upper()
                src_lower = (row.source or "").lower()
                ops = []
                # SQL DML detection
                if "SELECT " in src and "FROM" in src:
                    ops.append("R")
                if "INSERT " in src and "INTO" in src:
                    ops.append("C")
                if "UPDATE " in src and "SET" in src:
                    ops.append("U")
                if "DELETE " in src and "FROM" in src:
                    ops.append("D")
                # ORM read patterns
                if not ops or "R" not in ops:
                    if any(p in src_lower for p in (
                        "findall", "findby", "findone", "objects.filter",
                        "objects.get", ".query(", ".fetch(",
                    )):
                        ops.append("R")
                # ORM create patterns
                if "C" not in ops:
                    if any(p in src_lower for p in (
                        ".save(", ".create(", ".persist(",
                    )):
                        ops.append("C")
                # ORM update patterns
                if "U" not in ops:
                    if any(p in src_lower for p in (
                        ".update(", ".merge(",
                    )):
                        ops.append("U")
                # ORM delete patterns
                if "D" not in ops:
                    if any(p in src_lower for p in (
                        ".delete(", ".remove(",
                    )):
                        ops.append("D")
                if ops:
                    crud_matrix.append({
                        "name": row.name,
                        "file": row.file_path,
                        "ops": ", ".join(ops),
                    })

            if crud_matrix:
                lines += [
                    f"## CRUD Operations Matrix ({len(crud_matrix)} units with SQL)",
                    "",
                    "| Unit | File | Operations |",
                    "|------|------|------------|",
                ]
                for item in crud_matrix[:100]:
                    lines.append(f"| {item['name']} | {item['file']} | {item['ops']} |")
                lines.append("")
    except Exception as e:
        logger.warning("Failed to build CRUD matrix: %s", e)
        lines.append(f"CRUD matrix unavailable: {e}\n")

    # --- Cursor patterns ---
    try:
        with db.get_session() as session:
            cursor_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic data access iteration patterns
                        u.source ILIKE '%ResultSet%'
                        OR u.source ILIKE '%Cursor%'
                        OR u.source ILIKE '%Iterator%'
                        OR u.source ILIKE '%paginate%'
                        OR u.source ILIKE '%Pageable%'
                        OR u.source ILIKE '%ScrollableResults%'
                        OR u.source ILIKE '%lazy_load%'
                        OR u.source ILIKE '%LazyLoad%'
                        OR u.source ILIKE '%batch_size%'
                        OR u.source ILIKE '%@BatchSize%'
                        OR u.name ILIKE '%CURSOR%'
                        OR u.name ILIKE '%ITERATOR%'
                        OR u.name ILIKE '%PAGINATOR%'
                        OR u.name ILIKE '%RESULTSET%'
                        -- COBOL-specific cursor patterns
                        OR u.source ILIKE '%DECLARE%CURSOR%'
                        OR u.source ILIKE '%OPEN %CURSOR%'
                        OR u.source ILIKE '%FETCH %'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if cursor_rows:
            lines += [
                f"## Data Iteration / Cursor Patterns ({len(cursor_rows)} units)",
                "",
                "| Unit | File | Line |",
                "|------|------|------|",
            ]
            for row in cursor_rows:
                lines.append(f"| {row.name} | {row.file_path} | {row.start_line or ''} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query cursor patterns: %s", e)
        lines.append(f"Cursor pattern query unavailable: {e}\n")

    # --- Parameterized query / host variable patterns ---
    try:
        with db.get_session() as session:
            param_query_count = session.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM code_units u
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic parameterized query patterns
                        u.source ILIKE '%PreparedStatement%'
                        OR u.source ILIKE '%@Param%'
                        OR u.source ILIKE '%:param%'
                        OR u.source ILIKE '%$1%'
                        OR u.source ILIKE '%?%' AND u.source ILIKE '%execute%'
                        OR u.source ILIKE '%bind%'
                        OR u.source ILIKE '%parameterized%'
                        OR u.source ILIKE '%named_params%'
                        -- COBOL-specific host variable patterns
                        OR u.source ILIKE '%:WS-%'
                      )
                """),
                {"pid": pid},
            ).scalar() or 0

        if param_query_count > 0:
            lines += [
                f"## Parameterized Query / Host Variable Usage",
                "",
                f"**Units with parameterized queries or host variables**: {param_query_count}",
                "",
            ]
    except Exception as e:
        logger.warning("Failed to query parameterized query patterns: %s", e)

    # --- Stored procedure call edges ---
    try:
        with db.get_session() as session:
            sp_edges = session.execute(
                text("""
                    SELECT su.name AS caller_name, su.qualified_name AS caller_qname,
                           tu.name AS sp_name, tu.qualified_name AS sp_qname,
                           sf.file_path AS caller_file
                    FROM code_edges e
                    JOIN code_units su ON e.source_unit_id = su.unit_id
                    JOIN code_units tu ON e.target_unit_id = tu.unit_id
                    JOIN code_files sf ON su.file_id = sf.file_id
                    WHERE e.project_id = :pid AND e.edge_type = 'calls_sp'
                    ORDER BY su.name, tu.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if sp_edges:
            lines += [
                f"## Stored Procedure Calls ({len(sp_edges)})",
                "",
                "| Caller | Stored Procedure | Caller File |",
                "|--------|-----------------|-------------|",
            ]
            for edge in sp_edges:
                lines.append(
                    f"| {edge.caller_name} | {edge.sp_name} | {edge.caller_file} |"
                )
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query SP edges: %s", e)
        lines.append(f"Stored procedure call query unavailable: {e}\n")

    # --- Database platform-specific patterns ---
    try:
        with db.get_session() as session:
            db_platform_units = session.execute(
                text("""
                    SELECT u.name, u.unit_type, f.file_path
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic database platform patterns
                        u.source ILIKE '%Hibernate%'
                        OR u.source ILIKE '%JPA%'
                        OR u.source ILIKE '%SQLAlchemy%'
                        OR u.source ILIKE '%TypeORM%'
                        OR u.source ILIKE '%Prisma%'
                        OR u.source ILIKE '%Sequelize%'
                        OR u.source ILIKE '%ActiveRecord%'
                        OR u.source ILIKE '%Entity Framework%'
                        OR u.source ILIKE '%Dapper%'
                        OR u.source ILIKE '%MyBatis%'
                        OR u.source ILIKE '%Drizzle%'
                        OR u.source ILIKE '%Knex%'
                        OR u.source ILIKE '%connection_pool%'
                        OR u.source ILIKE '%ConnectionPool%'
                        OR u.source ILIKE '%HikariCP%'
                        -- DB2 / mainframe-specific patterns
                        OR u.source ILIKE '%DB2%'
                        OR u.source ILIKE '%SQLCODE%'
                        OR u.source ILIKE '%SQLCA%'
                        OR u.source ILIKE '%DCLGEN%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 50
                """),
                {"pid": pid},
            ).fetchall()

        if db_platform_units:
            lines += [
                f"## Database Platform Patterns ({len(db_platform_units)} units)",
                "",
                "| Unit | Type | File |",
                "|------|------|------|",
            ]
            for u in db_platform_units:
                lines.append(f"| {u.name} | {u.unit_type} | {u.file_path} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query database platform patterns: %s", e)

    if len(lines) <= 3:
        lines.append("No database architecture patterns detected in this codebase.")

    return "\n".join(lines)


# =============================================================================
# Chapter 7: Processing Flow & Call Hierarchies (NO LLM)
# =============================================================================


def generate_call_trees(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate call trees for top entry points using transitive callee queries."""
    pid = _pid(project_id)

    from ..asg_builder.queries import get_all_callees

    # Get top entry points (by analysis, up to 10)
    with db.get_session() as session:
        ep_rows = session.execute(
            text("""
                SELECT a.entry_unit_id, u.name, u.qualified_name,
                       f.file_path, a.total_units
                FROM deep_analyses a
                JOIN code_units u ON a.entry_unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE a.project_id = :pid
                ORDER BY a.total_units DESC
                LIMIT 10
            """),
            {"pid": pid},
        ).fetchall()

    lines = [
        f"# 7. {CHAPTER_TITLES[6]}",
        "",
        f"**Top {len(ep_rows)} entry points by call chain depth**",
        "",
    ]

    if not ep_rows:
        lines.append("No entry points available. Run deep analysis first.")
        return "\n".join(lines)

    for ep in ep_rows:
        unit_id = str(ep.entry_unit_id)
        lines += [
            f"## {ep.name}",
            f"File: `{ep.file_path}`",
            "",
        ]

        try:
            callees = get_all_callees(db, project_id, unit_id, max_depth=5)
            by_depth = callees.get("by_depth", {})
            total = callees.get("total_count", 0)

            if total == 0:
                lines.append("No callees detected.")
                lines.append("")
                continue

            lines.append(f"Total callees: {total}")
            lines.append("")
            lines.append("```")

            # Render as indented tree
            for depth in sorted(by_depth.keys()):
                for callee in by_depth[depth][:15]:  # Cap per depth
                    indent = "  " * depth
                    name = callee.get("name", "?")
                    qname = callee.get("qualified_name", "")
                    fp = callee.get("file_path", "")
                    display = qname if qname else name
                    if fp:
                        display += f" ({fp})"
                    lines.append(f"{indent}{display}")

            lines.append("```")
            lines.append("")
        except Exception as e:
            logger.warning("Failed to get callees for %s: %s", ep.name, e)
            lines.append(f"Error retrieving call tree: {e}")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# Chapter 8: Integration Architecture & External Systems (NO LLM)
# =============================================================================


def generate_integrations(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate integrations chapter from deep analysis results."""
    pid = _pid(project_id)

    with db.get_session() as session:
        rows = session.execute(
            text("SELECT result_json FROM deep_analyses WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchall()

    # Collect integrations from all analyses
    grouped: Dict[str, list] = defaultdict(list)
    for row in rows:
        rj = row.result_json if isinstance(row.result_json, dict) else {}
        integrations = rj.get("integrations", [])
        if not isinstance(integrations, list):
            continue
        for integ in integrations:
            if not isinstance(integ, dict):
                continue
            itype = integ.get("type", "unknown")
            grouped[itype].append(integ)

    total = sum(len(v) for v in grouped.values())

    lines = [
        f"# 8. {CHAPTER_TITLES[7]}",
        "",
        f"**Total Integrations Found**: {total}",
        "",
    ]

    if not grouped:
        lines.append("No integrations found in deep analysis results.")
        return "\n".join(lines)

    for itype in sorted(grouped.keys()):
        items = grouped[itype]
        lines += [
            f"## {itype} ({len(items)})",
            "",
            "| Name | Description | Evidence |",
            "|------|-------------|----------|",
        ]

        seen_names = set()
        for item in items:
            name = item.get("name", "Unnamed")
            if name in seen_names:
                continue
            seen_names.add(name)
            desc = (item.get("description", "") or "")[:100]
            evidence = item.get("evidence", item.get("evidence_refs", []))
            evidence_str = ""
            if isinstance(evidence, list) and evidence:
                parts = []
                for ev in evidence[:2]:
                    if isinstance(ev, dict):
                        parts.append(f"{ev.get('file', '')}:{ev.get('line', '')}")
                    elif isinstance(ev, str):
                        parts.append(ev)
                evidence_str = "; ".join(parts)
            lines.append(f"| {name} | {desc} | {evidence_str} |")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Chapter 9: Error Handling & Recovery Patterns (NO LLM)
# =============================================================================


def generate_error_handling(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate error handling patterns from abend, IO-status, RESP, and exception patterns."""
    pid = _pid(project_id)

    lines = [
        f"# 9. {CHAPTER_TITLES[8]}",
        "",
    ]

    # --- Abend patterns ---
    try:
        with db.get_session() as session:
            abend_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic exception/error handler patterns
                        u.name ILIKE '%ERROR%'
                        OR u.name ILIKE '%EXCEPTION%'
                        OR u.name ILIKE '%HANDLER%'
                        OR u.name ILIKE '%CATCH%'
                        OR u.name ILIKE '%RECOVER%'
                        OR u.name ILIKE '%ABORT%'
                        OR u.name ILIKE '%FAULT%'
                        OR u.source ILIKE '%catch (%'
                        OR u.source ILIKE '%catch(%'
                        OR u.source ILIKE '%except %'
                        OR u.source ILIKE '%except:%'
                        OR u.source ILIKE '%@ExceptionHandler%'
                        OR u.source ILIKE '%@ControllerAdvice%'
                        OR u.source ILIKE '%ErrorBoundary%'
                        OR u.source ILIKE '%BaseException%'
                        OR u.source ILIKE '%finally %'
                        OR u.source ILIKE '%finally{%'
                        -- COBOL-specific error handler patterns
                        OR u.name ILIKE '%ABEND%'
                        OR u.name ILIKE '%9999%'
                        OR u.name ILIKE '%Z-ABEND%'
                        OR u.source ILIKE '%ABEND%'
                        OR u.source ILIKE '%GOBACK%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if abend_rows:
            lines += [
                f"## Exception / Error Handler Units ({len(abend_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in abend_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No exception/error handler patterns detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query abend patterns: %s", e)
        lines.append(f"Abend pattern query unavailable: {e}\n")

    # --- IO-Status handling ---
    try:
        with db.get_session() as session:
            io_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic error propagation patterns
                        u.source ILIKE '%throw %'
                        OR u.source ILIKE '%throw(%'
                        OR u.source ILIKE '%throw new%'
                        OR u.source ILIKE '%raise %'
                        OR u.source ILIKE '%raise(%'
                        OR u.source ILIKE '%rethrow%'
                        OR u.source ILIKE '%throws %'
                        OR u.source ILIKE '%Promise.reject%'
                        OR u.source ILIKE '%reject(%'
                        OR u.source ILIKE '%propagate%'
                        OR u.name ILIKE '%THROW%'
                        OR u.name ILIKE '%RAISE%'
                        OR u.name ILIKE '%PROPAGATE%'
                        -- COBOL-specific I/O status error patterns
                        OR u.name ILIKE '%9910%'
                        OR u.name ILIKE '%IO-STATUS%'
                        OR u.name ILIKE '%IO STATUS%'
                        OR u.name ILIKE '%DISPLAY-IO%'
                        OR u.name ILIKE '%FILE-STATUS%'
                        OR u.source ILIKE '%FILE STATUS%'
                        OR u.source ILIKE '%IO-STATUS%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if io_rows:
            lines += [
                f"## Error Propagation Patterns ({len(io_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in io_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No error propagation patterns (throw/raise/reject) detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query IO-status patterns: %s", e)
        lines.append(f"IO-status pattern query unavailable: {e}\n")

    # --- CICS RESP handling ---
    try:
        with db.get_session() as session:
            resp_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic recovery patterns
                        u.source ILIKE '%retry%'
                        OR u.source ILIKE '%fallback%'
                        OR u.source ILIKE '%circuit_breaker%'
                        OR u.source ILIKE '%circuit-breaker%'
                        OR u.source ILIKE '%CircuitBreaker%'
                        OR u.source ILIKE '%recover%'
                        OR u.source ILIKE '%rollback%'
                        OR u.source ILIKE '%backoff%'
                        OR u.source ILIKE '%@Retryable%'
                        OR u.source ILIKE '%@Recover%'
                        OR u.source ILIKE '%@HystrixCommand%'
                        OR u.name ILIKE '%RETRY%'
                        OR u.name ILIKE '%FALLBACK%'
                        OR u.name ILIKE '%RECOVER%'
                        OR u.name ILIKE '%CIRCUIT%'
                        OR u.name ILIKE '%ROLLBACK%'
                        OR u.name ILIKE '%RESILIEN%'
                        -- COBOL/CICS-specific recovery patterns
                        OR u.source ILIKE '%RESP(%'
                        OR u.source ILIKE '%RESP2(%'
                        OR u.source ILIKE '%HANDLE CONDITION%'
                        OR u.source ILIKE '%HANDLE ABEND%'
                        OR u.source ILIKE '%HANDLE AID%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if resp_rows:
            lines += [
                f"## Recovery Patterns ({len(resp_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in resp_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No recovery patterns (retry/fallback/circuit-breaker) detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query CICS RESP patterns: %s", e)
        lines.append(f"CICS RESP pattern query unavailable: {e}\n")

    # --- General error/exception patterns ---
    try:
        with db.get_session() as session:
            err_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic logging/diagnostic in error context
                        u.name ILIKE '%LOG%ERROR%'
                        OR u.name ILIKE '%LOG%EXCEPTION%'
                        OR u.name ILIKE '%ERROR%LOG%'
                        OR u.name ILIKE '%ERROR%REPORT%'
                        OR u.name ILIKE '%DEBUG%'
                        OR u.name ILIKE '%WARN%'
                        OR u.name ILIKE '%TRACE%'
                        OR u.name ILIKE '%DIAGNOSTIC%'
                        OR u.source ILIKE '%logger.error%'
                        OR u.source ILIKE '%logger.warn%'
                        OR u.source ILIKE '%logging.error%'
                        OR u.source ILIKE '%logging.exception%'
                        OR u.source ILIKE '%console.error%'
                        OR u.source ILIKE '%traceback%'
                        OR u.source ILIKE '%stack_trace%'
                        OR u.source ILIKE '%stackTrace%'
                        OR u.source ILIKE '%printStackTrace%'
                        -- COBOL-specific error routine patterns
                        OR u.name ILIKE '%ERR-RTN%'
                        OR u.name ILIKE '%ERR-EXIT%'
                        OR u.source ILIKE '%ON EXCEPTION%'
                        OR u.source ILIKE '%NOT ON EXCEPTION%'
                        OR u.source ILIKE '%DISPLAY%ERR%'
                      )
                      AND u.name NOT ILIKE '%ABEND%'
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if err_rows:
            lines += [
                f"## Logging / Diagnostic Patterns in Error Context ({len(err_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in err_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query error patterns: %s", e)
        lines.append(f"Error pattern query unavailable: {e}\n")

    # --- Side effects from deep analyses ---
    try:
        with db.get_session() as session:
            da_rows = session.execute(
                text("SELECT result_json FROM deep_analyses WHERE project_id = :pid"),
                {"pid": pid},
            ).fetchall()

        error_side_effects = []
        for row in da_rows:
            rj = row.result_json if isinstance(row.result_json, dict) else {}
            for se in rj.get("side_effects", []):
                if isinstance(se, dict):
                    se_type = se.get("type", "").lower()
                    if any(kw in se_type for kw in ("error", "exception", "abort", "rollback")):
                        error_side_effects.append(se)

        if error_side_effects:
            lines += [
                f"## Error-Related Side Effects from Deep Analysis ({len(error_side_effects)})",
                "",
                "| Type | Description |",
                "|------|-------------|",
            ]
            seen = set()
            for se in error_side_effects[:50]:
                desc = (se.get("description", "") or "")[:120]
                key = (se.get("type", ""), desc)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"| {se.get('type', '')} | {desc} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to extract error side effects: %s", e)

    if len(lines) <= 3:
        lines.append("No error handling or recovery patterns detected in this codebase.")

    return "\n".join(lines)


# =============================================================================
# Chapter 10: Technology Stack & Platform Profile (NO LLM)
# =============================================================================


def generate_tech_stack(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate technology stack chapter from project info and ground truth."""
    pid = _pid(project_id)

    with db.get_session() as session:
        proj = session.execute(
            text("SELECT name, primary_language, languages, file_count, total_lines "
                 "FROM projects WHERE project_id = :pid"),
            {"pid": pid},
        ).fetchone()

        # Language breakdown
        lang_rows = session.execute(
            text("SELECT language, COUNT(*) AS cnt, SUM(line_count) AS total_lines "
                 "FROM code_files WHERE project_id = :pid "
                 "GROUP BY language ORDER BY cnt DESC"),
            {"pid": pid},
        ).fetchall()

        # Extract frameworks/patterns from deep analyses
        gt_rows = session.execute(
            text("SELECT result_json FROM deep_analyses WHERE project_id = :pid LIMIT 30"),
            {"pid": pid},
        ).fetchall()

    # Extract unique patterns and frameworks
    frameworks = set()
    patterns = set()
    for row in gt_rows:
        rj = row.result_json if isinstance(row.result_json, dict) else {}
        for integ in rj.get("integrations", []):
            if isinstance(integ, dict):
                name = integ.get("name", "")
                itype = integ.get("type", "")
                if name:
                    frameworks.add(name)
                if itype:
                    patterns.add(itype)

    lines = [
        f"# 10. {CHAPTER_TITLES[9]}",
        "",
        f"**Project**: {proj.name if proj else 'Unknown'}",
        f"**Primary Language**: {proj.primary_language if proj else 'N/A'}",
        "",
        "## Language Breakdown",
        "",
        "| Language | Files | Lines |",
        "|----------|-------|-------|",
    ]

    for row in lang_rows:
        lines.append(f"| {row.language or 'unknown'} | {row.cnt} | {row.total_lines or 0} |")

    if proj and proj.languages:
        detected = proj.languages if isinstance(proj.languages, list) else []
        if detected:
            lines += [
                "",
                "## Detected Languages",
                "",
                ", ".join(detected),
            ]

    if frameworks:
        lines += [
            "",
            "## Detected Frameworks & Libraries",
            "",
        ]
        for fw in sorted(frameworks):
            lines.append(f"- {fw}")

    if patterns:
        lines += [
            "",
            "## Detected Integration Patterns",
            "",
        ]
        for pat in sorted(patterns):
            lines.append(f"- {pat}")

    return "\n".join(lines)


# =============================================================================
# Chapter 11: Performance & Scalability Characteristics (NO LLM)
# =============================================================================


def generate_performance_characteristics(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate performance characteristics from complexity, fan-in, batch patterns, and loops."""
    pid = _pid(project_id)

    lines = [
        f"# 11. {CHAPTER_TITLES[10]}",
        "",
    ]

    # --- Complexity hotspots (top 20) -- performance risk indicators ---
    try:
        from ..asg_builder.queries import get_complexity_report
        complexity = get_complexity_report(db, project_id, limit=20, min_complexity=5)

        lines += [
            "## Complexity Hotspots (Performance Risk)",
            "",
        ]
        if complexity:
            lines += [
                "High cyclomatic complexity often correlates with performance-sensitive logic.",
                "",
                "| Name | File | Lines | Complexity |",
                "|------|------|-------|------------|",
            ]
            for item in complexity:
                lines.append(
                    f"| {item['name']} | {item['file_path']} | "
                    f"{item.get('line_count', '?')} | {item['complexity']} |"
                )
        else:
            lines.append("No high-complexity functions detected (threshold: 5).")
        lines.append("")
    except Exception as e:
        logger.warning("Failed to get complexity report for perf chapter: %s", e)
        lines.append(f"Complexity report unavailable: {e}\n")

    # --- Hot modules (highest fan-in) ---
    try:
        from ..asg_builder.queries import get_module_dependency_graph
        mod_graph = get_module_dependency_graph(db, project_id, level="directory", dir_depth=2)
        incoming: Dict[str, int] = defaultdict(int)
        for link in mod_graph.get("links", []):
            incoming[link["target"]] += link.get("weight", 1)

        top_fanin = sorted(incoming.items(), key=lambda x: -x[1])[:15]

        lines += [
            "## Hot Modules (Highest Fan-In)",
            "",
            "Modules with high fan-in are central to the system and may be performance bottlenecks.",
            "",
        ]
        if top_fanin:
            lines += [
                "| Module | Incoming Dependencies |",
                "|--------|----------------------|",
            ]
            for mod, count in top_fanin:
                lines.append(f"| {mod} | {count} |")
        else:
            lines.append("No module dependency data available.")
        lines.append("")
    except Exception as e:
        logger.warning("Failed to get module deps for perf chapter: %s", e)
        lines.append(f"Module dependency analysis unavailable: {e}\n")

    # --- Batch vs online program count ---
    try:
        with db.get_session() as session:
            batch_count = session.execute(
                text("""
                    SELECT COUNT(DISTINCT a.entry_unit_id) AS cnt
                    FROM deep_analyses a
                    WHERE a.project_id = :pid AND a.entry_type ILIKE '%batch%'
                """),
                {"pid": pid},
            ).scalar() or 0

            online_count = session.execute(
                text("""
                    SELECT COUNT(DISTINCT a.entry_unit_id) AS cnt
                    FROM deep_analyses a
                    WHERE a.project_id = :pid
                      AND (a.entry_type ILIKE '%online%' OR a.entry_type ILIKE '%cics%'
                           OR a.entry_type ILIKE '%transaction%')
                """),
                {"pid": pid},
            ).scalar() or 0

            total_ep = session.execute(
                text("SELECT COUNT(DISTINCT entry_unit_id) AS cnt FROM deep_analyses WHERE project_id = :pid"),
                {"pid": pid},
            ).scalar() or 0

        lines += [
            "## Program Classification",
            "",
            f"| Category | Count |",
            f"|----------|-------|",
            f"| Batch programs | {batch_count} |",
            f"| Online/CICS programs | {online_count} |",
            f"| Other/unclassified | {max(0, total_ep - batch_count - online_count)} |",
            f"| **Total entry points** | **{total_ep}** |",
            "",
        ]
    except Exception as e:
        logger.warning("Failed to classify programs: %s", e)
        lines.append(f"Program classification unavailable: {e}\n")

    # --- File I/O patterns (batch processing indicators) ---
    try:
        with db.get_session() as session:
            io_rows = session.execute(
                text("""
                    SELECT u.name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic I/O and data processing patterns
                        u.source ILIKE '%stream%'
                        OR u.source ILIKE '%buffer%'
                        OR u.source ILIKE '%reader%'
                        OR u.source ILIKE '%writer%'
                        OR u.source ILIKE '%batch%'
                        OR u.source ILIKE '%chunk%'
                        OR u.source ILIKE '%paginate%'
                        OR u.source ILIKE '%cursor%'
                        OR u.source ILIKE '%BufferedReader%'
                        OR u.source ILIKE '%InputStream%'
                        OR u.source ILIKE '%OutputStream%'
                        OR u.source ILIKE '%FileReader%'
                        OR u.source ILIKE '%FileWriter%'
                        OR u.name ILIKE '%STREAM%'
                        OR u.name ILIKE '%BUFFER%'
                        OR u.name ILIKE '%READER%'
                        OR u.name ILIKE '%WRITER%'
                        OR u.name ILIKE '%BATCH%'
                        OR u.name ILIKE '%CHUNK%'
                        OR u.name ILIKE '%PAGE%'
                        -- COBOL-specific file I/O patterns
                        OR u.source ILIKE '%OPEN INPUT%'
                        OR u.source ILIKE '%OPEN OUTPUT%'
                        OR u.source ILIKE '%OPEN I-O%'
                        OR u.source ILIKE '%READ %'
                        OR u.source ILIKE '%WRITE %'
                        OR u.source ILIKE '%REWRITE %'
                      )
                    LIMIT 50
                """),
                {"pid": pid},
            ).fetchall()

        if io_rows:
            lines += [
                f"## File I/O Patterns ({len(io_rows)} units)",
                "",
                "Units with I/O, streaming, or batch processing patterns.",
                "",
                "| Unit | File | Line |",
                "|------|------|------|",
            ]
            for row in io_rows[:30]:
                lines.append(f"| {row.name} | {row.file_path} | {row.start_line or ''} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query file I/O patterns: %s", e)

    # --- Loop patterns (PERFORM VARYING, PERFORM UNTIL) ---
    try:
        with db.get_session() as session:
            loop_rows = session.execute(
                text("""
                    SELECT u.name, f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic loop and concurrency patterns
                        u.source ILIKE '%FOR %'
                        OR u.source ILIKE '%WHILE %'
                        OR u.source ILIKE '%forEach%'
                        OR u.source ILIKE '%.map(%'
                        OR u.source ILIKE '%.reduce(%'
                        OR u.source ILIKE '%.filter(%'
                        OR u.source ILIKE '%iterate%'
                        OR u.source ILIKE '%.stream()%'
                        OR u.source ILIKE '%async %'
                        OR u.source ILIKE '%await %'
                        OR u.source ILIKE '%parallel%'
                        OR u.source ILIKE '%concurrent%'
                        OR u.source ILIKE '%thread%'
                        OR u.source ILIKE '%pool%'
                        OR u.source ILIKE '%cache%'
                        OR u.source ILIKE '%@Cacheable%'
                        OR u.source ILIKE '%index%'
                        OR u.name ILIKE '%ASYNC%'
                        OR u.name ILIKE '%PARALLEL%'
                        OR u.name ILIKE '%CONCURRENT%'
                        OR u.name ILIKE '%THREAD%'
                        OR u.name ILIKE '%POOL%'
                        OR u.name ILIKE '%CACHE%'
                        -- COBOL-specific loop patterns
                        OR u.source ILIKE '%PERFORM%VARYING%'
                        OR u.source ILIKE '%PERFORM%UNTIL%'
                        OR u.source ILIKE '%PERFORM%TIMES%'
                      )
                    LIMIT 50
                """),
                {"pid": pid},
            ).fetchall()

        if loop_rows:
            lines += [
                f"## Loop & Concurrency Patterns ({len(loop_rows)} units)",
                "",
                "Units containing iteration, concurrency, caching, or performance-sensitive constructs.",
                "",
                "| Unit | File | Line |",
                "|------|------|------|",
            ]
            for row in loop_rows[:30]:
                lines.append(f"| {row.name} | {row.file_path} | {row.start_line or ''} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query loop patterns: %s", e)

    if len(lines) <= 3:
        lines.append("No performance characteristics data available for this codebase.")

    return "\n".join(lines)


# =============================================================================
# Chapter 12: Technical Debt & Complexity Assessment (NO LLM)
# =============================================================================


def generate_code_quality(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate code quality and risk chapter from complexity, dead code, and module deps."""
    from ..asg_builder.queries import (
        get_complexity_report,
        get_dead_code,
        get_module_dependency_graph,
    )

    lines = [
        f"# 12. {CHAPTER_TITLES[11]}",
        "",
    ]

    # Complexity report (top 20)
    try:
        complexity = get_complexity_report(db, project_id, limit=20, min_complexity=3)
        lines += [
            "## High Complexity Functions (Top 20)",
            "",
            "| Name | File | Lines | Complexity |",
            "|------|------|-------|------------|",
        ]
        for item in complexity:
            lines.append(
                f"| {item['name']} | {item['file_path']} | "
                f"{item.get('line_count', '?')} | {item['complexity']} |"
            )
        if not complexity:
            lines.append("| (none above threshold) | | | |")
        lines.append("")
    except Exception as e:
        logger.warning("Failed to get complexity report: %s", e)
        lines.append(f"Complexity report unavailable: {e}\n")

    # Dead code
    try:
        dead_code = get_dead_code(db, project_id, limit=30)
        lines += [
            "## Potentially Dead Code",
            "",
            f"**Uncalled functions/methods found**: {len(dead_code)}",
            "",
        ]
        if dead_code:
            lines += [
                "| Name | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for item in dead_code[:30]:
                lines.append(
                    f"| {item['name']} | {item['unit_type']} | "
                    f"{item['file_path']} | {item.get('start_line', '')} |"
                )
        lines.append("")
    except Exception as e:
        logger.warning("Failed to get dead code report: %s", e)
        lines.append(f"Dead code analysis unavailable: {e}\n")

    # Most-depended-on modules
    try:
        mod_graph = get_module_dependency_graph(db, project_id, level="directory", dir_depth=2)
        # Count incoming edges per module
        incoming: Dict[str, int] = defaultdict(int)
        for link in mod_graph.get("links", []):
            incoming[link["target"]] += link.get("weight", 1)

        top_deps = sorted(incoming.items(), key=lambda x: -x[1])[:15]

        lines += [
            "## Most-Depended-On Modules",
            "",
            "| Module | Incoming Dependencies |",
            "|--------|----------------------|",
        ]
        for mod, count in top_deps:
            lines.append(f"| {mod} | {count} |")
        if not top_deps:
            lines.append("| (no module dependencies detected) | |")
        lines.append("")
    except Exception as e:
        logger.warning("Failed to get module dependencies: %s", e)
        lines.append(f"Module dependency analysis unavailable: {e}\n")

    return "\n".join(lines)


# =============================================================================
# Chapter 13: Non-Functional Requirements & Compliance (NO LLM)
# =============================================================================


def generate_nfr_compliance(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate NFR and compliance chapter from security, audit, and transaction patterns."""
    pid = _pid(project_id)

    lines = [
        f"# 13. {CHAPTER_TITLES[12]}",
        "",
    ]

    # --- Security patterns ---
    try:
        with db.get_session() as session:
            sec_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic security patterns
                        u.name ILIKE '%AUTH%'
                        OR u.name ILIKE '%LOGIN%'
                        OR u.name ILIKE '%PASSWORD%'
                        OR u.name ILIKE '%ENCRYPT%'
                        OR u.name ILIKE '%DECRYPT%'
                        OR u.name ILIKE '%SECURITY%'
                        OR u.name ILIKE '%TOKEN%'
                        OR u.name ILIKE '%SESSION%'
                        OR u.name ILIKE '%PERMISSION%'
                        OR u.name ILIKE '%ROLE%'
                        OR u.name ILIKE '%RBAC%'
                        OR u.name ILIKE '%CREDENTIAL%'
                        OR u.name ILIKE '%CIPHER%'
                        OR u.name ILIKE '%SECRET%'
                        OR u.name ILIKE '%HASH%'
                        OR u.name ILIKE '%BCRYPT%'
                        OR u.source ILIKE '%authenticate%'
                        OR u.source ILIKE '%authorize%'
                        OR u.source ILIKE '%@PreAuthorize%'
                        OR u.source ILIKE '%@Secured%'
                        OR u.source ILIKE '%oauth%'
                        OR u.source ILIKE '%jwt%'
                        OR u.source ILIKE '%bcrypt%'
                        OR u.source ILIKE '%aes%'
                        OR u.source ILIKE '%rsa%'
                        -- COBOL-specific security patterns
                        OR u.source ILIKE '%RACF%'
                        OR u.source ILIKE '%SIGNON%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if sec_rows:
            lines += [
                f"## Security Patterns ({len(sec_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in sec_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No explicit security-related patterns detected by name.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query security patterns: %s", e)
        lines.append(f"Security pattern query unavailable: {e}\n")

    # --- Audit / logging patterns ---
    try:
        with db.get_session() as session:
            audit_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic audit/observability patterns
                        u.name ILIKE '%AUDIT%'
                        OR u.name ILIKE '%LOG%'
                        OR u.name ILIKE '%TRAIL%'
                        OR u.name ILIKE '%TRACE%'
                        OR u.name ILIKE '%MONITOR%'
                        OR u.name ILIKE '%TELEMETRY%'
                        OR u.name ILIKE '%METRIC%'
                        OR u.name ILIKE '%EVENT%'
                        OR u.name ILIKE '%OBSERVER%'
                        OR u.source ILIKE '%logger%'
                        OR u.source ILIKE '%logging%'
                        OR u.source ILIKE '%sentry%'
                        OR u.source ILIKE '%datadog%'
                        OR u.source ILIKE '%prometheus%'
                        OR u.source ILIKE '%opentelemetry%'
                        -- COBOL-specific audit patterns
                        OR u.name ILIKE '%JOURNAL%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if audit_rows:
            lines += [
                f"## Audit & Logging Patterns ({len(audit_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in audit_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No audit/logging patterns detected by name.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query audit patterns: %s", e)
        lines.append(f"Audit pattern query unavailable: {e}\n")

    # --- Transaction integrity ---
    try:
        with db.get_session() as session:
            txn_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic transaction patterns
                        u.source ILIKE '%COMMIT%'
                        OR u.source ILIKE '%ROLLBACK%'
                        OR u.source ILIKE '%BEGIN TRANSACTION%'
                        OR u.source ILIKE '%@Transactional%'
                        OR u.source ILIKE '%atomic%'
                        OR u.source ILIKE '%savepoint%'
                        OR u.source ILIKE '%session.begin%'
                        OR u.source ILIKE '%transaction.begin%'
                        OR u.source ILIKE '%db.transaction%'
                        OR u.name ILIKE '%TRANSACTION%'
                        OR u.name ILIKE '%COMMIT%'
                        OR u.name ILIKE '%ROLLBACK%'
                        -- COBOL-specific transaction patterns
                        OR u.source ILIKE '%SYNCPOINT%'
                        OR u.source ILIKE '%SYNCPOINT ROLLBACK%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if txn_rows:
            lines += [
                f"## Transaction Integrity Patterns ({len(txn_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in txn_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No transaction integrity patterns detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query transaction patterns: %s", e)
        lines.append(f"Transaction pattern query unavailable: {e}\n")

    # --- PCI / sensitive data indicators ---
    try:
        with db.get_session() as session:
            pci_rows = session.execute(
                text("""
                    SELECT u.name, u.qualified_name, u.unit_type,
                           f.file_path, u.start_line
                    FROM code_units u
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE u.project_id = :pid
                      AND (
                        -- Language-agnostic sensitive data / compliance patterns
                        u.name ILIKE '%SSN%'
                        OR u.name ILIKE '%SOCIAL-SEC%'
                        OR u.name ILIKE '%SOCIAL_SEC%'
                        OR u.name ILIKE '%CREDIT-CARD%'
                        OR u.name ILIKE '%CREDIT_CARD%'
                        OR u.name ILIKE '%CARD-NUM%'
                        OR u.name ILIKE '%CARD_NUM%'
                        OR u.name ILIKE '%ACCOUNT-NUM%'
                        OR u.name ILIKE '%ACCOUNT_NUM%'
                        OR u.name ILIKE '%PCI%'
                        OR u.name ILIKE '%PII%'
                        OR u.name ILIKE '%HIPAA%'
                        OR u.name ILIKE '%GDPR%'
                        OR u.name ILIKE '%SOX%'
                        OR u.name ILIKE '%COMPLIANCE%'
                        OR u.name ILIKE '%CONSENT%'
                        OR u.name ILIKE '%ANONYMIZE%'
                        OR u.name ILIKE '%MASK%'
                        OR u.name ILIKE '%REDACT%'
                        OR u.name ILIKE '%SANITIZE%'
                        OR u.source ILIKE '%gdpr%'
                        OR u.source ILIKE '%pci%'
                        OR u.source ILIKE '%hipaa%'
                        OR u.source ILIKE '%anonymize%'
                        OR u.source ILIKE '%data_mask%'
                        OR u.source ILIKE '%data-mask%'
                      )
                    ORDER BY f.file_path, u.name
                    LIMIT 100
                """),
                {"pid": pid},
            ).fetchall()

        if pci_rows:
            lines += [
                f"## Sensitive Data / Compliance Indicators ({len(pci_rows)})",
                "",
                "| Unit | Type | File | Line |",
                "|------|------|------|------|",
            ]
            for row in pci_rows:
                lines.append(
                    f"| {row.name} | {row.unit_type} | {row.file_path} | "
                    f"{row.start_line or ''} |"
                )
            lines.append("")
        else:
            lines.append("No PCI/PII/GDPR/sensitive data indicators detected.")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to query PCI patterns: %s", e)

    # --- Deep analysis security/audit integrations ---
    try:
        with db.get_session() as session:
            da_rows = session.execute(
                text("SELECT result_json FROM deep_analyses WHERE project_id = :pid"),
                {"pid": pid},
            ).fetchall()

        sec_integrations = []
        for row in da_rows:
            rj = row.result_json if isinstance(row.result_json, dict) else {}
            for integ in rj.get("integrations", []):
                if isinstance(integ, dict):
                    itype = (integ.get("type", "") or "").lower()
                    if any(kw in itype for kw in ("security", "audit", "auth", "compliance")):
                        sec_integrations.append(integ)

        if sec_integrations:
            lines += [
                f"## Security/Audit Integrations from Deep Analysis ({len(sec_integrations)})",
                "",
                "| Type | Name | Description |",
                "|------|------|-------------|",
            ]
            seen = set()
            for integ in sec_integrations[:30]:
                name = integ.get("name", "")
                if name in seen:
                    continue
                seen.add(name)
                desc = (integ.get("description", "") or "")[:100]
                lines.append(f"| {integ.get('type', '')} | {name} | {desc} |")
            lines.append("")
    except Exception as e:
        logger.warning("Failed to extract security integrations: %s", e)

    if len(lines) <= 3:
        lines.append("No non-functional requirement or compliance patterns detected.")

    return "\n".join(lines)


# =============================================================================
# Chapter 14: Architecture Risks & Migration Gaps (LLM synthesis)
# =============================================================================


def generate_risks_and_gaps(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Generate risk assessment and migration gap analysis using LLM synthesis."""
    pid = _pid(project_id)

    # Gather risk indicators from various sources
    risk_data: Dict[str, Any] = {}

    # --- Complexity hotspots ---
    try:
        from ..asg_builder.queries import get_complexity_report
        complexity = get_complexity_report(db, project_id, limit=10, min_complexity=5)
        risk_data["complexity_hotspots"] = [
            {"name": c["name"], "file": c["file_path"], "complexity": c["complexity"]}
            for c in complexity
        ]
    except Exception as e:
        logger.warning("Failed to get complexity for risks: %s", e)
        risk_data["complexity_hotspots"] = []

    # --- Dead code count ---
    try:
        from ..asg_builder.queries import get_dead_code
        dead_code = get_dead_code(db, project_id, limit=100)
        risk_data["dead_code_count"] = len(dead_code)
    except Exception as e:
        logger.warning("Failed to get dead code for risks: %s", e)
        risk_data["dead_code_count"] = 0

    # --- High-coupling modules ---
    try:
        from ..asg_builder.queries import get_module_dependency_graph
        mod_graph = get_module_dependency_graph(db, project_id, level="directory", dir_depth=2)
        incoming: Dict[str, int] = defaultdict(int)
        outgoing: Dict[str, int] = defaultdict(int)
        for link in mod_graph.get("links", []):
            incoming[link["target"]] += link.get("weight", 1)
            outgoing[link["source"]] += link.get("weight", 1)

        # Modules with both high fan-in and fan-out are high coupling risks
        coupling_risks = []
        for mod in set(list(incoming.keys()) + list(outgoing.keys())):
            fi = incoming.get(mod, 0)
            fo = outgoing.get(mod, 0)
            if fi >= 3 or fo >= 5:
                coupling_risks.append({"module": mod, "fan_in": fi, "fan_out": fo})
        coupling_risks.sort(key=lambda x: -(x["fan_in"] + x["fan_out"]))
        risk_data["coupling_risks"] = coupling_risks[:15]
    except Exception as e:
        logger.warning("Failed to get coupling for risks: %s", e)
        risk_data["coupling_risks"] = []

    # --- Side effects and "further analysis needed" from deep analyses ---
    try:
        with db.get_session() as session:
            da_rows = session.execute(
                text("SELECT result_json FROM deep_analyses WHERE project_id = :pid"),
                {"pid": pid},
            ).fetchall()

        side_effects = []
        further_analysis = []
        for row in da_rows:
            rj = row.result_json if isinstance(row.result_json, dict) else {}
            for se in rj.get("side_effects", []):
                if isinstance(se, dict):
                    side_effects.append(se)
            # Check for notes about further analysis
            notes = rj.get("notes", rj.get("further_analysis", []))
            if isinstance(notes, list):
                for note in notes:
                    if isinstance(note, str):
                        further_analysis.append(note)
                    elif isinstance(note, dict):
                        further_analysis.append(note.get("description", str(note)))
            elif isinstance(notes, str) and notes:
                further_analysis.append(notes)

        risk_data["side_effects"] = side_effects[:30]
        risk_data["further_analysis"] = further_analysis[:20]
    except Exception as e:
        logger.warning("Failed to extract side effects for risks: %s", e)
        risk_data["side_effects"] = []
        risk_data["further_analysis"] = []

    # --- Missing analysis coverage ---
    try:
        with db.get_session() as session:
            total_files = session.execute(
                text("SELECT COUNT(*) FROM code_files WHERE project_id = :pid"),
                {"pid": pid},
            ).scalar() or 0

            analyzed_files = session.execute(
                text("""
                    SELECT COUNT(DISTINCT f.file_id)
                    FROM deep_analyses a
                    JOIN code_units u ON a.entry_unit_id = u.unit_id
                    JOIN code_files f ON u.file_id = f.file_id
                    WHERE a.project_id = :pid
                """),
                {"pid": pid},
            ).scalar() or 0

        risk_data["total_files"] = total_files
        risk_data["analyzed_files"] = analyzed_files
        risk_data["coverage_pct"] = round(
            (analyzed_files / total_files * 100) if total_files > 0 else 0, 1
        )
    except Exception as e:
        logger.warning("Failed to compute coverage for risks: %s", e)
        risk_data["total_files"] = 0
        risk_data["analyzed_files"] = 0
        risk_data["coverage_pct"] = 0

    # Build context for LLM
    context_parts = [
        f"Complexity hotspots (complexity >= 5): {len(risk_data['complexity_hotspots'])} functions",
    ]
    for c in risk_data["complexity_hotspots"][:10]:
        context_parts.append(f"  - {c['name']} ({c['file']}): complexity {c['complexity']}")

    context_parts.append(f"\nDead code: {risk_data['dead_code_count']} uncalled functions")

    context_parts.append(f"\nHigh-coupling modules: {len(risk_data['coupling_risks'])}")
    for cr in risk_data["coupling_risks"][:10]:
        context_parts.append(f"  - {cr['module']}: fan-in={cr['fan_in']}, fan-out={cr['fan_out']}")

    context_parts.append(
        f"\nAnalysis coverage: {risk_data['analyzed_files']}/{risk_data['total_files']} files "
        f"({risk_data['coverage_pct']}%)"
    )

    if risk_data["side_effects"]:
        context_parts.append(f"\nSide effects detected: {len(risk_data['side_effects'])}")
        seen_types = set()
        for se in risk_data["side_effects"][:15]:
            se_type = se.get("type", "unknown")
            desc = (se.get("description", "") or "")[:80]
            key = (se_type, desc)
            if key not in seen_types:
                seen_types.add(key)
                context_parts.append(f"  - [{se_type}] {desc}")

    if risk_data["further_analysis"]:
        context_parts.append(f"\nFurther analysis recommended:")
        for fa in risk_data["further_analysis"][:10]:
            context_parts.append(f"  - {fa[:120]}")

    context = "\n".join(context_parts)

    prompt = f"""You are a software architect writing a reverse engineering document.
Based on the following risk indicators and analysis gaps, write Chapter 14: {CHAPTER_TITLES[13]}.

Include:
1. A risk assessment table with severity levels (Critical / High / Medium / Low)
2. Migration risks: what could go wrong during modernization
3. Unknowns: areas where analysis coverage is insufficient
4. Recommended further analysis: specific areas needing deeper investigation
5. Mitigation strategies for the top risks

Write in markdown. Start with "# 14. {CHAPTER_TITLES[13]}". Be concise but thorough.
Base all findings on the provided data. Do not invent risks not supported by evidence.

Risk Intelligence:
{context}
"""

    try:
        from llama_index.core import Settings as LISettings
        response = LISettings.llm.complete(prompt)
        return str(response)
    except Exception as e:
        logger.error("LLM call failed for risks and gaps: %s", e)
        # Fallback: data-only risk list
        lines = [
            f"# 14. {CHAPTER_TITLES[13]}",
            "",
            "## Risk Summary",
            "",
            "| Category | Indicator | Count / Value |",
            "|----------|-----------|---------------|",
            f"| Complexity | High-complexity functions (>= 5) | {len(risk_data['complexity_hotspots'])} |",
            f"| Dead Code | Uncalled functions | {risk_data['dead_code_count']} |",
            f"| Coupling | High-coupling modules | {len(risk_data['coupling_risks'])} |",
            f"| Coverage | Files analyzed | {risk_data['analyzed_files']}/{risk_data['total_files']} ({risk_data['coverage_pct']}%) |",
            f"| Side Effects | Detected | {len(risk_data['side_effects'])} |",
            "",
        ]

        if risk_data["complexity_hotspots"]:
            lines += [
                "## Complexity Hotspots",
                "",
                "| Name | File | Complexity |",
                "|------|------|------------|",
            ]
            for c in risk_data["complexity_hotspots"]:
                lines.append(f"| {c['name']} | {c['file']} | {c['complexity']} |")
            lines.append("")

        if risk_data["coupling_risks"]:
            lines += [
                "## High-Coupling Modules",
                "",
                "| Module | Fan-In | Fan-Out |",
                "|--------|--------|---------|",
            ]
            for cr in risk_data["coupling_risks"]:
                lines.append(f"| {cr['module']} | {cr['fan_in']} | {cr['fan_out']} |")
            lines.append("")

        if risk_data["further_analysis"]:
            lines += [
                "## Recommended Further Analysis",
                "",
            ]
            for fa in risk_data["further_analysis"]:
                lines.append(f"- {fa[:150]}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Chapter dispatch
# =============================================================================

CHAPTER_GENERATORS = {
    1: generate_executive_summary,
    2: generate_architecture_overview,
    3: generate_entry_points,
    4: generate_functional_requirements,
    5: generate_data_model,
    6: generate_database_architecture,         # NEW
    7: generate_call_trees,                    # was 6
    8: generate_integrations,                  # was 7
    9: generate_error_handling,                # NEW
    10: generate_tech_stack,                   # was 9
    11: generate_performance_characteristics,  # NEW
    12: generate_code_quality,                 # was 8
    13: generate_nfr_compliance,               # NEW
    14: generate_risks_and_gaps,               # NEW
}
