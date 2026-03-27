"""Mainframe-specific chapter overrides for reverse engineering documentation.

When the project is predominantly COBOL/JCL/PL1, these generators replace the
generic chapters with deep, mainframe-aware functional specifications:
  - Per-program paragraph inventories, COMMAREA mappings, file access matrices
  - CICS transaction maps with BMS screen navigation
  - Business rules extracted with COBOL code evidence
  - Copybook layouts and fan-in analysis
  - VSAM file access matrices across all programs
  - PERFORM hierarchy trees
  - CICS command inventories and ABEND patterns

Each override function has the same signature as the generic generators:
    (db: DatabaseManager, project_id: str, pipeline=None) -> str
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text

from ..db import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pid(project_id: str) -> UUID:
    return UUID(project_id) if isinstance(project_id, str) else project_id


def _safe_source(row) -> str:
    """Return source text or empty string if NULL."""
    return row.source or "" if hasattr(row, "source") else ""


def _safe_meta(row) -> dict:
    """Return metadata dict, handling NULL/string/dict."""
    raw = getattr(row, "metadata", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _fetch_programs(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all program-level code_units for the project."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT u.unit_id, u.name, u.qualified_name, u.source,
                       u.unit_type, u.start_line, u.end_line,
                       f.file_path, u.metadata
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.project_id = :pid AND u.unit_type = 'program'
                ORDER BY u.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_paragraphs(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all paragraphs linked via 'contains' edges."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT cu.unit_id, cu.name, cu.qualified_name, cu.source,
                       cu.start_line, cu.end_line,
                       parent.name AS parent_name, parent.unit_id AS parent_unit_id
                FROM code_edges e
                JOIN code_units cu ON e.target_unit_id = cu.unit_id
                JOIN code_units parent ON e.source_unit_id = parent.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'contains'
                  AND cu.unit_type = 'paragraph'
                ORDER BY parent.name, cu.start_line
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_calls(db: DatabaseManager, pid: UUID) -> list:
    """Fetch call edges (paragraph-to-paragraph and program-to-program)."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS caller, tu.name AS callee,
                       su.qualified_name AS caller_qn, tu.qualified_name AS callee_qn,
                       su.unit_type AS caller_type, tu.unit_type AS callee_type
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'calls'
                ORDER BY su.name, tu.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_imports(db: DatabaseManager, pid: UUID) -> list:
    """Fetch import edges (program -> copybook)."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT su.name AS program, tu.name AS copybook,
                       su.unit_id AS prog_unit_id, tu.unit_id AS copy_unit_id
                FROM code_edges e
                JOIN code_units su ON e.source_unit_id = su.unit_id
                JOIN code_units tu ON e.target_unit_id = tu.unit_id
                WHERE e.project_id = :pid AND e.edge_type = 'imports'
                ORDER BY su.name, tu.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_copybooks(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all copybook code_units."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT u.unit_id, u.name, u.qualified_name, u.source,
                       u.start_line, u.end_line, f.file_path, u.metadata
                FROM code_units u
                JOIN code_files f ON u.file_id = f.file_id
                WHERE u.project_id = :pid AND u.unit_type = 'copybook'
                ORDER BY u.name
            """),
            {"pid": pid},
        ).fetchall()


def _fetch_deep_analyses(db: DatabaseManager, pid: UUID) -> list:
    """Fetch all deep analysis results for the project."""
    with db.get_session() as session:
        return session.execute(
            text("""
                SELECT a.analysis_id, a.entry_type, a.result_json, a.narrative,
                       a.confidence_score, a.coverage_pct,
                       u.name AS entry_name, u.unit_id AS entry_unit_id,
                       f.file_path
                FROM deep_analyses a
                JOIN code_units u ON a.entry_unit_id = u.unit_id
                JOIN code_files f ON u.file_id = f.file_id
                WHERE a.project_id = :pid
                ORDER BY u.name
            """),
            {"pid": pid},
        ).fetchall()


# ---------------------------------------------------------------------------
# Source-text extraction helpers
# ---------------------------------------------------------------------------


def _extract_cics_commands(source: str) -> List[Dict[str, str]]:
    """Extract EXEC CICS commands from COBOL source."""
    results = []
    for m in re.finditer(
        r"EXEC\s+CICS\s+(\w+)(?:\s+([^.]*?))?(?:\s*END-EXEC|\s*\.)",
        source, re.IGNORECASE | re.DOTALL,
    ):
        cmd = m.group(1).upper()
        args = (m.group(2) or "").strip()
        # Extract resource name from common patterns
        resource = ""
        res_match = re.search(
            r"(?:DATASET|FILE|MAP|MAPSET|QUEUE|TRANSID|PROGRAM)\s*\(\s*(\S+?)\s*\)",
            args, re.IGNORECASE,
        )
        if res_match:
            resource = res_match.group(1).strip("'\"")
        results.append({"command": cmd, "args": args[:120], "resource": resource})
    return results


def _extract_file_operations(source: str) -> List[str]:
    """Extract file I/O verbs from COBOL source."""
    ops = []
    for op in [
        "OPEN", "READ", "WRITE", "REWRITE", "DELETE", "CLOSE",
        "STARTBR", "READNEXT", "READPREV", "ENDBR", "START",
    ]:
        if re.search(rf"\b{op}\b", source, re.IGNORECASE):
            ops.append(op)
    return ops


def _extract_map_info(source: str) -> Dict[str, Optional[str]]:
    """Extract BMS MAP/MAPSET from source."""
    map_m = re.search(r"MAP\s*\(\s*'?(\w+)'?\s*\)", source, re.IGNORECASE)
    mapset_m = re.search(r"MAPSET\s*\(\s*'?(\w+)'?\s*\)", source, re.IGNORECASE)
    return {
        "map": map_m.group(1) if map_m else None,
        "mapset": mapset_m.group(1) if mapset_m else None,
    }


def _extract_commarea_fields(source: str) -> List[str]:
    """Extract COMMAREA field references (CDEMO-* pattern)."""
    return sorted(set(re.findall(r"(CDEMO-[\w-]+)", source)))


def _extract_dataset_names(source: str) -> List[str]:
    """Extract DATASET(...) references from CICS commands."""
    return list(set(
        m.group(1).strip("'\"")
        for m in re.finditer(
            r"DATASET\s*\(\s*([^)]+)\s*\)", source, re.IGNORECASE
        )
    ))


def _classify_program(meta: dict, source: str) -> str:
    """Classify program as batch/cics_online/utility/subroutine."""
    if meta:
        cat = meta.get("program_category", "")
        if cat:
            # Normalize: batch_program -> batch, cics_online stays
            if "batch" in cat:
                return "batch"
            if "cics" in cat:
                return "cics_online"
            return cat
    if re.search(r"EXEC\s+CICS", source, re.IGNORECASE):
        return "cics_online"
    return "batch"


def _extract_transaction_id(meta: dict, source: str, narrative: str = "") -> Optional[str]:
    """Extract CICS transaction ID from metadata, source, or deep analysis narrative.

    Strategy:
      1. Check metadata for explicit transaction_id
      2. Look for literal TRANSID('XXXX') in RETURN command
      3. Look for WS-TRANID variable and find its VALUE clause
      4. Parse from deep analysis narrative (e.g. "TRANSID CT00")
    """
    if meta:
        tid = meta.get("transaction_id")
        if tid:
            return tid

    # Strategy 2: literal in EXEC CICS RETURN TRANSID('XXXX')
    m = re.search(r"TRANSID\s*\(\s*'(\w+)'\s*\)", source, re.IGNORECASE)
    if m:
        return m.group(1)

    # Strategy 3: variable-based -- find TRANSID(WS-VAR) then VALUE of WS-VAR
    var_match = re.search(r"TRANSID\s*\(\s*(\w[\w-]*)\s*\)", source, re.IGNORECASE)
    if var_match:
        var_name = var_match.group(1)
        # Search for VALUE clause in working storage
        val_match = re.search(
            rf"{re.escape(var_name)}\s+PIC\s+\S+\s+VALUE\s+'(\w+)'",
            source, re.IGNORECASE,
        )
        if val_match:
            return val_match.group(1)

    # Strategy 4: parse from deep analysis narrative
    if narrative:
        # Patterns like "(CT00)", "TRANSID CT00", "transaction CT00"
        tid_match = re.search(
            r"(?:TRANSID|transaction)\s+(\w{2,4})\b",
            narrative, re.IGNORECASE,
        )
        if tid_match:
            return tid_match.group(1)
        # Pattern like "(XXXX)" after "program" or "transaction"
        paren_match = re.search(r"\(([A-Z][A-Z0-9]{1,3})\)", narrative)
        if paren_match:
            return paren_match.group(1)

    return None


# ---------------------------------------------------------------------------
# Chapter 1 Override: Program Inventory & System Classification
# ---------------------------------------------------------------------------


def generate_program_inventory(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Per-program catalog with type, transaction, files accessed, paragraph counts."""
    pid = _pid(project_id)
    lines = ["# 1. Program Inventory & System Classification\n"]

    try:
        programs = _fetch_programs(db, pid)
        paragraphs = _fetch_paragraphs(db, pid)
        imports = _fetch_imports(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        # Build indexes
        para_count: Dict[str, int] = defaultdict(int)
        for p in paragraphs:
            para_count[p.parent_name] += 1

        imports_by_prog: Dict[str, List[str]] = defaultdict(list)
        for imp in imports:
            imports_by_prog[imp.program].append(imp.copybook)

        # Build analysis lookup for file access and narrative
        analysis_by_name: Dict[str, dict] = {}
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else json.loads(a.result_json)
            analysis_by_name[a.entry_name] = {
                "result_json": rj,
                "narrative": a.narrative or "",
            }

        # Classify programs
        batch_progs = []
        cics_progs = []
        utility_progs = []
        sub_progs = []

        catalog_rows = []
        for prog in programs:
            source = _safe_source(prog)
            meta = _safe_meta(prog)

            category = _classify_program(meta, source)
            ainfo = analysis_by_name.get(prog.name, {})
            narrative = ainfo.get("narrative", "")
            trans_id = _extract_transaction_id(meta, source, narrative) or "N/A"
            datasets = _extract_dataset_names(source)
            n_paras = para_count.get(prog.name, 0)

            # Merge in deep analysis integrations for file names
            rj = ainfo.get("result_json", {})
            for integ in rj.get("integrations", []):
                desc = integ.get("description", "")
                ds_match = re.findall(r"(?:dataset|file)\s+(\w+)", desc, re.IGNORECASE)
                datasets.extend(ds_match)
            datasets = sorted(set(d for d in datasets if d))

            catalog_rows.append({
                "name": prog.name,
                "category": category,
                "trans_id": trans_id,
                "file_path": prog.file_path,
                "datasets": ", ".join(datasets) if datasets else "--",
                "n_paras": n_paras,
            })

            bucket = {
                "batch": batch_progs,
                "cics_online": cics_progs,
                "utility": utility_progs,
                "subroutine": sub_progs,
            }.get(category, utility_progs)
            bucket.append(prog.name)

        # Catalog table
        lines += [
            "## Program Catalog\n",
            "| Program | Type | Transaction | Source File | Files Accessed | Paragraphs |",
            "|---------|------|-------------|------------|---------------|------------|",
        ]
        for row in catalog_rows:
            lines.append(
                f"| {row['name']} | {row['category']} | {row['trans_id']} "
                f"| `{row['file_path']}` | {row['datasets']} | {row['n_paras']} |"
            )
        lines.append("")

        # Processing pattern classification
        lines.append("## Processing Pattern Classification\n")
        if batch_progs:
            lines.append(f"- **Batch** ({len(batch_progs)}): {', '.join(sorted(batch_progs))}")
        if cics_progs:
            lines.append(f"- **CICS Online** ({len(cics_progs)}): {', '.join(sorted(cics_progs))}")
        if utility_progs:
            lines.append(f"- **Utility** ({len(utility_progs)}): {', '.join(sorted(utility_progs))}")
        if sub_progs:
            lines.append(f"- **Subroutine** ({len(sub_progs)}): {', '.join(sorted(sub_progs))}")
        lines.append("")

        # Summary stats
        lines += [
            "## Summary\n",
            f"- **Total programs**: {len(programs)}",
            f"- **Total paragraphs**: {sum(para_count.values())}",
            f"- **Copybooks referenced**: {len(set(i.copybook for i in imports))}",
            f"- **Deep analyses available**: {len(analyses)}",
            "",
        ]

    except Exception as e:
        logger.error("Chapter 1 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 3 Override: CICS Transaction Map & Entry Points
# ---------------------------------------------------------------------------


def generate_cics_transaction_map(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Transaction-to-program-to-screen mapping with navigation flow."""
    pid = _pid(project_id)
    lines = ["# 3. CICS Transaction Map & Entry Points\n"]

    try:
        programs = _fetch_programs(db, pid)
        calls = _fetch_calls(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        # Build analysis lookup
        analysis_by_name: Dict[str, dict] = {}
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else json.loads(a.result_json)
            analysis_by_name[a.entry_name] = {
                "entry_type": a.entry_type,
                "narrative": a.narrative or "",
                "result_json": rj,
            }

        # Online programs (CICS)
        online_rows = []
        batch_rows = []

        for prog in programs:
            source = _safe_source(prog)
            meta = _safe_meta(prog)

            category = _classify_program(meta, source)
            ainfo = analysis_by_name.get(prog.name, {})
            narrative = ainfo.get("narrative", "")
            trans_id = _extract_transaction_id(meta, source, narrative)
            map_info = _extract_map_info(source)

            # Extract PF-key navigation from source
            pf_keys = []
            for m in re.finditer(
                r"DFHPF(\d+)",
                source, re.IGNORECASE,
            ):
                # Find what action follows this PF key check
                after = source[m.end():m.end() + 300]
                action_m = re.search(
                    r"(?:MOVE\s+'(\w+)'|PERFORM\s+(\w[\w-]*))",
                    after, re.IGNORECASE,
                )
                if action_m:
                    target = action_m.group(1) or action_m.group(2)
                    pf_keys.append(f"PF{m.group(1)}={target}")

            pf_str = ", ".join(sorted(set(pf_keys))[:5]) if pf_keys else "--"

            # Narrative snippet from deep analysis
            desc = narrative[:80].replace("\n", " ") if narrative else "--"

            if category == "cics_online":
                online_rows.append({
                    "trans_id": trans_id or "--",
                    "program": prog.name,
                    "map": map_info.get("map") or "--",
                    "mapset": map_info.get("mapset") or "--",
                    "desc": desc,
                    "pf_keys": pf_str,
                })
            elif category == "batch":
                # Batch entry points
                input_files = []
                output_files = []
                datasets = _extract_dataset_names(source)

                # For batch programs, also check OPEN INPUT/OUTPUT
                for om in re.finditer(
                    r"OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([\w-]+)",
                    source, re.IGNORECASE,
                ):
                    mode = om.group(1).upper()
                    fname = om.group(2)
                    if mode == "INPUT":
                        input_files.append(fname)
                    elif mode in ("OUTPUT", "EXTEND"):
                        output_files.append(fname)
                    else:
                        input_files.append(fname)
                        output_files.append(fname)

                # Fallback to CICS dataset analysis
                if not input_files and not output_files:
                    for ds in datasets:
                        if re.search(rf"READ\s+{re.escape(ds)}", source, re.IGNORECASE):
                            input_files.append(ds)
                        if re.search(rf"WRITE\s+{re.escape(ds)}", source, re.IGNORECASE):
                            output_files.append(ds)
                        if ds not in input_files and ds not in output_files:
                            input_files.append(ds)

                batch_rows.append({
                    "program": prog.name,
                    "input": ", ".join(sorted(set(input_files))) if input_files else "--",
                    "output": ", ".join(sorted(set(output_files))) if output_files else "--",
                    "desc": desc,
                })

        # Transaction -> Program -> Screen table
        if online_rows:
            lines += [
                "## Transaction -> Program -> Screen Mapping\n",
                "| Transaction ID | Program | BMS Map | Mapset | Description | PF-Key Nav |",
                "|---------------|---------|---------|--------|-------------|-----------|",
            ]
            for row in sorted(online_rows, key=lambda r: r["trans_id"]):
                lines.append(
                    f"| {row['trans_id']} | {row['program']} | {row['map']} "
                    f"| {row['mapset']} | {row['desc']} | {row['pf_keys']} |"
                )
            lines.append("")

        # Batch job entry points
        if batch_rows:
            lines += [
                "## Batch Job Entry Points\n",
                "| Program | Input Files | Output Files | Description |",
                "|---------|-------------|-------------|-------------|",
            ]
            for row in sorted(batch_rows, key=lambda r: r["program"]):
                lines.append(
                    f"| {row['program']} | {row['input']} | {row['output']} | {row['desc']} |"
                )
            lines.append("")

        # Screen navigation flow (derived from XCTL/RETURN TRANSID patterns)
        lines.append("## Screen Navigation Flow\n")
        lines.append("```")

        # Build transfer graph from source XCTL analysis
        xctl_graph: Dict[str, List[str]] = defaultdict(list)
        for prog in programs:
            source = _safe_source(prog)
            for m in re.finditer(
                r"EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*'?(\w+)'?\s*\)",
                source, re.IGNORECASE | re.DOTALL,
            ):
                target = m.group(1)
                xctl_graph[prog.name].append(target)
            # Also handle variable-based XCTL: PROGRAM(WS-VAR) -> find VALUE
            for m in re.finditer(
                r"EXEC\s+CICS\s+XCTL\s+PROGRAM\s*\(\s*(\w[\w-]*)\s*\)",
                source, re.IGNORECASE | re.DOTALL,
            ):
                var = m.group(1)
                if var.startswith("'"):
                    continue  # already handled
                # Find MOVE 'PROGNAME' TO var
                for mv in re.finditer(
                    rf"MOVE\s+'(\w+)'\s+TO\s+{re.escape(var)}",
                    source, re.IGNORECASE,
                ):
                    xctl_graph[prog.name].append(mv.group(1))

            # Pseudo-conversational RETURN TRANSID
            meta = _safe_meta(prog)
            ainfo = analysis_by_name.get(prog.name, {})
            tid = _extract_transaction_id(meta, source, ainfo.get("narrative", ""))
            if tid:
                xctl_graph[prog.name].append(f"[RETURN TRANSID({tid})]")

        # Render as tree starting from root programs
        rendered: Set[str] = set()

        def _render_nav(prog_name: str, indent: int = 0) -> None:
            prefix = "    " * indent + ("+--> " if indent > 0 else "")
            lines.append(f"{prefix}{prog_name}")
            if prog_name in rendered:
                return
            rendered.add(prog_name)
            for target in sorted(set(xctl_graph.get(prog_name, []))):
                if target.startswith("["):
                    lines.append(f"{'    ' * (indent + 1)}+--> {target}")
                else:
                    _render_nav(target, indent + 1)

        # Find root programs (not targeted by XCTL from others)
        all_targets = set()
        for targets in xctl_graph.values():
            all_targets.update(t for t in targets if not t.startswith("["))
        prog_names_set = {p.name for p in programs}
        roots = [p.name for p in programs
                 if p.name not in all_targets
                 and _classify_program(_safe_meta(p), _safe_source(p)) == "cics_online"]
        if not roots:
            roots = sorted(xctl_graph.keys())[:3]

        for root in sorted(roots):
            _render_nav(root)

        lines.append("```\n")

    except Exception as e:
        logger.error("Chapter 3 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 4 Override: Per-Program Business Rules with Code Evidence
# ---------------------------------------------------------------------------


def generate_per_program_specs(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Per-program functional specifications with paragraph inventories,
    business rules with COBOL evidence, file access, and COMMAREA mappings."""
    pid = _pid(project_id)
    lines = ["# 4. Business Rules & Functional Specifications\n"]
    lines.append(
        "Each program below includes: purpose, files accessed, business rules "
        "with COBOL code evidence, paragraph inventory, and COMMAREA mappings.\n"
    )

    try:
        programs = _fetch_programs(db, pid)
        paragraphs = _fetch_paragraphs(db, pid)
        calls = _fetch_calls(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        # Index paragraphs by parent program
        paras_by_prog: Dict[str, list] = defaultdict(list)
        for p in paragraphs:
            paras_by_prog[p.parent_name].append(p)

        # Index calls: caller -> [callee]
        calls_by_caller: Dict[str, List[str]] = defaultdict(list)
        for c in calls:
            calls_by_caller[c.caller].append(c.callee)

        # Index deep analyses by entry name
        analysis_by_name: Dict[str, dict] = {}
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else json.loads(a.result_json)
            analysis_by_name[a.entry_name] = {
                "entry_type": a.entry_type,
                "narrative": a.narrative or "",
                "result_json": rj,
                "confidence": a.confidence_score,
            }

        for prog in programs:
            source = _safe_source(prog)
            meta = _safe_meta(prog)

            category = _classify_program(meta, source)
            ainfo = analysis_by_name.get(prog.name, {})
            rj = ainfo.get("result_json", {})
            narrative = ainfo.get("narrative", "")

            lines.append(f"---\n\n## {prog.name} -- {category.replace('_', ' ').title()}\n")

            # Purpose from deep analysis narrative
            if narrative:
                purpose_line = narrative.split("\n")[0][:300]
                lines.append(f"### Purpose\n\n{purpose_line}\n")
            else:
                lines.append(f"### Purpose\n\n{category.replace('_', ' ').title()} program in `{prog.file_path}`.\n")

            # --- Files Accessed ---
            lines.append("### Files Accessed\n")
            datasets = _extract_dataset_names(source)
            cics_cmds = _extract_cics_commands(source)
            file_ops_global = _extract_file_operations(source)

            if datasets or any(c["command"] in ("READ", "WRITE", "REWRITE", "DELETE", "STARTBR", "READNEXT") for c in cics_cmds):
                lines += [
                    "| File | Access Mode | Operations |",
                    "|------|-----------|-----------|",
                ]
                # Determine per-dataset operations from CICS commands
                ds_ops: Dict[str, Set[str]] = defaultdict(set)
                for cmd in cics_cmds:
                    if cmd["command"] in ("READ", "READNEXT", "READPREV", "STARTBR", "ENDBR"):
                        if cmd["resource"]:
                            ds_ops[cmd["resource"]].add("Read")
                    elif cmd["command"] in ("WRITE",):
                        if cmd["resource"]:
                            ds_ops[cmd["resource"]].add("Write")
                    elif cmd["command"] in ("REWRITE",):
                        if cmd["resource"]:
                            ds_ops[cmd["resource"]].add("Rewrite")
                    elif cmd["command"] in ("DELETE",):
                        if cmd["resource"]:
                            ds_ops[cmd["resource"]].add("Delete")

                # For datasets found in source that aren't in CICS commands
                for ds in datasets:
                    if ds not in ds_ops:
                        ds_ops[ds] = set(file_ops_global) if file_ops_global else {"Access"}

                # Also detect batch file operations (OPEN INPUT/OUTPUT)
                for om in re.finditer(
                    r"OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([\w-]+)",
                    source, re.IGNORECASE,
                ):
                    mode = om.group(1).upper()
                    fname = om.group(2)
                    if mode == "INPUT":
                        ds_ops[fname].add("Read")
                    elif mode in ("OUTPUT", "EXTEND"):
                        ds_ops[fname].add("Write")
                    else:
                        ds_ops[fname].add("Read")
                        ds_ops[fname].add("Write")

                for ds_name in sorted(ds_ops.keys()):
                    ops = sorted(ds_ops[ds_name])
                    mode = "Read" if ops == ["Read"] else "/".join(ops)
                    lines.append(f"| {ds_name} | {mode} | {', '.join(ops)} |")
                lines.append("")
            else:
                lines.append("No file access detected.\n")

            # --- Integrations from deep analysis ---
            integrations = rj.get("integrations", [])
            if integrations and not datasets:
                lines.append("### Integration Points\n")
                for integ in integrations:
                    desc = integ.get("description", "")
                    lines.append(f"- {desc[:200]}")
                lines.append("")

            # --- Business Rules with Code Evidence ---
            lines.append("### Business Rules\n")
            br_list = rj.get("business_rules", [])
            if br_list:
                for idx, br in enumerate(br_list, 1):
                    severity = br.get("severity", "")
                    sev_badge = f" [{severity}]" if severity else ""
                    desc = br.get("description", "No description")
                    lines.append(f"**BR-{idx:03d}{sev_badge}**: {desc}\n")

                    # Code evidence
                    evidence_list = br.get("evidence", [])
                    for ev in evidence_list:
                        snippet = ev.get("snippet", "")
                        qn = ev.get("qualified_name", "")
                        loc = ""
                        if ev.get("file_path") and ev.get("start_line"):
                            loc = f" (`{ev['file_path']}:{ev['start_line']}-{ev.get('end_line', '')}`)"
                        if qn:
                            lines.append(f"*Source: {qn}{loc}*\n")
                        if snippet:
                            snippet_trimmed = snippet.strip()
                            if len(snippet_trimmed) > 600:
                                snippet_trimmed = snippet_trimmed[:600] + "\n  ..."
                            lines.append(f"```cobol\n{snippet_trimmed}\n```\n")
                lines.append("")
            else:
                # Fallback: extract IF/EVALUATE from source paragraphs
                prog_paras = paras_by_prog.get(prog.name, [])
                br_idx = 0
                for para in prog_paras:
                    para_src = _safe_source(para)
                    if not para_src:
                        continue
                    for m in re.finditer(
                        r"(IF\s+.+?)(?:\n\s{7,}(?:ELSE|END-IF|PERFORM|MOVE|DISPLAY)|\Z)",
                        para_src, re.IGNORECASE | re.DOTALL,
                    ):
                        br_idx += 1
                        condition = m.group(1).strip().replace("\n", "\n  ")[:300]
                        lines.append(f"**BR-{br_idx:03d}** ({para.name}):\n")
                        lines.append(f"```cobol\n{condition}\n```\n")
                        if br_idx >= 20:
                            break
                    if br_idx >= 20:
                        lines.append(f"*... truncated ({len(prog_paras)} paragraphs total)*\n")
                        break

                if br_idx == 0:
                    lines.append("No business rules extracted for this program.\n")

            # --- Paragraph Inventory ---
            prog_paras = paras_by_prog.get(prog.name, [])
            if prog_paras:
                lines.append("### Paragraph Inventory\n")
                lines += [
                    "| # | Paragraph | Line | Calls |",
                    "|---|-----------|------|-------|",
                ]
                for idx, para in enumerate(prog_paras, 1):
                    callees = calls_by_caller.get(para.name, [])
                    callees_str = ", ".join(callees[:6])
                    if len(callees) > 6:
                        callees_str += f" (+{len(callees) - 6})"
                    lines.append(
                        f"| {idx} | {para.name} | {para.start_line or '--'} "
                        f"| {callees_str or '--'} |"
                    )
                lines.append("")

            # --- COMMAREA Fields ---
            commarea = _extract_commarea_fields(source)
            if commarea:
                lines.append("### COMMAREA Fields Referenced\n")
                lines += [
                    "| Field | Usage Context |",
                    "|-------|--------------|",
                ]
                for field in commarea[:30]:
                    ctx_match = re.search(
                        rf"{re.escape(field)}\s+(?:TO|FROM|=|OF)\s+(\S+)",
                        source, re.IGNORECASE,
                    )
                    ctx = ctx_match.group(0)[:60] if ctx_match else "--"
                    lines.append(f"| {field} | `{ctx}` |")
                lines.append("")

            lines.append("")  # spacing between programs

    except Exception as e:
        logger.error("Chapter 4 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 5 Override: Copybook Layouts & Data Contracts
# ---------------------------------------------------------------------------


def generate_copybook_layouts(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Copybook field layouts, COMMAREA structures, and fan-in analysis."""
    pid = _pid(project_id)
    lines = ["# 5. Copybook Layouts & Data Contracts\n"]

    try:
        copybooks = _fetch_copybooks(db, pid)
        imports = _fetch_imports(db, pid)

        # Fan-in: how many programs use each copybook
        fanin: Dict[str, List[str]] = defaultdict(list)
        for imp in imports:
            fanin[imp.copybook].append(imp.program)

        for cb in copybooks:
            source = _safe_source(cb)
            users = sorted(set(fanin.get(cb.name, [])))

            lines.append(f"## {cb.name}\n")
            lines.append(f"**Source**: `{cb.file_path}`  ")
            lines.append(f"**Fan-in**: {len(users)} program(s): {', '.join(users) if users else 'none'}\n")

            if source:
                # Extract field definitions: level number, field name, PIC clause
                fields = []
                for m in re.finditer(
                    r"^\s*(\d{2})\s+([\w-]+)\s*(?:PIC(?:TURE)?\s+([\w()V9SX.,+-]+))?",
                    source, re.MULTILINE | re.IGNORECASE,
                ):
                    level = m.group(1)
                    name = m.group(2)
                    pic = m.group(3) or ""
                    fields.append({"level": level, "name": name, "pic": pic})

                if fields:
                    lines += [
                        "| Level | Field | PIC |",
                        "|-------|-------|-----|",
                    ]
                    for f in fields[:60]:
                        indent = "  " * max(0, (int(f["level"]) // 5) - 1) if f["level"].isdigit() else ""
                        lines.append(f"| {f['level']} | {indent}{f['name']} | {f['pic']} |")
                    if len(fields) > 60:
                        lines.append(f"| ... | *{len(fields) - 60} more fields* | |")
                    lines.append("")
                else:
                    trimmed = source.strip()[:1500]
                    lines.append(f"```cobol\n{trimmed}\n```\n")
            else:
                lines.append("*Source not available.*\n")

            lines.append("")

        # Copybook usage matrix
        if fanin:
            lines += [
                "## Copybook Usage Matrix\n",
                "| Copybook | Fan-in | Used By Programs |",
                "|----------|--------|-----------------|",
            ]
            for cb_name in sorted(fanin.keys(), key=lambda k: -len(fanin[k])):
                progs = sorted(set(fanin[cb_name]))
                progs_str = ", ".join(progs[:8])
                if len(progs) > 8:
                    progs_str += f" (+{len(progs) - 8})"
                lines.append(f"| {cb_name} | {len(progs)} | {progs_str} |")
            lines.append("")

    except Exception as e:
        logger.error("Chapter 5 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 6 Override: VSAM File Access Matrix
# ---------------------------------------------------------------------------


def generate_vsam_access_matrix(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Cross-program file access matrix showing R/W/RW per program per dataset."""
    pid = _pid(project_id)
    lines = ["# 6. VSAM File Access Matrix\n"]

    try:
        programs = _fetch_programs(db, pid)

        # Collect per-program, per-dataset operations
        matrix: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        dataset_keys: Dict[str, str] = {}

        for prog in programs:
            source = _safe_source(prog)
            if not source:
                continue

            cics_cmds = _extract_cics_commands(source)
            for cmd in cics_cmds:
                resource = cmd["resource"]
                if not resource:
                    continue
                if cmd["command"] in ("READ", "READNEXT", "READPREV", "STARTBR", "ENDBR"):
                    matrix[resource][prog.name].add("R")
                elif cmd["command"] == "WRITE":
                    matrix[resource][prog.name].add("W")
                elif cmd["command"] == "REWRITE":
                    matrix[resource][prog.name].add("W")
                    matrix[resource][prog.name].add("R")
                elif cmd["command"] == "DELETE":
                    matrix[resource][prog.name].add("D")

                # Try to extract key field
                key_match = re.search(
                    r"RIDFLD\s*\(\s*([^)]+)\s*\)", cmd["args"], re.IGNORECASE
                )
                if key_match and resource not in dataset_keys:
                    dataset_keys[resource] = key_match.group(1).strip()

            # Batch file operations (OPEN INPUT/OUTPUT)
            for om in re.finditer(
                r"OPEN\s+(INPUT|OUTPUT|I-O|EXTEND)\s+([\w-]+)",
                source, re.IGNORECASE,
            ):
                mode = om.group(1).upper()
                fname = om.group(2)
                if mode == "INPUT":
                    matrix[fname][prog.name].add("R")
                elif mode in ("OUTPUT", "EXTEND"):
                    matrix[fname][prog.name].add("W")
                else:
                    matrix[fname][prog.name].add("R")
                    matrix[fname][prog.name].add("W")

        if matrix:
            # Sort programs for column headers
            prog_names = sorted(set(
                pn for ds_progs in matrix.values() for pn in ds_progs.keys()
            ))

            lines.append("## File -> Program Access Matrix\n")
            header = "| File (DD) | Key |"
            sep = "|-----------|-----|"
            for pn in prog_names:
                header += f" {pn} |"
                sep += "------|"

            lines.append(header)
            lines.append(sep)

            for ds in sorted(matrix.keys()):
                key = dataset_keys.get(ds, "--")
                row = f"| {ds} | {key} |"
                for pn in prog_names:
                    ops = matrix[ds].get(pn, set())
                    if ops:
                        cell = "/".join(sorted(ops))
                    else:
                        cell = "--"
                    row += f" {cell} |"
                lines.append(row)
            lines.append("")

            # Per-file details
            lines.append("## Per-File Details\n")
            for ds in sorted(matrix.keys()):
                key = dataset_keys.get(ds, "unknown")
                writers = [pn for pn, ops in matrix[ds].items() if "W" in ops]
                readers = [pn for pn, ops in matrix[ds].items() if "R" in ops]
                all_progs = sorted(matrix[ds].keys())

                lines.append(f"### {ds}\n")
                lines.append(f"- **Key**: {key}")
                lines.append(f"- **Programs**: {', '.join(all_progs)}")
                if len(writers) > 1:
                    lines.append(f"- **Concurrency risk**: HIGH ({len(writers)} writers: {', '.join(writers)})")
                elif writers:
                    lines.append(f"- **Writers**: {', '.join(writers)}")
                lines.append("")
        else:
            lines.append("No VSAM/file access patterns detected.\n")

    except Exception as e:
        logger.error("Chapter 6 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 7 Override: Per-Program PERFORM Hierarchy
# ---------------------------------------------------------------------------


def generate_perform_hierarchy(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """PERFORM hierarchy trees for each program."""
    pid = _pid(project_id)
    lines = ["# 7. Processing Flow & PERFORM Hierarchy\n"]

    try:
        programs = _fetch_programs(db, pid)
        paragraphs = _fetch_paragraphs(db, pid)
        calls = _fetch_calls(db, pid)

        # Index paragraphs by parent
        paras_by_prog: Dict[str, List[str]] = defaultdict(list)
        for p in paragraphs:
            paras_by_prog[p.parent_name].append(p.name)

        # Index calls from paragraphs
        para_calls: Dict[str, List[str]] = defaultdict(list)
        for c in calls:
            para_calls[c.caller].append(c.callee)

        for prog in programs:
            prog_paras = set(paras_by_prog.get(prog.name, []))
            if not prog_paras:
                continue

            lines.append(f"## {prog.name}\n")
            lines.append("```")

            # Find root paragraphs (not called by other paragraphs in this program)
            called_paras: Set[str] = set()
            for para_name in prog_paras:
                for callee in para_calls.get(para_name, []):
                    if callee in prog_paras:
                        called_paras.add(callee)

            roots = sorted(prog_paras - called_paras)
            if not roots:
                main_candidates = [p for p in prog_paras if "MAIN" in p.upper() or "0000" in p]
                roots = main_candidates if main_candidates else sorted(prog_paras)[:1]

            rendered_in_tree: Set[str] = set()

            def _render_tree(para_name: str, indent: int = 0, visited: Optional[set] = None) -> None:
                if visited is None:
                    visited = set()
                prefix = "|   " * indent + "+-- " if indent > 0 else ""
                if para_name in visited:
                    lines.append(f"{prefix}{para_name} (recursive)")
                    return
                lines.append(f"{prefix}{para_name}")
                rendered_in_tree.add(para_name)
                visited.add(para_name)

                callees = [c for c in para_calls.get(para_name, []) if c in prog_paras]
                for callee in callees:
                    _render_tree(callee, indent + 1, visited.copy())

            for root in roots:
                _render_tree(root)

            orphans = sorted(prog_paras - rendered_in_tree)
            if orphans:
                lines.append(f"\n(Unreferenced paragraphs: {', '.join(orphans)})")

            lines.append("```\n")

    except Exception as e:
        logger.error("Chapter 7 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 8 Override: CICS Commands & Integration Patterns
# ---------------------------------------------------------------------------


def generate_cics_integration_patterns(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """CICS command inventory and program-to-program transfer map."""
    pid = _pid(project_id)
    lines = ["# 8. Integration Patterns & CICS Commands\n"]

    try:
        programs = _fetch_programs(db, pid)
        analyses = _fetch_deep_analyses(db, pid)

        cmd_rows = []
        xctl_rows = []

        for prog in programs:
            source = _safe_source(prog)
            if not source:
                continue

            cics_cmds = _extract_cics_commands(source)
            for cmd in cics_cmds:
                purpose = ""
                if cmd["command"] == "READ":
                    purpose = f"Read {cmd['resource']} by key"
                elif cmd["command"] == "REWRITE":
                    purpose = f"Update {cmd['resource']}"
                elif cmd["command"] == "WRITE":
                    purpose = f"Write to {cmd['resource']}"
                elif cmd["command"] == "DELETE":
                    purpose = f"Delete from {cmd['resource']}"
                elif cmd["command"] == "SEND":
                    purpose = f"Display screen {cmd['resource']}"
                elif cmd["command"] == "RECEIVE":
                    purpose = f"Get user input {cmd['resource']}"
                elif cmd["command"] == "RETURN":
                    purpose = "Pseudo-conversational return"
                elif cmd["command"] == "XCTL":
                    purpose = f"Transfer to {cmd['resource']}"
                    xctl_rows.append({
                        "from": prog.name,
                        "to": cmd["resource"],
                        "mechanism": "XCTL",
                    })
                elif cmd["command"] == "LINK":
                    purpose = f"Call {cmd['resource']}"
                    xctl_rows.append({
                        "from": prog.name,
                        "to": cmd["resource"],
                        "mechanism": "LINK",
                    })
                elif cmd["command"] == "WRITEQ":
                    purpose = f"Write to queue {cmd['resource']}"
                elif cmd["command"] == "READQ":
                    purpose = f"Read from queue {cmd['resource']}"
                elif cmd["command"] == "STARTBR":
                    purpose = f"Start browse {cmd['resource']}"
                elif cmd["command"] == "READNEXT":
                    purpose = f"Browse next {cmd['resource']}"
                else:
                    purpose = cmd["args"][:50] if cmd["args"] else ""

                cmd_rows.append({
                    "program": prog.name,
                    "command": f"EXEC CICS {cmd['command']}",
                    "resource": cmd["resource"] or "--",
                    "purpose": purpose,
                })

        # Command inventory table
        if cmd_rows:
            lines += [
                "## CICS Command Inventory\n",
                "| Program | Command | Resource | Purpose |",
                "|---------|---------|----------|---------|",
            ]
            for row in sorted(cmd_rows, key=lambda r: (r["program"], r["command"])):
                lines.append(
                    f"| {row['program']} | {row['command']} | {row['resource']} | {row['purpose']} |"
                )
            lines.append("")

        # Program-to-program transfers
        if xctl_rows:
            lines += [
                "## Program-to-Program Transfers\n",
                "| From | To | Mechanism |",
                "|------|-----|-----------|",
            ]
            seen = set()
            for row in sorted(xctl_rows, key=lambda r: (r["from"], r["to"])):
                key = (row["from"], row["to"], row["mechanism"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"| {row['from']} | {row['to']} | {row['mechanism']} |")
            lines.append("")

        # Deep analysis integrations (enrichment)
        ext_integrations = []
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else json.loads(a.result_json)
            for integ in rj.get("integrations", []):
                itype = integ.get("type", "")
                if itype not in ("database",):
                    ext_integrations.append({
                        "program": a.entry_name,
                        "type": itype,
                        "description": integ.get("description", "")[:150],
                    })

        if ext_integrations:
            lines += [
                "## External Integration Points\n",
                "| Program | Type | Description |",
                "|---------|------|-------------|",
            ]
            for row in ext_integrations:
                lines.append(f"| {row['program']} | {row['type']} | {row['description']} |")
            lines.append("")

        if not cmd_rows and not xctl_rows:
            lines.append("No CICS commands detected in the codebase.\n")

    except Exception as e:
        logger.error("Chapter 8 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chapter 9 Override: ABEND & Error Handling Patterns
# ---------------------------------------------------------------------------


def generate_abend_error_patterns(
    db: DatabaseManager, project_id: str, pipeline: Any = None,
) -> str:
    """Standardized ABEND patterns, file status handling, and RESP code checks."""
    pid = _pid(project_id)
    lines = ["# 9. Error Handling & Recovery\n"]

    try:
        programs = _fetch_programs(db, pid)
        paragraphs = _fetch_paragraphs(db, pid)

        paras_by_prog: Dict[str, list] = defaultdict(list)
        for p in paragraphs:
            paras_by_prog[p.parent_name].append(p)

        abend_patterns: Dict[str, List[str]] = defaultdict(list)
        resp_patterns: List[Dict] = []
        io_status_patterns: Dict[str, List[str]] = defaultdict(list)

        for prog in programs:
            source = _safe_source(prog)
            if not source:
                continue

            # ABEND / CEE3ABD calls
            if re.search(r"CEE3ABD|ABEND|9999-ABEND", source, re.IGNORECASE):
                abend_match = re.search(
                    r"((?:9999|ABEND)[\w-]*)\.\s*\n((?:\s+.+\n){1,8})",
                    source, re.IGNORECASE,
                )
                if abend_match:
                    pattern_key = abend_match.group(0).strip()[:300]
                    abend_patterns[pattern_key].append(prog.name)
                else:
                    abend_patterns["ABEND (pattern not extracted)"].append(prog.name)

            # IO-STATUS / FILE STATUS handling
            if re.search(r"IO-STATUS|FILE.STATUS|WS-IO-STAT", source, re.IGNORECASE):
                io_match = re.search(
                    r"((?:9910|DISPLAY-IO|IO-STATUS)[\w-]*)\.\s*\n((?:\s+.+\n){1,6})",
                    source, re.IGNORECASE,
                )
                if io_match:
                    pattern_key = io_match.group(0).strip()[:300]
                    io_status_patterns[pattern_key].append(prog.name)
                else:
                    io_status_patterns["IO-STATUS check (pattern not extracted)"].append(prog.name)

            # CICS RESP code handling
            for m in re.finditer(
                r"RESP\s*\(\s*([\w-]+)\s*\)(?:.*?RESP2\s*\(\s*([\w-]+)\s*\))?",
                source, re.IGNORECASE | re.DOTALL,
            ):
                resp_var = m.group(1)
                after_ctx = source[m.end():m.end() + 500]
                codes_handled = set()
                for code_m in re.finditer(
                    rf"(?:{re.escape(resp_var)}|DFHRESP)\s*(?:=|EQUAL)\s*(\w+)",
                    after_ctx, re.IGNORECASE,
                ):
                    codes_handled.add(code_m.group(1))
                for code_m in re.finditer(
                    r"DFHRESP\s*\(\s*(\w+)\s*\)", after_ctx, re.IGNORECASE,
                ):
                    codes_handled.add(code_m.group(1))

                if codes_handled:
                    resp_patterns.append({
                        "program": prog.name,
                        "resp_var": resp_var,
                        "codes": sorted(codes_handled),
                    })

        if abend_patterns:
            lines.append("## Standardized ABEND Patterns\n")
            for pattern, progs in abend_patterns.items():
                lines.append(f"### Used by: {', '.join(sorted(set(progs)))}\n")
                lines.append(f"```cobol\n{pattern}\n```\n")

        if io_status_patterns:
            lines.append("## File Status / IO-STATUS Handling\n")
            for pattern, progs in io_status_patterns.items():
                lines.append(f"### Used by: {', '.join(sorted(set(progs)))}\n")
                lines.append(f"```cobol\n{pattern}\n```\n")

        if resp_patterns:
            lines += [
                "## CICS RESP Code Handling\n",
                "| Program | RESP Variable | Codes Handled |",
                "|---------|--------------|--------------|",
            ]
            seen = set()
            for rp in resp_patterns:
                key = (rp["program"], rp["resp_var"], tuple(rp["codes"]))
                if key not in seen:
                    seen.add(key)
                    codes_str = ", ".join(rp["codes"])
                    lines.append(f"| {rp['program']} | {rp['resp_var']} | {codes_str} |")
            lines.append("")

        # Cross-cutting error concerns from deep analysis
        analyses = _fetch_deep_analyses(db, pid)
        cross_cutting = []
        for a in analyses:
            rj = a.result_json if isinstance(a.result_json, dict) else json.loads(a.result_json)
            for concern in rj.get("cross_cutting_concerns", []):
                # Normalize: concern may be a string or a dict
                if isinstance(concern, dict):
                    concern_text = concern.get("description", concern.get("text", str(concern)))
                else:
                    concern_text = str(concern)
                if any(kw in concern_text.lower() for kw in ("error", "exception", "abend", "recovery", "fault")):
                    cross_cutting.append({"program": a.entry_name, "concern": concern_text})

        if cross_cutting:
            lines += [
                "## Cross-Cutting Error Concerns (from Deep Analysis)\n",
                "| Program | Concern |",
                "|---------|---------|",
            ]
            for cc in cross_cutting:
                lines.append(f"| {cc['program']} | {cc['concern'][:120]} |")
            lines.append("")

        if not abend_patterns and not resp_patterns and not io_status_patterns:
            lines.append("No standardized error handling patterns detected.\n")

    except Exception as e:
        logger.error("Chapter 9 (mainframe) failed: %s", e, exc_info=True)
        lines.append(f"*Generation error: {e}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Override dispatch table
# ---------------------------------------------------------------------------

MAINFRAME_CHAPTER_OVERRIDES = {
    1: generate_program_inventory,
    3: generate_cics_transaction_map,
    4: generate_per_program_specs,
    5: generate_copybook_layouts,
    6: generate_vsam_access_matrix,
    7: generate_perform_hierarchy,
    8: generate_cics_integration_patterns,
    9: generate_abend_error_patterns,
}
