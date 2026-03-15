#!/usr/bin/env python3
"""One-time script: add .cpy copybook files + fix CICS entry type labels.

1. Incrementally ingest .cpy files from the source directory into an existing project
2. Update deep_analyses.entry_type from 'http_endpoint' to 'cics_transaction'
   for COBOL programs with program_category='cics_online'
3. Rebuild ASG edges to resolve COPY → copybook import edges

Usage:
    source venv/bin/activate
    python scripts/fix_cpy_and_cics.py
"""

import os
import sys
import uuid

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_URL", "postgresql://codeloom:codeloom@localhost:5432/codeloom_dev")

from sqlalchemy import create_engine, text

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ID = "3f5b69e1-babf-4ce3-822b-5a237a6895b2"
CPY_DIR = "/Users/bharath/Desktop/MF/awsmf/cpy"
PROJECT_ROOT = "/Users/bharath/Desktop/MF/awsmf"
# ────────────────────────────────────────────────────────────────────

engine = create_engine(os.environ["DATABASE_URL"])


def fix_cics_entry_type():
    """Update http_endpoint → cics_transaction for CICS COBOL programs."""
    with engine.begin() as conn:
        # Find CICS program unit_ids
        cics_units = conn.execute(text("""
            SELECT cu.unit_id, cu.name
            FROM code_units cu
            JOIN code_files cf ON cu.file_id = cf.file_id
            WHERE cf.project_id = :pid
              AND cu.unit_type = 'program'
              AND cu.metadata->>'program_category' = 'cics_online'
        """), {"pid": PROJECT_ID}).fetchall()

        if not cics_units:
            print("No CICS programs found — skipping entry_type fix")
            return

        unit_ids = [str(r.unit_id) for r in cics_units]
        print(f"Found {len(cics_units)} CICS programs: {[r.name for r in cics_units]}")

        # Update deep_analyses — use a subquery instead of ANY(array)
        result = conn.execute(text("""
            UPDATE deep_analyses
            SET entry_type = 'cics_transaction'
            WHERE project_id = :pid
              AND entry_type = 'http_endpoint'
              AND entry_unit_id IN (
                  SELECT cu.unit_id FROM code_units cu
                  JOIN code_files cf ON cu.file_id = cf.file_id
                  WHERE cf.project_id = :pid
                    AND cu.unit_type = 'program'
                    AND cu.metadata->>'program_category' = 'cics_online'
              )
        """), {"pid": PROJECT_ID})
        print(f"Updated {result.rowcount} deep_analyses rows: http_endpoint → cics_transaction")

        # Also update result_json where entry_type is stored
        conn.execute(text("""
            UPDATE deep_analyses
            SET result_json = jsonb_set(
                result_json,
                '{entry_point,entry_type}',
                '"cics_transaction"'
            )
            WHERE project_id = :pid
              AND entry_type = 'cics_transaction'
              AND result_json->'entry_point'->>'entry_type' = 'http_endpoint'
        """), {"pid": PROJECT_ID})
        print("Updated result_json entry_type references")


def ingest_copybooks():
    """Parse and store .cpy files into the existing project."""
    from codeloom.core.ast_parser import parse_file, detect_language
    from codeloom.core.ast_parser.utils import SUPPORTED_EXTENSIONS

    # Verify .cpy is now supported
    assert SUPPORTED_EXTENSIONS.get(".cpy") == "cobol", ".cpy not in SUPPORTED_EXTENSIONS!"

    # Collect .cpy files
    cpy_files = []
    for fname in os.listdir(CPY_DIR):
        if fname.lower().endswith(".cpy"):
            cpy_files.append(os.path.join(CPY_DIR, fname))

    if not cpy_files:
        print("No .cpy files found — skipping")
        return

    print(f"\nFound {len(cpy_files)} copybook files to ingest")

    with engine.begin() as conn:
        # Check which .cpy files are already ingested
        existing = conn.execute(text("""
            SELECT file_path FROM code_files WHERE project_id = :pid
        """), {"pid": PROJECT_ID}).fetchall()
        existing_paths = {r.file_path for r in existing}

        files_added = 0
        units_added = 0

        for fpath in sorted(cpy_files):
            rel_path = os.path.relpath(fpath, PROJECT_ROOT)

            if rel_path in existing_paths:
                print(f"  SKIP (exists): {rel_path}")
                continue

            # Parse with our updated COBOL parser
            parse_result = parse_file(fpath, PROJECT_ROOT)

            if not parse_result.units:
                print(f"  SKIP (no units): {rel_path}")
                continue

            # Read file for line count and size
            file_size = os.path.getsize(fpath)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            line_count = content.count("\n") + 1

            # Insert code_file
            import hashlib
            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            file_id = str(uuid.uuid4())
            conn.execute(text("""
                INSERT INTO code_files (file_id, project_id, file_path, language, line_count, size_bytes, file_hash)
                VALUES (:fid, :pid, :fp, :lang, :lc, :sz, :fh)
            """), {
                "fid": file_id,
                "pid": PROJECT_ID,
                "fp": rel_path,
                "lang": "cobol",
                "lc": line_count,
                "sz": file_size,
                "fh": file_hash,
            })
            files_added += 1

            # Insert code_units
            for unit in parse_result.units:
                unit_id = str(uuid.uuid4())
                import json
                conn.execute(text("""
                    INSERT INTO code_units (unit_id, file_id, unit_type, name, qualified_name,
                                           language, start_line, end_line, source, signature, metadata)
                    VALUES (:uid, :fid, :utype, :name, :qname, :lang, :sl, :el, :src, :sig, :meta::jsonb)
                """), {
                    "uid": unit_id,
                    "fid": file_id,
                    "utype": unit.unit_type,
                    "name": unit.name,
                    "qname": unit.qualified_name,
                    "lang": unit.language,
                    "sl": unit.start_line,
                    "el": unit.end_line,
                    "src": unit.source,
                    "sig": unit.signature,
                    "meta": json.dumps(unit.metadata) if unit.metadata else "{}",
                })
                units_added += 1

            print(f"  ADD: {rel_path} → {len(parse_result.units)} unit(s)")

        print(f"\nIngested {files_added} files, {units_added} units")


def rebuild_asg_edges():
    """Rebuild ASG edges to pick up COPY → copybook import edges."""
    print("\nRebuilding ASG edges...")
    from codeloom.core.db import DatabaseManager
    from codeloom.core.asg_builder import ASGBuilder

    db = DatabaseManager(os.environ["DATABASE_URL"])
    asg = ASGBuilder(db)
    edge_count = asg.build_edges(PROJECT_ID)
    print(f"ASG rebuilt: {edge_count} edges")


def update_project_file_count():
    """Update the project's file_count to include new copybooks."""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE projects
            SET file_count = (SELECT COUNT(*) FROM code_files WHERE project_id = :pid)
            WHERE project_id = :pid
        """), {"pid": PROJECT_ID})
        row = conn.execute(text(
            "SELECT file_count FROM projects WHERE project_id = :pid"
        ), {"pid": PROJECT_ID}).fetchone()
        print(f"Project file_count updated to {row.file_count}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fix: CICS entry type + incremental .cpy ingestion")
    print("=" * 60)

    print("\n── Step 1: Fix CICS entry_type labels ──")
    fix_cics_entry_type()

    print("\n── Step 2: Ingest .cpy copybook files ──")
    ingest_copybooks()

    print("\n── Step 3: Update project stats ──")
    update_project_file_count()

    print("\n── Step 4: Rebuild ASG edges ──")
    rebuild_asg_edges()

    print("\n" + "=" * 60)
    print("Done. Reload the UI to see changes.")
    print("=" * 60)
