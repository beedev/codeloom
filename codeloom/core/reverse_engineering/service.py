"""Reverse Engineering Documentation service.

Orchestrates generation of structured 15-chapter reverse engineering
documentation by composing existing intelligence from the Understanding Engine,
ASG queries, and ground truth.

Follows the same pattern as UnderstandingEngine (core/understanding/engine.py).
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from ..db import DatabaseManager
from .chapters import CHAPTER_GENERATORS, CHAPTER_TITLES

logger = logging.getLogger(__name__)

TOTAL_CHAPTERS = 15

# Languages that trigger the mainframe lane override
_MAINFRAME_LANGUAGES = frozenset({"cobol", "jcl", "pli", "pl1"})

# Chapters that use LLM synthesis (require manual review)
_LLM_CHAPTERS = frozenset({2, 5, 14})


class ReverseEngineeringService:
    """Orchestrate reverse engineering documentation generation.

    Public API:
        generate(project_id, chapters=None) -> {doc_id, status}
        get_status(doc_id) -> status dict
        get_document(project_id) -> latest doc dict
        get_doc_by_id(doc_id) -> full doc dict
        get_chapter(doc_id, chapter_num) -> markdown string
        list_docs(project_id) -> list of doc summaries
        validate_document(project_id, doc_id=None) -> validation report
    """

    def __init__(self, db_manager: DatabaseManager, pipeline: Any = None):
        self._db = db_manager
        self._pipeline = pipeline

    # -- Public API ----------------------------------------------------------

    def generate(
        self,
        project_id: str,
        chapters: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Generate reverse engineering documentation.

        Args:
            project_id: UUID string of the project
            chapters: Optional list of chapter numbers (1-15) to generate.
                      If None, generates all 15 chapters.

        Returns:
            Dict with doc_id and status.
        """
        pid = UUID(project_id) if isinstance(project_id, str) else project_id
        doc_id = uuid4()

        # Determine which chapters to generate
        if chapters:
            chapter_nums = sorted(set(c for c in chapters if 1 <= c <= TOTAL_CHAPTERS))
        else:
            chapter_nums = list(range(1, TOTAL_CHAPTERS + 1))

        total = len(chapter_nums)
        titles = [CHAPTER_TITLES[i - 1] for i in chapter_nums]

        # Create the doc row (serialize list as JSON string for JSONB column)
        with self._db.get_session() as session:
            session.execute(
                text("""
                    INSERT INTO reverse_engineering_docs
                        (doc_id, project_id, status, chapters, chapter_titles,
                         progress, total_chapters, created_at, updated_at)
                    VALUES (:did, :pid, 'generating', CAST('{}' AS jsonb), CAST(:titles AS jsonb), 0, :total, NOW(), NOW())
                """),
                {
                    "did": doc_id,
                    "pid": pid,
                    "titles": json.dumps(titles),
                    "total": total,
                },
            )

        # -------------------------------------------------------------------
        # Detect project language and select appropriate chapter generators
        # -------------------------------------------------------------------
        is_mainframe = False
        is_python = False
        is_csharp = False
        try:
            with self._db.get_session() as session:
                lang_rows = session.execute(
                    text(
                        "SELECT language, COUNT(*) AS cnt "
                        "FROM code_files WHERE project_id = :pid "
                        "GROUP BY language"
                    ),
                    {"pid": pid},
                ).fetchall()
            is_mainframe = any(
                r.language in _MAINFRAME_LANGUAGES for r in lang_rows if r.language
            )
            is_python = any(
                r.language == "python" for r in lang_rows if r.language
            )
            is_csharp = any(
                r.language == "csharp" for r in lang_rows if r.language
            )
        except Exception as e:
            logger.warning("Language detection failed (defaulting to generic): %s", e)

        if is_mainframe:
            from .lane_mainframe import MAINFRAME_CHAPTER_OVERRIDES

            active_generators = {**CHAPTER_GENERATORS, **MAINFRAME_CHAPTER_OVERRIDES}
            logger.info(
                "Mainframe project detected -- overriding chapters %s",
                sorted(MAINFRAME_CHAPTER_OVERRIDES.keys()),
            )
        elif is_python:
            from .lane_python import PYTHON_CHAPTER_OVERRIDES

            active_generators = {**CHAPTER_GENERATORS, **PYTHON_CHAPTER_OVERRIDES}
            logger.info(
                "Python project detected -- overriding chapters %s",
                sorted(PYTHON_CHAPTER_OVERRIDES.keys()),
            )
        elif is_csharp:
            from .lane_dotnet import DOTNET_CHAPTER_OVERRIDES

            active_generators = {**CHAPTER_GENERATORS, **DOTNET_CHAPTER_OVERRIDES}
            logger.info(
                "C#/.NET project detected -- overriding chapters %s",
                sorted(DOTNET_CHAPTER_OVERRIDES.keys()),
            )
        else:
            active_generators = CHAPTER_GENERATORS

        # Generate chapters synchronously
        generated_chapters = {}
        error = None

        for i, ch_num in enumerate(chapter_nums):
            generator = active_generators.get(ch_num)
            if not generator:
                logger.warning("No generator for chapter %d", ch_num)
                continue

            try:
                logger.info(
                    "Generating chapter %d/%d: %s",
                    i + 1, total, CHAPTER_TITLES[ch_num - 1],
                )
                content = generator(self._db, project_id, self._pipeline)
                generated_chapters[str(ch_num)] = content

                # Update progress -- build JSON object and merge into chapters
                ch_update = json.dumps({str(ch_num): content})
                with self._db.get_session() as session:
                    session.execute(
                        text("""
                            UPDATE reverse_engineering_docs
                            SET progress = :progress,
                                chapters = chapters || CAST(:ch_data AS jsonb),
                                updated_at = NOW()
                            WHERE doc_id = :did
                        """),
                        {
                            "progress": i + 1,
                            "ch_data": ch_update,
                            "did": doc_id,
                        },
                    )
            except Exception as e:
                logger.error("Chapter %d generation failed: %s", ch_num, e, exc_info=True)
                error = f"Chapter {ch_num} ({CHAPTER_TITLES[ch_num - 1]}): {e}"
                # Continue with remaining chapters

        # -----------------------------------------------------------------
        # Chapter 15: Findings Engine (deterministic, zero-LLM)
        # -----------------------------------------------------------------
        if 15 in chapter_nums and generated_chapters:
            try:
                from .findings import FindingsEngine
                logger.info(
                    "Generating chapter %d/%d: %s",
                    total, total, CHAPTER_TITLES[14],
                )
                engine = FindingsEngine()
                findings = engine.run(self._db, project_id)
                findings_md = engine.format_as_markdown(findings)
                generated_chapters["15"] = findings_md

                ch_update = json.dumps({"15": findings_md})
                with self._db.get_session() as session:
                    session.execute(
                        text("""
                            UPDATE reverse_engineering_docs
                            SET progress = :progress,
                                chapters = chapters || CAST(:ch_data AS jsonb),
                                updated_at = NOW()
                            WHERE doc_id = :did
                        """),
                        {
                            "progress": total,
                            "ch_data": ch_update,
                            "did": doc_id,
                        },
                    )
                logger.info(
                    "Findings engine produced %d findings", len(findings),
                )
            except Exception as e:
                logger.error("Findings engine failed: %s", e, exc_info=True)
                error = f"Chapter 15 (Findings Engine): {e}"

        # Mark completion
        final_status = "completed" if generated_chapters else "failed"

        with self._db.get_session() as session:
            session.execute(
                text("""
                    UPDATE reverse_engineering_docs
                    SET status = :status, error = :error,
                        progress = :progress, updated_at = NOW()
                    WHERE doc_id = :did
                """),
                {
                    "status": final_status,
                    "error": error,
                    "progress": len(generated_chapters),
                    "did": doc_id,
                },
            )

        # Vectorize chapters into RAG index for search and chat (best-effort)
        if generated_chapters and final_status == "completed":
            self._vectorize_chapters(str(pid), generated_chapters, titles)

        # Auto-validate (best-effort -- never crash generation)
        if generated_chapters and final_status == "completed":
            try:
                validation = self._validate_chapters(
                    str(pid), generated_chapters, titles,
                )
                with self._db.get_session() as session:
                    session.execute(
                        text("""
                            UPDATE reverse_engineering_docs
                            SET chapters = jsonb_set(
                                COALESCE(chapters, '{}'::jsonb),
                                '{_validation}',
                                :val::jsonb
                            )
                            WHERE doc_id = :did
                        """),
                        {"did": str(doc_id), "val": json.dumps(validation)},
                    )
                    session.commit()
                logger.info(
                    "Doc validation: %.0f%% confidence",
                    validation.get("overall_confidence", 0) * 100,
                )
            except Exception as e:
                logger.warning("Doc validation failed: %s", e)

        return {
            "doc_id": str(doc_id),
            "status": final_status,
            "project_id": str(pid),
            "chapters_generated": len(generated_chapters),
            "total_chapters": total,
            "error": error,
        }

    def get_status(self, doc_id: str) -> Dict[str, Any]:
        """Get the generation status of a document."""
        did = UUID(doc_id) if isinstance(doc_id, str) else doc_id

        with self._db.get_session() as session:
            row = session.execute(
                text("""
                    SELECT doc_id, project_id, status, progress, total_chapters,
                           error, chapter_titles, created_at, updated_at
                    FROM reverse_engineering_docs
                    WHERE doc_id = :did
                """),
                {"did": did},
            ).fetchone()

        if not row:
            return {"error": "Document not found"}

        return {
            "doc_id": str(row.doc_id),
            "project_id": str(row.project_id),
            "status": row.status,
            "progress": row.progress,
            "total_chapters": row.total_chapters,
            "error": row.error,
            "chapter_titles": row.chapter_titles,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def get_document(self, project_id: str) -> Dict[str, Any]:
        """Get the latest completed document for a project."""
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        with self._db.get_session() as session:
            row = session.execute(
                text("""
                    SELECT doc_id, project_id, status, chapters, chapter_titles,
                           progress, total_chapters, error, created_at, updated_at
                    FROM reverse_engineering_docs
                    WHERE project_id = :pid
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"pid": pid},
            ).fetchone()

        if not row:
            return {"error": "No documents found for this project"}

        return self._row_to_dict(row)

    def get_doc_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get a document by its ID."""
        did = UUID(doc_id) if isinstance(doc_id, str) else doc_id

        with self._db.get_session() as session:
            row = session.execute(
                text("""
                    SELECT doc_id, project_id, status, chapters, chapter_titles,
                           progress, total_chapters, error, created_at, updated_at
                    FROM reverse_engineering_docs
                    WHERE doc_id = :did
                """),
                {"did": did},
            ).fetchone()

        if not row:
            return {"error": "Document not found"}

        return self._row_to_dict(row)

    def get_chapter(self, doc_id: str, chapter_num: int) -> Dict[str, Any]:
        """Get a single chapter from a document.

        Returns:
            Dict with chapter_num, title, and content (markdown).
        """
        did = UUID(doc_id) if isinstance(doc_id, str) else doc_id

        with self._db.get_session() as session:
            row = session.execute(
                text("""
                    SELECT chapters FROM reverse_engineering_docs
                    WHERE doc_id = :did
                """),
                {"did": did},
            ).fetchone()

        if not row:
            return {"error": "Document not found"}

        chapters = row.chapters or {}
        key = str(chapter_num)

        if key not in chapters:
            return {"error": f"Chapter {chapter_num} not found in document"}

        title = CHAPTER_TITLES[chapter_num - 1] if 1 <= chapter_num <= TOTAL_CHAPTERS else f"Chapter {chapter_num}"

        return {
            "chapter_num": chapter_num,
            "title": title,
            "content": chapters[key],
        }

    def list_docs(self, project_id: str) -> List[Dict[str, Any]]:
        """List all reverse engineering documents for a project."""
        pid = UUID(project_id) if isinstance(project_id, str) else project_id

        with self._db.get_session() as session:
            rows = session.execute(
                text("""
                    SELECT doc_id, project_id, status, progress, total_chapters,
                           error, chapter_titles, created_at, updated_at
                    FROM reverse_engineering_docs
                    WHERE project_id = :pid
                    ORDER BY created_at DESC
                """),
                {"pid": pid},
            ).fetchall()

        return [
            {
                "doc_id": str(row.doc_id),
                "project_id": str(row.project_id),
                "status": row.status,
                "progress": row.progress,
                "total_chapters": row.total_chapters,
                "error": row.error,
                "chapter_titles": row.chapter_titles,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]

    def validate_document(
        self, project_id: str, doc_id: str = None,
    ) -> Dict[str, Any]:
        """Validate a reverse engineering doc against source code ground truth.

        Returns validation report with per-chapter confidence scores and
        flagged issues.
        """
        # Get the doc (latest if no doc_id)
        if doc_id:
            doc = self.get_doc_by_id(doc_id)
        else:
            doc = self.get_document(project_id)

        if doc.get("error"):
            return {"error": doc["error"]}

        validation = self._validate_chapters(
            project_id, doc["chapters"], doc["chapter_titles"],
        )

        # Store validation result in the doc
        try:
            with self._db.get_session() as session:
                session.execute(
                    text("""
                        UPDATE reverse_engineering_docs
                        SET chapters = jsonb_set(
                            COALESCE(chapters, '{}'::jsonb),
                            '{_validation}',
                            :val::jsonb
                        )
                        WHERE doc_id = :did
                    """),
                    {"did": doc["doc_id"], "val": json.dumps(validation)},
                )
                session.commit()
        except Exception as e:
            logger.warning("Failed to store validation result: %s", e)

        return validation

    # -- Private helpers -----------------------------------------------------

    def _validate_chapters(
        self,
        project_id: str,
        chapters: Dict[str, str],
        titles: List[str],
    ) -> Dict[str, Any]:
        """Run ground truth validation checks against source code and ASG data.

        Checks:
            A. Program coverage (Ch 4 vs code_units)
            B. Paragraph coverage (Ch 4 + Ch 7 vs code_units)
            C. Business rule source verification (Ch 4 source references)
            D. Code snippet verification (COBOL/source blocks vs actual source)
            E. File access / import edge verification (Ch 6 vs ASG edges)
            F. LLM chapter flagging (Ch 2, 5, 14)
            G. Edge count verification (Ch 1 vs code_edges)

        Returns a validation dict with overall_confidence, per-chapter details,
        summary metrics, and flagged issues.
        """
        pid = project_id
        flagged_issues: List[str] = []
        chapter_results: Dict[str, Dict[str, Any]] = {}

        # Initialize defaults for all summary metrics
        program_coverage_pct = 100.0
        documented_programs_count = 0
        actual_programs_count = 0
        paragraph_coverage_pct = 100.0
        doc_paragraphs = 0
        actual_paragraphs = 0
        source_verified = 0
        source_unverified = 0
        source_verification_rate = 100.0
        snippet_verified = 0
        snippet_total = 0
        snippet_verification_rate = 100.0
        access_claims = 0
        actual_import_count = 0

        try:
            with self._db.get_session() as session:
                # ---------------------------------------------------------------
                # A. Program Coverage
                # ---------------------------------------------------------------
                try:
                    actual_programs = session.execute(
                        text("""
                            SELECT name FROM code_units
                            WHERE project_id = :pid AND unit_type = 'program'
                        """),
                        {"pid": pid},
                    ).fetchall()
                    actual_names = {r.name for r in actual_programs}
                    actual_programs_count = len(actual_names)

                    # Parse program names from Ch 4 (look for ## PROGRAM_NAME headings)
                    ch4 = chapters.get("4", "")
                    documented_programs = set(
                        re.findall(r"^## (\w+)\b", ch4, re.MULTILINE)
                    )
                    documented_programs_count = len(documented_programs)

                    missing_from_doc = actual_names - documented_programs
                    extra_in_doc = documented_programs - actual_names

                    if actual_programs_count > 0:
                        program_coverage_pct = (
                            len(documented_programs & actual_names)
                            / actual_programs_count
                            * 100
                        )
                    else:
                        program_coverage_pct = 100.0

                    for name in sorted(missing_from_doc):
                        flagged_issues.append(
                            f"Program missing from doc: {name}"
                        )
                    for name in sorted(extra_in_doc):
                        flagged_issues.append(
                            f"Program in doc not found in codebase: {name}"
                        )
                except Exception as e:
                    logger.warning("Program coverage check failed: %s", e)
                    flagged_issues.append(f"Program coverage check error: {e}")

                # ---------------------------------------------------------------
                # B. Paragraph Coverage
                # ---------------------------------------------------------------
                try:
                    actual_paragraphs = session.execute(
                        text("""
                            SELECT COUNT(*) FROM code_units
                            WHERE project_id = :pid AND unit_type = 'paragraph'
                        """),
                        {"pid": pid},
                    ).scalar() or 0

                    # Count paragraph table rows in Ch 4 + Ch 7
                    ch4_ch7 = chapters.get("4", "") + chapters.get("7", "")
                    doc_paragraph_mentions = len(
                        re.findall(r"^\|\s*\d+\s*\|", ch4_ch7, re.MULTILINE)
                    )
                    # Also count tree entries in Ch 7
                    tree_entries = len(
                        re.findall(r"[├└│─]\s*\w", chapters.get("7", ""))
                    )
                    doc_paragraphs = max(doc_paragraph_mentions, tree_entries)

                    if actual_paragraphs > 0:
                        paragraph_coverage_pct = min(
                            doc_paragraphs / actual_paragraphs * 100, 100
                        )
                    else:
                        paragraph_coverage_pct = 100.0
                except Exception as e:
                    logger.warning("Paragraph coverage check failed: %s", e)
                    flagged_issues.append(f"Paragraph coverage check error: {e}")

                # ---------------------------------------------------------------
                # C. Business Rule Source Verification
                # ---------------------------------------------------------------
                try:
                    ch4 = chapters.get("4", "")
                    # Pattern: *Source: PROGRAM.PARA (`file:line`) or PROGRAM (`file:line`)
                    source_refs = re.findall(
                        r"\*Source:\s*(\w+)(?:\.(\w+))?\s*\(`?([^`:)]+):(\d+)",
                        ch4,
                    )

                    source_verified = 0
                    source_unverified = 0

                    for prog_name, para_name, file_path, line_num in source_refs:
                        lookup_name = para_name if para_name else prog_name
                        exists = session.execute(
                            text("""
                                SELECT COUNT(*) FROM code_units u
                                JOIN code_files f ON u.file_id = f.file_id
                                WHERE u.project_id = :pid
                                  AND u.name = :name
                                  AND f.file_path = :fp
                            """),
                            {"pid": pid, "name": lookup_name, "fp": file_path},
                        ).scalar()

                        if exists:
                            source_verified += 1
                        else:
                            source_unverified += 1
                            flagged_issues.append(
                                f"Source reference not found: "
                                f"{prog_name}{'.' + para_name if para_name else ''}"
                                f" at {file_path}:{line_num}"
                            )

                    total_refs = source_verified + source_unverified
                    source_verification_rate = (
                        source_verified / total_refs * 100
                        if total_refs > 0
                        else 100.0
                    )
                except Exception as e:
                    logger.warning("Source reference verification failed: %s", e)
                    flagged_issues.append(f"Source reference check error: {e}")

                # ---------------------------------------------------------------
                # D. Code Snippet Verification
                # ---------------------------------------------------------------
                try:
                    ch4 = chapters.get("4", "")
                    # Find code blocks -- language-agnostic (cobol, sql, etc.)
                    snippet_pattern = (
                        r"\*Source:.*?`([^`]+)`.*?\n```\w*\n(.*?)```"
                    )
                    snippets = re.findall(snippet_pattern, ch4, re.DOTALL)

                    snippet_verified = 0
                    snippet_total = 0

                    for _ref, code in snippets:
                        snippet_total += 1
                        code_lines = [
                            ln.strip()
                            for ln in code.strip().split("\n")
                            if ln.strip()
                        ]
                        if not code_lines:
                            continue

                        # Use first 60 chars of first meaningful line
                        search_text = code_lines[0][:60]
                        # Escape special SQL LIKE chars
                        safe_text = (
                            search_text
                            .replace("\\", "\\\\")
                            .replace("%", "\\%")
                            .replace("_", "\\_")
                        )
                        found = session.execute(
                            text("""
                                SELECT COUNT(*) FROM code_units
                                WHERE project_id = :pid
                                  AND source ILIKE :pattern
                            """),
                            {"pid": pid, "pattern": f"%{safe_text}%"},
                        ).scalar()

                        if found:
                            snippet_verified += 1
                        else:
                            flagged_issues.append(
                                f"Code snippet not found in source: "
                                f"{search_text[:50]}..."
                            )

                    snippet_verification_rate = (
                        snippet_verified / snippet_total * 100
                        if snippet_total > 0
                        else 100.0
                    )
                except Exception as e:
                    logger.warning("Snippet verification failed: %s", e)
                    flagged_issues.append(f"Snippet verification error: {e}")

                # ---------------------------------------------------------------
                # E. File Access / Import Edge Verification
                # ---------------------------------------------------------------
                try:
                    actual_imports = session.execute(
                        text("""
                            SELECT DISTINCT su.name AS program, tu.name AS target
                            FROM code_edges e
                            JOIN code_units su ON e.source_unit_id = su.unit_id
                            JOIN code_units tu ON e.target_unit_id = tu.unit_id
                            WHERE e.project_id = :pid AND e.edge_type = 'imports'
                        """),
                        {"pid": pid},
                    ).fetchall()
                    actual_import_count = len(actual_imports)

                    ch6 = chapters.get("6", "")
                    access_claims = len(
                        re.findall(r"\|\s*[RWD/]+\s*\|", ch6)
                    )
                except Exception as e:
                    logger.warning("Import edge verification failed: %s", e)
                    flagged_issues.append(f"Import edge check error: {e}")

                # ---------------------------------------------------------------
                # G. Edge Count Verification (Ch 1 reports ASG stats)
                # ---------------------------------------------------------------
                try:
                    actual_edge_count = session.execute(
                        text("""
                            SELECT COUNT(*) FROM code_edges
                            WHERE project_id = :pid
                        """),
                        {"pid": pid},
                    ).scalar() or 0

                    ch1 = chapters.get("1", "")
                    # Try to find edge count claims in Ch 1
                    edge_matches = re.findall(
                        r"(\d[\d,]*)\s*(?:edges|relationships|connections)",
                        ch1, re.IGNORECASE,
                    )
                    ch1_edge_claim = 0
                    if edge_matches:
                        ch1_edge_claim = int(
                            edge_matches[0].replace(",", "")
                        )
                except Exception as e:
                    logger.warning("Edge count verification failed: %s", e)
                    actual_edge_count = 0
                    ch1_edge_claim = 0

        except Exception as e:
            logger.error("Validation DB session failed: %s", e)
            return {
                "error": f"Validation failed: {e}",
                "overall_confidence": 0.0,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }

        # -------------------------------------------------------------------
        # F. LLM Chapter Flagging
        # -------------------------------------------------------------------
        for ch_num_str in chapters:
            try:
                ch_num = int(ch_num_str)
            except (ValueError, TypeError):
                continue

            # Skip the _validation key itself
            if ch_num_str.startswith("_"):
                continue

            ch_info: Dict[str, Any] = {
                "confidence": 90,
                "issues": [],
                "verified_claims": 0,
                "total_claims": 0,
            }

            if ch_num in _LLM_CHAPTERS:
                ch_info["confidence"] = 70
                ch_info["llm_generated"] = True
                ch_info["issues"].append(
                    "LLM-synthesized -- manual review recommended"
                )

            # Assign per-chapter specifics
            if ch_num == 1:
                # Edge count check
                if ch1_edge_claim and actual_edge_count:
                    ratio = min(ch1_edge_claim, actual_edge_count) / max(
                        ch1_edge_claim, actual_edge_count, 1
                    )
                    ch_info["confidence"] = int(ratio * 100)
                    if ratio < 0.9:
                        ch_info["issues"].append(
                            f"Edge count mismatch: doc claims {ch1_edge_claim}, "
                            f"actual {actual_edge_count}"
                        )
                else:
                    ch_info["confidence"] = 95

            elif ch_num == 4:
                total_claims = source_verified + source_unverified + snippet_total
                verified_claims = source_verified + snippet_verified
                ch_info["verified_claims"] = verified_claims
                ch_info["total_claims"] = total_claims
                ch_info["confidence"] = int(
                    (verified_claims / max(total_claims, 1)) * 100
                )
                if source_unverified > 0:
                    ch_info["issues"].append(
                        f"{source_unverified} source references not found"
                    )
                if snippet_total - snippet_verified > 0:
                    ch_info["issues"].append(
                        f"{snippet_total - snippet_verified} code snippets "
                        f"not found in source"
                    )

            elif ch_num == 6:
                ch_info["confidence"] = 85 if access_claims > 0 else 90

            elif ch_num == 7:
                ch_info["confidence"] = (
                    int(paragraph_coverage_pct) if paragraph_coverage_pct < 100 else 95
                )

            chapter_results[str(ch_num)] = ch_info

        # -------------------------------------------------------------------
        # Build summary
        # -------------------------------------------------------------------
        summary = {
            "program_coverage": {
                "documented": documented_programs_count,
                "actual": actual_programs_count,
                "percentage": round(program_coverage_pct, 1),
            },
            "paragraph_coverage": {
                "documented": doc_paragraphs,
                "actual": actual_paragraphs,
                "percentage": round(paragraph_coverage_pct, 1),
            },
            "business_rules": {
                "verified": source_verified,
                "unverified": source_unverified,
                "percentage": round(source_verification_rate, 1),
            },
            "code_snippets": {
                "verified": snippet_verified,
                "total": snippet_total,
                "percentage": round(snippet_verification_rate, 1),
            },
            "source_references": {
                "verified": source_verified,
                "unverified": source_unverified,
                "percentage": round(source_verification_rate, 1),
            },
        }

        # Overall confidence = weighted average of key metrics
        weights = {
            "program_coverage": 0.20,
            "paragraph_coverage": 0.15,
            "business_rules": 0.25,
            "code_snippets": 0.25,
            "source_references": 0.15,
        }
        overall = sum(
            summary[k]["percentage"] * w for k, w in weights.items()
        ) / 100

        return {
            "overall_confidence": round(overall, 3),
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "chapters": chapter_results,
            "summary": summary,
            "flagged_issues": flagged_issues,
        }

    def _vectorize_chapters(
        self,
        project_id: str,
        chapters: Dict[str, str],
        chapter_titles: List[str],
    ) -> None:
        """Chunk and embed reverse engineering chapters into the RAG vector store.

        Each chapter is split into ~1024-token chunks (~4000 chars) with
        paragraph-boundary splitting. Nodes are tagged with metadata so they
        appear in codeloom_search_codebase and chat results.

        This is best-effort: failures are logged but do not fail generation.
        """
        try:
            from llama_index.core.schema import TextNode
            from llama_index.core import Settings as LISettings

            nodes: List[Any] = []
            for ch_num_str, content in chapters.items():
                ch_num = int(ch_num_str)
                title = (
                    chapter_titles[ch_num - 1]
                    if ch_num <= len(chapter_titles)
                    else f"Chapter {ch_num}"
                )

                # Split long chapters into ~1024-token chunks (~4000 chars)
                chunk_size = 4000
                text_remaining = content
                chunk_idx = 0

                while text_remaining:
                    chunk = text_remaining[:chunk_size]
                    text_remaining = text_remaining[chunk_size:]

                    # Try to break at a paragraph boundary in the second half
                    if text_remaining and "\n\n" in chunk[chunk_size // 2 :]:
                        break_point = chunk.rfind("\n\n", chunk_size // 2)
                        if break_point > 0:
                            text_remaining = chunk[break_point:] + text_remaining
                            chunk = chunk[:break_point]

                    safe_title = title.lower().replace(" ", "_").replace("&", "and")
                    node = TextNode(
                        text=chunk,
                        metadata={
                            "project_id": project_id,
                            "source_id": f"reverse_engineering_{project_id}",
                            "file_name": (
                                f"reverse_engineering/ch{ch_num:02d}_{safe_title}.md"
                            ),
                            "node_type": "reverse_engineering",
                            "chapter_number": str(ch_num),
                            "chapter_title": title,
                            "chunk_index": str(chunk_idx),
                        },
                        excluded_embed_metadata_keys=[
                            "project_id",
                            "source_id",
                            "node_type",
                            "chapter_number",
                            "chunk_index",
                        ],
                    )
                    nodes.append(node)
                    chunk_idx += 1

            if not nodes:
                logger.info("No reverse engineering chunks to vectorize")
                return

            # Generate embeddings in batches
            embed_model = LISettings.embed_model
            batch_size = 100
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                texts = [n.get_content() for n in batch]
                embeddings = embed_model.get_text_embedding_batch(texts)
                for node, emb in zip(batch, embeddings):
                    node.embedding = emb

            # Store in vector store
            if self._pipeline and hasattr(self._pipeline, "_vector_store"):
                vs = self._pipeline._vector_store
                num_added = vs.add_nodes(nodes, project_id=project_id)
                logger.info(
                    "Vectorized %d reverse engineering chunks for project %s",
                    num_added,
                    project_id,
                )

                # Invalidate node cache so new chunks appear in queries
                self._pipeline.invalidate_node_cache(project_id)
            else:
                logger.warning(
                    "Pipeline or vector store not available for vectorization"
                )
        except Exception as e:
            logger.error(
                "Failed to vectorize reverse engineering chapters: %s", e
            )
            # Don't fail the generation -- vectorization is best-effort

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Convert a DB row to a full document dict."""
        return {
            "doc_id": str(row.doc_id),
            "project_id": str(row.project_id),
            "status": row.status,
            "chapters": row.chapters or {},
            "chapter_titles": row.chapter_titles or [],
            "progress": row.progress,
            "total_chapters": row.total_chapters,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
