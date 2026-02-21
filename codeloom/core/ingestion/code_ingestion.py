"""Code Ingestion Service.

Orchestrates: upload/clone/path → extract → AST parse → chunk → embed → store.
Supports zip upload, git clone, and local directory ingestion.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4, UUID

from ..ast_parser import parse_file, detect_language, should_skip_directory
from ..ast_parser.models import ParseResult
from ..code_chunker import CodeChunker
from ..db import DatabaseManager
from ..db.models import CodeFile, CodeUnit as CodeUnitModel, Project

logger = logging.getLogger(__name__)

# Limits
MAX_FILE_SIZE_MB = 50
MAX_FILES = 500


@dataclass
class IngestionResult:
    """Summary of a code ingestion run."""

    project_id: str
    files_processed: int = 0
    files_skipped: int = 0
    units_extracted: int = 0
    chunks_created: int = 0
    embeddings_stored: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class CodeIngestionService:
    """Orchestrates code ingestion: source → parse → chunk → embed → store."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store,
        max_file_size_mb: int = MAX_FILE_SIZE_MB,
        max_files: int = MAX_FILES,
        max_tokens_per_chunk: int = 1024,
    ):
        self._db = db_manager
        self._vector_store = vector_store
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._max_files = max_files
        self._chunker = CodeChunker(max_tokens=max_tokens_per_chunk)

    # ── Public entry points ──────────────────────────────────────────────

    def ingest_zip(
        self,
        zip_path: str,
        project_id: str,
        user_id: str,
    ) -> IngestionResult:
        """Extract a zip file and ingest the codebase.

        Args:
            zip_path: Path to the uploaded .zip file
            project_id: UUID of the project
            user_id: UUID of the user (for audit)

        Returns:
            IngestionResult with counts and errors
        """
        start = time.time()
        result = IngestionResult(project_id=project_id)
        temp_dir = None

        try:
            temp_dir = tempfile.mkdtemp(prefix="codeloom_ingest_")
            logger.info(f"Extracting zip to {temp_dir}")

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_dir)

            result = self._ingest_directory(temp_dir, project_id, user_id)

            # Update project source metadata
            self._update_project_source(project_id, "zip", None, None)

        except zipfile.BadZipFile:
            result.errors.append("Invalid zip file")
            self._update_project_status(project_id, "error")
        except Exception as e:
            logger.error(f"Zip ingestion failed: {e}")
            result.errors.append(str(e))
            self._update_project_status(project_id, "error")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        result.elapsed_seconds = time.time() - start
        self._log_result(result)
        return result

    def ingest_git(
        self,
        repo_url: str,
        branch: str,
        project_id: str,
        user_id: str,
    ) -> IngestionResult:
        """Clone a git repo and ingest the codebase.

        Args:
            repo_url: Git repository URL (https or ssh)
            branch: Branch to clone
            project_id: UUID of the project
            user_id: UUID of the user (for audit)

        Returns:
            IngestionResult with counts and errors
        """
        start = time.time()
        result = IngestionResult(project_id=project_id)
        temp_dir = None

        try:
            temp_dir = tempfile.mkdtemp(prefix="codeloom_git_")
            logger.info(f"Cloning {repo_url} (branch: {branch}) to {temp_dir}")

            proc = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                error_msg = proc.stderr.strip() or f"git clone failed with code {proc.returncode}"
                result.errors.append(error_msg)
                self._update_project_status(project_id, "error")
                result.elapsed_seconds = time.time() - start
                return result

            logger.info(f"Clone complete, starting ingestion")
            result = self._ingest_directory(temp_dir, project_id, user_id)

            # Update project source metadata
            self._update_project_source(project_id, "git", repo_url, branch)

        except subprocess.TimeoutExpired:
            result.errors.append("Git clone timed out (300s limit)")
            self._update_project_status(project_id, "error")
        except FileNotFoundError:
            result.errors.append("git is not installed or not in PATH")
            self._update_project_status(project_id, "error")
        except Exception as e:
            logger.error(f"Git ingestion failed: {e}")
            result.errors.append(str(e))
            self._update_project_status(project_id, "error")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        result.elapsed_seconds = time.time() - start
        self._log_result(result)
        return result

    def ingest_local(
        self,
        dir_path: str,
        project_id: str,
        user_id: str,
    ) -> IngestionResult:
        """Ingest from a local directory path (read-only, no copy).

        Args:
            dir_path: Absolute path to the local directory
            project_id: UUID of the project
            user_id: UUID of the user (for audit)

        Returns:
            IngestionResult with counts and errors
        """
        start = time.time()
        result = IngestionResult(project_id=project_id)

        if not os.path.isdir(dir_path):
            result.errors.append(f"Directory does not exist: {dir_path}")
            self._update_project_status(project_id, "error")
            result.elapsed_seconds = time.time() - start
            return result

        try:
            logger.info(f"Ingesting local directory: {dir_path}")
            result = self._ingest_directory(dir_path, project_id, user_id)

            # Update project source metadata
            self._update_project_source(project_id, "local", dir_path, None)

        except Exception as e:
            logger.error(f"Local ingestion failed: {e}")
            result.errors.append(str(e))
            self._update_project_status(project_id, "error")

        result.elapsed_seconds = time.time() - start
        self._log_result(result)
        return result

    # ── Core ingestion logic ─────────────────────────────────────────────

    def _ingest_directory(
        self,
        root_dir: str,
        project_id: str,
        user_id: str,
    ) -> IngestionResult:
        """Core ingestion: walk directory → parse → chunk → embed → store.

        Shared by ingest_zip, ingest_git, and ingest_local.

        Args:
            root_dir: Root directory containing source files
            project_id: UUID of the project
            user_id: UUID of the user

        Returns:
            IngestionResult with counts and errors
        """
        result = IngestionResult(project_id=project_id)

        source_files = self._collect_files(root_dir)
        logger.info(f"Found {len(source_files)} source files")

        if len(source_files) > self._max_files:
            result.errors.append(
                f"Too many files ({len(source_files)}). Maximum is {self._max_files}."
            )
            return result

        self._update_project_status(project_id, "parsing")

        all_nodes = []
        total_lines = 0
        languages_seen = set()

        for file_path in source_files:
            try:
                file_result = self._process_file(
                    file_path, root_dir, project_id, result
                )
                if file_result:
                    nodes, line_count, language = file_result
                    all_nodes.extend(nodes)
                    total_lines += line_count
                    if language:
                        languages_seen.add(language)
                    result.files_processed += 1
                else:
                    result.files_skipped += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                result.errors.append(f"{file_path}: {e}")
                result.files_skipped += 1

        result.chunks_created = len(all_nodes)

        if all_nodes:
            try:
                from llama_index.core import Settings as LISettings

                embed_model = LISettings.embed_model
                logger.info(f"Generating embeddings for {len(all_nodes)} chunks...")

                texts = [node.get_content() for node in all_nodes]
                BATCH_SIZE = 100
                all_embeddings = []
                for i in range(0, len(texts), BATCH_SIZE):
                    batch = texts[i : i + BATCH_SIZE]
                    batch_embeddings = embed_model.get_text_embedding_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(
                        f"Embedded batch {i // BATCH_SIZE + 1}/"
                        f"{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}"
                    )

                for node, embedding in zip(all_nodes, all_embeddings):
                    node.embedding = embedding

                logger.info(f"Embeddings generated, storing {len(all_nodes)} nodes...")
                self._vector_store.add_nodes(all_nodes, project_id=project_id)
                result.embeddings_stored = len(all_nodes)
                logger.info(f"Stored {len(all_nodes)} embeddings for project {project_id}")
            except Exception as e:
                logger.error(f"Failed to embed/store: {e}")
                result.errors.append(f"Embedding failed: {e}")

        primary_language = max(languages_seen, key=lambda l: l == "python") if languages_seen else None
        self._update_project_stats(
            project_id=project_id,
            file_count=result.files_processed,
            total_lines=total_lines,
            primary_language=primary_language,
            languages=sorted(languages_seen),
            ast_status="complete",
        )

        # Build ASG edges after all files are parsed and stored
        try:
            self._update_asg_status(project_id, "building")
            from ..asg_builder import ASGBuilder
            asg = ASGBuilder(self._db)
            edge_count = asg.build_edges(project_id)
            self._update_asg_status(project_id, "complete")
            logger.info(f"ASG built: {edge_count} edges for project {project_id}")
        except Exception as e:
            logger.error(f"ASG build failed: {e}")
            self._update_asg_status(project_id, "error")

        return result

    # ── Private helpers ──────────────────────────────────────────────────

    def _collect_files(self, root_dir: str) -> List[str]:
        """Walk directory tree and collect supported source files."""
        files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if not should_skip_directory(d)]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                    if size > self._max_file_size_bytes:
                        continue
                except OSError:
                    continue

                language = detect_language(fname)
                if language:
                    files.append(full_path)

        return files

    def _process_file(
        self,
        file_path: str,
        project_root: str,
        project_id: str,
        result: IngestionResult,
    ):
        """Process a single file: parse → enrich → store records → chunk.

        Returns:
            Tuple of (nodes, line_count, language) or None if skipped
        """
        parse_result = parse_file(file_path, project_root)

        if not parse_result.units and not parse_result.imports:
            return None

        # Semantic enrichment: add structured params/return_type/modifiers
        try:
            source_text = None
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_text = f.read()
            if source_text and parse_result.units:
                from ..ast_parser.enricher import SemanticEnricher
                enricher = SemanticEnricher()
                enricher.enrich_units(parse_result.units, source_text, parse_result.language)

                # Bridge enrichment (JavaParser / Roslyn) — graceful if unavailable
                self._run_bridge_enrichment(file_path, parse_result.units, parse_result.language)
        except Exception as e:
            logger.warning(f"Semantic enrichment failed for {file_path}: {e}")

        rel_path = parse_result.file_path
        file_size = os.path.getsize(file_path)

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        file_id = str(uuid4())
        with self._db.get_session() as session:
            code_file = CodeFile(
                file_id=UUID(file_id),
                project_id=UUID(project_id),
                file_path=rel_path,
                language=parse_result.language,
                line_count=parse_result.line_count,
                size_bytes=file_size,
                file_hash=file_hash,
            )
            session.add(code_file)

            # Stamp file-level imports on the first unit per file so
            # the ASG builder can detect cross-file import edges.
            imports_stamped = False

            for unit in parse_result.units:
                unit_id = uuid4()
                unit.metadata["unit_id"] = str(unit_id)
                if unit.parent_name:
                    unit.metadata["parent_name"] = unit.parent_name
                if not imports_stamped and parse_result.imports:
                    unit.metadata["file_imports"] = parse_result.imports
                    imports_stamped = True
                code_unit = CodeUnitModel(
                    unit_id=unit_id,
                    file_id=UUID(file_id),
                    project_id=UUID(project_id),
                    unit_type=unit.unit_type,
                    name=unit.name,
                    qualified_name=unit.qualified_name,
                    language=unit.language,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    signature=unit.signature,
                    docstring=unit.docstring,
                    source=unit.source,
                    unit_metadata=unit.metadata or {},
                )
                session.add(code_unit)

            result.units_extracted += len(parse_result.units)

        nodes = self._chunker.chunk_file(parse_result, project_id, file_id)

        return nodes, parse_result.line_count, parse_result.language

    def _run_bridge_enrichment(self, file_path: str, units, language: str) -> None:
        """Run optional JavaParser/Roslyn bridge enrichment.

        Silently skips if bridges are unavailable (no Java/dotnet installed).
        """
        try:
            if language == "java":
                from ..ast_parser.bridges.java_bridge import JavaParserBridge
                bridge = JavaParserBridge()
                if bridge.is_available():
                    bridge.enrich(file_path, units)
            elif language == "csharp":
                from ..ast_parser.bridges.dotnet_bridge import RoslynBridge
                bridge = RoslynBridge()
                if bridge.is_available():
                    bridge.enrich(file_path, units)
        except ImportError:
            pass  # Bridges not yet installed
        except Exception as e:
            logger.debug(f"Bridge enrichment skipped for {file_path}: {e}")

    def _update_asg_status(self, project_id: str, status: str):
        """Update project's asg_status."""
        try:
            with self._db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()
                if project:
                    project.asg_status = status
        except Exception as e:
            logger.error(f"Failed to update ASG status: {e}")

    def _update_project_status(self, project_id: str, status: str):
        """Update project's ast_status."""
        try:
            with self._db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()
                if project:
                    project.ast_status = status
        except Exception as e:
            logger.error(f"Failed to update project status: {e}")

    def _update_project_stats(
        self,
        project_id: str,
        file_count: int,
        total_lines: int,
        primary_language: Optional[str],
        languages: list,
        ast_status: str,
    ):
        """Update project statistics after ingestion."""
        try:
            with self._db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()
                if project:
                    project.file_count = file_count
                    project.total_lines = total_lines
                    project.primary_language = primary_language
                    project.languages = languages
                    project.ast_status = ast_status
                    project.last_synced_at = datetime.utcnow()
                    logger.info(f"Updated project stats: {file_count} files, {total_lines} lines")
        except Exception as e:
            logger.error(f"Failed to update project stats: {e}")

    def _update_project_source(
        self,
        project_id: str,
        source_type: str,
        source_url: Optional[str],
        repo_branch: Optional[str],
    ):
        """Update project source tracking fields."""
        try:
            with self._db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()
                if project:
                    project.source_type = source_type
                    project.source_url = source_url
                    project.repo_branch = repo_branch
        except Exception as e:
            logger.error(f"Failed to update project source: {e}")

    def _log_result(self, result: IngestionResult):
        """Log ingestion result summary."""
        logger.info(
            f"Ingestion complete: {result.files_processed} files, "
            f"{result.units_extracted} units, {result.chunks_created} chunks, "
            f"{result.embeddings_stored} embeddings in {result.elapsed_seconds:.1f}s"
        )
