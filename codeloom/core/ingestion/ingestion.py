import os
import re
import logging
import hashlib
import pickle
import asyncio
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms.llm import LLM
from dotenv import load_dotenv
from tqdm import tqdm

from ...setting import get_settings, RAGSettings
from .synopsis_manager import SynopsisManager
from ..db import DatabaseManager
from ..project import ProjectManager
# Lazy import: ContextualRetrievalService loaded only when contextual retrieval is enabled
# (avoids hard dependency on deleted transformations module)

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class DocumentReader:
    """Handles reading various document formats."""

    SUPPORTED_TEXT_FORMATS = ('.pdf', '.epub', '.txt')
    SUPPORTED_MARKDOWN_FORMATS = ('.md', '.markdown')
    SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.tiff', '.png', '.gif', '.webp')

    def __init__(self):
        self._textract_available = self._check_textract_available()
        self._docx_available = self._check_docx_available()
        self._pptx_available = self._check_pptx_available()
        self._vision_manager = self._init_vision_manager()

    def _init_vision_manager(self):
        """Initialize VisionManager for image processing."""
        try:
            from ..vision import get_vision_manager
            manager = get_vision_manager()
            if manager.is_available():
                logger.info("VisionManager initialized for image processing")
                return manager
            else:
                logger.debug("VisionManager not available (no API keys configured)")
                return None
        except Exception as e:
            logger.debug(f"VisionManager not available: {e}")
            return None

    def _check_textract_available(self) -> bool:
        """Check if AWS Textract credentials are available."""
        return bool(
            os.getenv("AWS_ACCESS_KEY_ID") and
            os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    def _check_docx_available(self) -> bool:
        """Check if docx support is available."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return True
        except ImportError:
            return False

    def _check_pptx_available(self) -> bool:
        """Check if pptx support is available."""
        try:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            return True
        except ImportError:
            return False

    def read(self, file_path: str) -> str:
        """Read document and return extracted text."""
        file_name = Path(file_path).name.lower()

        if file_name.endswith(self.SUPPORTED_TEXT_FORMATS):
            return self._read_pdf_like(file_path)
        elif file_name.endswith(self.SUPPORTED_MARKDOWN_FORMATS):
            return self._read_markdown(file_path)
        elif file_name.endswith(".docx"):
            return self._read_docx(file_path)
        elif file_name.endswith(".pptx"):
            return self._read_pptx(file_path)
        elif file_name.endswith(self.SUPPORTED_IMAGE_FORMATS):
            return self._read_image(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_name}")
            return ""

    def _read_pdf_like(self, file_path: str) -> str:
        """Read PDF using pymupdf4llm for table-aware markdown extraction."""
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

    def _read_markdown(self, file_path: str) -> str:
        """Read markdown files as plain text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading markdown {file_path}: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error reading markdown {file_path}: {e}")
            return ""

    def _read_docx(self, file_path: str) -> str:
        """Read DOCX files."""
        if not self._docx_available:
            logger.warning("DOCX support not available. Install langchain-community.")
            return ""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            document = loader.load()
            return " ".join([page.page_content for page in document])
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""

    def _read_pptx(self, file_path: str) -> str:
        """Read PPTX files."""
        if not self._pptx_available:
            logger.warning("PPTX support not available. Install langchain-community.")
            return ""
        try:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(file_path)
            document = loader.load()
            return " ".join([page.page_content for page in document])
        except Exception as e:
            logger.error(f"Error reading PPTX {file_path}: {e}")
            return ""

    def _read_image(self, file_path: str) -> str:
        """Read image files using VisionManager or AWS Textract OCR.

        Uses VisionManager (Gemini/OpenAI Vision) as primary method,
        falls back to AWS Textract if vision providers are not available.
        """
        # Try VisionManager first (Gemini/OpenAI Vision)
        if self._vision_manager:
            try:
                logger.debug(f"Processing image with VisionManager: {file_path}")
                result = self._vision_manager.analyze_image(file_path)

                # Combine description and extracted text
                content_parts = []
                if result.description:
                    content_parts.append(f"Image Description: {result.description}")
                if result.text_content and result.text_content.lower() != "no text found":
                    content_parts.append(f"Extracted Text: {result.text_content}")

                if content_parts:
                    logger.info(f"Successfully processed image with {result.provider}: {file_path}")
                    return "\n\n".join(content_parts)
            except Exception as e:
                logger.warning(f"VisionManager failed for {file_path}: {e}")
                # Fall through to Textract

        # Fall back to AWS Textract
        if not self._textract_available:
            if not self._vision_manager:
                logger.warning(
                    "Image processing not available. Configure either:\n"
                    "- GOOGLE_API_KEY or OPENAI_API_KEY for Vision providers, or\n"
                    "- AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for Textract"
                )
            return ""

        try:
            from langchain_community.document_loaders import AmazonTextractPDFLoader
            loader = AmazonTextractPDFLoader(file_path)
            document = loader.load()
            return " ".join([page.page_content for page in document])
        except Exception as e:
            logger.error(f"Error reading image {file_path} with Textract: {e}")
            return ""


class TextProcessor:
    """Handles basic text cleanup and normalization."""

    # Pattern for normalizing whitespace
    WHITESPACE_PATTERN = re.compile(r'\s+')

    @classmethod
    def filter_text(cls, text: str) -> str:
        """Basic text cleanup - normalize whitespace only.

        Preserves all characters including unicode, special characters, etc.
        Only normalizes excessive whitespace to improve readability.
        """
        if not text:
            return ""
        # Just normalize whitespace - keep all characters including unicode
        normalized_text = cls.WHITESPACE_PATTERN.sub(' ', text.strip())
        return normalized_text


class NodeCache:
    """Handles caching of processed nodes to disk."""

    def __init__(self, cache_dir: str = "data/node_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time."""
        file_path = Path(file_path)
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            content = f"{file_path.name}_{mtime}"
        else:
            content = file_path.name
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, file_path: str) -> Optional[List[BaseNode]]:
        """Get cached nodes for a file."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache for {file_path}: {e}")
        return None

    def set(self, file_path: str, nodes: List[BaseNode]) -> None:
        """Cache nodes for a file."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(nodes, f)
        except Exception as e:
            logger.warning(f"Error caching nodes for {file_path}: {e}")

    def clear(self) -> None:
        """Clear all cached nodes."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")


class LocalDataIngestion:
    """Handles document ingestion with parallel processing and caching."""

    def __init__(
        self,
        setting: RAGSettings | None = None,
        max_workers: int = 4,
        use_cache: bool = False,  # Disabled by default - use fresh DB queries instead
        db_manager: Optional[DatabaseManager] = None,
        vector_store = None,
        transformation_callback: Optional[callable] = None,
    ) -> None:
        self._setting = setting or get_settings()
        self._node_store: dict[str, List[BaseNode]] = {}
        self._ingested_file: List[str] = []
        self._max_workers = max_workers

        # Clean up deprecated cache files on startup
        self._cleanup_old_cache()

        # Initialize components
        self._reader = DocumentReader()
        self._processor = TextProcessor()
        self._cache = NodeCache() if use_cache else None  # Disabled - causes stale data issues
        self._synopsis_manager = SynopsisManager()

        # Project integration
        self._db_manager = db_manager
        self._project_manager = ProjectManager(db_manager) if db_manager else None

        # Vector store for pgvector persistence
        self._vector_store = vector_store

        # Callback for queuing AI transformations after upload
        # Signature: callback(source_id: str, document_text: str, project_id: str, file_name: str)
        self._transformation_callback = transformation_callback

        # Contextual retrieval enrichment (Anthropic approach)
        # Enriches chunks with LLM-generated context to improve retrieval for structured content
        self._contextual_retrieval_enabled = self._setting.contextual_retrieval.enabled
        self._context_service = None
        if self._contextual_retrieval_enabled:
            try:
                from ..transformations.context_service import ContextualRetrievalService
                self._context_service = ContextualRetrievalService(
                    batch_size=self._setting.contextual_retrieval.batch_size,
                    max_concurrency=self._setting.contextual_retrieval.max_concurrency
                )
                logger.info("Contextual retrieval enabled - chunks will be enriched with LLM context")
            except ImportError:
                logger.warning("ContextualRetrievalService not available - contextual retrieval disabled")
                self._contextual_retrieval_enabled = False

        # Create splitter once
        self._splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex
        )

    def _cleanup_old_cache(self) -> None:
        """Clean up deprecated node cache files on startup."""
        cache_dir = Path("data/node_cache")
        if cache_dir.exists():
            deleted_count = 0
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} deprecated node cache files")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file contents for duplicate detection."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _enrich_nodes_with_context(
        self,
        nodes: List[BaseNode],
        doc_title: str
    ) -> List[BaseNode]:
        """Enrich nodes with LLM-generated contextual content.

        Uses the ContextualRetrievalService to generate context descriptions
        for each chunk, which improves retrieval for structured content.

        Args:
            nodes: List of nodes to enrich
            doc_title: Document title/filename for context

        Returns:
            List of enriched nodes (or original nodes if enrichment fails)
        """
        if not self._context_service:
            return nodes

        # Convert to TextNodes if needed
        text_nodes = []
        for node in nodes:
            if isinstance(node, TextNode):
                text_nodes.append(node)
            else:
                # Convert BaseNode to TextNode
                text_nodes.append(TextNode(
                    text=node.get_content(),
                    metadata=node.metadata.copy() if hasattr(node, 'metadata') else {}
                ))

        # Run async enrichment in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self._context_service.enrich_chunks(text_nodes, doc_title)
            )

            if result.success_count > 0:
                logger.info(
                    f"Context enrichment: {result.success_count}/{len(nodes)} chunks enriched"
                )
                if result.errors:
                    logger.warning(f"Context enrichment errors: {result.errors[:3]}")
                return result.enriched_nodes
            else:
                logger.warning("Context enrichment produced no results, using original nodes")
                return nodes

        except Exception as e:
            logger.error(f"Context enrichment async error: {e}")
            return nodes

    def _process_single_file(
        self,
        input_file: str,
        embed_nodes: bool = True,
        embed_model: Any | None = None
    ) -> tuple[str, List[BaseNode], str]:
        """Process a single file and return (filename, nodes, document_text).

        Args:
            input_file: Path to the file
            embed_nodes: Whether to embed nodes
            embed_model: Embedding model to use

        Returns:
            Tuple of (filename, nodes, document_text)
        """
        file_name = Path(input_file).name
        file_path = Path(input_file)

        # Calculate current file hash for cache validation
        current_file_hash = self._calculate_file_hash(input_file)

        # Check memory cache first - but validate hash to detect file changes
        if file_name in self._node_store:
            cached_nodes = self._node_store[file_name]
            # Check if file has changed by comparing hash
            if cached_nodes and len(cached_nodes) > 0:
                cached_hash = cached_nodes[0].metadata.get("file_hash", "")
                if cached_hash == current_file_hash:
                    return file_name, cached_nodes, ""
                else:
                    # File changed, invalidate memory cache
                    logger.info(f"File {file_name} changed, invalidating memory cache")
                    del self._node_store[file_name]

        # Check disk cache (uses mtime in key, so auto-invalidates on file change)
        if self._cache:
            cached_nodes = self._cache.get(input_file)
            if cached_nodes:
                logger.debug(f"Using cached nodes for {file_name}")
                return file_name, cached_nodes, ""

        # Read and process document
        raw_text = self._reader.read(input_file)
        if not raw_text:
            logger.warning(f"No text extracted from {file_name}")
            return file_name, [], ""

        filtered_text = self._processor.filter_text(raw_text)

        # Calculate file metadata
        file_hash = self._calculate_file_hash(input_file)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        upload_timestamp = datetime.now().isoformat()

        # Create metadata
        metadata = {
            "file_name": file_name,
            "file_hash": file_hash,
            "file_size": file_size,
            "upload_timestamp": upload_timestamp,
        }

        # Create document and split into nodes
        document = Document(
            text=filtered_text.strip(),
            metadata=metadata
        )

        nodes = self._splitter([document], show_progress=False)

        # Apply contextual retrieval enrichment if enabled
        # This adds LLM-generated context to each chunk to improve retrieval
        if self._context_service and nodes:
            try:
                logger.info(f"Enriching {len(nodes)} chunks with contextual content for {file_name}")
                nodes = self._enrich_nodes_with_context(nodes, file_name)
                logger.info(f"Context enrichment complete for {file_name}")
            except Exception as e:
                logger.warning(f"Context enrichment failed for {file_name}: {e} - using original chunks")

        # Embed nodes if requested
        if embed_nodes and nodes:
            model = embed_model or Settings.embed_model
            if model:
                try:
                    nodes = model(nodes, show_progress=False)
                except Exception as e:
                    logger.error(f"Error embedding nodes for {file_name}: {e}")

        # Cache to disk
        if self._cache and nodes:
            self._cache.set(input_file, nodes)

        return file_name, nodes, filtered_text

    def store_nodes(
        self,
        input_files: list[str],
        embed_nodes: bool = True,
        embed_model: Any | None = None,
        project_id: Optional[str] = None,
        user_id: str = "default"
    ) -> List[BaseNode]:
        """Process multiple files with parallel execution and enhanced metadata.

        Args:
            input_files: List of file paths to process
            embed_nodes: Whether to embed nodes
            embed_model: Embedding model to use
            project_id: Project ID for document organization
            user_id: User ID for multi-user support (default: "default")

        Returns:
            List of processed nodes with enhanced metadata
        """
        return_nodes: List[BaseNode] = []
        # Don't clear _ingested_file to allow multiple offerings to accumulate
        # self._ingested_file = []

        if not input_files:
            return return_nodes

        if embed_nodes:
            Settings.embed_model = embed_model or Settings.embed_model

        # Log metadata info
        if project_id:
            logger.info(
                f"Processing {len(input_files)} files for project: {project_id}"
            )

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_file,
                    input_file,
                    embed_nodes,
                    embed_model
                ): input_file
                for input_file in input_files
            }

            # Use tqdm for progress tracking
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Ingesting documents"
            ):
                try:
                    file_name, nodes, document_text = future.result()
                    if nodes:
                        # Register document in project database if project_id provided
                        source_id = None
                        if project_id and self._project_manager:
                            try:
                                # Read file content for hash calculation
                                input_file = futures[future]
                                with open(input_file, 'rb') as f:
                                    file_content = f.read()

                                # Determine file type
                                file_type = Path(input_file).suffix.lstrip('.')

                                # Register in database
                                source_id = self._project_manager.add_document(
                                    project_id=project_id,
                                    file_name=file_name,
                                    file_content=file_content,
                                    file_type=file_type,
                                    chunk_count=len(nodes)
                                )
                                logger.info(f"Registered {file_name} in project {project_id} (source_id: {source_id})")

                                # Queue AI transformations if callback is set
                                if self._transformation_callback and source_id and document_text:
                                    try:
                                        self._transformation_callback(
                                            source_id=source_id,
                                            document_text=document_text,
                                            project_id=project_id,
                                            file_name=file_name
                                        )
                                        logger.debug(f"Queued transformations for {file_name}")
                                    except Exception as te:
                                        logger.warning(f"Failed to queue transformations for {file_name}: {te}")

                            except ValueError as e:
                                # Duplicate document detected - but still need to create embeddings
                                # Try to get existing source_id so we can continue with embedding creation
                                logger.info(f"Document {file_name} already registered, looking up source_id for embedding creation")
                                try:
                                    docs = self._project_manager.get_documents(project_id)
                                    for doc in docs:
                                        if doc.get('file_name') == file_name:
                                            source_id = doc.get('source_id')
                                            logger.info(f"Found existing source_id: {source_id} for {file_name}")
                                            # Track this source for chunk_count update after embedding creation
                                            if not hasattr(self, '_sources_needing_chunk_update'):
                                                self._sources_needing_chunk_update = {}
                                            self._sources_needing_chunk_update[source_id] = len(nodes)
                                            break
                                except Exception as lookup_err:
                                    logger.warning(f"Could not look up existing source_id for {file_name}: {lookup_err}")
                                # Continue with embedding creation (don't skip)
                            except Exception as e:
                                logger.error(f"Error registering {file_name} in database: {e}")
                                raise

                        # Add project metadata to all nodes
                        if project_id:
                            for node in nodes:
                                if hasattr(node, 'metadata'):
                                    node.metadata["project_id"] = project_id
                                    node.metadata["user_id"] = user_id
                                    node.metadata["tree_level"] = 0  # Mark as leaf node for RAPTOR
                                    if source_id:
                                        node.metadata["source_id"] = source_id

                        self._node_store[file_name] = nodes
                        self._ingested_file.append(file_name)
                        return_nodes.extend(nodes)

                        # Log node count with metadata
                        chunk_count = len(nodes)
                        logger.debug(
                            f"Processed {file_name}: {chunk_count} chunks, "
                            f"Project='{project_id}', Source ID='{source_id}'"
                        )
                except Exception as e:
                    input_file = futures[future]
                    logger.error(f"Error processing {input_file}: {e}")

        # Persist new nodes to pgvector using incremental add (with duplicate detection)
        # This is O(n) for n new nodes and handles duplicates gracefully
        if self._vector_store and project_id and return_nodes:
            try:
                logger.info(f"Adding {len(return_nodes)} new nodes to pgvector for project {project_id}")

                # Use add_nodes() for incremental add - handles duplicates automatically
                added_count = self._vector_store.add_nodes(return_nodes, project_id=project_id)

                # Verify nodes are persisted
                project_nodes = self._vector_store.get_nodes_by_project_sql(project_id)
                logger.info(
                    f"✓ Added {added_count} nodes to pgvector "
                    f"(total {len(project_nodes)} nodes for project {project_id})"
                )

                # Update chunk_count for any documents that were duplicates
                if hasattr(self, '_sources_needing_chunk_update') and self._sources_needing_chunk_update:
                    for source_id, chunk_count in self._sources_needing_chunk_update.items():
                        try:
                            self._project_manager.update_document_chunk_count(source_id, chunk_count)
                        except Exception as update_err:
                            logger.warning(f"Failed to update chunk_count for {source_id}: {update_err}")
                    self._sources_needing_chunk_update.clear()

            except Exception as e:
                logger.error(f"Error persisting nodes to pgvector: {e}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info(f"Total nodes created: {len(return_nodes)}")
        return return_nodes

    def reset(self) -> None:
        """Reset in-memory node store."""
        self._node_store.clear()
        self._ingested_file.clear()

    def clear_cache(self) -> None:
        """Clear both memory and disk cache."""
        self.reset()
        if self._cache:
            self._cache.clear()

    def check_nodes_exist(self) -> bool:
        """Check if any nodes are stored."""
        return len(self._node_store) > 0

    def get_all_nodes(self) -> List[BaseNode]:
        """Get all stored nodes."""
        return_nodes = []
        for nodes in self._node_store.values():
            return_nodes.extend(nodes)
        return return_nodes

    def get_ingested_nodes(self) -> List[BaseNode]:
        """Get nodes from currently ingested files."""
        return_nodes = []
        for file in self._ingested_file:
            if file in self._node_store:
                return_nodes.extend(self._node_store[file])
        return return_nodes

    def set_transformation_callback(self, callback: callable) -> None:
        """Set callback for queuing AI transformations after document upload.

        Args:
            callback: Function with signature:
                callback(source_id: str, document_text: str, project_id: str, file_name: str)
        """
        self._transformation_callback = callback
        logger.info("Transformation callback set for document processing")
