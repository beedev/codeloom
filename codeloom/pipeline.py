import logging
import os
import threading
import time
from typing import Optional, Dict, Tuple, List

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole

from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    PGVectorStore,
    get_system_prompt
)
from .core.db import DatabaseManager
from .core.notebook import NotebookManager
from .core.conversation import ConversationStore
from .core.observability import QueryLogger, get_token_counter
from .core.transformations import TransformationWorker, TransformationJob
from .core.raptor import RAPTORWorker, RAPTORJob
from .core.memory import SessionMemoryService
from .core.constants import DEFAULT_USER_ID
from .core.utils import unwrap_llm
from .setting import get_settings, QueryTimeSettings

logger = logging.getLogger(__name__)


class LocalRAGPipeline:
    """
    Main RAG pipeline orchestrating model, embedding, ingestion, and chat engine.

    MULTI-USER ARCHITECTURE
    =======================
    This class contains both single-user (global state) and multi-user (stateless) patterns.

    MULTI-USER SAFE METHODS (Thread-safe, no global state mutation):
    - stateless_query()           - Fast query with explicit user_id/notebook_id
    - stateless_query_streaming() - Streaming version of above
    - _get_cached_nodes()         - Thread-safe node cache access

    SINGLE-USER METHODS (Mutate global state - NOT safe for concurrent users):
    - switch_notebook()           - Sets _current_notebook_id, _current_user_id
    - set_engine()                - Sets _current_offering_filter
    - set_chat_mode()             - Sets engine mode
    - query()                     - Uses global engine state

    GLOBAL STATE (Document for awareness):
    - _current_notebook_id: Active notebook for single-user session
    - _current_user_id: Active user for single-user session
    - _current_offering_filter: Active notebook filter for retrieval

    RECOMMENDED USAGE:
    - For API/multi-user: Use stateless_query() with explicit IDs
    - For UI session: Use switch_notebook() + query() pattern
    - For concurrent users: ONLY use stateless_* methods

    Optimized for:
    - Single model/embedding initialization
    - Cached vector index with TTL (5 min)
    - Thread-safe node cache for multi-user access
    - Proper logging
    """

    def __init__(
        self,
        host: str = "host.docker.internal",
        database_url: Optional[str] = None
    ) -> None:
        self._host = host
        self._ollama_host = os.getenv("OLLAMA_HOST", host)
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._query_engine = None
        self._simple_chat_engine = None  # For general chat without retrieval
        self._settings = get_settings()

        # Engine state management for conversation history preservation
        self._engine_initialized: bool = False
        self._current_offering_filter: Optional[list[str]] = None  # Track current filter for isolation

        # Notebook context tracking (NotebookLM architecture)
        # NOTE: These are single-user state - use stateless_query() for multi-user
        self._current_notebook_id: Optional[str] = None
        self._current_user_id: str = DEFAULT_USER_ID  # Default user for single-user mode

        # Session memory for cross-request conversation persistence
        self._session_memory = SessionMemoryService(
            max_messages_per_session=100,
            session_ttl_hours=24
        )
        logger.info("Session memory service initialized")

        # Node cache for performance optimization (avoids reloading nodes from DB)
        # Cache format: {notebook_id: (nodes, timestamp, node_count)}
        self._node_cache: Dict[str, Tuple[List[TextNode], float, int]] = {}
        self._node_cache_ttl = 300  # 5 minutes TTL
        self._node_cache_lock = threading.Lock()  # Thread-safe cache access for multi-user

        # Sales mode components (optional, disabled by default)
        # These are legacy components for sales enablement features that are not currently implemented
        self._query_classifier = None  # Optional: Query classification for sales mode
        self._offering_analyzer = None  # Optional: Offering analysis for sales mode

        # Database and conversation management (optional, only if database_url provided)
        self._db_manager: Optional[DatabaseManager] = None
        self._notebook_manager: Optional[NotebookManager] = None
        self._conversation_store: Optional[ConversationStore] = None
        self._query_logger: Optional[QueryLogger] = None
        self._transformation_worker: Optional[TransformationWorker] = None
        self._raptor_worker: Optional[RAPTORWorker] = None
        if database_url:
            # Increased pool size for better concurrency under load
            # pool_size=20: 20 persistent connections
            # max_overflow=30: Up to 30 additional connections under high load
            self._db_manager = DatabaseManager(
                database_url,
                pool_size=20,
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=1800  # Recycle connections every 30 minutes
            )
            self._db_manager.init_db()
            self._notebook_manager = NotebookManager(self._db_manager)
            self._conversation_store = ConversationStore(self._db_manager)
            self._query_logger = QueryLogger(db_manager=self._db_manager)
            logger.info(f"Database initialized with notebook management, conversation persistence, and query logging")
        else:
            # Initialize in-memory query logger even without database
            self._query_logger = QueryLogger()
            logger.info("Query logger initialized (in-memory mode, notebook features disabled)")

        # Initialize components once - using PGVectorStore (pgvector)
        self._vector_store = PGVectorStore(
            host=host,
            setting=self._settings,
            persist=True
        )
        self._engine = LocalChatEngine(
            setting=self._settings,
            host=host
        )

        # Initialize TransformationWorker for AI transformations (if database available)
        # Skip in multi-worker mode (Gunicorn) - asyncio doesn't fork well
        transformation_callback = None
        skip_background_workers = os.getenv("DISABLE_BACKGROUND_WORKERS", "").lower() in ("true", "1", "yes")
        if self._db_manager and not skip_background_workers:
            self._transformation_worker = TransformationWorker(
                db_manager=self._db_manager,
                embed_callback=self._embed_transformation,  # Will embed transformation content
                poll_interval=10.0,
                max_concurrent=2,
            )
            self._transformation_worker.start()
            logger.info("TransformationWorker started for AI transformations")
        elif skip_background_workers:
            logger.info("TransformationWorker disabled (DISABLE_BACKGROUND_WORKERS=true)")

            # Create callback for ingestion to queue transformation jobs
            def transformation_callback(source_id: str, document_text: str, notebook_id: str, file_name: str):
                """Queue a transformation job when a document is ingested."""
                job = TransformationJob(
                    source_id=source_id,
                    document_text=document_text,
                    notebook_id=notebook_id,
                    file_name=file_name,
                )
                self._transformation_worker.queue_job(job)
                logger.debug(f"Queued transformation job for source: {source_id}")

        self._ingestion = LocalDataIngestion(
            setting=self._settings,
            max_workers=4,
            use_cache=False,  # Disabled - using pgvector for persistence
            db_manager=self._db_manager,
            vector_store=self._vector_store,
            transformation_callback=transformation_callback,
        )

        # Initialize models once and cache in Settings
        self._default_model = LocalRAGModel.set(
            model_name=self._model_name,
            host=host,
            setting=self._settings
        )
        # Settings.llm requires a proper LlamaIndex LLM instance
        # If using a wrapper (e.g., GroqWithBackoff), extract the raw LLM
        if hasattr(self._default_model, 'get_raw_llm'):
            Settings.llm = self._default_model.get_raw_llm()
        else:
            Settings.llm = self._default_model
        Settings.embed_model = LocalEmbedding.set(
            host=host,
            setting=self._settings
        )

        # Initialize RAPTORWorker for hierarchical tree building (if database available)
        # Skip in multi-worker mode (Gunicorn) - asyncio doesn't fork well
        if self._db_manager and not skip_background_workers:
            self._raptor_worker = RAPTORWorker(
                db_manager=self._db_manager,
                vector_store=self._vector_store,
                poll_interval=15.0,  # Check for pending trees every 15 seconds
                max_concurrent=1,  # Tree building is resource-intensive
            )
            self._raptor_worker.start()
            logger.info("RAPTORWorker started for hierarchical tree building")
        elif skip_background_workers:
            logger.info("RAPTORWorker disabled (DISABLE_BACKGROUND_WORKERS=true)")

        logger.info(f"Pipeline initialized - Host: {host}")
        logger.debug(f"LLM Model: {self._model_name or self._settings.ollama.llm}")
        logger.debug(f"Embed Model: {self._settings.ingestion.embed_llm}")

    def switch_notebook(
        self,
        notebook_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Switch to a different notebook with fresh session memory.

        Session-only mode:
        1. Clear existing in-memory conversation
        2. Update notebook context
        3. Recreate engine with notebook-filtered nodes
        4. Start fresh (no DB history loading)

        Args:
            notebook_id: UUID of the notebook to switch to
            user_id: UUID of the user (defaults to current user)

        Raises:
            ValueError: If conversation store is not initialized (database_url not provided)
        """
        if not self._conversation_store:
            raise ValueError(
                "Conversation persistence not available. "
                "Initialize pipeline with database_url to enable notebook features."
            )

        user_id = user_id or self._current_user_id

        # Step 1: Clear engine's in-memory conversation buffer
        if self._query_engine:
            try:
                self._query_engine.reset()
                logger.info("Cleared engine memory buffer")
            except Exception as e:
                logger.debug(f"No conversation to clear: {e}")

        # Step 2: Update notebook context
        self._current_notebook_id = notebook_id
        self._current_user_id = user_id
        logger.info(f"Switched to notebook: {notebook_id}")

        # Step 3: Get history from session memory (cross-request persistence)
        # This allows resuming conversations when switching back to a notebook
        chat_history = self._session_memory.get_history(user_id, notebook_id)
        if chat_history:
            logger.info(f"Restored {len(chat_history)} messages from session memory")
        else:
            logger.info("Starting fresh session (no previous history)")

        # Step 4: Load ONLY nodes for this notebook using cached SQL filtering
        # Uses caching to avoid repeated DB queries (100-300ms saved on cache hit)
        logger.info(f"Loading nodes for notebook {notebook_id}")
        notebook_nodes = self._get_cached_nodes(notebook_id)
        logger.info(f"Got {len(notebook_nodes)} nodes for notebook {notebook_id}")

        # Step 5: Recreate engine with notebook context and chat history
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=notebook_nodes,
            language=self._language,
            chat_history=chat_history
        )

        # Update engine state
        self._engine_initialized = True

        logger.info(
            f"Engine recreated for notebook {notebook_id} with "
            f"{len(notebook_nodes)} nodes (session-only memory)"
        )

    def get_model_name(self) -> str:
        return self._model_name

    def get_llm(self):
        """Get the current LLM instance for use by services like SQL Chat."""
        return Settings.llm

    def get_embed_model(self):
        """Get the current embedding model instance."""
        return Settings.embed_model

    def _get_cached_nodes(self, notebook_id: str) -> List[TextNode]:
        """
        Get nodes for a notebook with caching.

        Caches nodes per notebook with TTL to avoid repeated DB queries.
        Cache is invalidated when:
        - TTL expires (5 minutes)
        - Node count changes (document added/removed)

        Thread-safe for multi-user concurrent access.

        Args:
            notebook_id: UUID of the notebook

        Returns:
            List of TextNode objects for the notebook
        """
        with self._node_cache_lock:
            current_time = time.time()

            # Check cache
            if notebook_id in self._node_cache:
                nodes, timestamp, cached_count = self._node_cache[notebook_id]

                # Check TTL
                if current_time - timestamp < self._node_cache_ttl:
                    logger.debug(f"Cache hit for notebook {notebook_id}: {len(nodes)} nodes")
                    return nodes
                else:
                    logger.debug(f"Cache expired for notebook {notebook_id}")

            # Cache miss - load from DB
            start_time = time.time()
            nodes = self._vector_store.get_nodes_by_notebook_sql(notebook_id)
            load_time_ms = int((time.time() - start_time) * 1000)

            # Store in cache
            self._node_cache[notebook_id] = (nodes, current_time, len(nodes))
            logger.info(f"Cached {len(nodes)} nodes for notebook {notebook_id} (loaded in {load_time_ms}ms)")

            return nodes

    def invalidate_node_cache(self, notebook_id: Optional[str] = None) -> None:
        """
        Invalidate node cache for a notebook or all notebooks.

        Call this after document upload/delete to ensure fresh nodes.
        Thread-safe for multi-user concurrent access.

        Args:
            notebook_id: Specific notebook to invalidate, or None for all
        """
        with self._node_cache_lock:
            if notebook_id:
                if notebook_id in self._node_cache:
                    del self._node_cache[notebook_id]
                    logger.debug(f"Invalidated node cache for notebook {notebook_id}")
            else:
                self._node_cache.clear()
                logger.debug("Invalidated all node caches")

    def set_model_name(self, model_name: str) -> None:
        self._model_name = model_name
        logger.debug(f"Model name set to: {model_name}")

    def get_language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        self._language = language
        logger.debug(f"Language set to: {language}")

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def set_system_prompt(self, system_prompt: Optional[str] = None) -> None:
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language,
            is_rag_prompt=self._ingestion.check_nodes_exist()
        )
        logger.debug("System prompt updated")

    def set_model(self) -> None:
        """Update the LLM model with current settings."""
        self._default_model = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._ollama_host,
            setting=self._settings
        )
        # Unwrap LLM wrappers (e.g., GroqWithBackoff) for LlamaIndex Settings compatibility
        Settings.llm = unwrap_llm(self._default_model)
        logger.info(f"Model updated: {self._model_name}")

    def reset_engine(self) -> None:
        """Reset the chat engine without documents."""
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=[],
            language=self._language
        )
        logger.debug("Engine reset (no documents)")

    def reset_documents(self) -> None:
        """Reset all ingested documents."""
        self._ingestion.reset()
        self._vector_store.reset()
        self.invalidate_node_cache()  # Clear all node caches
        logger.info("Documents reset")

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self._query_engine:
            self._query_engine.reset()
            logger.debug("Conversation cleared")

    def reset_conversation(self) -> None:
        """Reset conversation and switch to non-RAG mode."""
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )
        logger.info("Conversation reset to non-RAG mode")

    def set_embed_model(self, model_name: str) -> None:
        """Update the embedding model."""
        Settings.embed_model = LocalEmbedding.set(
            model_name=model_name,
            host=self._ollama_host,
            setting=self._settings
        )
        logger.info(f"Embedding model updated: {model_name}")

    def pull_model(self, model_name: str):
        """Pull an LLM model from Ollama."""
        logger.info(f"Pulling model: {model_name}")
        return LocalRAGModel.pull(self._ollama_host, model_name)

    def pull_embed_model(self, model_name: str):
        """Pull an embedding model from Ollama."""
        logger.info(f"Pulling embedding model: {model_name}")
        return LocalEmbedding.pull(self._ollama_host, model_name)

    def check_exist(self, model_name: str) -> bool:
        """Check if an LLM model exists on Ollama."""
        return LocalRAGModel.check_model_exist(self._ollama_host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        """Check if an embedding model exists on Ollama."""
        return LocalEmbedding.check_model_exist(self._ollama_host, model_name)

    def _embed_transformation(
        self,
        text: str,
        node_type: str,
        source_id: str,
        notebook_id: str
    ) -> None:
        """
        Embed transformation content and store in vector store.

        Called by TransformationWorker after generating transformations.
        This allows transformation content (summaries, insights, questions)
        to be retrieved alongside regular document chunks.

        Args:
            text: The transformation text to embed
            node_type: Type of transformation ("summary", "insight", "question")
            source_id: Source document UUID
            notebook_id: Notebook UUID for filtering
        """
        try:
            from llama_index.core.schema import TextNode
            from uuid import UUID
            import uuid

            # Look up the source document to get file_name
            file_name = None
            if self._db_manager:
                try:
                    with self._db_manager.get_session() as session:
                        from .core.db.models import NotebookSource
                        source = session.query(NotebookSource).filter(
                            NotebookSource.source_id == UUID(source_id)
                        ).first()
                        if source:
                            file_name = source.file_name
                except Exception as e:
                    logger.warning(f"Could not lookup file_name for source {source_id}: {e}")

            # Create a TextNode with transformation metadata including file_name
            node = TextNode(
                text=text,
                id_=str(uuid.uuid4()),
                metadata={
                    "source_id": source_id,
                    "notebook_id": notebook_id,
                    "node_type": node_type,
                    "file_name": file_name,  # Parent document name for source display
                }
            )

            # Generate embedding for the node using Settings.embed_model
            embed_model = Settings.embed_model
            if embed_model:
                try:
                    # Generate embedding for this node
                    embedding = embed_model.get_text_embedding(text)
                    node.embedding = embedding
                    logger.debug(f"Generated embedding for {node_type} (dim={len(embedding)})")
                except Exception as e:
                    logger.error(f"Error generating embedding for {node_type}: {e}")
                    return  # Don't add node without embedding
            else:
                logger.warning("No embed model available for transformation embedding")
                return

            # Add to vector store with proper metadata
            if hasattr(self._vector_store, 'add_transformation_nodes'):
                self._vector_store.add_transformation_nodes(
                    nodes=[node],
                    notebook_id=notebook_id,
                    source_id=source_id
                )
                logger.debug(f"Embedded {node_type} for source {source_id}")
            else:
                logger.warning("Vector store does not support add_transformation_nodes")

        except Exception as e:
            logger.error(f"Error embedding transformation for source {source_id}: {e}")

    def store_nodes(
        self,
        input_files: Optional[list[str]] = None,
        notebook_id: Optional[str] = None,
        user_id: str = "default"
    ) -> None:
        """
        Process and store document nodes with metadata.

        Args:
            input_files: List of file paths to process
            notebook_id: Notebook UUID for NotebookLM-style isolation (optional)
            user_id: User identifier (defaults to "default")

        Uses parallel processing and caching for efficiency.
        """
        if not input_files:
            logger.warning("No input files provided")
            return

        logger.info(f"Processing {len(input_files)} files")
        self._ingestion.store_nodes(
            input_files=input_files,
            notebook_id=notebook_id,
            user_id=user_id
        )

        # Invalidate node cache for this notebook (nodes have changed)
        if notebook_id:
            self.invalidate_node_cache(notebook_id)

        logger.info("Document processing complete")

    def set_chat_mode(self, system_prompt: Optional[str] = None, force_reset: bool = False) -> None:
        """Configure chat mode with current documents and settings.

        Args:
            system_prompt: Optional system prompt to set
            force_reset: If True, force rebuild of chat engine (needed after document upload)
        """
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine(force_reset=force_reset)
        logger.debug(f"Chat mode configured (force_reset={force_reset})")

    def load_notebook_documents(self, notebook_ids: list[str]) -> int:
        """
        Load and ingest documents from specified notebooks.

        Args:
            notebook_ids: List of notebook IDs to load documents from

        Returns:
            Number of documents loaded
        """
        if not self._notebook_manager:
            logger.warning("Cannot load notebook documents - notebook manager not available")
            return 0

        import os
        from pathlib import Path

        total_loaded = 0

        for notebook_id in notebook_ids:
            try:
                # Get documents for this notebook from database
                documents = self._notebook_manager.get_documents(notebook_id)

                if not documents:
                    logger.info(f"No documents found for notebook {notebook_id}")
                    continue

                logger.info(f"Found {len(documents)} documents for notebook {notebook_id}")

                # Load each document from disk and ingest with notebook_id metadata
                for doc in documents:
                    file_name = doc['file_name']
                    file_path = os.path.join(self._ingestion._data_dir, file_name)

                    if not os.path.exists(file_path):
                        logger.warning(f"Document file not found: {file_path}")
                        continue

                    # Ingest with notebook_id metadata
                    logger.info(f"Loading document: {file_name} with notebook_id={notebook_id}")
                    self._ingestion.store_nodes(
                        input_files=[file_path],
                        notebook_id=notebook_id
                    )
                    total_loaded += 1

                logger.info(f"Loaded {total_loaded} documents for notebook {notebook_id}")

            except Exception as e:
                logger.error(f"Error loading documents for notebook {notebook_id}: {e}")
                continue

        return total_loaded

    def set_engine(
        self,
        offering_filter: Optional[list[str]] = None,
        force_reset: bool = False
    ) -> None:
        """Set up the chat engine with current nodes, optionally filtered by offerings.
        Only recreates engine if filter changes or force_reset is True.

        Args:
            offering_filter: List of offering names/IDs or notebook IDs to filter by
            force_reset: Force recreation of engine even if filter unchanged
        """
        # Check if filter has changed (critical for notebook isolation)
        filter_changed = offering_filter != self._current_offering_filter
        if filter_changed:
            logger.info(f"Filter changed: {self._current_offering_filter} -> {offering_filter}")

        # Recreate engine if not initialized, filter changed, or force_reset requested
        if not self._engine_initialized or filter_changed or force_reset:
            logger.info(f"Creating new engine with filter: {offering_filter}")

            # PRESERVE chat history from session memory (cross-request persistence)
            preserved_history = []
            # Determine notebook_id: use current or extract from offering_filter
            notebook_id_for_history = self._current_notebook_id
            if not notebook_id_for_history and offering_filter and len(offering_filter) == 1:
                # Single notebook filter - use it for history lookup
                notebook_id_for_history = offering_filter[0]

            if notebook_id_for_history:
                # Get history from session memory service
                preserved_history = self._session_memory.get_history(
                    user_id=self._current_user_id,
                    notebook_id=notebook_id_for_history
                )
                if preserved_history:
                    logger.info(f"✓ Restored {len(preserved_history)} messages from session memory for notebook {notebook_id_for_history}")
                else:
                    logger.debug(f"No history in session memory for notebook {notebook_id_for_history}")
            else:
                # Fallback: try to extract from existing engine memory buffer
                if self._query_engine is not None and hasattr(self._query_engine, 'memory'):
                    try:
                        preserved_history = self._query_engine.memory.get_all()
                        logger.info(f"✓ Preserved {len(preserved_history)} messages from engine memory")
                    except Exception as e:
                        logger.warning(f"❌ Could not extract chat history: {e}")
                        preserved_history = []

            # Load nodes using cached SQL filtering (O(log n) vs O(n))
            # Uses caching to avoid repeated DB queries (100-300ms saved on cache hit)
            if offering_filter:
                # offering_filter can contain notebook_ids, offering_names, or offering_ids
                # Try cached SQL-based loading for notebook_ids (most common case)
                nodes = []
                for filter_id in offering_filter:
                    # Try to load as notebook_id first (UUID format) with caching
                    notebook_nodes = self._get_cached_nodes(filter_id)
                    if notebook_nodes:
                        nodes.extend(notebook_nodes)
                        logger.debug(f"Got {len(notebook_nodes)} cached nodes for notebook {filter_id}")

                # If no nodes found via notebook_id, fall back to offering filter
                # This handles legacy offering_name/offering_id filters
                if not nodes:
                    logger.info("No notebook nodes found, falling back to offering filter")
                    all_nodes = self._vector_store.load_all_nodes()
                    for node in all_nodes:
                        metadata = node.metadata or {}
                        node_offering_name = metadata.get("offering_name")
                        node_offering_id = metadata.get("offering_id")
                        if (node_offering_name and node_offering_name in offering_filter) or \
                           (node_offering_id and node_offering_id in offering_filter):
                            nodes.append(node)
                    logger.info(f"Filtered {len(nodes)} nodes using offering filter")
                else:
                    logger.info(f"Loaded {len(nodes)} nodes using SQL filter (filter={offering_filter})")
            else:
                # No filter - load all nodes (only when truly needed)
                logger.info("Loading all nodes from pgvector (no filter)")
                nodes = self._vector_store.load_all_nodes()
                logger.info(f"Loaded {len(nodes)} total nodes")


            # Create new engine WITH preserved chat history
            logger.info(f"Creating new engine with {len(preserved_history)} preserved messages")

            # Determine notebook_id for RAPTOR-aware retrieval
            # Use current_notebook_id if set, otherwise extract from offering_filter
            notebook_id = self._current_notebook_id
            if not notebook_id and offering_filter and len(offering_filter) == 1:
                # Single notebook filter - use it as notebook_id for RAPTOR
                notebook_id = offering_filter[0]

            self._query_engine = self._engine.set_engine(
                llm=self._default_model,
                nodes=nodes,
                language=self._language,
                offering_filter=offering_filter,
                vector_store=self._vector_store,
                chat_history=preserved_history,
                notebook_id=notebook_id,
            )

            # Verify the new engine has the history (optional - session memory handles persistence)
            if hasattr(self._query_engine, 'memory'):
                new_history_count = len(self._query_engine.memory.get_all())
                logger.info(f"✓ New engine memory buffer contains {new_history_count} messages")
                if new_history_count != len(preserved_history):
                    logger.error(f"❌ MISMATCH: Preserved {len(preserved_history)} but new engine has {new_history_count}!")
            # Note: Engine may not have memory attribute - session memory handles history externally

            # Update state
            self._engine_initialized = True
            self._current_offering_filter = offering_filter  # Track current filter for isolation

            filter_msg = f" (filtered by {len(offering_filter)} offerings)" if offering_filter else ""
            logger.info(f"New engine created with {len(nodes)} nodes{filter_msg}")
        else:
            logger.info("Using existing engine - filter unchanged")

    def get_history(self, chatbot: list[dict]) -> list[ChatMessage]:
        """Convert chatbot history to ChatMessage format."""
        history = []
        for chat in chatbot:
            role_str = chat.get('role')
            content = chat.get('content')
            if role_str and content:
                role = MessageRole.USER if role_str == 'user' else MessageRole.ASSISTANT
                history.append(ChatMessage(role=role, content=content))
        return history

    def _is_follow_up_query(self, message: str, chatbot: list) -> bool:
        """
        Detect if a query is a follow-up to previous conversation.

        Args:
            message: Current user message
            chatbot: Conversation history

        Returns:
            True if this is a follow-up query, False if new problem statement
        """
        # If no conversation history, it's not a follow-up
        if not chatbot or len(chatbot) == 0:
            return False

        # Follow-up indicators
        follow_up_keywords = [
            "more details", "explain", "tell me more", "elaborate",
            "what about", "how about", "can you", "could you",
            "specifically", "example", "clarify", "expand"
        ]

        message_lower = message.lower()

        # Check for follow-up keywords
        has_follow_up_keyword = any(kw in message_lower for kw in follow_up_keywords)

        # Check for question words without problem keywords
        question_words = ["what", "how", "why", "when", "where", "which"]
        problem_keywords = ["problem", "issue", "challenge", "need help", "struggling"]

        has_question = any(qw in message_lower for qw in question_words)
        has_problem = any(pk in message_lower for pk in problem_keywords)

        # If it's a question without problem keywords, it's likely a follow-up
        if has_question and not has_problem:
            return True

        # If it has follow-up keywords, it's a follow-up
        if has_follow_up_keyword:
            return True

        # If message is short (< 50 chars) and conversational, likely follow-up
        if len(message) < 50 and has_question:
            return True

        return False

    def save_message(
        self,
        notebook_id: str,
        user_id: str,
        role: str,
        content: str
    ) -> Optional[str]:
        """
        Save a single conversation message to the database.

        Args:
            notebook_id: UUID of the notebook
            user_id: UUID of the user
            role: Message role ('user' or 'assistant')
            content: Message content text

        Returns:
            conversation_id (UUID) if successful, None if persistence not available

        Raises:
            ValueError: If role is not 'user' or 'assistant'
        """
        if not self._conversation_store:
            logger.warning("Conversation persistence not available - message not saved")
            return None

        try:
            conversation_id = self._conversation_store.save_message(
                notebook_id=notebook_id,
                user_id=user_id,
                role=role,
                content=content
            )
            logger.debug(f"Saved {role} message to notebook {notebook_id}")
            return conversation_id
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return None

    def save_conversation_exchange(
        self,
        notebook_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str
    ) -> bool:
        """
        Save a complete conversation exchange (user query + assistant response).

        This is a convenience method for saving both messages in a single call.
        Typically called by the UI layer after a query completes and the streamed
        response has been fully received.

        Args:
            notebook_id: UUID of the notebook
            user_id: UUID of the user
            user_message: User's query text
            assistant_message: Assistant's response text

        Returns:
            True if both messages saved successfully, False otherwise
        """
        if not self._conversation_store:
            logger.warning("Conversation persistence not available - exchange not saved")
            return False

        try:
            # Save user message
            user_id_saved = self.save_message(
                notebook_id=notebook_id,
                user_id=user_id,
                role="user",
                content=user_message
            )

            # Save assistant message
            assistant_id_saved = self.save_message(
                notebook_id=notebook_id,
                user_id=user_id,
                role="assistant",
                content=assistant_message
            )

            success = user_id_saved is not None and assistant_id_saved is not None
            if success:
                logger.info(f"Saved conversation exchange to notebook {notebook_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to save conversation exchange: {e}")
            return False

    def query(
        self,
        mode: str,
        message: str,
        chatbot: list[list[str]],
        query_settings: Optional[QueryTimeSettings] = None
    ) -> StreamingAgentChatResponse:
        """
        Execute a query against the chat engine.

        Note: This method does NOT automatically save messages to the database.
        If conversation persistence is enabled (database_url provided), the UI
        should call save_conversation_exchange() after the streaming response completes.

        Args:
            mode: "chat" for conversational, other for single Q&A
            message: User message
            chatbot: Conversation history
            query_settings: Optional per-request settings (search style, depth, temperature)

        Returns:
            Streaming response from the chat engine
        """
        logger.debug(f"Query mode: {mode}, message length: {len(message)}")

        # Apply query-time settings if provided
        if query_settings:
            logger.debug(
                f"Query settings: bm25={query_settings.bm25_weight:.2f}, "
                f"vector={query_settings.vector_weight:.2f}, "
                f"top_k={query_settings.similarity_top_k}, "
                f"temp={query_settings.temperature:.2f}"
            )
            # Apply retrieval settings to engine
            self._engine.set_query_settings(query_settings)

            # Apply temperature to LLM if different from default
            # Note: This updates the model for this query
            if hasattr(self._default_model, 'temperature'):
                original_temp = getattr(self._default_model, 'temperature', 0.7)
                if abs(query_settings.temperature - original_temp) > 0.01:
                    self._default_model.temperature = query_settings.temperature
                    logger.debug(f"LLM temperature adjusted: {original_temp} -> {query_settings.temperature}")

        # Start timing for query logging
        start_time = time.time()

        try:
            # Execute query
            if mode == "chat":
                # ChatMemoryBuffer automatically manages conversation history
                # DO NOT pass history parameter - it replaces internal memory
                response = self._query_engine.stream_chat(message)
            else:
                # Reset memory for single Q&A mode
                self._query_engine.reset()
                response = self._query_engine.stream_chat(message)

            # Log query execution with token counting
            # Note: Prompt tokens counted immediately, completion tokens require post-stream analysis
            if self._query_logger:
                response_time_ms = int((time.time() - start_time) * 1000)

                # Count prompt tokens using TokenCounter
                token_counter = get_token_counter()
                prompt_tokens = token_counter.count_tokens(message)

                self._query_logger.log_query(
                    notebook_id=self._current_notebook_id,  # Use current notebook or None
                    user_id=self._current_user_id,
                    query_text=message,
                    model_name=self._default_model.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,  # TODO: Count after stream completes (requires UI-side counting)
                    response_time_ms=response_time_ms
                )

            return response
        finally:
            # Clear query settings after use (revert to defaults)
            if query_settings:
                self._engine.clear_query_settings()

    def store_conversation_exchange(
        self,
        user_message: str,
        assistant_message: str,
        notebook_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Store a conversation exchange in session memory.

        Called by web routes after streaming response completes.

        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            notebook_id: Notebook ID (defaults to current)
            user_id: User ID (defaults to current)
        """
        nb_id = notebook_id or self._current_notebook_id
        u_id = user_id or self._current_user_id

        if nb_id:
            self._session_memory.add_exchange(
                user_id=u_id,
                notebook_id=nb_id,
                user_message=user_message,
                assistant_message=assistant_message
            )
            msg_count = self._session_memory.get_message_count(u_id, nb_id)
            logger.info(f"✓ Stored exchange in session memory for notebook {nb_id} (total: {msg_count} messages)")
        else:
            logger.warning("No notebook context - skipping session memory storage")

    def get_session_history(
        self,
        notebook_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list:
        """
        Get conversation history from session memory.

        Args:
            notebook_id: Notebook ID (defaults to current)
            user_id: User ID (defaults to current)
            limit: Maximum messages to return

        Returns:
            List of ChatMessage objects
        """
        nb_id = notebook_id or self._current_notebook_id
        u_id = user_id or self._current_user_id

        if nb_id:
            return self._session_memory.get_history(u_id, nb_id, limit)
        return []

    def clear_session_history(
        self,
        notebook_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Clear conversation history from session memory.

        Args:
            notebook_id: Notebook ID (defaults to current)
            user_id: User ID (defaults to current)

        Returns:
            True if session was cleared
        """
        nb_id = notebook_id or self._current_notebook_id
        u_id = user_id or self._current_user_id

        if nb_id:
            return self._session_memory.clear_session(u_id, nb_id)
        return False

    # =========================================================================
    # STATELESS FAST QUERY (Multi-user safe, 4-8x faster than full pipeline)
    # =========================================================================

    def stateless_query(
        self,
        message: str,
        notebook_id: str,
        user_id: str,
        include_history: bool = True,
        max_history: int = 10,
        max_sources: int = 6,
        query_settings: Optional[QueryTimeSettings] = None,
    ) -> Dict:
        """
        Execute a fast, stateless query using the API pattern.

        This method is thread-safe and multi-user safe - no global state is mutated.
        Uses the same fast retrieval pattern as /api/query:
        1. Get cached nodes (thread-safe)
        2. Fast retrieval (per-request retriever)
        3. Get RAPTOR summaries
        4. Build context with history
        5. Single LLM call

        ~4-8x faster than full pipeline (5-10s vs 40s).

        Args:
            message: User's query string
            notebook_id: UUID of the notebook to query
            user_id: UUID of the user
            include_history: Whether to include conversation history
            max_history: Maximum history turns to include
            max_sources: Maximum number of sources to return
            query_settings: Optional per-request settings

        Returns:
            Dict with response, sources, metadata

        Example:
            result = pipeline.stateless_query(
                message="What are the key findings?",
                notebook_id="uuid",
                user_id="uuid",
            )
            print(result["response"])
        """
        from .core.stateless import (
            fast_retrieve,
            get_raptor_summaries,
            build_context_with_history,
            format_sources,
            execute_query,
            load_conversation_history,
            save_conversation_turn,
        )

        start_time = time.time()
        timings = {}

        # Step 1: Get cached nodes (thread-safe)
        t1 = time.time()
        nodes = self._get_cached_nodes(notebook_id)
        timings["1_node_cache_ms"] = int((time.time() - t1) * 1000)

        if not nodes:
            logger.warning(f"No nodes found for notebook {notebook_id}")
            return {
                "success": False,
                "response": "No documents found in this notebook.",
                "sources": [],
                "metadata": {"error": "No documents"}
            }

        # Step 2: Load conversation history (from DB)
        t2 = time.time()
        conversation_history = []
        if include_history and self._conversation_store:
            conversation_history = load_conversation_history(
                conversation_store=self._conversation_store,
                notebook_id=notebook_id,
                user_id=user_id,
                max_history=max_history,
            )
        timings["2_load_history_ms"] = int((time.time() - t2) * 1000)

        # Step 3: Fast retrieval (per-request, no global state)
        t3 = time.time()
        retrieval_results = []
        if self._engine and self._engine._retriever:
            retrieval_results = fast_retrieve(
                nodes=nodes,
                query=message,
                notebook_id=notebook_id,
                vector_store=self._vector_store,
                retriever_factory=self._engine._retriever,
                llm=Settings.llm,
                top_k=max_sources,
            )
        timings["3_fast_retrieval_ms"] = int((time.time() - t3) * 1000)

        # Step 4: Get RAPTOR summaries
        t4 = time.time()
        raptor_summaries = get_raptor_summaries(
            query=message,
            notebook_id=notebook_id,
            vector_store=self._vector_store,
            embed_model=Settings.embed_model,
            top_k=5,
        )
        timings["4_raptor_summaries_ms"] = int((time.time() - t4) * 1000)

        # Step 5: Build context with history
        t5 = time.time()
        context = build_context_with_history(
            retrieval_results=retrieval_results,
            raptor_summaries=raptor_summaries,
            conversation_history=conversation_history,
            max_history=max_history,
            max_summaries=3,
            max_chunks=max_sources,
        )
        timings["5_context_building_ms"] = int((time.time() - t5) * 1000)

        # Step 6: Apply temperature if provided
        llm = Settings.llm
        if query_settings and hasattr(llm, 'temperature'):
            # Note: This doesn't mutate Settings.llm - it uses the same instance
            # For true isolation, would need per-request LLM instantiation
            pass  # Temperature applied at completion time

        # Step 7: Execute query (single LLM call)
        t7 = time.time()
        response_text = execute_query(
            query=message,
            context=context,
            llm=llm,
        )
        timings["7_llm_completion_ms"] = int((time.time() - t7) * 1000)

        # Step 8: Save conversation turn to DB
        t8 = time.time()
        if self._conversation_store:
            save_conversation_turn(
                conversation_store=self._conversation_store,
                notebook_id=notebook_id,
                user_id=user_id,
                user_message=message,
                assistant_response=response_text,
            )
        timings["8_save_history_ms"] = int((time.time() - t8) * 1000)

        # Step 9: Format sources
        sources = format_sources(retrieval_results, max_sources)

        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Stateless query completed in {execution_time_ms}ms, {len(sources)} sources")

        return {
            "success": True,
            "response": response_text,
            "sources": sources,
            "metadata": {
                "execution_time_ms": execution_time_ms,
                "model": self._default_model.model if self._default_model else "unknown",
                "retrieval_strategy": "combined_raptor",
                "node_count": len(nodes),
                "raptor_summaries_used": len(raptor_summaries) if raptor_summaries else 0,
                "history_turns_used": len(conversation_history) // 2,
                "timings": timings,
            }
        }

    def stateless_query_streaming(
        self,
        message: str,
        notebook_id: str,
        user_id: str,
        include_history: bool = True,
        max_history: int = 10,
        max_sources: int = 6,
    ):
        """
        Execute a fast, stateless query with streaming response.

        Same as stateless_query but yields response chunks for SSE streaming.

        Args:
            message: User's query string
            notebook_id: UUID of the notebook to query
            user_id: UUID of the user
            include_history: Whether to include conversation history
            max_history: Maximum history turns to include
            max_sources: Maximum number of sources to return

        Yields:
            Dict events: {"type": "sources"|"content"|"done", ...}
        """
        from .core.stateless import (
            fast_retrieve,
            get_raptor_summaries,
            build_context_with_history,
            format_sources,
            execute_query_streaming,
            load_conversation_history,
            save_conversation_turn,
        )

        response_text = ""

        try:
            # Step 1-5: Same as stateless_query
            nodes = self._get_cached_nodes(notebook_id)
            if not nodes:
                yield {"type": "error", "error": "No documents found"}
                return

            conversation_history = []
            if include_history and self._conversation_store:
                conversation_history = load_conversation_history(
                    conversation_store=self._conversation_store,
                    notebook_id=notebook_id,
                    user_id=user_id,
                    max_history=max_history,
                )

            retrieval_results = []
            if self._engine and self._engine._retriever:
                retrieval_results = fast_retrieve(
                    nodes=nodes,
                    query=message,
                    notebook_id=notebook_id,
                    vector_store=self._vector_store,
                    retriever_factory=self._engine._retriever,
                    llm=Settings.llm,
                    top_k=max_sources,
                )

            raptor_summaries = get_raptor_summaries(
                query=message,
                notebook_id=notebook_id,
                vector_store=self._vector_store,
                embed_model=Settings.embed_model,
            )

            context = build_context_with_history(
                retrieval_results=retrieval_results,
                raptor_summaries=raptor_summaries,
                conversation_history=conversation_history,
                max_history=max_history,
            )

            # Yield sources first
            sources = format_sources(retrieval_results, max_sources)
            yield {"type": "sources", "sources": sources}

            # Stream response
            for chunk in execute_query_streaming(message, context, Settings.llm):
                response_text += chunk
                yield {"type": "content", "content": chunk}

            # Save to conversation store after streaming completes
            if self._conversation_store:
                save_conversation_turn(
                    conversation_store=self._conversation_store,
                    notebook_id=notebook_id,
                    user_id=user_id,
                    user_message=message,
                    assistant_response=response_text,
                )

            yield {"type": "done"}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "error": str(e)}

    def get_session_memory_stats(self) -> dict:
        """Get session memory statistics."""
        return self._session_memory.get_stats()

    def chat_without_retrieval(
        self,
        message: str,
        chatbot: list[list[str]],
        query_settings: Optional[QueryTimeSettings] = None
    ) -> StreamingAgentChatResponse:
        """
        Direct chat with LLM without any document retrieval.
        Used when no notebook is selected (General Chat mode).

        This bypasses the entire retrieval pipeline for faster responses
        when the user just wants to chat with the LLM directly.

        Args:
            message: User message
            chatbot: Conversation history (for context, though SimpleChatEngine manages its own memory)
            query_settings: Optional per-request settings (only temperature applies for non-RAG chat)

        Returns:
            Streaming response from SimpleChatEngine
        """
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.memory import ChatMemoryBuffer

        logger.debug(f"General chat (no retrieval): message length {len(message)}")

        # Apply temperature from query settings if provided
        if query_settings and hasattr(self._default_model, 'temperature'):
            original_temp = getattr(self._default_model, 'temperature', 0.7)
            if abs(query_settings.temperature - original_temp) > 0.01:
                self._default_model.temperature = query_settings.temperature
                logger.debug(f"LLM temperature adjusted: {original_temp} -> {query_settings.temperature}")

        # Create SimpleChatEngine on first use (lazy initialization)
        if self._simple_chat_engine is None:
            logger.info("Initializing SimpleChatEngine for general chat mode")
            # Get dynamic token limit based on LLM's context window
            try:
                model_context_window = getattr(self._default_model.metadata, 'context_window', None)
                if model_context_window and model_context_window > 0:
                    token_limit = int(model_context_window * 0.6)
                    logger.debug(f"Dynamic token limit: {token_limit} (60% of {model_context_window})")
                else:
                    token_limit = self._settings.ollama.chat_token_limit
            except Exception:
                token_limit = self._settings.ollama.chat_token_limit

            memory = ChatMemoryBuffer.from_defaults(
                token_limit=token_limit
            )
            self._simple_chat_engine = SimpleChatEngine.from_defaults(
                llm=self._default_model,
                memory=memory,
                system_prompt=self._system_prompt
            )

        # Start timing for query logging
        start_time = time.time()

        # Execute direct chat (no retrieval)
        response = self._simple_chat_engine.stream_chat(message)

        # Log query execution
        if self._query_logger:
            response_time_ms = int((time.time() - start_time) * 1000)
            token_counter = get_token_counter()
            prompt_tokens = token_counter.count_tokens(message)

            self._query_logger.log_query(
                notebook_id=None,  # NULL for general chat (no notebook)
                user_id=self._current_user_id,
                query_text=message,
                model_name=self._default_model.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                response_time_ms=response_time_ms
            )

        return response

    def query_sales_mode(
        self,
        message: str,
        selected_offerings: Optional[list[str]] = None,
        selected_notebooks: Optional[list[str]] = None,
        chatbot: list = None
    ) -> StreamingAgentChatResponse:
        """
        Execute sales enablement query with intelligent mode detection (Hybrid Architecture).

        IMPORTANT: This feature requires query_classifier and offering_analyzer components
        which are not currently implemented. This method will raise NotImplementedError
        until these components are provided.

        HYBRID MODE SUPPORT:
        - Traditional: Use selected_offerings (backward compatible)
        - Notebook: Use selected_notebooks (NotebookLM architecture)
        - Conversation-aware: Follow-up detection via conversation history

        Workflow:
        1. Classify query (problem_solving vs pitch modes)
        2. If problem_solving + follow-up: use existing context
        3. If problem_solving + new: analyze ALL offerings/notebooks and recommend bundle
        4. If pitch: use selected offerings/notebooks
        5. Generate response with context from offerings/notebooks

        Note: This method does NOT automatically save messages to the database.
        If conversation persistence is enabled (database_url provided), the UI
        should call save_conversation_exchange() after the streaming response completes.

        Args:
            message: User's query
            selected_offerings: Pre-selected offerings for pitch mode (backward compatibility)
            selected_notebooks: Pre-selected notebooks for pitch mode (hybrid architecture)
            chatbot: Conversation history

        Returns:
            Streaming response with sales-optimized content

        Raises:
            NotImplementedError: Sales mode components not yet implemented
        """
        # Check if sales mode components are initialized
        if self._query_classifier is None or self._offering_analyzer is None:
            logger.warning(
                "Sales mode requested but components not initialized. "
                "query_classifier and offering_analyzer are required for this feature. "
                "Falling back to standard query mode."
            )
            raise NotImplementedError(
                "Sales mode is not currently available. "
                "This feature requires query_classifier and offering_analyzer components "
                "which have not been implemented. "
                "Please use the standard query() method instead."
            )

        logger.info(f"Sales mode query: {message[:100]}...")

        # Step 1: Classify the query with hybrid support
        classification = self._query_classifier.classify(
            query=message,
            selected_offerings=selected_offerings,
            selected_notebooks=selected_notebooks,
            conversation_history=chatbot
        )

        mode = classification["mode"]
        logger.info(f"Query classified as: {mode} (confidence: {classification['confidence']})")

        # Get all nodes for analysis
        all_nodes = self._ingestion.get_ingested_nodes()

        # Get list of available offerings from nodes
        available_offerings = list(set([
            node.metadata.get("offering_name")
            for node in all_nodes
            if node.metadata.get("offering_name")
        ]))

        logger.info(f"Available offerings: {available_offerings}")

        # Step 2: Handle different modes
        if mode == "problem_solving":
            # Check if this is a follow-up query (from classifier)
            is_follow_up = classification.get("is_follow_up", False)
            use_all_notebooks = classification.get("use_all_notebooks", False)

            if is_follow_up:
                logger.info("Detected follow-up query - skipping analysis, using existing engine")
                # Use existing engine and filter - no need to run analysis again
                # For follow-ups, use notebooks if provided, otherwise offerings
                offering_filter = selected_notebooks if selected_notebooks else selected_offerings
                # ChatMemoryBuffer in CondensePlusContextChatEngine will handle conversation history automatically
                response_prefix = ""
            else:
                # Analyze problem and recommend offering bundle
                logger.info(f"New problem - running offering analysis (use_all_notebooks={use_all_notebooks})")

                # Determine analysis mode: notebook mode vs traditional mode
                if self._notebook_manager and use_all_notebooks:
                    # Notebook mode: analyze all notebooks using document nodes
                    logger.info("Using notebook mode for offering analysis")
                    analysis_result = self._offering_analyzer.analyze_problem(
                        problem_description=classification["problem_description"],
                        user_id="default_user",  # TODO: Get from user context
                        nodes=all_nodes,
                        customer_name=classification.get("customer_name"),
                        industry=classification.get("industry"),
                        top_n=3
                    )
                else:
                    # Traditional mode: use pre-generated synopses
                    logger.info("Using traditional mode for offering analysis")

                    # Load pre-generated synopses
                    all_synopses = self._ingestion.get_all_synopses()

                    # Extract synopsis text for available offerings
                    offering_synopses = {}
                    for offering_name in available_offerings:
                        # Find synopsis by offering name
                        for offering_id, synopsis_data in all_synopses.items():
                            if synopsis_data.get("offering_name") == offering_name:
                                offering_synopses[offering_name] = synopsis_data.get("synopsis", "")
                                break

                    logger.info(f"Loaded {len(offering_synopses)} synopses for problem analysis")

                    # If no synopses available, log warning
                    if not offering_synopses:
                        logger.warning("No synopses available for problem analysis. Please ensure synopses are generated after document upload.")

                    analysis_result = self._offering_analyzer.analyze_problem(
                        problem_description=classification["problem_description"],
                        offering_synopses=offering_synopses,
                        customer_name=classification.get("customer_name"),
                        industry=classification.get("industry"),
                        top_n=3
                    )

                recommended_offerings = analysis_result["recommended_offerings"]
                logger.info(f"Recommended offerings: {recommended_offerings}")

                # Generate high-level implementation plan
                implementation_plan = ""
                if recommended_offerings:
                    logger.info("Generating high-level implementation plan...")
                    implementation_plan = self._offering_analyzer.generate_implementation_plan(
                        recommended_offerings=recommended_offerings,
                        offering_synopses=offering_synopses,
                        problem_description=classification["problem_description"],
                        customer_name=classification.get("customer_name"),
                        industry=classification.get("industry")
                    )
                    logger.info(f"Implementation plan generated: {len(implementation_plan)} chars")

                # Refine plan with detailed offering content
                refined_plan = ""
                if implementation_plan and recommended_offerings:
                    logger.info("Refining plan with detailed offering content...")
                    refined_plan = self._refine_plan_with_details(
                        implementation_plan=implementation_plan,
                        recommended_offerings=recommended_offerings,
                        all_nodes=all_nodes,
                        problem_description=classification["problem_description"]
                    )
                    logger.info(f"Refined plan generated: {len(refined_plan)} chars")

                # Use recommended offerings as filter
                offering_filter = recommended_offerings if recommended_offerings else None

                # Let LLM handle response format adaptation based on system prompt
                # Build context message with offering recommendations for LLM
                logger.info("Providing offering context to LLM for adaptive response generation")

                explanations = analysis_result.get("explanations", {})
                bundle_strategy = analysis_result.get("bundle_strategy", "")

                llm_context = f"\n\n**System Context for Response:**\n"
                llm_context += f"Based on the problem analysis, the recommended offerings are:\n"
                for i, offering in enumerate(recommended_offerings, 1):
                    explanation = explanations.get(offering, "")
                    llm_context += f"{i}. {offering}: {explanation}\n"

                llm_context += f"\n**Bundle Strategy:** {bundle_strategy}\n"

                if implementation_plan:
                    llm_context += f"\n**Implementation Overview:** {implementation_plan[:500]}...\n"

                llm_context += f"\n**Note:** Adapt your response format based on the user's query. "
                llm_context += f"If they ask for a specific format (elevator pitch, summary, detailed, etc.), provide it. "
                llm_context += f"Otherwise, use the default comprehensive format with recommended offerings, strategy, and implementation approach.\n"

                # Append context to message for LLM
                message = message + llm_context
                response_prefix = ""

        elif mode == "offering_summary":
            # Comprehensive retrieval for specific offering
            offering_name = classification.get("offering_mentioned")
            if offering_name:
                logger.info(f"Offering summary mode for: {offering_name}")
                offering_filter = [offering_name]
                response_prefix = f"## {offering_name} - Comprehensive Summary\n\n"
            else:
                logger.warning("Offering summary requested but no offering mentioned")
                # Hybrid mode: use notebooks if provided, otherwise offerings
                offering_filter = selected_notebooks if selected_notebooks else selected_offerings
                response_prefix = ""

        else:  # pitch_specific or pitch_generic
            # Hybrid mode: use notebooks if provided, otherwise offerings
            logger.info(f"Pitch mode: using selected {'notebooks' if selected_notebooks else 'offerings'}")
            offering_filter = selected_notebooks if selected_notebooks else selected_offerings
            response_prefix = ""

        # Step 3: Set engine with offering filter
        self.set_engine(offering_filter=offering_filter)

        # Step 4: Generate response
        # CondensePlusContextChatEngine's ChatMemoryBuffer automatically manages conversation history
        # We only reset() for the first message in a new conversation
        if not chatbot:
            # First message in conversation - reset memory buffer
            self._query_engine.reset()
            logger.info("Starting new conversation - reset memory buffer")

        # Start timing for query logging
        start_time = time.time()

        # ChatMemoryBuffer will automatically:
        # 1. Store this query and response in memory
        # 2. Use CONDENSED_CONTEXT_PROMPT to condense follow-up questions with history
        # 3. Retrieve relevant context from the memory buffer
        response = self._query_engine.stream_chat(message)

        # Log query execution (with placeholder token counts for streaming responses)
        if self._query_logger:
            response_time_ms = int((time.time() - start_time) * 1000)
            self._query_logger.log_query(
                notebook_id=self._current_notebook_id,  # Use current notebook or None
                user_id=self._current_user_id,
                query_text=message,
                model_name=self._default_model.model,
                prompt_tokens=0,  # TODO: Extract from LangSmith or implement token counting
                completion_tokens=0,  # TODO: Extract from LangSmith or implement token counting
                response_time_ms=response_time_ms
            )

        # Add prefix to response if needed
        if response_prefix:
            # Note: For streaming responses, we'll need to prepend this in the UI layer
            # Store it as an attribute for the UI to access
            response.response_prefix = response_prefix

        return response

    def _refine_plan_with_details(
        self,
        implementation_plan: str,
        recommended_offerings: list[str],
        all_nodes: list,
        problem_description: str
    ) -> str:
        """
        Refine the implementation plan by incorporating detailed offering content.

        This sends the high-level plan along with detailed node content to the LLM
        for sensitization and refinement with specific technical details.

        Args:
            implementation_plan: High-level implementation plan
            recommended_offerings: List of recommended offering names
            all_nodes: All document nodes
            problem_description: Customer's problem statement

        Returns:
            Refined implementation plan with detailed technical content
        """
        # Get detailed content for recommended offerings
        offering_details = {}
        for offering_name in recommended_offerings:
            # Find all nodes for this offering
            offering_nodes = [
                node for node in all_nodes
                if node.metadata.get("offering_name") == offering_name
            ]

            # Combine node content (limit to prevent token overflow)
            if offering_nodes:
                combined_content = "\n\n".join([node.get_content() for node in offering_nodes[:10]])
                # Limit to ~5000 chars per offering
                offering_details[offering_name] = combined_content[:5000]

        # Build context for refinement
        detailed_context = "\n\n---\n\n".join([
            f"# {offering_name}\n\n{content}"
            for offering_name, content in offering_details.items()
        ])

        refinement_prompt = f"""You are a senior solutions architect refining an implementation plan with technical details.

**Customer Problem:**
{problem_description}

**High-Level Implementation Plan:**
{implementation_plan}

**Detailed Offering Documentation:**
{detailed_context}

Your Task:
Refine the implementation plan by:
1. Adding specific technical details from the documentation
2. Incorporating actual features, APIs, and capabilities mentioned in the offering details
3. Providing concrete configuration examples where applicable
4. Highlighting specific integration points and technical dependencies
5. Adding realistic timelines based on the technical complexity

Keep the same structure as the high-level plan but enrich each phase with:
- Specific technical components from the documentation
- Actual feature names and capabilities
- Configuration considerations
- Technical prerequisites and dependencies

Write in a professional, technically accurate tone (400-500 words).

Refined Implementation Plan:"""

        try:
            response = self._default_model.complete(refinement_prompt)
            refined_plan = response.text.strip()
            logger.debug(f"Refined plan: {len(refined_plan)} chars")
            return refined_plan
        except Exception as e:
            logger.error(f"Error refining plan: {e}")
            return implementation_plan  # Fallback to original plan

    def _format_problem_solving_response(
        self,
        analysis_result: dict,
        classification: dict,
        implementation_plan: str = "",
        refined_plan: str = ""
    ) -> str:
        """Format problem-solving analysis results for display."""
        recommended = analysis_result["recommended_offerings"]
        scores = analysis_result["offering_scores"]
        explanations = analysis_result["offering_explanations"]
        bundle_strategy = analysis_result["bundle_strategy"]

        # Build formatted response
        output = "# 🎯 Recommended Solution Bundle\n\n"

        # Customer context
        if classification.get("customer_name"):
            output += f"**Customer:** {classification['customer_name']}\n"
        if classification.get("industry"):
            output += f"**Industry:** {classification['industry']}\n"
        output += "\n"

        # Recommended offerings
        output += "## 📊 Top Recommended Offerings\n\n"
        for i, offering in enumerate(recommended, 1):
            explanation = explanations.get(offering, "No explanation available")
            output += f"### {i}. {offering}\n"
            output += f"{explanation}\n\n"

        # Bundle strategy
        output += "## 💡 Bundle Strategy\n\n"
        output += f"{bundle_strategy}\n\n"

        # Implementation plan (if available)
        if implementation_plan:
            output += "---\n\n"
            output += "## 🗺️ High-Level Implementation Plan\n\n"
            output += f"{implementation_plan}\n\n"

        # Refined plan (if available)
        if refined_plan:
            output += "---\n\n"
            output += "## 🔧 Detailed Implementation Roadmap\n\n"
            output += f"{refined_plan}\n\n"

        output += "---\n\n"
        output += "## 📝 Detailed Information\n\n"

        return output

    @property
    def transformation_worker(self) -> Optional[TransformationWorker]:
        """Get the TransformationWorker instance for API integration."""
        return self._transformation_worker

    @property
    def raptor_worker(self) -> Optional[RAPTORWorker]:
        """Get the RAPTORWorker instance for API integration."""
        return self._raptor_worker

    def queue_raptor_build(self, source_id: str, notebook_id: str, file_name: str) -> bool:
        """Queue a RAPTOR tree build for a source.

        Args:
            source_id: Source ID to build tree for
            notebook_id: Notebook containing the source
            file_name: File name for logging

        Returns:
            True if job was queued successfully
        """
        if not self._raptor_worker:
            logger.warning("RAPTOR worker not available")
            return False

        job = RAPTORJob(
            source_id=source_id,
            notebook_id=notebook_id,
            file_name=file_name
        )
        self._raptor_worker.queue_job(job)
        return True

    def shutdown(self) -> None:
        """
        Gracefully shutdown the pipeline and its components.

        Stops background workers and releases resources.
        """
        logger.info("Shutting down RAG pipeline...")

        # Stop transformation worker
        if self._transformation_worker:
            try:
                self._transformation_worker.stop()
                logger.info("TransformationWorker stopped")
            except Exception as e:
                logger.error(f"Error stopping TransformationWorker: {e}")

        # Stop RAPTOR worker
        if self._raptor_worker:
            try:
                self._raptor_worker.stop()
                logger.info("RAPTORWorker stopped")
            except Exception as e:
                logger.error(f"Error stopping RAPTORWorker: {e}")

        logger.info("Pipeline shutdown complete")
