"""Simple Flask-based web interface for RAG Chatbot."""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Generator

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file, send_from_directory
from sqlalchemy import text

from ..pipeline import LocalRAGPipeline
from ..setting import QueryTimeSettings
from ..core.plugins import get_configured_image_provider, register_default_plugins
from ..core.metadata import MetadataManager
from ..api.routes.web_content import create_web_content_routes
from ..api.routes.studio import create_studio_routes
from ..api.routes.vision import create_vision_routes
from ..api.routes.transformations import create_transformation_routes
from ..api.routes.agents import create_agent_routes
from ..api.routes.analytics import create_analytics_routes
from ..api.routes.sql_chat import create_sql_chat_routes
from ..api.routes.query import create_query_routes
from ..api.routes.chat_v2 import create_chat_v2_routes
from ..api.routes.admin import create_admin_routes
from ..api.routes.auth import create_auth_routes
from ..api.routes.settings import create_settings_routes
from ..api.routes.quiz import create_quiz_routes
from ..core.ingestion import WebContentIngestion, SynopsisManager
from ..core.studio import StudioManager
from ..core.constants import DEFAULT_USER_ID

logger = logging.getLogger(__name__)

# CORS allowed origins for development
ALLOWED_CORS_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5173',
]


class FlaskChatbotUI:
    """Flask-based UI for the RAG chatbot."""

    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        host: str = "host.docker.internal",
        data_dir: str = "data/data",
        upload_dir: str = "uploads",
        db_manager=None,
        project_manager=None
    ):
        self._pipeline = pipeline
        self._host = host
        self._ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self._data_dir = Path(data_dir)
        self._upload_dir = Path(upload_dir)
        self._processed_files: list[str] = []  # Track processed files

        # Project feature
        self._db_manager = db_manager
        self._project_manager = project_manager

        # Initialize image generation provider via plugin system
        register_default_plugins()
        try:
            self._image_provider = get_configured_image_provider()
            logger.info(f"Initialized image provider: {self._image_provider.name}")
        except Exception as e:
            logger.warning(f"Image generation not available: {e}")
            self._image_provider = None

        # Initialize metadata manager
        self._metadata_manager = MetadataManager(config_dir="data/config")

        # Document metadata file
        self._doc_metadata_file = Path("data/config/documents_metadata.json")
        self._doc_metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Ensure directories exist
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Flask app with React frontend
        template_dir = Path(__file__).parent.parent / "templates"
        # React frontend build directory
        frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
        self._frontend_dist = frontend_dist
        self._app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(frontend_dist / "assets") if frontend_dist.exists() else None,
            static_url_path="/assets"
        )

        # Configure session secret key for authentication
        self._app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

        # Store db_manager in app extensions for access from decorators (RBAC)
        if not hasattr(self._app, 'extensions'):
            self._app.extensions = {}
        self._app.extensions['db_manager'] = self._db_manager

        # Add CORS support for React dev server
        self._setup_cors()
        self._setup_routes()

        # pgvector Persistence: Load nodes from persistent storage instead of re-ingesting
        # Documents are persisted to pgvector during upload, no need to reload from disk
        # The pipeline will load nodes from pgvector when set_engine() is called
        # self._reload_documents_with_metadata()  # DISABLED - using pgvector persistence

        # Initialize few-shot examples for SQL Chat (background, non-blocking)
        self._init_few_shot_background()

        logger.info("Flask UI initialized (using pgvector persistence)")

    def _reload_documents_with_metadata(self) -> None:
        """Reload all documents from data and uploads directories with metadata from JSON."""
        try:
            # Load document metadata
            doc_metadata = self._load_document_metadata()

            if not doc_metadata:
                logger.info("No document metadata found")
                return

            # Find all documents in data and uploads directories
            doc_files = []
            for directory in [self._data_dir, self._upload_dir]:
                if directory.exists():
                    for file_path in directory.iterdir():
                        if file_path.is_file() and file_path.name in doc_metadata:
                            doc_files.append(str(file_path))

            if not doc_files:
                logger.info("No documents found in data or uploads directories")
                return

            logger.info(f"Reloading {len(doc_files)} documents with metadata")

            # Process each document with its metadata
            for doc_path in doc_files:
                filename = os.path.basename(doc_path)
                metadata = doc_metadata[filename]

                it_practice = metadata.get("it_practice")
                offering_name = metadata.get("offering_name")
                offering_id = metadata.get("offering_id", "")

                # Auto-generate offering_id from offering_name if empty
                if not offering_id and offering_name and offering_name != "N/A":
                    offering_id = offering_name.lower().replace(" ", "-").replace("_", "-")

                # Store nodes with metadata
                self._pipeline.store_nodes(
                    input_files=[doc_path],
                    it_practice=it_practice if it_practice != "N/A" else None,
                    offering_name=offering_name if offering_name != "N/A" else None,
                    offering_id=offering_id if offering_id else None
                )

                # Track processed file
                if filename not in self._processed_files:
                    self._processed_files.append(filename)

                logger.info(f"Reloaded {filename} with Practice='{it_practice}', Offering='{offering_name}'")

            # Set chat mode after loading documents
            self._pipeline.set_chat_mode()
            logger.info(f"Successfully reloaded {len(doc_files)} documents with metadata")

        except Exception as e:
            logger.error(f"Error reloading documents with metadata: {e}")

    def _init_few_shot_background(self) -> None:
        """Initialize few-shot SQL examples in background if not loaded.

        Downloads and embeds the Gretel synthetic_text_to_sql dataset (~100K examples)
        for few-shot retrieval in Text-to-SQL. Runs in a background thread to avoid
        blocking app startup.
        """
        if not self._db_manager:
            logger.debug("Few-shot init skipped: no database manager")
            return

        import threading

        def load_few_shot():
            import asyncio
            try:
                from ..core.sql_chat.few_shot_setup import FewShotSetup

                embed_model = self._pipeline.get_embed_model()
                if not embed_model:
                    logger.warning("Few-shot init skipped: no embed model available")
                    return

                setup = FewShotSetup(self._db_manager, embed_model)

                # Check if already loaded (requires MIN_REQUIRED_EXAMPLES = 50000)
                if setup.is_initialized():
                    count = setup.get_example_count()
                    logger.info(f"Few-shot examples already loaded: {count}")
                    return

                # Clear partial data and start fresh
                current_count = setup.get_example_count()
                if current_count > 0:
                    logger.info(f"Clearing {current_count} partial examples, loading full dataset...")
                    setup.clear_examples()

                logger.info("Starting few-shot dataset initialization (background)...")

                # Run async initialize in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Load limited examples for faster startup (configurable via env)
                    max_examples = int(os.getenv("FEW_SHOT_MAX_EXAMPLES", "100000"))
                    success = loop.run_until_complete(setup.initialize(max_examples=max_examples))
                    if success:
                        logger.info(f"Few-shot initialization complete: {setup.get_example_count()} examples")
                    else:
                        logger.error("Few-shot initialization failed")
                finally:
                    loop.close()
            except ImportError as e:
                logger.debug(f"Few-shot init skipped: {e}")
            except Exception as e:
                logger.error(f"Few-shot background init failed: {e}")

        thread = threading.Thread(target=load_few_shot, daemon=True, name="few-shot-loader")
        thread.start()
        logger.debug("Few-shot background loader thread started")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _load_document_metadata(self) -> dict:
        """Load document metadata from JSON file."""
        if self._doc_metadata_file.exists():
            try:
                with open(self._doc_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading document metadata: {e}")
                return {}
        return {}

    def _save_document_metadata(self, metadata: dict) -> None:
        """Save document metadata to JSON file."""
        try:
            with open(self._doc_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")

    def _add_document_metadata(self, filename: str, it_practice: str, offering_name: str, offering_id: str) -> None:
        """Add metadata for a document."""
        metadata = self._load_document_metadata()
        metadata[filename] = {
            "it_practice": it_practice or "N/A",
            "offering_name": offering_name or "N/A",
            "offering_id": offering_id or ""
        }
        self._save_document_metadata(metadata)

    def _remove_document_metadata(self, filename: str) -> None:
        """Remove metadata for a document."""
        metadata = self._load_document_metadata()
        if filename in metadata:
            del metadata[filename]
            self._save_document_metadata(metadata)

    def _get_document_metadata(self, filename: str) -> dict:
        """Get metadata for a document."""
        metadata = self._load_document_metadata()
        return metadata.get(filename, {
            "it_practice": "N/A",
            "offering_name": "N/A",
            "offering_id": ""
        })

    def _is_image_generation_request(self, message: str) -> bool:
        """Check if message is requesting image generation.

        NOTE: Image generation is now only available in Content Studio.
        This method always returns False to disable in-chat image generation.
        To re-enable, implement intent detection logic here.
        """
        # Image generation disabled in chat - use Studio instead
        return False


    def _create_image_prompt_with_context(self, user_message: str, document_content: str) -> str:
        """Create a crisp, summarized image prompt using LLM.

        Flow: User request + Document context → LLM summarization → Clean image prompt

        Args:
            user_message: User's original request
            document_content: Retrieved content from RAG documents

        Returns:
            Crisp, summarized prompt optimized for image generation
        """
        try:
            # Use the pipeline's configured LLM
            llm = self._pipeline._default_model

            # Truncate document content if too long (keep key info)
            max_context = 2000
            doc_summary = document_content[:max_context] if len(document_content) > max_context else document_content

            summarization_prompt = f"""Analyze this request and context, then create a SHORT image generation prompt.

User wants: "{user_message}"

Context from documents:
{doc_summary}

Create a crisp image prompt (2-3 sentences max) that:
1. Captures the main visual concept
2. Describes style (infographic, diagram, illustration, etc.)
3. Specifies key visual elements and colors

Output ONLY the image prompt, nothing else. Keep it under 100 words."""

            response = llm.complete(summarization_prompt)
            image_prompt = str(response).strip()

            # Clean up - remove quotes if LLM wrapped the response
            if image_prompt.startswith('"') and image_prompt.endswith('"'):
                image_prompt = image_prompt[1:-1]

            logger.info(f"User request: {user_message[:100]}...")
            logger.info(f"Generated image prompt: {image_prompt}")

            return image_prompt

        except Exception as e:
            logger.error(f"Error creating image prompt: {e}")
            # Fallback: create simple prompt from user message
            return f"Professional infographic showing: {user_message}"

    def _extract_text_structure(self, user_message: str, document_content: str) -> dict:
        """Extract structured text elements from document content for overlay.

        Args:
            user_message: User's original request
            document_content: Retrieved content from RAG documents

        Returns:
            Dictionary with structured text for overlay
        """
        try:
            # Use the pipeline's configured LLM
            llm = self._pipeline._default_model

            extraction_prompt = f"""Extract key information from the document for creating an infographic.

User wants: "{user_message}"

Document content:
\"\"\"
{document_content[:2000]}
\"\"\"

Extract and structure the information as follows:
1. A clear, concise TITLE (5-8 words max)
2. 3-4 SECTIONS, each with:
   - heading: Short heading (2-4 words)
   - content: Brief description (10-15 words)

Focus on the most important information that addresses the user's request.

Output format (JSON):
{{
    "title": "Main Title Here",
    "sections": [
        {{"heading": "Section 1", "content": "Brief description here"}},
        {{"heading": "Section 2", "content": "Brief description here"}},
        {{"heading": "Section 3", "content": "Brief description here"}}
    ]
}}

Output ONLY valid JSON, nothing else."""

            response = llm.complete(extraction_prompt)

            # Parse JSON response
            import json
            text_structure = json.loads(str(response).strip())

            logger.info(f"Extracted text structure: {text_structure}")
            return text_structure

        except Exception as e:
            logger.error(f"Error extracting text structure: {e}")
            # Fallback structure
            return {
                "title": "Retail Solutions",
                "sections": [
                    {"heading": "Overview", "content": document_content[:100]},
                ]
            }

    def _setup_cors(self):
        """Set up CORS headers for React dev server."""

        @self._app.after_request
        def add_cors_headers(response):
            # Allow React dev server origins
            origin = request.headers.get('Origin', '')

            if origin in ALLOWED_CORS_ORIGINS:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, PATCH, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
                response.headers['Access-Control-Allow-Credentials'] = 'true'

            return response

        @self._app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
        @self._app.route('/<path:path>', methods=['OPTIONS'])
        def handle_options(path):
            """Handle preflight OPTIONS requests."""
            response = self._app.make_default_options_response()
            origin = request.headers.get('Origin', '')

            if origin in ALLOWED_CORS_ORIGINS:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, PATCH, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
                response.headers['Access-Control-Allow-Credentials'] = 'true'

            return response

    def _setup_routes(self):
        """Set up Flask routes."""

        @self._app.route("/")
        def index():
            """Serve React frontend."""
            if self._frontend_dist.exists():
                return send_from_directory(str(self._frontend_dist), "index.html")
            return render_template("index.html")

        @self._app.route("/chat", methods=["POST"])
        def chat():
            data = request.json
            message = data.get("message", "")
            history = data.get("history", [])
            model = data.get("model", "")
            mode = data.get("mode", "chat")
            # Hybrid mode: support both offerings (traditional) and projects
            selected_offerings = data.get("selected_offerings", [])
            selected_projects = data.get("selected_projects", [])

            # Query settings for per-request tuning
            query_settings_raw = data.get("query_settings", {})
            # Create QueryTimeSettings from frontend values using the class factory method
            # This handles the conversion: search_style → weights, result_depth → top_k, temperature → 0-2.0
            query_settings_obj = QueryTimeSettings.from_frontend(
                search_style=query_settings_raw.get("search_style", 50),
                result_depth=query_settings_raw.get("result_depth", "balanced"),
                temperature=query_settings_raw.get("temperature", 20)
            )

            logger.debug(f"Query settings: bm25={query_settings_obj.bm25_weight:.2f}, "
                        f"vec={query_settings_obj.vector_weight:.2f}, "
                        f"top_k={query_settings_obj.similarity_top_k}, "
                        f"temp={query_settings_obj.temperature:.2f}")

            # Support React frontend's project_id (singular) format
            project_id = data.get("project_id")
            if project_id and not selected_projects:
                selected_projects = [project_id]

            if not message:
                return jsonify({"error": "No message provided"}), 400

            # Log hybrid selection
            if selected_projects:
                logger.info(f"Received query with {len(selected_projects)} selected projects: {selected_projects}")
            elif selected_offerings:
                logger.info(f"Received query with {len(selected_offerings)} selected offerings: {selected_offerings}")
            else:
                logger.info("Received query with no selections")

            # Set model if provided and different
            if model and model != self._pipeline.get_model_name():
                try:
                    self._pipeline.set_model_name(model)
                    self._pipeline.set_model()
                    # Set engine with hybrid filter support (projects or offerings)
                    # Note: query_sales_mode() will manage engine state during query execution
                    offering_filter = selected_projects if selected_projects else selected_offerings
                    if offering_filter:
                        self._pipeline.set_engine(offering_filter=offering_filter)
                    else:
                        self._pipeline.set_engine()
                except Exception as e:
                    logger.error(f"Error setting model: {e}")

            # Ensure model is set
            if not self._pipeline.get_model_name():
                self._pipeline.set_model_name("")
                self._pipeline.set_model()

            # Engine management is now handled automatically by query_sales_mode()
            # which preserves conversation history when offerings change

            # STEP 1: Always query documents first to get context
            def process_request() -> Generator[str, None, None]:
                import time as timing_module
                start_time = timing_module.time()
                timings = {}

                try:
                    # Query RAG pipeline to get relevant document content
                    yield f"data: {json.dumps({'token': ''})}\n\n"

                    # Pre-extract sources BEFORE streaming (same pattern as /api/chat)
                    # This is necessary because source_nodes is empty after streaming completes
                    t1 = timing_module.time()
                    pre_extracted_sources = []
                    if selected_projects:
                        try:
                            from llama_index.core.schema import QueryBundle
                            from llama_index.core import Settings

                            # Stage 3: Get cached nodes
                            t_cache = timing_module.time()
                            nodes = []
                            for nb_id in selected_projects:
                                nb_nodes = self._pipeline._get_cached_nodes(nb_id)
                                if nb_nodes:
                                    nodes.extend(nb_nodes)
                            timings["3_node_cache_ms"] = int((timing_module.time() - t_cache) * 1000)

                            if nodes and hasattr(self._pipeline, '_engine') and self._pipeline._engine is not None:
                                # Stage 2: Set engine / project switch
                                t_switch = timing_module.time()
                                self._pipeline.set_engine(offering_filter=selected_projects, force_reset=True)
                                timings["2_project_switch_ms"] = int((timing_module.time() - t_switch) * 1000)

                                if hasattr(self._pipeline._engine, '_retriever'):
                                    # Stage 4: Create retriever
                                    t_retriever = timing_module.time()
                                    retriever = self._pipeline._engine._retriever.get_retrievers(
                                        llm=Settings.llm,
                                        language="eng",
                                        nodes=nodes,
                                        offering_filter=selected_projects,
                                        vector_store=self._pipeline._vector_store,
                                        project_id=project_id
                                    )
                                    timings["4_retriever_creation_ms"] = int((timing_module.time() - t_retriever) * 1000)

                                    # Stage 5: Chunk retrieval
                                    t_retrieve = timing_module.time()
                                    query_bundle = QueryBundle(query_str=message)
                                    retrieval_results = retriever.retrieve(query_bundle)
                                    timings["5_chunk_retrieval_ms"] = int((timing_module.time() - t_retrieve) * 1000)

                                    # Stage 6: Format sources
                                    t_format = timing_module.time()
                                    seen_sources = set()
                                    for node_with_score in retrieval_results[:6]:
                                        node = node_with_score.node
                                        metadata = node.metadata or {}
                                        # Skip transformation nodes
                                        if metadata.get('node_type') in ('summary', 'insight', 'question'):
                                            continue
                                        filename = metadata.get('file_name') or metadata.get('filename') or 'Unknown'
                                        page = metadata.get('page_label') or metadata.get('page') or metadata.get('page_number')
                                        source_key = f"{filename}:{page}" if page else filename
                                        if source_key not in seen_sources:
                                            seen_sources.add(source_key)
                                            source_info = {
                                                'filename': filename,
                                                'score': float(round(node_with_score.score, 3)) if node_with_score.score else None,
                                                'snippet': node.text[:200] + '...' if len(node.text) > 200 else node.text
                                            }
                                            if page:
                                                source_info['page'] = page
                                            pre_extracted_sources.append(source_info)
                                    timings["6_source_formatting_ms"] = int((timing_module.time() - t_format) * 1000)
                                    logger.info(f"Pre-extracted {len(pre_extracted_sources)} sources from retriever")
                        except Exception as e:
                            logger.warning(f"Could not pre-extract sources: {e}")

                    # Choose query method based on selection
                    t_query = timing_module.time()
                    if selected_projects:
                        # PROJECT MODE: Use simple query() method for direct Q&A with project documents
                        # Engine already set during source extraction above
                        # force_reset=False since we already set it
                        if not hasattr(self._pipeline, '_engine') or self._pipeline._engine is None:
                            self._pipeline.set_engine(offering_filter=selected_projects, force_reset=True)
                        rag_response = self._pipeline.query(
                            mode=mode,
                            message=message,
                            chatbot=history,
                            query_settings=query_settings_obj
                        )
                    elif selected_offerings:
                        # SALES MODE: Use query_sales_mode for intelligent classification
                        # Only use when offerings are explicitly selected
                        rag_response = self._pipeline.query_sales_mode(
                            message=message,
                            selected_offerings=selected_offerings,
                            selected_projects=selected_projects,
                            chatbot=history
                        )
                    else:
                        # GENERAL CHAT MODE: Bypass retriever, direct LLM chat
                        # Uses SimpleChatEngine without loading any documents
                        rag_response = self._pipeline.chat_without_retrieval(
                            message=message,
                            chatbot=history,
                            query_settings=query_settings_obj
                        )
                    timings["6b_query_execution_ms"] = int((timing_module.time() - t_query) * 1000)

                    # Get the full response text from RAG
                    t_llm = timing_module.time()
                    document_context = ""

                    # Check if there's a response prefix from sales mode
                    response_prefix = getattr(rag_response, 'response_prefix', '')
                    if response_prefix:
                        # Stream the prefix first
                        for token in response_prefix:
                            yield f"data: {json.dumps({'token': token})}\n\n"

                    for token in rag_response.response_gen:
                        document_context += token
                    timings["7_llm_completion_ms"] = int((timing_module.time() - t_llm) * 1000)

                    logger.info(f"Retrieved document context: {document_context[:200]}...")

                    # STEP 2: Check if image generation is requested
                    if self._is_image_generation_request(message) and self._image_provider:
                        msg = "\n\n**Generating image based on your request and document context...**\n"
                        yield f"data: {json.dumps({'token': msg})}\n\n"

                        # Create enhanced prompt using document context
                        enhanced_prompt = self._create_image_prompt_with_context(
                            user_message=message,
                            document_content=document_context
                        )

                        yield f"data: {json.dumps({'token': 'Creating visual representation...'})}\n\n"

                        try:
                            # Generate image using the plugin-based provider
                            image_paths = self._image_provider.generate(
                                prompt=enhanced_prompt,
                                num_images=1,
                                aspect_ratio="16:9"  # Default to landscape for infographics
                            )

                            if image_paths:
                                for idx, path in enumerate(image_paths):
                                    image_url = f"/image/{os.path.basename(path)}"
                                    success_msg = f"\n Image {idx + 1} generated!\n"
                                    yield f"data: {json.dumps({'token': success_msg})}\n\n"
                                    yield f"data: {json.dumps({'image': image_url, 'message': 'Generated image'})}\n\n"
                            else:
                                warn_msg = "\n No images were generated. Please try a different prompt.\n"
                                yield f"data: {json.dumps({'token': warn_msg})}\n\n"

                        except Exception as img_error:
                            logger.error(f"Image generation failed: {img_error}")
                            err_msg = f"\n Image generation failed: {str(img_error)}\n"
                            yield f"data: {json.dumps({'token': err_msg})}\n\n"

                        # Store in session memory for cross-request persistence
                        if project_id:
                            try:
                                self._pipeline.store_conversation_exchange(
                                    user_message=message,
                                    assistant_message=document_context,
                                    project_id=project_id
                                )
                            except Exception as save_err:
                                logger.warning(f"Failed to store in session memory: {save_err}")

                        # Log query to QueryLogger for metrics (image generation path)
                        if hasattr(self._pipeline, '_query_logger') and self._pipeline._query_logger:
                            try:
                                from codeloom.core.observability.token_counter import get_token_counter
                                token_counter = get_token_counter()
                                prompt_tokens = token_counter.count_tokens(message)
                                completion_tokens = token_counter.count_tokens(document_context)
                                model_name = self._pipeline.get_model_name() or "unknown"
                                user_id = data.get("user_id", "00000000-0000-0000-0000-000000000001")
                                img_execution_time_ms = int((timing_module.time() - start_time) * 1000)

                                self._pipeline._query_logger.log_query(
                                    project_id=project_id,
                                    user_id=user_id,
                                    query_text=message,
                                    model_name=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    response_time_ms=timings.get("7_llm_completion_ms", img_execution_time_ms)
                                )
                            except Exception as log_err:
                                logger.warning(f"Failed to log query metrics: {log_err}")

                        yield f"data: {json.dumps({'done': True})}\n\n"

                    elif self._is_image_generation_request(message) and not self._image_provider:
                        # Image generation requested but provider not available
                        config_msg = "\n Image generation is not configured. Please set up GOOGLE_API_KEY in your environment.\n"
                        yield f"data: {json.dumps({'token': config_msg})}\n\n"
                        # Fall through to return the text response
                        for token in document_context:
                            yield f"data: {json.dumps({'token': token})}\n\n"

                        # Store in session memory for cross-request persistence
                        if project_id:
                            try:
                                self._pipeline.store_conversation_exchange(
                                    user_message=message,
                                    assistant_message=document_context,
                                    project_id=project_id
                                )
                            except Exception as save_err:
                                logger.warning(f"Failed to store in session memory: {save_err}")

                        # Log query to QueryLogger for metrics
                        if hasattr(self._pipeline, '_query_logger') and self._pipeline._query_logger:
                            try:
                                from codeloom.core.observability.token_counter import get_token_counter
                                token_counter = get_token_counter()
                                prompt_tokens = token_counter.count_tokens(message)
                                completion_tokens = token_counter.count_tokens(document_context)
                                model_name = self._pipeline.get_model_name() or "unknown"
                                user_id = data.get("user_id", "00000000-0000-0000-0000-000000000001")
                                no_img_execution_time_ms = int((timing_module.time() - start_time) * 1000)

                                self._pipeline._query_logger.log_query(
                                    project_id=project_id,
                                    user_id=user_id,
                                    query_text=message,
                                    model_name=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    response_time_ms=timings.get("7_llm_completion_ms", no_img_execution_time_ms)
                                )
                            except Exception as log_err:
                                logger.warning(f"Failed to log query metrics: {log_err}")

                        yield f"data: {json.dumps({'done': True})}\n\n"

                    else:
                        # STEP 5: Return normal RAG response
                        # Time the response streaming
                        t_stream = timing_module.time()
                        for token in document_context:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        timings["8_response_streaming_ms"] = int((timing_module.time() - t_stream) * 1000)

                        # STEP 6: Use pre-extracted sources (extracted before streaming)
                        # Note: source_nodes from rag_response is empty after streaming completes
                        # So we use pre_extracted_sources which were retrieved BEFORE the chat
                        sources = pre_extracted_sources
                        logger.info(f"Using {len(sources)} pre-extracted sources for response")

                        # Store in session memory for cross-request persistence
                        if project_id:
                            try:
                                self._pipeline.store_conversation_exchange(
                                    user_message=message,
                                    assistant_message=document_context,
                                    project_id=project_id
                                )
                            except Exception as save_err:
                                logger.warning(f"Failed to store in session memory: {save_err}")

                        # Calculate total execution time
                        execution_time_ms = int((timing_module.time() - start_time) * 1000)

                        # Log query to QueryLogger for metrics
                        if hasattr(self._pipeline, '_query_logger') and self._pipeline._query_logger:
                            try:
                                from codeloom.core.observability.token_counter import get_token_counter
                                token_counter = get_token_counter()
                                prompt_tokens = token_counter.count_tokens(message)
                                completion_tokens = token_counter.count_tokens(document_context)
                                model_name = self._pipeline.get_model_name() or "unknown"
                                user_id = data.get("user_id", "00000000-0000-0000-0000-000000000001")

                                self._pipeline._query_logger.log_query(
                                    project_id=project_id,
                                    user_id=user_id,
                                    query_text=message,
                                    model_name=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    response_time_ms=timings.get("7_llm_completion_ms", execution_time_ms)
                                )
                            except Exception as log_err:
                                logger.warning(f"Failed to log query metrics: {log_err}")

                        # Send sources and metadata with the done signal
                        metadata = {
                            'execution_time_ms': execution_time_ms,
                            'timings': timings,
                            'node_count': len(pre_extracted_sources)
                        }
                        yield f"data: {json.dumps({'done': True, 'sources': sources, 'metadata': metadata})}\n\n"

                except Exception as e:
                    logger.error(f"Error during chat: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(
                stream_with_context(process_request()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )

        @self._app.route("/upload", methods=["POST"])
        def upload():
            if "files" not in request.files:
                return jsonify({"success": False, "error": "No files provided"})

            files = request.files.getlist("files")
            if not files or files[0].filename == "":
                return jsonify({"success": False, "error": "No files selected"})

            # Get project_id from form (project-based architecture)
            project_id = request.form.get("project_id", "")
            user_id = "00000000-0000-0000-0000-000000000001"  # Default user

            # Legacy metadata (still captured for backwards compatibility)
            it_practice = request.form.get("it_practice", "")
            offering_name = request.form.get("offering_name", "")
            offering_id = ""
            if offering_name:
                offering_id = offering_name.lower().replace(" ", "-").replace("_", "-")

            uploaded_files = []
            try:
                for file in files:
                    if file.filename:
                        # Save file
                        filepath = self._upload_dir / file.filename
                        file.save(str(filepath))
                        uploaded_files.append(str(filepath))
                        logger.info(f"Uploaded: {file.filename} [Project: {project_id}]")

                # Process documents with project-based architecture
                if uploaded_files:
                    self._pipeline.store_nodes(
                        input_files=uploaded_files,
                        project_id=project_id if project_id else None,
                        user_id=user_id
                    )
                    self._pipeline.set_chat_mode()
                    # Track processed files and save metadata
                    for f in uploaded_files:
                        filename = os.path.basename(f)
                        if filename not in self._processed_files:
                            self._processed_files.append(filename)
                        # Save document metadata (legacy support)
                        self._add_document_metadata(filename, it_practice, offering_name, offering_id)
                    logger.info(f"Processed {len(uploaded_files)} documents for project {project_id}")

                return jsonify({
                    "success": True,
                    "count": len(uploaded_files),
                    "files": [os.path.basename(f) for f in uploaded_files],
                    "all_files": self._processed_files,
                    "project_id": project_id,
                    "metadata": {
                        "it_practice": it_practice,
                        "offering_name": offering_name,
                        "offering_id": offering_id
                    }
                })

            except Exception as e:
                logger.error(f"Error uploading files: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/clear", methods=["POST"])
        def clear():
            try:
                self._pipeline.clear_conversation()
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error clearing conversation: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/reset", methods=["POST"])
        def reset():
            try:
                self._pipeline.reset_documents()
                self._pipeline.reset_conversation()

                # Clear upload directory
                for file in self._upload_dir.iterdir():
                    try:
                        file.unlink()
                    except Exception:
                        pass

                # Clear tracked files
                self._processed_files.clear()

                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error resetting: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/model", methods=["POST"])
        def set_model():
            data = request.json
            model = data.get("model", "")

            if not model:
                return jsonify({"success": False, "error": "No model specified"})

            try:
                # Skip existence check for API-based models
                is_api_model = (
                    model.startswith("gpt-") or
                    model.startswith("claude-") or
                    model.startswith("gemini-")
                )

                # Check if model exists (only for Ollama models)
                if not is_api_model and not self._pipeline.check_exist(model):
                    return jsonify({
                        "success": False,
                        "error": f"Model {model} not found. Pull it first."
                    })

                self._pipeline.set_model_name(model)
                self._pipeline.set_model()
                self._pipeline.set_engine()
                logger.info(f"Model set to: {model}")

                return jsonify({"success": True, "model": model})

            except Exception as e:
                logger.error(f"Error setting model: {e}")
                return jsonify({"success": False, "error": str(e)})

        # DEPRECATED: Old Flask templates - React SPA handles these routes now
        # @self._app.route("/projects")
        # def projects_page():
        #     """Serve the projects management page."""
        #     return render_template("projects.html")

        # @self._app.route("/documents")
        # def documents_page():
        #     """Serve the documents management page."""
        #     return render_template("documents.html")

        @self._app.route("/api/documents/list", methods=["GET"])
        def list_documents():
            """Get list of uploaded documents with metadata."""
            try:
                documents = []
                # Get all files from upload directory
                if self._upload_dir.exists():
                    for file_path in self._upload_dir.iterdir():
                        if file_path.is_file():
                            # Get file size
                            file_size = file_path.stat().st_size
                            size_str = self._format_file_size(file_size)

                            # Get document metadata
                            doc_metadata = self._get_document_metadata(file_path.name)
                            documents.append({
                                "name": file_path.name,
                                "size": size_str,
                                "it_practice": doc_metadata.get("it_practice", "N/A"),
                                "offering_name": doc_metadata.get("offering_name", "N/A")
                            })

                return jsonify({
                    "success": True,
                    "documents": documents,
                    "count": len(documents)
                })
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/documents/<filename>", methods=["DELETE"])
        def delete_document(filename):
            """Delete a document."""
            try:
                file_path = self._upload_dir / filename

                if not file_path.exists():
                    return jsonify({
                        "success": False,
                        "error": "File not found"
                    }), 404

                # Delete the file
                file_path.unlink()

                # Remove from processed files list
                if filename in self._processed_files:
                    self._processed_files.remove(filename)

                # Remove document metadata
                self._remove_document_metadata(filename)

                logger.info(f"Deleted document: {filename}")

                return jsonify({
                    "success": True,
                    "message": f"Document '{filename}' deleted successfully"
                })

            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/health", methods=["GET"])
        def health():
            """Comprehensive health check endpoint (MVP 6)."""
            import time
            import psutil
            import requests

            start_time = time.time()
            health_status = {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "components": {}
            }

            # Check database connection
            try:
                if self._db_manager:
                    with self._db_manager.get_session() as session:
                        session.execute(text("SELECT 1"))
                    health_status["components"]["database"] = {
                        "status": "healthy",
                        "type": "postgresql"
                    }
                else:
                    health_status["components"]["database"] = {
                        "status": "not_configured",
                        "type": "none"
                    }
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Check Ollama connection
            try:
                ollama_url = f"http://{self._ollama_host}:11434/api/tags"
                response = requests.get(ollama_url, timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    health_status["components"]["ollama"] = {
                        "status": "healthy",
                        "models_available": len(models)
                    }
                else:
                    health_status["components"]["ollama"] = {
                        "status": "degraded",
                        "http_status": response.status_code
                    }
            except Exception as e:
                health_status["components"]["ollama"] = {
                    "status": "unavailable",
                    "error": str(e)
                }

            # Check vector store
            try:
                if self._pipeline and self._pipeline._vector_store:
                    stats = self._pipeline._vector_store.get_collection_stats()
                    health_status["components"]["vector_store"] = {
                        "status": "healthy",
                        "type": "pgvector",
                        "document_count": stats.get("count", 0)
                    }
                else:
                    health_status["components"]["vector_store"] = {
                        "status": "not_initialized"
                    }
            except Exception as e:
                health_status["components"]["vector_store"] = {
                    "status": "error",
                    "error": str(e)
                }

            # System resources
            try:
                health_status["system"] = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            except Exception:
                health_status["system"] = {"status": "unavailable"}

            # Response time
            health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

            # Overall status
            component_statuses = [c.get("status") for c in health_status["components"].values()]
            if "unhealthy" in component_statuses:
                health_status["status"] = "unhealthy"
            elif "degraded" in component_statuses or "unavailable" in component_statuses:
                health_status["status"] = "degraded"

            status_code = 200 if health_status["status"] == "healthy" else 503
            return jsonify(health_status), status_code

        @self._app.route("/api/health", methods=["GET"])
        def api_health():
            """Simple health check for load balancers."""
            return jsonify({"status": "ok"})

        @self._app.route("/generate-image", methods=["POST"])
        @self._app.route("/image/generate", methods=["POST"])
        @self._app.route("/api/image/generate", methods=["POST"])
        def generate_image():
            try:
                if not self._image_provider:
                    return jsonify({
                        "success": False,
                        "error": "Image generation not configured. Set GOOGLE_API_KEY."
                    })

                data = request.json
                prompt = data.get("prompt", "")
                num_images = data.get("num_images", 1)
                aspect_ratio = data.get("aspect_ratio", "1:1")

                if not prompt or not prompt.strip():
                    return jsonify({"success": False, "error": "Prompt cannot be empty"})

                # Generate images using plugin provider
                image_paths = self._image_provider.generate(
                    prompt=prompt,
                    num_images=num_images,
                    aspect_ratio=aspect_ratio
                )

                # Get image info
                images_info = []
                for path in image_paths:
                    info = self._image_provider.get_image_info(path)
                    info["path"] = path
                    info["url"] = f"/image/{Path(path).name}"
                    images_info.append(info)

                return jsonify({
                    "success": True,
                    "count": len(image_paths),
                    "images": images_info
                })

            except Exception as e:
                logger.error(f"Error generating image: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/image/<filename>", methods=["GET"])
        def get_image(filename):
            try:
                if not self._image_provider:
                    return jsonify({"error": "Image provider not configured"}), 503

                image_dir = self._image_provider.output_dir
                filepath = image_dir / filename

                if not filepath.exists():
                    return jsonify({"error": "Image not found"}), 404

                return send_file(str(filepath), mimetype="image/png")

            except Exception as e:
                logger.error(f"Error serving image: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route("/images", methods=["GET"])
        def list_images():
            try:
                if not self._image_provider:
                    return jsonify({
                        "success": True,
                        "count": 0,
                        "images": [],
                        "message": "Image provider not configured"
                    })

                image_paths = self._image_provider.list_generated_images()
                images_info = []

                for path in image_paths:
                    info = self._image_provider.get_image_info(path)
                    info["url"] = f"/image/{Path(path).name}"
                    images_info.append(info)

                return jsonify({
                    "success": True,
                    "count": len(images_info),
                    "images": images_info
                })

            except Exception as e:
                logger.error(f"Error listing images: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/clear-images", methods=["POST"])
        def clear_images():
            try:
                if not self._image_provider:
                    return jsonify({
                        "success": True,
                        "deleted_count": 0,
                        "message": "Image provider not configured"
                    })

                deleted_count = self._image_provider.clear_images()
                return jsonify({
                    "success": True,
                    "deleted_count": deleted_count
                })

            except Exception as e:
                logger.error(f"Error clearing images: {e}")
                return jsonify({"success": False, "error": str(e)})

        # === Sales Enablement Metadata Endpoints ===

        @self._app.route("/api/practices", methods=["GET"])
        def get_practices():
            """Get all IT practices."""
            try:
                practices = self._metadata_manager.get_all_practices()
                return jsonify({
                    "success": True,
                    "practices": practices,
                    "count": len(practices)
                })
            except Exception as e:
                logger.error(f"Error getting practices: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/offerings", methods=["GET"])
        def get_offerings():
            """Get all unique offerings from uploaded documents."""
            try:
                # Load document metadata
                metadata = self._load_document_metadata()

                # Extract unique offerings with their IT practices
                offerings_map = {}
                for filename, doc_meta in metadata.items():
                    offering_name = doc_meta.get("offering_name", "")
                    it_practice = doc_meta.get("it_practice", "")

                    # Skip N/A or empty offerings
                    if offering_name and offering_name != "N/A":
                        if offering_name not in offerings_map:
                            offerings_map[offering_name] = {
                                "name": offering_name,
                                "it_practice": it_practice,
                                "document_count": 0
                            }
                        offerings_map[offering_name]["document_count"] += 1

                # Convert to list sorted by name
                offerings = sorted(offerings_map.values(), key=lambda x: x["name"])

                return jsonify({
                    "success": True,
                    "offerings": offerings,
                    "count": len(offerings)
                })
            except Exception as e:
                logger.error(f"Error getting offerings: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/models", methods=["GET"])
        def get_available_models():
            """Get all available LLM models from config file and Ollama.

            Models are configured in config/models.yaml. For Ollama, models
            can be auto-detected from the running server or filtered by whitelist.
            """
            try:
                from codeloom.core.model import LocalRAGModel
                from codeloom.setting import get_models_settings

                models = []
                models_config = get_models_settings()

                # Provider name mapping for display
                provider_display = {
                    "ollama": "Ollama",
                    "openai": "OpenAI",
                    "anthropic": "Anthropic",
                    "google": "Google",
                    "groq": "Groq"
                }

                # Process each provider from config
                for provider_key, provider_config in models_config.providers.items():
                    if not provider_config.enabled:
                        continue

                    # Check API key requirement
                    if provider_config.requires_api_key and not models_config.has_api_key(provider_key):
                        continue

                    provider_name = provider_display.get(provider_key, provider_key.title())
                    provider_type = provider_config.type

                    if provider_key == "ollama":
                        # For Ollama, auto-detect models from server
                        ollama_models = LocalRAGModel.list_available_models(self._ollama_host)

                        # If whitelist is defined in config, filter models
                        whitelist = [m.name for m in provider_config.models] if provider_config.models else None

                        for model in ollama_models:
                            if whitelist is None or model in whitelist:
                                models.append({
                                    "name": model,
                                    "provider": provider_name,
                                    "type": provider_type
                                })
                    else:
                        # For API providers, use models from config
                        for model_config in provider_config.models:
                            if model_config.enabled:
                                models.append({
                                    "name": model_config.name,
                                    "display_name": model_config.display_name or model_config.name,
                                    "provider": provider_name,
                                    "type": provider_type
                                })

                # Remove duplicates while preserving order
                seen = set()
                unique_models = []
                for model in models:
                    if model["name"] not in seen:
                        seen.add(model["name"])
                        unique_models.append(model)

                return jsonify({
                    "success": True,
                    "models": unique_models,
                    "count": len(unique_models),
                    "default_model": models_config.default_model,
                    "default_provider": models_config.default_provider
                })

            except Exception as e:
                logger.error(f"Error getting models: {e}")
                return jsonify({"success": False, "error": str(e), "models": []})
        @self._app.route("/api/practices/<practice_name>/offerings", methods=["GET"])
        def get_practice_offerings(practice_name):
            """Get offerings for a specific practice."""
            try:
                offerings = self._metadata_manager.get_offerings_by_practice(practice_name)
                return jsonify({
                    "success": True,
                    "practice": practice_name,
                    "offerings": offerings,
                    "count": len(offerings)
                })
            except Exception as e:
                logger.error(f"Error getting practice offerings: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/practices", methods=["POST"])
        def add_practice():
            """Add a new IT practice."""
            try:
                data = request.json
                practice_name = data.get("name", "")

                if not practice_name:
                    return jsonify({"success": False, "error": "Practice name required"})

                success = self._metadata_manager.add_practice(practice_name)

                if success:
                    return jsonify({
                        "success": True,
                        "practice": practice_name
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Practice already exists or invalid name"
                    })

            except Exception as e:
                logger.error(f"Error adding practice: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/offerings", methods=["POST"])
        def add_offering():
            """Add a new offering."""
            try:
                data = request.json
                practice = data.get("practice", "")
                offering_name = data.get("name", "")
                description = data.get("description", "")

                if not practice or not offering_name:
                    return jsonify({
                        "success": False,
                        "error": "Practice and offering name required"
                    })

                offering_id = self._metadata_manager.add_offering(
                    practice=practice,
                    offering_name=offering_name,
                    description=description
                )

                if offering_id:
                    return jsonify({
                        "success": True,
                        "offering_id": offering_id,
                        "offering_name": offering_name,
                        "practice": practice
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Failed to add offering (practice may not exist or offering already exists)"
                    })

            except Exception as e:
                logger.error(f"Error adding offering: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/metadata/stats", methods=["GET"])
        def get_metadata_stats():
            """Get metadata statistics."""
            try:
                stats = self._metadata_manager.get_statistics()
                return jsonify({
                    "success": True,
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return jsonify({"success": False, "error": str(e)})

        # Register web content routes (for web search and scraping)
        try:
            self._web_ingestion = WebContentIngestion(
                project_manager=self._project_manager,
                setting=self._pipeline._settings if self._pipeline else None,
            )
            create_web_content_routes(self._app, self._web_ingestion, self._pipeline)
            logger.info("Web content routes registered")
        except Exception as e:
            logger.warning(f"Web content routes not available: {e}")
            self._web_ingestion = None

        # Register Content Studio routes
        try:
            self._studio_manager = StudioManager(self._db_manager)
            self._synopsis_manager = SynopsisManager()  # Create fresh instance
            # Pass pipeline for retrieval-based content selection (same as Chat)
            create_studio_routes(
                self._app,
                self._studio_manager,
                self._synopsis_manager,
                pipeline=self._pipeline,
            )
            logger.info("Content Studio routes registered")
            if self._pipeline:
                logger.info("Content Studio using pipeline retriever for content selection")
        except Exception as e:
            logger.warning(f"Content Studio routes not available: {e}")
            self._studio_manager = None

        # Register Vision API routes
        try:
            create_vision_routes(self._app, upload_folder=str(self._upload_dir))
            logger.info("Vision API routes registered")
        except Exception as e:
            logger.warning(f"Vision API routes not available: {e}")

        # Register AI Transformation routes
        try:
            if self._db_manager:
                # Get transformation worker from pipeline if available
                transformation_worker = None
                if hasattr(self._pipeline, 'transformation_worker'):
                    transformation_worker = self._pipeline.transformation_worker

                create_transformation_routes(
                    self._app,
                    self._db_manager,
                    transformation_worker=transformation_worker
                )
                logger.info("Transformation API routes registered")
                if transformation_worker:
                    logger.info("Transformation retry enabled via TransformationWorker")
        except Exception as e:
            logger.warning(f"Transformation API routes not available: {e}")

        # Register Agent API routes
        try:
            create_agent_routes(self._app, self._pipeline)
            logger.info("Agent API routes registered")
        except Exception as e:
            logger.warning(f"Agent API routes not available: {e}")

        # Register Analytics routes (pass pipeline for LLM access in dashboard generation)
        try:
            create_analytics_routes(self._app, db_manager=self._db_manager, pipeline=self._pipeline)
            logger.info("Analytics API routes registered")
        except Exception as e:
            logger.warning(f"Analytics API routes not available: {e}")

        # Register SQL Chat (Chat with Data) routes
        try:
            create_sql_chat_routes(
                self._app,
                pipeline=self._pipeline,
                db_manager=self._db_manager,
                project_manager=self._project_manager
            )
            logger.info("SQL Chat API routes registered")
        except Exception as e:
            logger.warning(f"SQL Chat API routes not available: {e}")

        # Register Simple Query API routes (programmatic access)
        try:
            create_query_routes(
                self._app,
                pipeline=self._pipeline,
                db_manager=self._db_manager,
                project_manager=self._project_manager
            )
            logger.info("Simple Query API routes registered (/api/query)")
        except Exception as e:
            logger.warning(f"Query API routes not available: {e}")

        # Register V2 Chat API routes (fast pattern with memory)
        try:
            create_chat_v2_routes(
                self._app,
                pipeline=self._pipeline,
                db_manager=self._db_manager,
                project_manager=self._project_manager,
                conversation_store=self._pipeline._conversation_store
            )
            logger.info("V2 Chat API routes registered (/api/v2/chat)")
        except Exception as e:
            logger.warning(f"V2 Chat API routes not available: {e}")

        # Register Auth API routes (login, logout, password, API key)
        try:
            create_auth_routes(
                self._app,
                db_manager=self._db_manager
            )
            logger.info("Auth API routes registered (/api/auth)")
        except Exception as e:
            logger.warning(f"Auth API routes not available: {e}")

        # Register Admin API routes (RBAC management)
        try:
            create_admin_routes(
                self._app,
                db_manager=self._db_manager,
                project_manager=self._project_manager,
                pipeline=self._pipeline
            )
            logger.info("Admin API routes registered (/api/admin)")

            # Register RBAC session cleanup
            from ..core.auth.rbac import cleanup_rbac_session
            self._app.teardown_appcontext(cleanup_rbac_session)
        except Exception as e:
            logger.warning(f"Admin API routes not available: {e}")

        # Register Settings API routes (runtime configuration)
        try:
            create_settings_routes(self._app)
            logger.info("Settings API routes registered (/api/settings)")
        except Exception as e:
            logger.warning(f"Settings API routes not available: {e}")

        # Register Quiz API routes
        try:
            create_quiz_routes(
                self._app,
                pipeline=self._pipeline,
                db_manager=self._db_manager,
                project_manager=self._project_manager
            )
            logger.info("Quiz API routes registered (/api/quiz)")
        except Exception as e:
            logger.warning(f"Quiz API routes not available: {e}")

        # === Query Logging & Observability Endpoints ===

        @self._app.route("/api/usage-stats", methods=["GET"])
        def get_usage_stats():
            """Get usage statistics for current session."""
            try:
                if not self._pipeline._query_logger:
                    return jsonify({
                        "success": False,
                        "error": "Query logger not initialized"
                    })

                stats = self._pipeline._query_logger.get_usage_stats()
                return jsonify({
                    "success": True,
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Error getting usage stats: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/recent-queries", methods=["GET"])
        def get_recent_queries():
            """Get recent query history."""
            try:
                if not self._pipeline._query_logger:
                    return jsonify({
                        "success": False,
                        "error": "Query logger not initialized"
                    })

                limit = request.args.get("limit", 50, type=int)
                recent = self._pipeline._query_logger.get_recent_logs(limit=limit)

                # Convert datetime objects to strings for JSON serialization
                for log in recent:
                    if "timestamp" in log:
                        log["timestamp"] = log["timestamp"].isoformat()

                return jsonify({
                    "success": True,
                    "queries": recent,
                    "count": len(recent)
                })
            except Exception as e:
                logger.error(f"Error getting recent queries: {e}")
                return jsonify({"success": False, "error": str(e)})

        @self._app.route("/api/model-pricing", methods=["GET"])
        def get_model_pricing():
            """Get pricing information for all supported models."""
            try:
                if not self._pipeline._query_logger:
                    return jsonify({
                        "success": False,
                        "error": "Query logger not initialized"
                    })

                models = self._pipeline._query_logger.list_supported_models()
                pricing = {}
                for model in models:
                    model_pricing = self._pipeline._query_logger.get_model_pricing(model)
                    if model_pricing:
                        pricing[model] = model_pricing

                return jsonify({
                    "success": True,
                    "pricing": pricing,
                    "count": len(pricing)
                })
            except Exception as e:
                logger.error(f"Error getting model pricing: {e}")
                return jsonify({"success": False, "error": str(e)})

        # =============================================
        # Project API Routes
        # =============================================

        @self._app.route("/api/projects", methods=["GET"])
        def list_projects():
            """List all projects for the default user."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                # Use default user ID (UUID format)
                user_id = "00000000-0000-0000-0000-000000000001"
                projects = self._project_manager.list_projects(user_id)

                return jsonify({
                    "success": True,
                    "projects": projects,
                    "count": len(projects)
                })
            except Exception as e:
                logger.error(f"Error listing projects: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects", methods=["POST"])
        def create_project():
            """Create a new project."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                data = request.json
                name = data.get("name")
                description = data.get("description", "")

                if not name:
                    return jsonify({
                        "success": False,
                        "error": "Project name is required"
                    }), 400

                # Use default user ID
                user_id = "00000000-0000-0000-0000-000000000001"

                project_data = self._project_manager.create_project(
                    user_id=user_id,
                    name=name,
                    description=description
                )

                return jsonify({
                    "success": True,
                    "project": {
                        "id": project_data["id"],
                        "name": project_data["name"]
                    },
                    "message": f"Project '{name}' created successfully"
                })
            except Exception as e:
                logger.error(f"Error creating project: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>", methods=["PUT"])
        def update_project(project_id):
            """Update a project's name or description."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                data = request.json or {}
                name = data.get("name")
                description = data.get("description")

                if not name and description is None:
                    return jsonify({
                        "success": False,
                        "error": "No update data provided (name or description required)"
                    }), 400

                success = self._project_manager.update_project(
                    project_id=project_id,
                    name=name,
                    description=description
                )

                if success:
                    # Get updated project details
                    project = self._project_manager.get_project(project_id)
                    return jsonify({
                        "success": True,
                        "project": project,
                        "message": "Project updated successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Project not found"
                    }), 404
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400
            except Exception as e:
                logger.error(f"Error updating project: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>", methods=["DELETE"])
        def delete_project(project_id):
            """Delete a project."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                success = self._project_manager.delete_project(project_id)

                if success:
                    return jsonify({
                        "success": True,
                        "message": f"Project deleted successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Failed to delete project"
                    }), 404
            except Exception as e:
                logger.error(f"Error deleting project: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents", methods=["POST"])
        def upload_to_project(project_id):
            """Upload documents to a project."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                # Get uploaded files
                if 'files' not in request.files:
                    return jsonify({
                        "success": False,
                        "error": "No files provided"
                    }), 400

                files = request.files.getlist('files')
                if not files or files[0].filename == '':
                    return jsonify({
                        "success": False,
                        "error": "No files selected"
                    }), 400

                uploaded_files = []

                # Collect file paths for batch processing
                file_paths = []
                file_info = {}

                for file in files:
                    if file and file.filename:
                        # Save file to data directory
                        file_path = os.path.join(str(self._data_dir), file.filename)
                        file.save(file_path)

                        # Get file stats for response
                        file_size = os.path.getsize(file_path)

                        file_paths.append(file_path)
                        file_info[file.filename] = file_size

                # Process all files at once with project_id metadata
                # store_nodes() handles:
                # 1. Document registration in PostgreSQL with correct chunk_count
                # 2. Node metadata (project_id, source_id, user_id)
                # 3. pgvector persistence
                logger.info(f"Processing {len(file_paths)} files for project {project_id}")
                returned_nodes = self._pipeline.store_nodes(
                    input_files=file_paths,
                    project_id=project_id,
                    user_id="00000000-0000-0000-0000-000000000001"  # Default user UUID
                )

                # Build response with file information
                for filename, file_size in file_info.items():
                    # Track processed file
                    if filename not in self._processed_files:
                        self._processed_files.append(filename)

                    # Get source_id from database for this file
                    docs = self._project_manager.get_documents(project_id)
                    source_id = None
                    for doc in docs:
                        if doc['file_name'] == filename:
                            source_id = doc['source_id']
                            break

                    uploaded_files.append({
                        "file_name": filename,
                        "source_id": source_id,
                        "file_size": file_size
                    })

                    logger.info(f"Uploaded {filename} to project {project_id} (source_id: {source_id})")

                # Force rebuild chat engine after loading documents (load nodes from pgvector)
                logger.info("Rebuilding chat engine with newly loaded documents from pgvector")
                self._pipeline.set_chat_mode(force_reset=True)

                return jsonify({
                    "success": True,
                    "uploaded": uploaded_files,
                    "count": len(uploaded_files),
                    "message": f"Successfully uploaded {len(uploaded_files)} document(s)"
                })
            except Exception as e:
                logger.error(f"Error uploading to project: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents", methods=["GET"])
        def list_project_documents(project_id):
            """List all documents in a project."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                documents = self._project_manager.get_documents(project_id)

                return jsonify({
                    "success": True,
                    "documents": documents,
                    "count": len(documents)
                })
            except Exception as e:
                logger.error(f"Error listing project documents: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents/<source_id>", methods=["DELETE"])
        def delete_project_document(project_id, source_id):
            """Delete a document from a project."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                # Delete from PostgreSQL database
                success = self._project_manager.remove_document(project_id, source_id)

                if not success:
                    return jsonify({
                        "success": False,
                        "error": "Document not found or deletion failed"
                    }), 404

                # Delete from pgvector embeddings table
                if self._pipeline and self._pipeline._vector_store:
                    pgvector_success = self._pipeline._vector_store.delete_document_nodes(source_id)

                    if not pgvector_success:
                        logger.warning(f"pgvector deletion failed for document {source_id}, but PostgreSQL deletion succeeded")

                logger.info(f"Deleted document {source_id} from project {project_id}")

                return jsonify({
                    "success": True,
                    "message": f"Document deleted successfully"
                })
            except Exception as e:
                logger.error(f"Error deleting project document: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents/<source_id>", methods=["PATCH"])
        def update_project_document(project_id, source_id):
            """Update document properties (e.g., active status)."""
            try:
                if not self._project_manager:
                    return jsonify({
                        "success": False,
                        "error": "Project feature not available"
                    }), 503

                data = request.json or {}
                active = data.get("active")

                if active is None:
                    return jsonify({
                        "success": False,
                        "error": "No update data provided"
                    }), 400

                # Update the document active status
                updated_doc = self._project_manager.update_document_active(
                    project_id=project_id,
                    source_id=source_id,
                    active=active
                )

                if not updated_doc:
                    return jsonify({
                        "success": False,
                        "error": "Document not found"
                    }), 404

                return jsonify({
                    "success": True,
                    "document": {
                        "source_id": updated_doc['source_id'],
                        "file_name": updated_doc['file_name'],
                        "file_type": updated_doc.get('file_type'),
                        "chunk_count": updated_doc.get('chunk_count'),
                        "active": updated_doc['active']
                    },
                    "message": "Document status updated"
                })
            except Exception as e:
                logger.error(f"Error updating project document: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents/<source_id>/content", methods=["GET"])
        def get_document_content(project_id, source_id):
            """Get the full content of a document by reconstructing from chunks."""
            try:
                if not self._db_manager:
                    return jsonify({
                        "success": False,
                        "error": "Database not available"
                    }), 503

                # Query data_embeddings for all chunks of this document
                with self._db_manager.get_session() as session:
                    # Get chunks ordered by chunk_index, filtering by source_id and project_id
                    # Exclude RAPTOR tree nodes (only level 0 chunks)
                    result = session.execute(
                        text("""
                            SELECT text, metadata_
                            FROM data_embeddings
                            WHERE metadata_->>'source_id' = :source_id
                              AND metadata_->>'project_id' = :project_id
                              AND (metadata_->>'raptor_level' IS NULL OR metadata_->>'raptor_level' = '0')
                            ORDER BY COALESCE((metadata_->>'chunk_index')::int, 0)
                        """),
                        {"source_id": source_id, "project_id": project_id}
                    )
                    rows = result.fetchall()

                if not rows:
                    return jsonify({
                        "success": False,
                        "error": "Document content not found"
                    }), 404

                # Reconstruct document content from chunks
                content_parts = []
                filename = ""
                file_type = ""

                for row in rows:
                    content_parts.append(row.text)
                    if row.metadata_:
                        metadata = row.metadata_ if isinstance(row.metadata_, dict) else {}
                        if not filename:
                            filename = metadata.get('file_name', metadata.get('filename', ''))
                        if not file_type:
                            file_type = metadata.get('file_type', '')

                # Join chunks with double newlines for readability
                full_content = "\n\n".join(content_parts)

                return jsonify({
                    "success": True,
                    "content": full_content,
                    "chunk_count": len(rows),
                    "filename": filename,
                    "file_type": file_type
                })

            except Exception as e:
                logger.error(f"Error getting document content: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/documents/<source_id>/pdf", methods=["GET"])
        def get_document_pdf(project_id, source_id):
            """Serve the original PDF file for a document."""
            try:
                if not self._db_manager:
                    return jsonify({
                        "success": False,
                        "error": "Database not available"
                    }), 503

                # Get the document info from code_files
                with self._db_manager.get_session() as session:
                    result = session.execute(
                        text("""
                            SELECT filename, file_path
                            FROM code_files
                            WHERE source_id = :source_id
                              AND project_id = :project_id
                        """),
                        {"source_id": source_id, "project_id": project_id}
                    )
                    row = result.fetchone()

                if not row:
                    return jsonify({
                        "success": False,
                        "error": "Document not found"
                    }), 404

                filename = row.filename
                file_path = row.file_path

                # Check if it's a PDF
                if not filename.lower().endswith('.pdf'):
                    return jsonify({
                        "success": False,
                        "error": "Document is not a PDF"
                    }), 400

                # Try to find the file
                # First check if file_path is provided and exists
                if file_path and Path(file_path).exists():
                    return send_file(
                        file_path,
                        mimetype='application/pdf',
                        as_attachment=False,
                        download_name=filename
                    )

                # Try uploads directory
                upload_path = self._upload_dir / filename
                if upload_path.exists():
                    return send_file(
                        str(upload_path),
                        mimetype='application/pdf',
                        as_attachment=False,
                        download_name=filename
                    )

                # Try data directory
                data_path = self._data_dir / filename
                if data_path.exists():
                    return send_file(
                        str(data_path),
                        mimetype='application/pdf',
                        as_attachment=False,
                        download_name=filename
                    )

                # File not found
                return jsonify({
                    "success": False,
                    "error": "PDF file not found on disk"
                }), 404

            except Exception as e:
                logger.error(f"Error serving PDF: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/conversations", methods=["GET"])
        def get_project_conversations(project_id):
            """Get conversation history for a project."""
            try:
                # Use the same default user UUID as the rest of the system
                default_user_id = "00000000-0000-0000-0000-000000000001"
                user_id = request.args.get("user_id", default_user_id)
                limit = request.args.get("limit", 100, type=int)
                offset = request.args.get("offset", 0, type=int)

                # Get conversation history from the store
                history = self._pipeline._conversation_store.get_conversation_history(
                    project_id=project_id,
                    user_id=user_id,
                    limit=limit,
                    offset=offset
                )

                # Transform to frontend-friendly format
                messages = []
                for entry in history:
                    messages.append({
                        "id": entry.get("conversation_id", ""),
                        "role": entry.get("role", "user"),
                        "content": entry.get("content", ""),
                        "timestamp": entry.get("timestamp", "").isoformat() if hasattr(entry.get("timestamp", ""), "isoformat") else str(entry.get("timestamp", ""))
                    })

                return jsonify({
                    "success": True,
                    "messages": messages,
                    "count": len(messages)
                })
            except Exception as e:
                logger.error(f"Error getting conversation history: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self._app.route("/api/projects/<project_id>/clear-chat", methods=["POST"])
        def clear_project_chat(project_id):
            """Clear all conversation history for a project."""
            try:
                # Reset in-memory ChatMemoryBuffer (critical for session-only mode)
                self._pipeline.clear_conversation()
                logger.info(f"Reset in-memory conversation buffer for project {project_id}")

                return jsonify({
                    "success": True,
                    "message": "Conversation memory cleared"
                })
            except Exception as e:
                logger.error(f"Error clearing conversation: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        # Documentation route - serve mkdocs static site
        @self._app.route("/docs")
        @self._app.route("/docs/")
        def serve_docs_index():
            """Serve documentation homepage."""
            docs_dir = Path(__file__).parent.parent.parent / "site"
            if docs_dir.exists():
                return send_from_directory(str(docs_dir), "index.html")
            return jsonify({"error": "Documentation not built. Run 'mkdocs build' first."}), 404

        @self._app.route("/docs/<path:path>")
        def serve_docs(path):
            """Serve documentation static files."""
            docs_dir = Path(__file__).parent.parent.parent / "site"
            if docs_dir.exists():
                file_path = docs_dir / path
                if file_path.exists() and file_path.is_file():
                    return send_from_directory(str(docs_dir), path)
                # For directory paths, serve index.html
                index_path = docs_dir / path / "index.html"
                if index_path.exists():
                    return send_from_directory(str(docs_dir / path), "index.html")
            return jsonify({"error": "Not found"}), 404

        # React SPA catch-all route - must be last to not interfere with API routes
        @self._app.route("/<path:path>")
        def serve_react_app(path):
            """Serve React frontend assets or fallback to index.html for SPA routing."""
            # Skip API and other backend routes
            if path.startswith(('api/', 'chat', 'upload', 'clear', 'reset', 'model',
                               'projects', 'documents', 'health', 'generate-image',
                               'image/', 'images', 'clear-images', 'outputs/', 'docs')):
                return jsonify({"error": "Not found"}), 404

            if self._frontend_dist.exists():
                # Check if it's a static file request (js, css, images, etc.)
                file_path = self._frontend_dist / path
                if file_path.exists() and file_path.is_file():
                    return send_from_directory(str(self._frontend_dist), path)
                # For SPA routing, return index.html
                return send_from_directory(str(self._frontend_dist), "index.html")
            return render_template("index.html")

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the Flask application."""
        logger.info(f"Starting Flask server on {host}:{port}")
        self._app.run(host=host, port=port, debug=debug, threaded=True)

    def get_app(self):
        """Return the Flask app for WSGI servers."""
        return self._app
