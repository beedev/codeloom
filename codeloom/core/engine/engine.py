import logging
from typing import List, Optional, Union

from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
# PromptTemplate not needed - llama-index 0.10.x expects string for prompts

from .retriever import LocalRetriever
from ..prompt import get_condense_prompt
from ...setting import get_settings, RAGSettings, QueryTimeSettings

logger = logging.getLogger(__name__)


def _unwrap_llm(llm):
    """Extract raw LlamaIndex LLM from wrapper classes like GroqWithBackoff.

    LlamaIndex's resolve_llm() requires instances of the LLM base class.
    Wrapper classes (e.g., GroqWithBackoff for rate limiting) must be
    unwrapped before passing to components like SimpleChatEngine.

    Args:
        llm: LLM instance or wrapper

    Returns:
        Raw LlamaIndex LLM instance
    """
    if hasattr(llm, 'get_raw_llm'):
        raw_llm = llm.get_raw_llm()
        logger.debug(f"Unwrapped LLM: {type(llm).__name__} → {type(raw_llm).__name__}")
        return raw_llm
    return llm


class LocalChatEngine:
    """
    Factory for creating chat engines with or without document context.

    - SimpleChatEngine: For general conversation without documents
    - CondensePlusContextChatEngine: For RAG with document retrieval
    """

    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        self._setting = setting or get_settings()
        self._retriever = LocalRetriever(self._setting, host)
        self._host = host
        logger.debug("LocalChatEngine initialized")

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        language: str = "eng",
        offering_filter: Optional[List[str]] = None,
        vector_store=None,
        chat_history: Optional[List] = None,
        notebook_id: Optional[str] = None,
    ) -> Union[CondensePlusContextChatEngine, SimpleChatEngine]:
        """
        Create appropriate chat engine based on available documents.

        Args:
            llm: Language model for chat
            nodes: Document nodes (empty for simple chat)
            language: Language code for prompts
            offering_filter: List of offering names to filter by (Sales Pitch mode)
            vector_store: Optional LocalVectorStore for metadata filtering
            chat_history: Optional chat history to preserve when recreating engine
            notebook_id: Notebook ID for RAPTOR-aware retrieval

        Returns:
            Configured chat engine
        """
        # Unwrap LLM wrappers (e.g., GroqWithBackoff) for LlamaIndex compatibility
        llm = _unwrap_llm(llm)

        # Session-only memory: Limit based on CHAT_TOKEN_LIMIT setting
        # Must account for: system prompt (~500) + context prompt (~1000) + exchanges
        # Default 32K tokens allows larger context windows for RAPTOR summaries + chunks
        token_limit = self._setting.ollama.chat_token_limit
        logger.debug(f"Session-only memory with {token_limit} token limit (from CHAT_TOKEN_LIMIT)")

        # Truncate chat history if it exceeds token limit
        # LlamaIndex throws "Initial token count exceeds token limit" otherwise
        safe_history = []
        if chat_history:
            # Estimate ~4 chars per token, keep only recent messages
            estimated_tokens = 0
            max_history_tokens = int(token_limit * 0.5)  # Reserve 50% for new context

            # Process in reverse (most recent first) and take what fits
            for msg in reversed(chat_history):
                msg_tokens = len(str(msg.content)) // 4 + 10  # Rough estimate + overhead
                if estimated_tokens + msg_tokens > max_history_tokens:
                    logger.info(f"Truncating chat history at {len(safe_history)} messages to fit token limit")
                    break
                safe_history.insert(0, msg)  # Insert at front to maintain order
                estimated_tokens += msg_tokens

            if len(safe_history) < len(chat_history):
                logger.info(f"Chat history truncated: {len(chat_history)} -> {len(safe_history)} messages")

        # Create memory buffer and set history using set() method
        # Note: chat_history constructor param doesn't work in LlamaIndex 0.10.x
        memory = ChatMemoryBuffer(token_limit=token_limit)
        if safe_history:
            memory.set(safe_history)
            logger.info(f"ChatMemoryBuffer loaded with {len(memory.get_all())} messages from session")

        # Simple chat engine (no documents)
        if not nodes:
            logger.debug("Creating SimpleChatEngine (no documents)")
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=memory
            )

        # RAG chat engine with document retrieval
        filter_msg = f" (filtered by offerings: {offering_filter})" if offering_filter else ""
        logger.debug(f"Creating CondensePlusContextChatEngine with {len(nodes)} nodes{filter_msg}")

        # Use RAPTOR-aware retrieval when notebook_id is available
        # This enables hierarchical retrieval for better summary/detail query handling
        if notebook_id and vector_store:
            logger.info(f"Using RAPTOR-aware retrieval for notebook {notebook_id}")
            retriever = self._retriever.get_combined_raptor_retriever(
                llm=llm,
                language=language,
                nodes=nodes,
                vector_store=vector_store,
                notebook_id=notebook_id,
                source_ids=None,  # Will check all sources in notebook
            )
        else:
            logger.debug("Using standard retrieval (no notebook_id)")
            retriever = self._retriever.get_retrievers(
                llm=llm,
                language=language,
                nodes=nodes,
                offering_filter=offering_filter,
                vector_store=vector_store
            )

        # Get custom condense prompt for preserving customer context
        # Note: llama-index 0.10.x expects string, not PromptTemplate
        condense_prompt_str = get_condense_prompt(language)

        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory,
            condense_prompt=condense_prompt_str
        )

    def clear_retriever_cache(self) -> None:
        """Clear the retriever's index cache."""
        self._retriever.clear_cache()
        logger.debug("Retriever cache cleared")

    def set_query_settings(self, settings: Optional[QueryTimeSettings]) -> None:
        """Set query-time settings for the next retrieval operation.

        These settings override defaults from config files for a single query.
        Call this before each stream_chat() call to apply per-request settings.

        Args:
            settings: QueryTimeSettings instance, or None to use defaults
        """
        self._retriever.set_query_settings(settings)

    def clear_query_settings(self) -> None:
        """Clear query-time settings, reverting to defaults."""
        self._retriever.clear_query_settings()
