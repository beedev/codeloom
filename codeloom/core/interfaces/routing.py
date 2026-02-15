"""Abstract interfaces for document routing components.

This module defines interfaces for the two-stage LLM document routing system.
The routing system determines whether to:
- Synthesize directly from document summaries (DIRECT_SYNTHESIS)
- Deep-dive into specific documents (DEEP_DIVE)
- Analyze multiple documents (MULTI_DOC_ANALYSIS)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class RoutingStrategy(Enum):
    """Strategy for handling a user query.

    DIRECT_SYNTHESIS: Answer directly from document summaries, skip retrieval.
                      Used for "summarize all", "overview", "main themes" queries.

    DEEP_DIVE: Retrieve chunks from 1-3 specific documents.
               Used for "what does doc X say about Y?" queries.

    MULTI_DOC_ANALYSIS: Retrieve from multiple/all documents.
                        Used for "compare", "cross-reference" queries.
    """
    DIRECT_SYNTHESIS = "direct_synthesis"
    DEEP_DIVE = "deep_dive"
    MULTI_DOC_ANALYSIS = "multi_doc_analysis"


@dataclass
class DocumentSummary:
    """Summary information for a single document."""
    source_id: str
    file_name: str
    dense_summary: str
    key_insights: List[str] = field(default_factory=list)
    reflection_questions: List[str] = field(default_factory=list)
    chunk_count: int = 0
    transformation_status: str = "pending"


@dataclass
class RoutingResult:
    """Result of the document routing decision.

    Attributes:
        strategy: The routing strategy to use
        selected_document_ids: Document IDs for DEEP_DIVE/MULTI_DOC (empty for DIRECT_SYNTHESIS)
        direct_response: Pre-generated response for DIRECT_SYNTHESIS (None otherwise)
        reasoning: Explanation of why this strategy was chosen
        confidence: Confidence score 0.0-1.0
        metadata: Additional routing metadata (e.g., query intent, complexity)
    """
    strategy: RoutingStrategy
    selected_document_ids: List[str] = field(default_factory=list)
    direct_response: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def requires_retrieval(self) -> bool:
        """Check if this routing result requires Stage 2 retrieval."""
        return self.strategy != RoutingStrategy.DIRECT_SYNTHESIS

    @property
    def is_multi_doc(self) -> bool:
        """Check if this routing targets multiple documents."""
        return self.strategy == RoutingStrategy.MULTI_DOC_ANALYSIS or len(self.selected_document_ids) > 1


class IDocumentRoutingService(ABC):
    """Interface for two-stage document routing service.

    The routing service analyzes user queries against document summaries
    to determine the optimal retrieval strategy before executing RAG.

    Stage 1: Analyze query + summaries → Routing decision
    Stage 2: Execute retrieval based on routing (if needed)
    """

    @abstractmethod
    def route_query(
        self,
        query: str,
        notebook_id: str,
        user_id: Optional[str] = None
    ) -> RoutingResult:
        """Analyze query and determine routing strategy.

        This is Stage 1 of the two-stage routing process. It analyzes the
        user query against all document summaries to determine the best
        approach for answering.

        Args:
            query: User query text
            notebook_id: Notebook UUID to route within
            user_id: Optional user UUID for access control

        Returns:
            RoutingResult with strategy, selected documents, and optional direct response

        Raises:
            ValueError: If notebook_id is invalid
            RuntimeError: If LLM call fails
        """
        pass

    @abstractmethod
    def get_notebook_summaries(
        self,
        notebook_id: str,
        active_only: bool = True
    ) -> List[DocumentSummary]:
        """Retrieve all document summaries for a notebook.

        Args:
            notebook_id: Notebook UUID
            active_only: If True, only return active documents (default)

        Returns:
            List of DocumentSummary objects with summary and insights

        Raises:
            ValueError: If notebook_id is invalid
        """
        pass

    @abstractmethod
    def synthesize_from_summaries(
        self,
        query: str,
        summaries: List[DocumentSummary]
    ) -> str:
        """Generate a synthesized response directly from document summaries.

        Used for DIRECT_SYNTHESIS strategy where retrieval is not needed.

        Args:
            query: User query text
            summaries: List of document summaries to synthesize from

        Returns:
            Synthesized response text

        Raises:
            RuntimeError: If LLM call fails
        """
        pass


class IRoutingPrompts(ABC):
    """Interface for routing prompt templates.

    Separates prompt logic from service implementation for easier testing
    and customization.
    """

    @abstractmethod
    def get_routing_prompt(
        self,
        query: str,
        summaries_text: str,
        document_count: int
    ) -> str:
        """Generate the Stage 1 routing prompt.

        Args:
            query: User query text
            summaries_text: Formatted document summaries
            document_count: Number of documents in notebook

        Returns:
            Complete prompt for LLM routing decision
        """
        pass

    @abstractmethod
    def get_synthesis_prompt(
        self,
        query: str,
        summaries_text: str
    ) -> str:
        """Generate the synthesis prompt for DIRECT_SYNTHESIS.

        Args:
            query: User query text
            summaries_text: Formatted document summaries

        Returns:
            Complete prompt for LLM synthesis
        """
        pass

    @abstractmethod
    def format_summaries_for_llm(
        self,
        summaries: List[DocumentSummary]
    ) -> str:
        """Format document summaries for LLM consumption.

        Args:
            summaries: List of DocumentSummary objects

        Returns:
            Formatted string with all summaries
        """
        pass
