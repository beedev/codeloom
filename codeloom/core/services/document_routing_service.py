"""Service for two-stage LLM document routing.

This module implements the two-stage document routing system that intelligently
decides how to handle user queries:

Stage 1: Analyze query against document summaries to determine routing strategy
Stage 2: Execute retrieval based on routing decision (if needed)

Strategies:
- DIRECT_SYNTHESIS: Answer from summaries, skip retrieval
- DEEP_DIVE: Retrieve from 1-3 specific documents
- MULTI_DOC_ANALYSIS: Retrieve from multiple/all documents
"""

import json
import logging
import re
from typing import List, Optional
from uuid import UUID

from .base import BaseService
from ..interfaces.routing import (
    IDocumentRoutingService,
    RoutingStrategy,
    RoutingResult,
    DocumentSummary,
)
from ..prompt.routing_prompts import (
    get_routing_prompt,
    get_synthesis_prompt,
    format_summaries,
)
from ..db.models import CodeFile

logger = logging.getLogger(__name__)


class DocumentRoutingService(BaseService, IDocumentRoutingService):
    """Two-stage LLM document routing service.

    Analyzes user queries against document summaries to determine the optimal
    retrieval strategy before executing RAG.

    This service implements intelligent routing that:
    - Uses pre-generated document summaries for quick analysis
    - Avoids unnecessary retrieval for summary-type queries
    - Focuses retrieval on relevant documents for specific queries
    - Enables multi-document analysis for comparison queries
    """

    def __init__(self, pipeline=None, db_manager=None, project_manager=None):
        """Initialize document routing service.

        Args:
            pipeline: LocalRAGPipeline instance for LLM operations
            db_manager: DatabaseManager for accessing document summaries
            project_manager: ProjectManager for project operations
        """
        super().__init__(pipeline, db_manager, project_manager)
        self._logger = logging.getLogger(__name__)

    def get_project_summaries(
        self,
        project_id: str,
        active_only: bool = True
    ) -> List[DocumentSummary]:
        """Retrieve all document summaries for a project.

        Args:
            project_id: Project UUID
            active_only: If True, only return active documents

        Returns:
            List of DocumentSummary objects

        Raises:
            ValueError: If project_id is invalid
        """
        self._validate_database_available()

        try:
            with self._db_manager.get_session() as session:
                query = session.query(CodeFile).filter(
                    CodeFile.project_id == UUID(project_id)
                )

                if active_only:
                    query = query.filter(CodeFile.active == True)

                sources = query.all()

                summaries = []
                for source in sources:
                    summaries.append(DocumentSummary(
                        source_id=str(source.source_id),
                        file_name=source.file_name,
                        dense_summary=source.dense_summary or "",
                        key_insights=source.key_insights or [],
                        reflection_questions=source.reflection_questions or [],
                        chunk_count=source.chunk_count or 0,
                        transformation_status=source.transformation_status or "pending"
                    ))

                self._log_operation(
                    "get_project_summaries",
                    project_id=project_id,
                    document_count=len(summaries)
                )

                return summaries

        except Exception as e:
            self._log_error("get_project_summaries", e, project_id=project_id)
            raise

    def route_query(
        self,
        query: str,
        project_id: str,
        user_id: Optional[str] = None
    ) -> RoutingResult:
        """Analyze query and determine routing strategy (Stage 1).

        This is the core routing method that:
        1. Fetches all document summaries for the project
        2. Sends summaries + query to LLM for routing decision
        3. Returns routing result with strategy and selected documents

        Args:
            query: User query text
            project_id: Project UUID to route within
            user_id: Optional user UUID for access control

        Returns:
            RoutingResult with strategy, selected documents, and optional response

        Raises:
            ValueError: If project_id is invalid
            RuntimeError: If LLM call fails
        """
        self._log_operation(
            "route_query",
            query_length=len(query),
            project_id=project_id
        )

        try:
            # Get document summaries
            summaries = self.get_project_summaries(project_id)

            # Handle empty project
            if not summaries:
                self._logger.info(f"No documents in project {project_id}, returning empty result")
                return RoutingResult(
                    strategy=RoutingStrategy.DIRECT_SYNTHESIS,
                    selected_document_ids=[],
                    direct_response="This project doesn't have any documents yet. Please upload some documents to get started.",
                    reasoning="Empty project - no documents to query",
                    confidence=1.0,
                    metadata={"empty_project": True}
                )

            # Check if any documents have summaries
            docs_with_summaries = [s for s in summaries if s.dense_summary]
            if not docs_with_summaries:
                self._logger.info(f"No summaries available, falling back to MULTI_DOC_ANALYSIS")
                return RoutingResult(
                    strategy=RoutingStrategy.MULTI_DOC_ANALYSIS,
                    selected_document_ids=[s.source_id for s in summaries],
                    reasoning="No document summaries available - using all documents",
                    confidence=0.5,
                    metadata={"no_summaries": True}
                )

            # Format summaries for LLM
            formatted_summaries = format_summaries(summaries)

            # Build routing prompt
            routing_prompt = get_routing_prompt(
                query=query,
                summaries_text=formatted_summaries,
                document_count=len(summaries)
            )

            # Call LLM for routing decision
            llm_response = self._call_llm(routing_prompt)

            # Parse routing response
            result = self._parse_routing_response(llm_response, summaries)

            # If DIRECT_SYNTHESIS, generate synthesized response
            if result.strategy == RoutingStrategy.DIRECT_SYNTHESIS:
                result.direct_response = self.synthesize_from_summaries(query, summaries)

            self._log_operation(
                "route_query_complete",
                strategy=result.strategy.value,
                selected_count=len(result.selected_document_ids),
                confidence=result.confidence
            )

            return result

        except Exception as e:
            self._log_error("route_query", e, project_id=project_id)
            # Fallback to MULTI_DOC_ANALYSIS on error
            return RoutingResult(
                strategy=RoutingStrategy.MULTI_DOC_ANALYSIS,
                selected_document_ids=[s.source_id for s in summaries] if summaries else [],
                reasoning=f"Routing failed: {str(e)} - falling back to full retrieval",
                confidence=0.3,
                metadata={"error": str(e)}
            )

    def synthesize_from_summaries(
        self,
        query: str,
        summaries: List[DocumentSummary]
    ) -> str:
        """Generate synthesized response from document summaries.

        Used for DIRECT_SYNTHESIS strategy where retrieval is not needed.

        Args:
            query: User query text
            summaries: List of document summaries to synthesize from

        Returns:
            Synthesized response text

        Raises:
            RuntimeError: If LLM call fails
        """
        self._log_operation(
            "synthesize_from_summaries",
            query_length=len(query),
            summary_count=len(summaries)
        )

        try:
            # Format summaries for synthesis
            formatted_summaries = format_summaries(summaries)

            # Build synthesis prompt
            synthesis_prompt = get_synthesis_prompt(
                query=query,
                summaries_text=formatted_summaries
            )

            # Call LLM for synthesis
            response = self._call_llm(synthesis_prompt)

            self._logger.info(f"Generated synthesis response: {len(response)} chars")
            return response

        except Exception as e:
            self._log_error("synthesize_from_summaries", e)
            raise RuntimeError(f"Synthesis failed: {str(e)}") from e

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call LLM with the given prompt.

        Uses the pipeline's default model for inference.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        if not self._pipeline or not self._pipeline._default_model:
            raise RuntimeError("LLM not available - pipeline not initialized")

        try:
            # Use the pipeline's LLM directly
            response = self._pipeline._default_model.complete(prompt)
            return response.text.strip()
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM call failed: {str(e)}") from e

    def _parse_routing_response(
        self,
        response: str,
        summaries: List[DocumentSummary]
    ) -> RoutingResult:
        """Parse LLM routing response into RoutingResult.

        Args:
            response: LLM response text (expected to be JSON)
            summaries: List of document summaries for validation

        Returns:
            RoutingResult with parsed strategy and documents
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                self._logger.warning(f"No JSON found in routing response: {response[:200]}")
                return self._fallback_routing_result(summaries)

            json_str = json_match.group()
            data = json.loads(json_str)

            # Parse strategy
            strategy_str = data.get("strategy", "MULTI_DOC_ANALYSIS").upper()
            strategy_map = {
                "DIRECT_SYNTHESIS": RoutingStrategy.DIRECT_SYNTHESIS,
                "DEEP_DIVE": RoutingStrategy.DEEP_DIVE,
                "MULTI_DOC_ANALYSIS": RoutingStrategy.MULTI_DOC_ANALYSIS,
            }
            strategy = strategy_map.get(strategy_str, RoutingStrategy.MULTI_DOC_ANALYSIS)

            # Parse selected documents
            selected_ids = data.get("selected_documents", [])

            # Validate document IDs exist
            valid_ids = {s.source_id for s in summaries}
            validated_ids = [id for id in selected_ids if id in valid_ids]

            # If DEEP_DIVE or MULTI_DOC but no valid documents, adjust strategy
            if strategy in (RoutingStrategy.DEEP_DIVE, RoutingStrategy.MULTI_DOC_ANALYSIS):
                if not validated_ids:
                    # Use all documents if none were validly selected
                    validated_ids = [s.source_id for s in summaries]

            return RoutingResult(
                strategy=strategy,
                selected_document_ids=validated_ids,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
                metadata={
                    "raw_response": response[:500],
                    "original_selected": selected_ids
                }
            )

        except json.JSONDecodeError as e:
            self._logger.warning(f"Failed to parse routing JSON: {e}")
            return self._fallback_routing_result(summaries)
        except Exception as e:
            self._logger.warning(f"Error parsing routing response: {e}")
            return self._fallback_routing_result(summaries)

    def _fallback_routing_result(
        self,
        summaries: List[DocumentSummary]
    ) -> RoutingResult:
        """Create fallback routing result when parsing fails.

        Falls back to MULTI_DOC_ANALYSIS with all documents.

        Args:
            summaries: List of document summaries

        Returns:
            Fallback RoutingResult
        """
        return RoutingResult(
            strategy=RoutingStrategy.MULTI_DOC_ANALYSIS,
            selected_document_ids=[s.source_id for s in summaries],
            reasoning="Fallback to full retrieval due to routing parse error",
            confidence=0.5,
            metadata={"fallback": True}
        )
