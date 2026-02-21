"""Service for smart document suggestions based on query gaps.

This module provides intelligent document suggestions by analyzing query patterns,
document coverage, and knowledge gaps within a project.
"""

import logging
from typing import Optional

from .base import BaseService
from ..agents import DocumentAnalyzer

logger = logging.getLogger(__name__)


class SuggestionService(BaseService):
    """Service for generating smart document suggestions.

    Analyzes query patterns and document coverage to suggest:
    - Web search queries for missing knowledge areas
    - Document upload recommendations
    - Coverage improvements

    Integrates with DocumentAnalyzer agent for gap detection.
    """

    def __init__(self, pipeline=None, db_manager=None, project_manager=None):
        """Initialize suggestion service.

        Args:
            pipeline: LocalRAGPipeline instance for RAG operations
            db_manager: Optional DatabaseManager for persistence
            project_manager: Optional ProjectManager for project operations
        """
        super().__init__(pipeline, db_manager, project_manager)
        self._document_analyzer = DocumentAnalyzer(pipeline)

    def get_source_suggestions(
        self,
        project_id: str,
        query: str,
        query_history: Optional[list[str]] = None
    ) -> dict:
        """Get suggestions for documents to add based on query and history.

        Analyzes current project documents and query patterns to identify
        knowledge gaps and suggest relevant sources to add.

        Args:
            project_id: UUID of the project
            query: Current user query
            query_history: Optional list of previous queries

        Returns:
            Dictionary containing:
                - has_suggestions (bool): Whether suggestions are available
                - suggestions (list): List of suggestion dictionaries with:
                    - type: "web_search" or "upload"
                    - reason: Why this suggestion is made
                    - search_query: Suggested search query (for web_search type)
                    - priority: "high", "medium", or "low"
                - coverage_score (float): Document coverage score (0.0-1.0)

        Example:
            >>> suggestions = service.get_source_suggestions(
            ...     project_id="123",
            ...     query="What are the latest cloud trends?",
            ...     query_history=["AWS features", "Azure pricing"]
            ... )
            >>> if suggestions["has_suggestions"]:
            ...     for s in suggestions["suggestions"]:
            ...         print(f"{s['type']}: {s['reason']}")
        """
        try:
            self._log_operation(
                "get_source_suggestions",
                project_id=project_id,
                query_length=len(query),
                history_size=len(query_history) if query_history else 0
            )

            # Get current documents from project
            documents = self._get_project_documents(project_id)

            # Prepare context for analysis
            context = {
                "project_id": project_id,
                "documents": documents,
                "query_history": query_history or [query],
                "current_query": query
            }

            # Analyze coverage and gaps
            analysis = self._document_analyzer.analyze(context)

            # Generate suggestions from identified gaps
            suggestions = []
            for gap in analysis.get("gaps", []):
                suggestions.append({
                    "type": "web_search",
                    "reason": f"No documents covering: {gap}",
                    "search_query": gap,
                    "priority": self._calculate_priority(gap, analysis)
                })

            # Add general upload suggestion if coverage is very low
            coverage_score = analysis.get("coverage_score", 1.0)
            if coverage_score < 0.3 and len(documents) < 2:
                suggestions.append({
                    "type": "upload",
                    "reason": "Low document coverage - consider uploading relevant files",
                    "search_query": None,
                    "priority": "high"
                })

            result = {
                "has_suggestions": len(suggestions) > 0,
                "suggestions": suggestions[:5],  # Limit to top 5 suggestions
                "coverage_score": coverage_score
            }

            self._log_operation(
                "suggestions_generated",
                count=len(suggestions),
                coverage=coverage_score
            )

            return result

        except Exception as e:
            self._log_error("get_source_suggestions", e, project_id=project_id)
            # Return safe fallback
            return {
                "has_suggestions": False,
                "suggestions": [],
                "coverage_score": 1.0
            }

    def _get_project_documents(self, project_id: str) -> list[dict]:
        """Get documents for a project.

        Args:
            project_id: UUID of the project

        Returns:
            List of document metadata dictionaries
        """
        if not self.project_manager:
            logger.warning("ProjectManager not available, returning empty document list")
            return []

        try:
            docs = self.project_manager.get_code_files(project_id)
            return [
                {
                    "name": d.filename,
                    "active": d.active,
                    "source_type": d.source_type
                }
                for d in docs
            ]
        except Exception as e:
            self._log_error("_get_project_documents", e, project_id=project_id)
            return []

    def _calculate_priority(self, gap: str, analysis: dict) -> str:
        """Calculate priority level for a suggestion.

        Args:
            gap: Gap description
            analysis: Analysis results from DocumentAnalyzer

        Returns:
            Priority level: "high", "medium", or "low"
        """
        coverage = analysis.get("coverage_score", 1.0)

        # High priority if coverage is very low
        if coverage < 0.3:
            return "high"

        # Medium priority if coverage is moderate
        if coverage < 0.7:
            return "medium"

        # Low priority otherwise
        return "low"
