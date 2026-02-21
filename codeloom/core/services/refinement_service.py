"""Service for query refinement and improvement.

This module provides query analysis and refinement suggestions to help users
craft better queries for improved retrieval results.
"""

import logging
from typing import Optional

from .base import BaseService
from ..agents import QueryAnalyzer

logger = logging.getLogger(__name__)


class RefinementService(BaseService):
    """Service for query refinement and improvement.

    Analyzes user queries to:
    - Classify intent
    - Assess complexity
    - Suggest refinements for better results
    - Generate alternative query formulations

    Integrates with QueryAnalyzer agent for intelligent query understanding.
    """

    def __init__(self, pipeline=None, db_manager=None, project_manager=None):
        """Initialize refinement service.

        Args:
            pipeline: LocalRAGPipeline instance for RAG operations
            db_manager: Optional DatabaseManager for persistence
            project_manager: Optional ProjectManager for project operations
        """
        super().__init__(pipeline, db_manager, project_manager)
        self._query_analyzer = QueryAnalyzer(pipeline)

    def analyze_query(self, query: str) -> dict:
        """Analyze query and suggest refinements.

        Examines query structure, intent, and clarity to determine if
        refinements would improve retrieval quality.

        Args:
            query: User query text

        Returns:
            Dictionary containing:
                - intent (str): Classified intent category
                - confidence (float): Confidence in classification (0.0-1.0)
                - needs_refinement (bool): Whether query should be refined
                - refinements (list[str]): List of refinement suggestions
                - reason (str|None): Why refinement is suggested
                - complexity (float): Query complexity score (0.0-1.0)

        Example:
            >>> analysis = service.analyze_query("Tell me about that thing")
            >>> if analysis["needs_refinement"]:
            ...     print(f"Reason: {analysis['reason']}")
            ...     for refinement in analysis["refinements"]:
            ...         print(f"- {refinement}")
        """
        try:
            self._log_operation("analyze_query", query_length=len(query))

            # Get analysis from QueryAnalyzer agent
            analysis = self._query_analyzer.analyze(query)

            # Determine if refinement is needed
            confidence = analysis.get("confidence", 1.0)
            complexity = analysis.get("complexity", 0.0)
            needs_refinement = confidence < 0.7 or complexity > 0.8

            # Determine reason for refinement
            reason = None
            if confidence < 0.7:
                reason = "Query may be too vague or ambiguous"
            elif complexity > 0.8:
                reason = "Query is too complex - consider breaking into simpler questions"

            result = {
                "intent": analysis.get("intent", "unknown"),
                "confidence": confidence,
                "needs_refinement": needs_refinement,
                "refinements": analysis.get("suggested_refinements", []),
                "reason": reason,
                "complexity": complexity
            }

            self._log_operation(
                "query_analyzed",
                intent=result["intent"],
                confidence=confidence,
                needs_refinement=needs_refinement
            )

            return result

        except Exception as e:
            self._log_error("analyze_query", e, query_length=len(query))
            # Return safe fallback
            return {
                "intent": "unknown",
                "confidence": 0.5,
                "needs_refinement": False,
                "refinements": [],
                "reason": None,
                "complexity": 0.5
            }

    def get_refined_queries(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> list[str]:
        """Get list of refined query suggestions.

        Generates alternative formulations of the query that may yield
        better retrieval results.

        Args:
            query: Original query text
            context: Optional context dictionary with:
                - history: Conversation history
                - project_context: Project metadata
                - user_preferences: User preferences

        Returns:
            List of refined query strings

        Example:
            >>> refined = service.get_refined_queries(
            ...     "how does it work",
            ...     context={"history": ["cloud storage", "AWS S3"]}
            ... )
            >>> for q in refined:
            ...     print(f"Alternative: {q}")
        """
        try:
            self._log_operation("get_refined_queries", query_length=len(query))

            # Prepare context for analyzer
            analyzer_context = {
                "query": query,
                **(context or {})
            }

            # Get suggestions from QueryAnalyzer
            suggestions = self._query_analyzer.suggest(analyzer_context)

            # Extract refined queries
            refined_queries = []
            for suggestion in suggestions:
                if "text" in suggestion:
                    refined_queries.append(suggestion["text"])

            # If no suggestions, try to generate basic refinements
            if not refined_queries:
                refined_queries = self._generate_basic_refinements(query)

            result = refined_queries[:3]  # Limit to top 3 refinements

            self._log_operation("refinements_generated", count=len(result))

            return result

        except Exception as e:
            self._log_error("get_refined_queries", e, query_length=len(query))
            # Return safe fallback - original query
            return [query]

    def _generate_basic_refinements(self, query: str) -> list[str]:
        """Generate basic query refinements when analyzer doesn't produce suggestions.

        Args:
            query: Original query

        Returns:
            List of basic refinement suggestions
        """
        refinements = []

        # Add specificity prompt for short queries
        if len(query.split()) < 5:
            refinements.append(f"{query} - please provide specific details")

        # Add context prompt for question queries
        if any(w in query.lower() for w in ["what", "how", "why", "when", "where"]):
            refinements.append(f"{query} in this context")

        # Add comparison variant for descriptive queries
        if len(query.split()) >= 3 and "?" not in query:
            refinements.append(f"What are the key aspects of {query}?")

        return refinements[:3]

    def suggest_follow_ups(
        self,
        query: str,
        response: str,
        history: Optional[list[dict]] = None
    ) -> list[str]:
        """Suggest follow-up questions based on query and response.

        Args:
            query: Original query
            response: Assistant response
            history: Optional conversation history

        Returns:
            List of suggested follow-up question strings
        """
        try:
            self._log_operation("suggest_follow_ups", query_length=len(query))

            # Get query intent
            analysis = self._query_analyzer.analyze(query)
            intent = analysis.get("intent", "unknown")

            follow_ups = []

            # Generate intent-specific follow-ups
            if intent == "factual":
                follow_ups.append(f"Can you provide more details about {query}?")
                follow_ups.append(f"What are examples related to {query}?")
            elif intent == "comparison":
                follow_ups.append("Which option is recommended for my use case?")
                follow_ups.append("What are the trade-offs between these options?")
            elif intent == "action":
                follow_ups.append("What are the steps to accomplish this?")
                follow_ups.append("What are potential challenges I might face?")
            elif intent == "clarification":
                follow_ups.append("Can you explain this in simpler terms?")
                follow_ups.append("What are real-world examples?")

            result = follow_ups[:3]  # Limit to top 3

            self._log_operation("follow_ups_generated", count=len(result))

            return result

        except Exception as e:
            self._log_error("suggest_follow_ups", e, query_length=len(query))
            return []
