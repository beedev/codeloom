"""Feedback service for collecting and aggregating RAG response quality signals.

User feedback (thumbs up/down, star ratings, category tags) is stored in the
rag_feedback table and optionally forwarded to Langfuse as trace scores so that
quality degradation is visible in the observability dashboard.

Ported from dbn-v2 with adaptations:
  - dbnotebook → codeloom imports
  - notebook_id → project_id (CodeLoom uses projects, not notebooks)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from .base import BaseService

logger = logging.getLogger(__name__)


class FeedbackService(BaseService):
    """Service for collecting user feedback on RAG chat responses.

    Stores feedback in the database and optionally forwards scores to Langfuse.
    """

    def submit_feedback(
        self,
        trace_id: str,
        query_id: str,
        user_id: str,
        project_id: str,
        rating: Optional[int] = None,
        helpful: Optional[bool] = None,
        user_message: Optional[str] = None,
        feedback_category: Optional[str] = None,
    ) -> str:
        """Store feedback in the database and forward score to Langfuse.

        Args:
            trace_id: Langfuse trace ID returned by the chat endpoint.
            query_id: Internal query UUID returned by the chat endpoint.
            user_id: UUID of the user submitting feedback.
            project_id: UUID of the project the query was made against.
            rating: Optional star rating 1-5.
            helpful: Optional boolean thumbs up/down signal.
            user_message: Optional free-text explanation.
            feedback_category: One of: inaccurate | irrelevant | incomplete | helpful | other.

        Returns:
            str: UUID of the created rag_feedback row.
        """
        self._validate_database_available()

        if rating is not None and not (1 <= rating <= 5):
            raise ValueError(f"rating must be between 1 and 5, got {rating}")

        if rating is None and helpful is None:
            raise ValueError("At least one of 'rating' or 'helpful' must be provided")

        from codeloom.core.db.models import RAGFeedback

        with self._db_manager.get_session() as session:
            existing = (
                session.query(RAGFeedback)
                .filter(RAGFeedback.trace_id == trace_id)
                .first()
            )
            if existing is not None:
                raise ValueError(
                    f"Feedback for trace_id='{trace_id}' already exists "
                    f"(feedback_id={existing.feedback_id})"
                )

            feedback_id = uuid4()
            record = RAGFeedback(
                feedback_id=feedback_id,
                trace_id=trace_id,
                query_id=query_id,
                user_id=user_id,
                project_id=project_id,
                rating=rating,
                helpful=helpful,
                user_message=user_message[:500] if user_message else None,
                feedback_category=feedback_category,
                created_at=datetime.utcnow(),
            )
            try:
                session.add(record)
                session.flush()
            except IntegrityError:
                session.rollback()
                raise ValueError(f"Feedback for trace_id='{trace_id}' already exists")

        self._logger.info(
            f"Feedback stored | feedback_id={feedback_id} | trace_id={trace_id} | "
            f"rating={rating} | helpful={helpful} | category={feedback_category}"
        )

        # Forward numeric score to Langfuse (non-blocking)
        if rating is not None and trace_id:
            try:
                from codeloom.core.observability import get_tracer
                tracer = get_tracer()
                normalized = (rating - 1) / 4.0  # Map 1-5 → 0.0-1.0
                tracer.log_score(
                    trace_id=trace_id,
                    name="user_rating",
                    value=normalized,
                    comment=f"Star rating: {rating}/5{' — ' + user_message[:100] if user_message else ''}",
                )
            except Exception as score_err:
                logger.debug(f"Langfuse score logging failed (non-fatal): {score_err}")

        if helpful is not None and trace_id:
            try:
                from codeloom.core.observability import get_tracer
                tracer = get_tracer()
                tracer.log_score(
                    trace_id=trace_id,
                    name="user_helpful",
                    value=1.0 if helpful else 0.0,
                    comment=f"Helpful: {helpful}",
                )
            except Exception as score_err:
                logger.debug(f"Langfuse helpful score logging failed (non-fatal): {score_err}")

        return str(feedback_id)

    def annotate_nodes(
        self,
        feedback_id: str,
        nodes: List[Dict],
        requesting_user_id: Optional[str] = None,
    ) -> bool:
        """Attach per-node relevance annotations to an existing feedback record."""
        self._validate_database_available()

        from codeloom.core.db.models import RAGFeedback, RAGFeedbackNode

        with self._db_manager.get_session() as session:
            feedback = (
                session.query(RAGFeedback)
                .filter(RAGFeedback.feedback_id == feedback_id)
                .first()
            )
            if feedback is None:
                raise ValueError(f"feedback_id='{feedback_id}' not found")

            if requesting_user_id and str(feedback.user_id) != str(requesting_user_id):
                raise PermissionError("Feedback record does not belong to the requesting user")

            for node_data in nodes:
                node_id = node_data.get("node_id")
                if not node_id:
                    continue
                annotation = RAGFeedbackNode(
                    feedback_id=feedback_id,
                    node_id=str(node_id),
                    node_rank=node_data.get("node_rank"),
                    was_relevant=node_data.get("was_relevant"),
                )
                session.add(annotation)

        self._logger.info(
            f"Node annotations stored | feedback_id={feedback_id} | count={len(nodes)}"
        )
        return True

    def get_feedback_stats(
        self,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 7,
    ) -> Dict:
        """Aggregate feedback statistics over a rolling time window."""
        self._validate_database_available()

        from codeloom.core.db.models import RAGFeedback

        cutoff = datetime.utcnow() - timedelta(days=days)

        with self._db_manager.get_session() as session:
            q = session.query(RAGFeedback).filter(RAGFeedback.created_at >= cutoff)
            if project_id:
                q = q.filter(RAGFeedback.project_id == project_id)
            if user_id:
                q = q.filter(RAGFeedback.user_id == user_id)
            records = q.all()

        total = len(records)
        if total == 0:
            return {
                "total_feedback": 0,
                "avg_rating": None,
                "helpful_ratio": None,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "category_breakdown": {},
                "period_days": days,
            }

        ratings = [r.rating for r in records if r.rating is not None]
        helpfuls = [r.helpful for r in records if r.helpful is not None]

        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
        helpful_ratio = (
            round(sum(1 for h in helpfuls if h) / len(helpfuls), 3) if helpfuls else None
        )

        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            if 1 <= r <= 5:
                distribution[r] += 1

        category_breakdown: Dict[str, int] = {}
        for record in records:
            cat = record.feedback_category or "unspecified"
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

        return {
            "total_feedback": total,
            "avg_rating": avg_rating,
            "helpful_ratio": helpful_ratio,
            "rating_distribution": distribution,
            "category_breakdown": category_breakdown,
            "period_days": days,
        }
