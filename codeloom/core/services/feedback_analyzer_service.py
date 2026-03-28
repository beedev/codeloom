"""Feedback analyzer service — derives RAG parameter recommendations from user feedback.

Reads rag_feedback records for a project, groups them by query intent and
retrieval quality signals, and writes actionable insights to rag_feedback_insights.
These insights are later consumed by ParameterOptimizerService to tune retrieval.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseService

logger = logging.getLogger(__name__)


class FeedbackAnalyzerService(BaseService):
    """Analyses rag_feedback records and writes insights to rag_feedback_insights.

    Designed to be called periodically by FeedbackAnalyzerWorker. All methods
    are synchronous (no asyncio) and wrap database access in the existing
    SQLAlchemy session context.
    """

    # Minimum feedback records needed before producing recommendations
    MIN_SAMPLE_SIZE: int = 5
    # Insight TTL — insights expire after 24 hours, triggering a fresh analysis
    INSIGHT_TTL_HOURS: int = 24

    def run_full_analysis(
        self,
        project_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, int]:
        """Run a full analysis pass and persist insights.

        Args:
            project_id: Optional UUID to scope analysis to one project.
                         When None, all projects with sufficient feedback are analysed.
            days: Rolling window of feedback records to consider.

        Returns:
            Dict mapping project_id -> number of insights written.
        """
        self._validate_database_available()

        from codeloom.core.db.models import RAGFeedback
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            q = session.query(RAGFeedback.project_id).filter(
                RAGFeedback.created_at >= cutoff
            )
            if project_id:
                q = q.filter(RAGFeedback.project_id == project_id)

            # Collect distinct project_ids that have enough feedback
            from sqlalchemy import func
            count_q = (
                session.query(
                    RAGFeedback.project_id,
                    func.count(RAGFeedback.feedback_id).label("cnt"),
                )
                .filter(RAGFeedback.created_at >= cutoff)
            )
            if project_id:
                count_q = count_q.filter(RAGFeedback.project_id == project_id)
            count_q = count_q.group_by(RAGFeedback.project_id)
            project_counts = {str(row.project_id): row.cnt for row in count_q.all()}

        summary: Dict[str, int] = {}
        for proj_id, cnt in project_counts.items():
            if cnt < self.MIN_SAMPLE_SIZE:
                self._logger.debug(
                    f"Skipping project {proj_id} — only {cnt} feedback records "
                    f"(min={self.MIN_SAMPLE_SIZE})"
                )
                continue
            written = self._analyse_project(proj_id, days=days)
            summary[proj_id] = written
            self._logger.info(
                f"Feedback analysis complete | project={proj_id} | insights_written={written}"
            )

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyse_project(self, project_id: str, days: int) -> int:
        """Analyse feedback for a single project and persist insights.

        Returns:
            Number of insight rows written.
        """
        records = self._fetch_feedback(project_id, days)
        insights: List[Dict] = []

        # 1. Overall quality
        overall = self._overall_quality(records)
        if overall:
            insights.append(overall)

        # 2. Helpfulness analysis (no-context: only helpful flag)
        helpful_insight = self._helpfulness_analysis(records)
        if helpful_insight:
            insights.append(helpful_insight)

        # 3. Category breakdown — surface dominant problems
        cat_insights = self._category_breakdown(records)
        insights.extend(cat_insights)

        # 4. Retrieval quality hints from low ratings
        retrieval_insight = self._retrieval_quality(records)
        if retrieval_insight:
            insights.append(retrieval_insight)

        if not insights:
            return 0

        return self._persist_insights(project_id, insights)

    def _fetch_feedback(self, project_id: str, days: int) -> List:
        """Return RAGFeedback ORM objects for the given project and time window."""
        from codeloom.core.db.models import RAGFeedback
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            records = (
                session.query(RAGFeedback)
                .filter(
                    RAGFeedback.project_id == project_id,
                    RAGFeedback.created_at >= cutoff,
                )
                .all()
            )
            # Detach from session before returning
            session.expunge_all()
        return records

    def _overall_quality(self, records: List) -> Optional[Dict]:
        """Compute overall avg_rating and helpful_ratio insight."""
        ratings = [r.rating for r in records if r.rating is not None]
        helpfuls = [r.helpful for r in records if r.helpful is not None]

        if not ratings and not helpfuls:
            return None

        avg_rating = round(sum(ratings) / len(ratings), 3) if ratings else None
        helpful_ratio = (
            round(sum(1 for h in helpfuls if h) / len(helpfuls), 3)
            if helpfuls else None
        )

        # Choose the representative metric_value
        metric_value = avg_rating if avg_rating is not None else (
            round(helpful_ratio * 5, 3) if helpful_ratio is not None else None
        )

        return {
            "analysis_type": "overall",
            "metric_name": "overall_quality",
            "dimension_value": "all",
            "metric_value": metric_value,
            "sample_count": len(records),
            "confidence": min(1.0, len(records) / 50),
            "recommended_action": self._quality_action(avg_rating, helpful_ratio),
            "recommended_param_value": None,
        }

    def _helpfulness_analysis(self, records: List) -> Optional[Dict]:
        """Insight when helpful_ratio falls below a threshold."""
        helpfuls = [r.helpful for r in records if r.helpful is not None]
        if len(helpfuls) < self.MIN_SAMPLE_SIZE:
            return None

        helpful_ratio = sum(1 for h in helpfuls if h) / len(helpfuls)
        if helpful_ratio >= 0.7:
            return None  # Good enough — no action needed

        return {
            "analysis_type": "retrieval_quality",
            "metric_name": "helpful_ratio",
            "dimension_value": "low_helpfulness",
            "metric_value": round(helpful_ratio, 3),
            "sample_count": len(helpfuls),
            "confidence": min(1.0, len(helpfuls) / 30),
            "recommended_action": "increase_top_k",
            "recommended_param_value": None,
        }

    def _category_breakdown(self, records: List) -> List[Dict]:
        """Produce one insight per category that appears frequently (>20% of records)."""
        total = len(records)
        if total == 0:
            return []

        category_counts: Dict[str, int] = {}
        for r in records:
            cat = r.feedback_category or "unspecified"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        insights = []
        for cat, cnt in category_counts.items():
            fraction = cnt / total
            if fraction < 0.2:
                continue
            action = self._category_action(cat, fraction)
            insights.append({
                "analysis_type": "intent_breakdown",
                "metric_name": "category_fraction",
                "dimension_value": cat,
                "metric_value": round(fraction, 3),
                "sample_count": cnt,
                "confidence": min(1.0, cnt / 20),
                "recommended_action": action,
                "recommended_param_value": None,
            })
        return insights

    def _retrieval_quality(self, records: List) -> Optional[Dict]:
        """Suggest lower similarity threshold when many low-star ratings occur."""
        rated = [r for r in records if r.rating is not None]
        if len(rated) < self.MIN_SAMPLE_SIZE:
            return None

        low_fraction = sum(1 for r in rated if r.rating <= 2) / len(rated)
        if low_fraction < 0.3:
            return None  # Less than 30% low ratings — no action

        # Recommend lower similarity threshold to surface more candidates
        return {
            "analysis_type": "retrieval_quality",
            "metric_name": "low_rating_fraction",
            "dimension_value": "rating_leq_2",
            "metric_value": round(low_fraction, 3),
            "sample_count": len(rated),
            "confidence": min(1.0, len(rated) / 30),
            "recommended_action": "lower_similarity_threshold",
            "recommended_param_value": 0.55,
        }

    def _persist_insights(self, project_id: str, insights: List[Dict]) -> int:
        """Write insight rows to rag_feedback_insights.

        Existing non-expired insights for the project are deleted first so that
        fresh analysis always wins.
        """
        from codeloom.core.db.models import RAGFeedbackInsight

        expires_at = datetime.utcnow() + timedelta(hours=self.INSIGHT_TTL_HOURS)

        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            # Clear old insights for this project
            session.query(RAGFeedbackInsight).filter(
                RAGFeedbackInsight.project_id == project_id
            ).delete(synchronize_session=False)

            for data in insights:
                row = RAGFeedbackInsight(
                    insight_id=uuid4(),
                    project_id=project_id,
                    analysis_type=data.get("analysis_type"),
                    metric_name=data.get("metric_name"),
                    dimension_value=data.get("dimension_value"),
                    metric_value=data.get("metric_value"),
                    sample_count=data.get("sample_count"),
                    confidence=data.get("confidence"),
                    recommended_action=data.get("recommended_action"),
                    recommended_param_value=data.get("recommended_param_value"),
                    created_at=datetime.utcnow(),
                    expires_at=expires_at,
                )
                session.add(row)

        return len(insights)

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_action(
        avg_rating: Optional[float],
        helpful_ratio: Optional[float],
    ) -> str:
        if avg_rating is not None and avg_rating < 2.5:
            return "adjust_strategy"
        if helpful_ratio is not None and helpful_ratio < 0.5:
            return "increase_top_k"
        return "none"

    @staticmethod
    def _category_action(category: str, fraction: float) -> str:
        mapping: Dict[str, str] = {
            "inaccurate": "adjust_strategy",
            "irrelevant": "lower_similarity_threshold",
            "incomplete": "increase_top_k",
            "helpful": "none",
            "other": "none",
        }
        return mapping.get(category, "none")
