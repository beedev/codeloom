"""Parameter optimizer service — reads feedback insights and produces adaptive retrieval settings.

Reads non-expired rag_feedback_insights for a project, translates recommended actions
into concrete retrieval parameter deltas, and writes them to rag_adaptive_settings.
Pipeline.stateless_query_streaming() reads these settings at query time.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import uuid4

from .base import BaseService

logger = logging.getLogger(__name__)

# Hard limits to prevent out-of-range adaptive values
_MIN_TOP_K = 3
_MAX_TOP_K = 30
_DEFAULT_TOP_K = 10
_MIN_SIMILARITY = 0.3
_MAX_SIMILARITY = 0.9
_DEFAULT_SIMILARITY = 0.7
# How long adaptive settings rows stay valid before being re-derived
_SETTING_TTL_HOURS = 24


class ParameterOptimizerService(BaseService):
    """Translates feedback insights into adaptive retrieval parameters.

    Designed to be called after FeedbackAnalyzerService.run_full_analysis().
    Produces rows in rag_adaptive_settings which the pipeline reads at query time.
    """

    def apply_recommendations(
        self,
        project_id: str,
        intent: Optional[str] = None,
    ) -> bool:
        """Read insights for a project and write updated adaptive settings.

        Args:
            project_id: UUID of the project.
            intent: Optional query intent class to scope the settings.
                    When None, the settings apply to all intents.

        Returns:
            True if new settings were written, False if no actionable insights.
        """
        self._validate_database_available()

        actions = self._get_actionable_insights(project_id)
        if not actions:
            self._logger.debug(f"No actionable insights for project {project_id}")
            return False

        params = self._compute_params(actions)
        if not params:
            return False

        self._write_adaptive_settings(project_id, intent, params)
        self._logger.info(
            f"Adaptive settings written | project={project_id} | params={params}"
        )
        return True

    def get_adaptive_settings(
        self,
        project_id: str,
        intent: Optional[str] = None,
    ) -> Optional[Dict]:
        """Return the most recent non-expired adaptive settings for a project.

        Args:
            project_id: UUID of the project.
            intent: Optional intent class to look up. Falls back to intent=None row.

        Returns:
            Dict of parameter overrides (top_k, similarity_threshold, etc.)
            or None if no valid settings exist.
        """
        self._validate_database_available()

        from codeloom.core.db.models import RAGAdaptiveSetting

        now = datetime.utcnow()
        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            # Try exact intent match first, then fall back to generic (intent=None)
            for lookup_intent in ([intent, None] if intent else [None]):
                q = (
                    session.query(RAGAdaptiveSetting)
                    .filter(
                        RAGAdaptiveSetting.project_id == project_id,
                        RAGAdaptiveSetting.intent == lookup_intent,
                        (RAGAdaptiveSetting.expires_at > now)
                        | (RAGAdaptiveSetting.expires_at.is_(None)),
                    )
                    .order_by(RAGAdaptiveSetting.created_at.desc())
                    .first()
                )
                if q is not None:
                    result: Dict = {}
                    if q.top_k is not None:
                        result["top_k"] = q.top_k
                    if q.similarity_threshold is not None:
                        result["similarity_threshold"] = q.similarity_threshold
                    if q.bm25_weight is not None:
                        result["bm25_weight"] = q.bm25_weight
                    if q.vector_weight is not None:
                        result["vector_weight"] = q.vector_weight
                    if q.reranker_top_k is not None:
                        result["reranker_top_k"] = q.reranker_top_k
                    return result if result else None

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_actionable_insights(self, project_id: str) -> list:
        """Fetch non-expired, actionable insight rows for a project."""
        from codeloom.core.db.models import RAGFeedbackInsight

        now = datetime.utcnow()
        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            rows = (
                session.query(RAGFeedbackInsight)
                .filter(
                    RAGFeedbackInsight.project_id == project_id,
                    RAGFeedbackInsight.recommended_action != "none",
                    (RAGFeedbackInsight.expires_at > now)
                    | (RAGFeedbackInsight.expires_at.is_(None)),
                )
                .all()
            )
            session.expunge_all()
        return rows

    def _compute_params(self, actions: list) -> Dict:
        """Translate a list of recommended actions into concrete parameter deltas.

        Multiple insights may recommend different actions; all are applied and
        clamped to safe ranges.
        """
        top_k = _DEFAULT_TOP_K
        similarity = _DEFAULT_SIMILARITY

        for action_row in actions:
            action = action_row.recommended_action
            param_value = action_row.recommended_param_value
            confidence = action_row.confidence or 0.5

            if action == "increase_top_k":
                # Scale delta by confidence: +1..+5
                delta = max(1, min(5, round(confidence * 5)))
                top_k = min(_MAX_TOP_K, top_k + delta)

            elif action == "lower_similarity_threshold":
                new_sim = param_value if param_value is not None else (similarity - 0.1)
                similarity = max(_MIN_SIMILARITY, new_sim)

            elif action == "adjust_strategy":
                # Moderate increase in top_k and lower similarity for broader recall
                top_k = min(_MAX_TOP_K, top_k + 3)
                similarity = max(_MIN_SIMILARITY, similarity - 0.05)

        params: Dict = {}
        if top_k != _DEFAULT_TOP_K:
            params["top_k"] = top_k
        if abs(similarity - _DEFAULT_SIMILARITY) > 0.005:
            params["similarity_threshold"] = round(similarity, 3)

        return params

    def _write_adaptive_settings(
        self,
        project_id: str,
        intent: Optional[str],
        params: Dict,
    ) -> None:
        """Persist adaptive parameter settings, replacing existing rows for this project+intent."""
        from codeloom.core.db.models import RAGAdaptiveSetting

        expires_at = datetime.utcnow() + timedelta(hours=_SETTING_TTL_HOURS)

        with self._db_manager.get_session() as session:  # type: ignore[union-attr]
            # Replace existing rows for this project+intent combination
            session.query(RAGAdaptiveSetting).filter(
                RAGAdaptiveSetting.project_id == project_id,
                RAGAdaptiveSetting.intent == intent,
            ).delete(synchronize_session=False)

            row = RAGAdaptiveSetting(
                setting_id=uuid4(),
                project_id=project_id,
                intent=intent,
                top_k=params.get("top_k"),
                similarity_threshold=params.get("similarity_threshold"),
                bm25_weight=params.get("bm25_weight"),
                vector_weight=params.get("vector_weight"),
                reranker_top_k=params.get("reranker_top_k"),
                source="feedback_analysis",
                created_at=datetime.utcnow(),
                expires_at=expires_at,
            )
            session.add(row)
