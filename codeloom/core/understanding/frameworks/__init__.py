"""Framework detection and analysis registry."""

import logging
from typing import Any, Dict, List

from ...db import DatabaseManager
from .base import FrameworkContext
from .spring import SpringAnalyzer
from .aspnet import AspNetAnalyzer

logger = logging.getLogger(__name__)

# Registry of framework analyzers (order = detection priority)
_ANALYZERS = [
    SpringAnalyzer,
    AspNetAnalyzer,
]


def detect_and_analyze(
    db: DatabaseManager,
    project_id: str,
) -> List[Dict[str, Any]]:
    """Detect frameworks and return analysis contexts.

    Runs each registered analyzer's detect() method. For detected
    frameworks, runs analyze() and returns the contexts.

    Args:
        db: DatabaseManager instance
        project_id: UUID string

    Returns:
        List of serialized FrameworkContext dicts
    """
    contexts = []

    for analyzer_cls in _ANALYZERS:
        try:
            analyzer = analyzer_cls(db)
            if analyzer.detect(project_id):
                ctx = analyzer.analyze(project_id)
                contexts.append({
                    "framework_name": ctx.framework_name,
                    "framework_type": ctx.framework_type,
                    "version": ctx.version,
                    "di_registrations": ctx.di_registrations[:20],
                    "middleware_pipeline": ctx.middleware_pipeline[:10],
                    "security_config": ctx.security_config,
                    "transaction_boundaries": ctx.transaction_boundaries[:20],
                    "aop_pointcuts": ctx.aop_pointcuts[:10],
                    "analysis_hints": ctx.analysis_hints,
                })
                logger.info(f"Detected framework: {ctx.framework_name}")
        except Exception as e:
            logger.warning(f"Framework detection failed for {analyzer_cls.__name__}: {e}")

    return contexts
