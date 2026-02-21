"""Deep Understanding Engine — trace execution paths and extract structured knowledge.

Public API:
    UnderstandingEngine  — orchestrator (start_analysis, get_results, etc.)
    ChainTracer          — entry point detection + call tree tracing
    ChainAnalyzer        — tiered LLM analysis
    UnderstandingWorker  — background job processor
"""

from .engine import UnderstandingEngine
from .chain_tracer import ChainTracer
from .analyzer import ChainAnalyzer
from .worker import UnderstandingWorker

__all__ = [
    "UnderstandingEngine",
    "ChainTracer",
    "ChainAnalyzer",
    "UnderstandingWorker",
]
