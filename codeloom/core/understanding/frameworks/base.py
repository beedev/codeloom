"""Base classes for framework-specific analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...db import DatabaseManager


@dataclass
class FrameworkContext:
    """Framework-specific context to inject into analysis prompts."""
    framework_name: str           # e.g. "Spring Boot 2.7", "ASP.NET Core 6"
    framework_type: str           # "spring", "aspnet", "django", "express"
    version: Optional[str] = None

    # Discovered configuration
    di_registrations: List[str] = field(default_factory=list)
    middleware_pipeline: List[str] = field(default_factory=list)
    security_config: Dict[str, Any] = field(default_factory=dict)
    transaction_boundaries: List[str] = field(default_factory=list)
    aop_pointcuts: List[str] = field(default_factory=list)

    # Hints for the LLM
    analysis_hints: List[str] = field(default_factory=list)


class FrameworkAnalyzer(ABC):
    """Abstract base for framework-specific analyzers."""

    def __init__(self, db: DatabaseManager):
        self._db = db

    @abstractmethod
    def detect(self, project_id: str) -> bool:
        """Return True if this framework is detected in the project."""
        ...

    @abstractmethod
    def analyze(self, project_id: str) -> FrameworkContext:
        """Analyze framework-specific patterns and return context."""
        ...
