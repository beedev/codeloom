"""UML diagram generation for MVP-scoped migration analysis.

Generates 7 diagram types per Functional MVP:
  Structural (ASG-driven): Class, Package, Component
  Behavioral (LLM-generated): Sequence, Use Case, Activity, Deployment

Public API:
  DiagramService — orchestrator for on-demand generation and caching
"""

from .service import DiagramService

__all__ = ["DiagramService"]
