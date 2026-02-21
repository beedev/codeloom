"""
Database module for CodeLoom.

Exports:
- DatabaseManager: Database connection and session management
- get_database_manager: Factory function for DatabaseManager
- wait_for_db: Database availability checker with retry logic
- Models: User, Project, CodeFile, CodeUnit, CodeEdge, Conversation, QueryLog
- Base: SQLAlchemy declarative base
"""

from .db import DatabaseManager, get_database_manager, wait_for_db
from .models import (
    Base,
    User,
    Project,
    CodeFile,
    CodeUnit,
    CodeEdge,
    MigrationPlan,
    MigrationPhase,
    ProjectAccess,
    Conversation,
    QueryLog,
    EmbeddingConfig,
    Role,
    UserRole,
    DeepAnalysisJob,
    DeepAnalysis,
    AnalysisUnit,
)

__all__ = [
    # Database management
    "DatabaseManager",
    "get_database_manager",
    "wait_for_db",

    # ORM models
    "Base",
    "User",
    "Project",
    "CodeFile",
    "CodeUnit",
    "CodeEdge",
    "MigrationPlan",
    "MigrationPhase",
    "ProjectAccess",
    "Conversation",
    "QueryLog",
    "EmbeddingConfig",
    "Role",
    "UserRole",
    "DeepAnalysisJob",
    "DeepAnalysis",
    "AnalysisUnit",
]
