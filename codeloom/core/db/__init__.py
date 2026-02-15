"""
Database module for Notebook Architecture

Exports:
- DatabaseManager: Database connection and session management
- get_database_manager: Factory function for DatabaseManager
- wait_for_db: Database availability checker with retry logic
- Models: User, Notebook, NotebookSource, Conversation, QueryLog
- Base: SQLAlchemy declarative base
"""

from .db import DatabaseManager, get_database_manager, wait_for_db
from .models import Base, User, Notebook, NotebookSource, Conversation, QueryLog

__all__ = [
    # Database management
    "DatabaseManager",
    "get_database_manager",
    "wait_for_db",

    # ORM models
    "Base",
    "User",
    "Notebook",
    "NotebookSource",
    "Conversation",
    "QueryLog",
]
