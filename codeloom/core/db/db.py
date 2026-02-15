"""
Database Manager for Notebook Architecture

Provides connection pooling, session management, and database health checks
for PostgreSQL database using SQLAlchemy.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import logging
import time
from typing import Generator, Optional

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection manager with connection pooling and health checks.

    Features:
    - Connection pooling for efficient resource usage
    - Context manager for automatic session cleanup
    - Health check and retry logic
    - Database initialization
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """
        Initialize database manager with connection pool.

        Args:
            database_url: PostgreSQL connection string (postgresql://user:pass@host:port/db)
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_timeout: Timeout for getting connection from pool (seconds)
            pool_recycle: Recycle connections after this many seconds
            echo: Log all SQL statements (for debugging)
        """
        self.database_url = database_url
        self.echo = echo

        # Create engine with connection pooling
        logger.info(f"Creating database engine with pool_size={pool_size}, max_overflow={max_overflow}")
        self.engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
            echo=echo
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        logger.info("Database manager initialized successfully")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic cleanup.

        Usage:
            with db_manager.get_session() as session:
                user = session.query(User).first()

        Yields:
            Session: SQLAlchemy session

        Handles:
            - Automatic commit on success
            - Automatic rollback on exception
            - Session cleanup in all cases
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_session_maker(self) -> sessionmaker:
        """
        Get the session maker for manual session management.

        Returns:
            sessionmaker: SQLAlchemy session maker
        """
        return self.SessionLocal

    def init_db(self, drop_existing: bool = False):
        """
        Initialize database by creating all tables.

        Args:
            drop_existing: If True, drop all existing tables first (DANGEROUS!)
        """
        try:
            if drop_existing:
                logger.warning("Dropping all existing tables...")
                Base.metadata.drop_all(self.engine)

            logger.info("Creating database tables...")
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check database connectivity and health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with self.get_session() as session:
                # Simple query to check connectivity
                session.execute(text("SELECT 1"))
            logger.info("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_connection_info(self) -> dict:
        """
        Get information about the database connection pool.

        Returns:
            dict: Connection pool statistics
        """
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "timeout": pool._timeout,
        }

    def close(self):
        """
        Close all database connections and dispose of the engine.
        """
        logger.info("Closing database connections...")
        self.engine.dispose()
        logger.info("Database connections closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources"""
        self.close()


def get_database_manager(
    host: str = "localhost",
    port: int = 5432,
    database: str = "dbnotebook_dev",
    user: str = "postgres",
    password: str = "root",
    **kwargs
) -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        user: Database user
        password: Database password
        **kwargs: Additional arguments passed to DatabaseManager

    Returns:
        DatabaseManager: Configured database manager instance
    """
    database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return DatabaseManager(database_url, **kwargs)


def wait_for_db(
    db_manager: DatabaseManager,
    max_retries: int = 5,
    retry_delay: int = 2
) -> bool:
    """
    Wait for database to become available with exponential backoff.

    Args:
        db_manager: DatabaseManager instance
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)

    Returns:
        bool: True if database becomes available, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        logger.info(f"Database connection attempt {attempt}/{max_retries}...")

        if db_manager.health_check():
            logger.info("Database is ready")
            return True

        if attempt < max_retries:
            delay = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
            logger.warning(f"Database not ready. Retrying in {delay} seconds...")
            time.sleep(delay)

    logger.error(f"Failed to connect to database after {max_retries} attempts")
    return False
