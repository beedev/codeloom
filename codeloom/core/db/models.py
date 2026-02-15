"""
SQLAlchemy ORM Models for Notebook Architecture

This module defines the database models for the NotebookLM-style document chatbot:
- Users: User accounts (multi-user ready, start with default user)
- Notebooks: Document collections with isolated embeddings
- NotebookSources: Files uploaded to notebooks with metadata
- Conversations: Persistent conversation history per notebook
- QueryLogs: Query logging for observability and cost tracking
- AnalyticsSessions: Analytics dashboard sessions with uploaded Excel data
- DatabaseConnection: External database connections for Chat with Data
- SQLChatSession: Chat sessions for SQL queries
- SQLQueryHistory: History of executed SQL queries
- SQLFewShotExample: Few-shot examples from Gretel dataset
- SQLQueryTelemetry: Telemetry data for SQL query observability

RBAC (Role-Based Access Control):
- Role: Role definitions (admin, user, viewer) with permissions
- UserRole: Maps users to roles (many-to-many)
- NotebookAccess: Grants users access to specific notebooks
- SQLConnectionAccess: Grants users access to specific SQL connections
"""

from sqlalchemy import Column, String, Integer, Text, TIMESTAMP, ForeignKey, BigInteger, Index, TypeDecorator, Boolean
from sqlalchemy.orm import relationship, declarative_base, backref
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID, JSONB
import uuid
from datetime import datetime

Base = declarative_base()


# UUID type that works with both PostgreSQL and SQLite
class UUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's UUID type when available, otherwise stores as String(36).
    """
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgreSQL_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return value
            else:
                return uuid.UUID(value)


class User(Base):
    """User model for multi-user support (start with single default user)"""
    __tablename__ = "users"

    user_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True)
    password_hash = Column(String(255), nullable=True)  # bcrypt hashed password
    api_key = Column(String(255), nullable=True)  # Per-user API key for programmatic access
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    last_active = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    notebooks = relationship("Notebook", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user", cascade="all, delete-orphan")
    analytics_sessions = relationship("AnalyticsSession", back_populates="user", cascade="all, delete-orphan")
    user_roles = relationship("UserRole", foreign_keys="[UserRole.user_id]", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


class Notebook(Base):
    """Notebook model - collection of documents with isolated embeddings"""
    __tablename__ = "notebooks"
    __table_args__ = (
        Index('idx_user_notebooks', 'user_id', 'created_at'),
    )

    notebook_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    document_count = Column(Integer, default=0, nullable=False)

    # Relationships
    user = relationship("User", back_populates="notebooks")
    sources = relationship("NotebookSource", back_populates="notebook", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="notebook", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="notebook", cascade="all, delete-orphan")
    analytics_sessions = relationship("AnalyticsSession", back_populates="notebook")

    def __repr__(self):
        return f"<Notebook(notebook_id={self.notebook_id}, name='{self.name}', docs={self.document_count})>"


class NotebookSource(Base):
    """Document sources (files) uploaded to notebooks"""
    __tablename__ = "notebook_sources"
    __table_args__ = (
        Index('idx_notebook_sources', 'notebook_id', 'upload_timestamp'),
        Index('idx_source_hash', 'notebook_id', 'file_hash', unique=True),
    )

    source_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA256 for duplicate detection
    file_size = Column(BigInteger)  # File size in bytes
    file_type = Column(String(50))  # PDF, DOCX, TXT, etc.
    chunk_count = Column(Integer)  # Number of chunks/nodes created
    upload_timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    active = Column(Boolean, default=True, nullable=False)  # Toggle for including in RAG retrieval

    # AI Transformations
    dense_summary = Column(Text, nullable=True)  # 300-500 word comprehensive summary
    key_insights = Column(JSONB, nullable=True)  # ["insight1", "insight2", ...] - 5-10 actionable insights
    reflection_questions = Column(JSONB, nullable=True)  # ["q1", "q2", ...] - 5-7 thought-provoking questions

    # Transformation processing status
    transformation_status = Column(String(20), default='pending', nullable=False)  # pending|processing|completed|failed
    transformation_error = Column(Text, nullable=True)  # Error message if transformation failed
    transformed_at = Column(TIMESTAMP, nullable=True)  # When transformation completed

    # RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) status
    raptor_status = Column(String(20), default='pending', nullable=False)  # pending|building|completed|failed
    raptor_error = Column(Text, nullable=True)  # Error message if RAPTOR build failed
    raptor_built_at = Column(TIMESTAMP, nullable=True)  # When RAPTOR tree was built

    # Relationships
    notebook = relationship("Notebook", back_populates="sources")

    def __repr__(self):
        return f"<NotebookSource(source_id={self.source_id}, file_name='{self.file_name}', chunks={self.chunk_count}, active={self.active}, transform_status='{self.transformation_status}')>"


class Conversation(Base):
    """Persistent conversation history per notebook"""
    __tablename__ = "conversations"
    __table_args__ = (
        Index('idx_notebook_conversations', 'notebook_id', 'timestamp'),
        Index('idx_user_conversations', 'user_id', 'timestamp'),
        Index('idx_session_conversations', 'session_id', 'timestamp'),
    )

    conversation_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(), nullable=True)  # Groups messages into conversations (NULL = UI legacy)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    notebook = relationship("Notebook", back_populates="conversations")
    user = relationship("User", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(conversation_id={self.conversation_id}, role='{self.role}', timestamp={self.timestamp})>"


class QueryLog(Base):
    """Query logs for observability and cost tracking"""
    __tablename__ = "query_logs"
    __table_args__ = (
        Index('idx_query_logs_timestamp', 'timestamp'),
        Index('idx_query_logs_notebook', 'notebook_id', 'timestamp'),
        Index('idx_query_logs_user', 'user_id', 'timestamp'),
    )

    log_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="SET NULL"))
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    model_name = Column(String(100))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    response_time_ms = Column(Integer)  # Response time in milliseconds
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    notebook = relationship("Notebook", back_populates="query_logs")
    user = relationship("User", back_populates="query_logs")

    def __repr__(self):
        return f"<QueryLog(log_id={self.log_id}, model='{self.model_name}', tokens={self.total_tokens}, time={self.response_time_ms}ms)>"


class GeneratedContent(Base):
    """Generated content from Content Studio (infographics, mind maps, etc.)"""
    __tablename__ = "generated_content"
    __table_args__ = (
        Index('idx_generated_content_user', 'user_id', 'created_at'),
        Index('idx_generated_content_notebook', 'source_notebook_id', 'created_at'),
        Index('idx_generated_content_type', 'content_type', 'created_at'),
    )

    content_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    source_notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="SET NULL"), nullable=True)
    content_type = Column(String(50), nullable=False)  # 'infographic', 'mindmap', 'summary', etc.
    title = Column(String(500))
    prompt_used = Column(Text)  # The prompt used to generate the content
    file_path = Column(String(1000))  # Path to the generated file (e.g., outputs/studio/xxx.png)
    thumbnail_path = Column(String(1000))  # Optional thumbnail for gallery preview
    content_metadata = Column(JSONB)  # Additional generation params, model used, etc.
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", backref="generated_content")
    source_notebook = relationship("Notebook", backref="generated_content")

    def __repr__(self):
        return f"<GeneratedContent(content_id={self.content_id}, type='{self.content_type}', title='{self.title}')>"


class EmbeddingConfig(Base):
    """Tracks active embedding model to prevent mixing incompatible embeddings.

    CRITICAL: Different embedding models produce incompatible vector spaces even with
    the same dimensions. This table ensures only one model is used across all documents.
    Switching models requires re-embedding all documents.
    """
    __tablename__ = "embedding_config"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)  # e.g., "text-embedding-3-small", "nomic-embed-text-v1.5"
    provider = Column(String(50), nullable=False)  # e.g., "openai", "huggingface"
    dimensions = Column(Integer, nullable=False)  # Vector dimensions (768, 1536, etc.)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<EmbeddingConfig(provider='{self.provider}', model='{self.model_name}', dim={self.dimensions})>"


class AnalyticsSession(Base):
    """Analytics dashboard session with uploaded Excel data."""
    __tablename__ = "analytics_sessions"
    __table_args__ = (
        Index("idx_analytics_sessions_user", "user_id"),
        Index("idx_analytics_sessions_notebook", "notebook_id"),
        Index("idx_analytics_sessions_created", "created_at"),
    )

    session_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)  # MD5 for deduplication
    row_count = Column(Integer)
    column_count = Column(Integer)
    column_info = Column(JSONB)  # Column names, types, stats
    data_json = Column(JSONB)  # Parsed Excel data (for smaller datasets)
    profile_report_path = Column(String(500))  # Path to ydata HTML report
    dashboard_config = Column(JSONB)  # AI-generated dashboard configuration
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="analytics_sessions")
    notebook = relationship("Notebook", back_populates="analytics_sessions")

    def __repr__(self):
        return f"<AnalyticsSession(session_id={self.session_id}, filename='{self.filename}', rows={self.row_count}, cols={self.column_count})>"


# =============================================================================
# SQL Chat (Chat with Data) Models
# =============================================================================

class DatabaseConnection(Base):
    """External database connections for Chat with Data feature."""
    __tablename__ = "database_connections"
    __table_args__ = (
        Index("idx_db_connections_user", "user_id"),
    )

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    name = Column(String(200), nullable=False)
    db_type = Column(String(20), nullable=False)  # postgresql, mysql, sqlite
    host = Column(String(255), nullable=True)
    port = Column(Integer, nullable=True)
    database_name = Column(String(200), nullable=True)
    username = Column(String(100), nullable=True)
    password_encrypted = Column(Text, nullable=True)  # Fernet encrypted
    masking_policy = Column(JSONB, nullable=True)  # {mask_columns, redact_columns, hash_columns}
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    last_used_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    sessions = relationship("SQLChatSession", back_populates="connection", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DatabaseConnection(id={self.id}, name='{self.name}', type='{self.db_type}')>"


class SQLChatSession(Base):
    """Chat sessions for SQL queries against external databases."""
    __tablename__ = "sql_chat_sessions"
    __table_args__ = (
        Index("idx_sql_sessions_user", "user_id"),
        Index("idx_sql_sessions_connection", "connection_id"),
    )

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    connection_id = Column(UUID(), ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    last_query_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    connection = relationship("DatabaseConnection", back_populates="sessions")
    query_history = relationship("SQLQueryHistory", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SQLChatSession(id={self.id}, connection_id={self.connection_id})>"


class SQLQueryHistory(Base):
    """History of executed SQL queries in chat sessions."""
    __tablename__ = "sql_query_history"
    __table_args__ = (
        Index("idx_sql_history_session", "session_id"),
        Index("idx_sql_history_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(), ForeignKey("sql_chat_sessions.id", ondelete="CASCADE"), nullable=False)
    user_query = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=False)
    execution_time_ms = Column(Integer, nullable=True)
    row_count = Column(Integer, nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("SQLChatSession", back_populates="query_history")

    def __repr__(self):
        return f"<SQLQueryHistory(id={self.id}, success={self.success}, rows={self.row_count})>"


class SQLFewShotExample(Base):
    """Few-shot examples from Gretel dataset for Text-to-SQL learning."""
    __tablename__ = "sql_few_shot_examples"
    __table_args__ = (
        Index("idx_few_shot_domain", "domain"),
        Index("idx_few_shot_complexity", "complexity"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sql_prompt = Column(Text, nullable=False)
    sql_query = Column(Text, nullable=False)
    sql_context = Column(Text, nullable=True)
    complexity = Column(String(50), nullable=True)  # basic SQL, joins, aggregation, subqueries, window functions
    domain = Column(String(100), nullable=True)  # finance, healthcare, retail, etc.
    # Note: embedding column is added via migration (vector type not available in SQLAlchemy)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SQLFewShotExample(id={self.id}, domain='{self.domain}', complexity='{self.complexity}')>"


class SQLQueryTelemetry(Base):
    """Telemetry data for SQL query execution observability."""
    __tablename__ = "sql_query_telemetry"
    __table_args__ = (
        Index("idx_telemetry_session", "session_id"),
        Index("idx_telemetry_created", "created_at"),
        Index("idx_telemetry_intent", "intent"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(), nullable=False)
    user_query = Column(Text, nullable=True)
    generated_sql = Column(Text, nullable=True)
    intent = Column(String(50), nullable=True)  # lookup, aggregation, comparison, trend, top_k
    confidence_score = Column(Integer, nullable=True)  # Stored as int (score * 100)
    retry_count = Column(Integer, default=0, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    row_count = Column(Integer, nullable=True)
    cost_estimate = Column(Integer, nullable=True)  # Stored as int (cost * 100)
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<SQLQueryTelemetry(id={self.id}, intent='{self.intent}', success={self.success})>"


# =============================================================================
# RBAC (Role-Based Access Control) Models
# =============================================================================

class Role(Base):
    """Role definitions for RBAC.

    Built-in roles:
    - admin: Full access to all features and user management
    - user: Standard access to own notebooks and assigned resources
    - viewer: Read-only access to assigned notebooks
    """
    __tablename__ = "roles"

    role_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)  # admin, user, viewer
    description = Column(String(255), nullable=True)
    permissions = Column(JSONB, nullable=False, default=list)  # List of permission strings
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    user_roles = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Role(role_id={self.role_id}, name='{self.name}')>"


class UserRole(Base):
    """Maps users to roles (many-to-many)."""
    __tablename__ = "user_roles"
    __table_args__ = (
        Index("idx_user_roles_user", "user_id"),
        Index("idx_user_roles_role", "role_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    role_id = Column(UUID(), ForeignKey("roles.role_id", ondelete="CASCADE"), nullable=False)
    assigned_by = Column(UUID(), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    assigned_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="user_roles")
    role = relationship("Role", back_populates="user_roles")
    assigned_by_user = relationship("User", foreign_keys=[assigned_by])

    def __repr__(self):
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"


class NotebookAccess(Base):
    """Grants users access to specific notebooks.

    Access levels:
    - owner: Full control including delete and share
    - editor: Can edit documents and chat
    - viewer: Read-only access, can view documents and chat history
    """
    __tablename__ = "notebook_access"
    __table_args__ = (
        Index("idx_notebook_access_notebook", "notebook_id"),
        Index("idx_notebook_access_user", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    access_level = Column(String(20), nullable=False, default="viewer")  # owner, editor, viewer
    granted_by = Column(UUID(), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    granted_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    notebook = relationship("Notebook", backref="access_grants")
    user = relationship("User", foreign_keys=[user_id], backref="notebook_access")
    granted_by_user = relationship("User", foreign_keys=[granted_by])

    def __repr__(self):
        return f"<NotebookAccess(notebook_id={self.notebook_id}, user_id={self.user_id}, level='{self.access_level}')>"


class SQLConnectionAccess(Base):
    """Grants users access to specific SQL connections.

    Access levels:
    - owner: Full control including delete and share
    - user: Can query the database
    - viewer: Read-only access to query history
    """
    __tablename__ = "sql_connection_access"
    __table_args__ = (
        Index("idx_sql_access_connection", "connection_id"),
        Index("idx_sql_access_user", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    connection_id = Column(UUID(), ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    access_level = Column(String(20), nullable=False, default="user")  # owner, user, viewer
    granted_by = Column(UUID(), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    granted_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    connection = relationship("DatabaseConnection", backref="access_grants")
    user = relationship("User", foreign_keys=[user_id], backref="sql_connection_access")
    granted_by_user = relationship("User", foreign_keys=[granted_by])

    def __repr__(self):
        return f"<SQLConnectionAccess(connection_id={self.connection_id}, user_id={self.user_id}, level='{self.access_level}')>"


# =============================================================================
# Quiz Models
# =============================================================================

class Quiz(Base):
    """Quiz configuration created by admin/user.

    Quizzes are generated from notebook content and can be shared via link.
    Supports adaptive difficulty that adjusts based on user performance.
    Supports extended questions (beyond notebook content) and code-based questions.
    """
    __tablename__ = "quizzes"
    __table_args__ = (
        Index("idx_quizzes_user", "user_id"),
        Index("idx_quizzes_notebook", "notebook_id"),
        Index("idx_quizzes_created", "created_at"),
    )

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    notebook_id = Column(UUID(), ForeignKey("notebooks.notebook_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    num_questions = Column(Integer, default=10, nullable=False)
    difficulty_mode = Column(String(20), default='adaptive', nullable=False)  # adaptive|easy|medium|hard
    time_limit_minutes = Column(Integer, nullable=True)  # Optional time limit
    llm_model = Column(String(100), nullable=True)  # Optional LLM model for question generation
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Extended question generation options
    question_source = Column(String(20), default='notebook_only', nullable=False)  # notebook_only|extended
    include_code_questions = Column(Boolean, default=False, nullable=False)  # Enable code-based questions

    # Relationships - passive_deletes=True lets database CASCADE handle deletion
    notebook = relationship("Notebook", backref=backref("quizzes", passive_deletes=True))
    creator = relationship("User", backref=backref("created_quizzes", passive_deletes=True))
    attempts = relationship("QuizAttempt", back_populates="quiz", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Quiz(id={self.id}, title='{self.title}', questions={self.num_questions}, mode='{self.difficulty_mode}')>"


class QuizAttempt(Base):
    """Individual quiz attempt by a test-taker.

    Tracks progress through the quiz, answers given, and final score.
    For adaptive quizzes, current_difficulty adjusts based on performance.
    """
    __tablename__ = "quiz_attempts"
    __table_args__ = (
        Index("idx_quiz_attempts_quiz", "quiz_id"),
        Index("idx_quiz_attempts_started", "started_at"),
        Index("idx_quiz_attempts_email", "taker_email"),
    )

    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    quiz_id = Column(UUID(), ForeignKey("quizzes.id", ondelete="CASCADE"), nullable=False)
    taker_name = Column(String(255), nullable=False)
    taker_email = Column(String(255), nullable=True)  # Optional email for session resumption
    score = Column(Integer, default=0, nullable=False)
    total_questions = Column(Integer, nullable=False)
    answers_json = Column(JSONB, nullable=True)  # [{question, user_answer, correct_answer, correct, topic, explanation}]
    started_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    completed_at = Column(TIMESTAMP, nullable=True)
    current_question = Column(Integer, default=0, nullable=False)  # 0-indexed question number
    current_difficulty = Column(Integer, default=2, nullable=False)  # 1=easy, 2=medium, 3=hard

    # Relationships
    quiz = relationship("Quiz", back_populates="attempts")

    def __repr__(self):
        return f"<QuizAttempt(id={self.id}, taker='{self.taker_name}', score={self.score}/{self.total_questions})>"
