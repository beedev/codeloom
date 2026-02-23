"""
SQLAlchemy ORM Models for CodeLoom

Code intelligence and migration platform models:
- Users: User accounts with auth and RBAC
- Projects: Codebase projects (replaces projects)
- CodeFile: Individual files within a project
- CodeUnit: AST-parsed code units (functions, classes, methods)
- CodeEdge: ASG relationship edges between code units
- MigrationPlan: Migration configurations
- MigrationPhase: Phase tracking and output
- Conversation: Chat history per project
- QueryLog: Query observability and cost tracking
- EmbeddingConfig: Tracks active embedding model
- Role, UserRole: RBAC core
- ProjectAccess: Grants users access to specific projects
"""

from sqlalchemy import (
    Column, String, Integer, Float, Text, TIMESTAMP, ForeignKey, BigInteger,
    Index, TypeDecorator, Boolean, UniqueConstraint,
)
from sqlalchemy.orm import relationship, declarative_base
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


# =============================================================================
# Core User Model
# =============================================================================

class User(Base):
    """User model for multi-user support."""
    __tablename__ = "users"

    user_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True)
    password_hash = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    last_active = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="user", cascade="all, delete-orphan")
    user_roles = relationship("UserRole", foreign_keys="[UserRole.user_id]", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


# =============================================================================
# CodeLoom Project Models
# =============================================================================

class Project(Base):
    """Codebase project - replaces projects for code context."""
    __tablename__ = "projects"
    __table_args__ = (
        Index('idx_user_projects', 'user_id', 'created_at'),
    )

    project_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    primary_language = Column(String(50))           # python, javascript, java, csharp
    languages = Column(JSONB, default=list)          # all detected languages
    file_count = Column(Integer, default=0)
    total_lines = Column(Integer, default=0)
    ast_status = Column(String(20), default='pending', nullable=False)   # pending, parsing, complete, error
    asg_status = Column(String(20), default='pending', nullable=False)
    source_type = Column(String(20), default='zip')                      # zip, git, local
    source_url = Column(String(2048), nullable=True)                     # git URL or local path
    repo_branch = Column(String(255), nullable=True)                     # git branch
    last_synced_at = Column(TIMESTAMP, nullable=True)                    # last ingestion time
    deep_analysis_status = Column(String(20), default='none', nullable=False)  # none|pending|running|completed|failed
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="projects")
    files = relationship("CodeFile", back_populates="project", cascade="all, delete-orphan")
    code_units = relationship("CodeUnit", back_populates="project", cascade="all, delete-orphan")
    code_edges = relationship("CodeEdge", back_populates="project", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")
    query_logs = relationship("QueryLog", back_populates="project", cascade="all, delete-orphan")
    migration_plans = relationship("MigrationPlan", back_populates="source_project", cascade="all, delete-orphan")
    deep_analysis_jobs = relationship("DeepAnalysisJob", back_populates="project", cascade="all, delete-orphan")
    deep_analyses = relationship("DeepAnalysis", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(project_id={self.project_id}, name='{self.name}', lang='{self.primary_language}', files={self.file_count})>"


class CodeFile(Base):
    """Individual files within a project."""
    __tablename__ = "code_files"
    __table_args__ = (
        Index('idx_code_files_project', 'project_id'),
        UniqueConstraint('project_id', 'file_path', name='uq_project_file_path'),
    )

    file_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    file_path = Column(String(1024), nullable=False)    # relative path within project
    language = Column(String(50))
    file_hash = Column(String(64))                      # MD5 for change detection
    line_count = Column(Integer)
    size_bytes = Column(Integer)
    raptor_status = Column(String(20), default='pending')       # pending|building|completed|failed
    raptor_error = Column(Text, nullable=True)
    raptor_built_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="files")
    code_units = relationship("CodeUnit", back_populates="file", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CodeFile(file_id={self.file_id}, path='{self.file_path}', lang='{self.language}')>"


class CodeUnit(Base):
    """AST-parsed code units (functions, classes, methods, modules)."""
    __tablename__ = "code_units"
    __table_args__ = (
        Index('idx_code_units_file', 'file_id'),
        Index('idx_code_units_project', 'project_id'),
        Index('idx_code_units_type', 'unit_type'),
        Index('idx_code_units_name', 'name'),
    )

    unit_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    file_id = Column(UUID(), ForeignKey("code_files.file_id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    unit_type = Column(String(50), nullable=False)          # function, class, method, module, interface
    name = Column(String(255), nullable=False)
    qualified_name = Column(String(1024))                   # module.class.method
    language = Column(String(50))
    start_line = Column(Integer)
    end_line = Column(Integer)
    signature = Column(Text)
    docstring = Column(Text)
    source = Column(Text)
    unit_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    file = relationship("CodeFile", back_populates="code_units")
    project = relationship("Project", back_populates="code_units")
    outgoing_edges = relationship("CodeEdge", foreign_keys="[CodeEdge.source_unit_id]", back_populates="source_unit", cascade="all, delete-orphan")
    incoming_edges = relationship("CodeEdge", foreign_keys="[CodeEdge.target_unit_id]", back_populates="target_unit", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CodeUnit(unit_id={self.unit_id}, type='{self.unit_type}', name='{self.name}')>"


class CodeEdge(Base):
    """ASG relationship edges between code units."""
    __tablename__ = "code_edges"
    __table_args__ = (
        Index('idx_code_edges_project', 'project_id'),
        Index('idx_code_edges_source', 'source_unit_id'),
        Index('idx_code_edges_target', 'target_unit_id'),
        Index('idx_code_edges_type', 'edge_type'),
        UniqueConstraint('project_id', 'source_unit_id', 'target_unit_id', 'edge_type', name='uq_code_edge'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    source_unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"), nullable=False)
    target_unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"), nullable=False)
    edge_type = Column(String(50), nullable=False)      # contains, imports, calls, inherits, implements, overrides
    edge_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="code_edges")
    source_unit = relationship("CodeUnit", foreign_keys=[source_unit_id], back_populates="outgoing_edges")
    target_unit = relationship("CodeUnit", foreign_keys=[target_unit_id], back_populates="incoming_edges")

    def __repr__(self):
        return f"<CodeEdge(id={self.id}, type='{self.edge_type}', {self.source_unit_id} -> {self.target_unit_id})>"


# =============================================================================
# Migration Models
# =============================================================================

class MigrationPlan(Base):
    """Migration plan configuration."""
    __tablename__ = "migration_plans"
    __table_args__ = (
        Index('idx_migration_plans_user', 'user_id'),
        Index('idx_migration_plans_project', 'source_project_id'),
    )

    plan_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    source_project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="SET NULL"))
    target_brief = Column(Text, nullable=False)             # Target architecture description
    target_stack = Column(JSONB, nullable=False)             # {languages, frameworks, versions}
    constraints = Column(JSONB, default=dict)                # {timeline, team_size, risk_tolerance}
    status = Column(String(20), default='draft', nullable=False)  # draft, in_progress, complete, abandoned
    current_phase = Column(Integer, default=0, nullable=False)
    discovery_metadata = Column(JSONB, default=dict)         # {clustering_params, sp_analysis, total_mvps}
    framework_docs = Column(JSONB, nullable=True)            # {framework_name: {content, source, fetched_at}}
    migration_type = Column(String(30), default='framework_migration', nullable=False)  # version_upgrade, framework_migration, rewrite
    pipeline_version = Column(Integer, default=2, nullable=False)  # 1=old 6-phase, 2=new 4-phase
    asset_strategies = Column(JSONB, nullable=True)               # {lang: {strategy, target}} per-file-type migration strategies
    migration_lane_id = Column(String(100), nullable=True)       # auto-detected lane, e.g. "struts_to_springboot"
    lane_versions = Column(JSONB, nullable=True)                 # {"struts_to_springboot": "1.0.0"} — recorded on first execution
    batch_executions = Column(JSONB, default=list)                # [{batch_id, status, mvp_results, ...}]
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User")
    source_project = relationship("Project", back_populates="migration_plans")
    phases = relationship("MigrationPhase", back_populates="plan", cascade="all, delete-orphan")
    mvps = relationship("FunctionalMVP", back_populates="plan", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MigrationPlan(plan_id={self.plan_id}, status='{self.status}', phase={self.current_phase})>"


class MigrationPhase(Base):
    """Migration phase tracking and output.

    Phase numbering:
    - Phases 1-2: Plan-level (Discovery, Architecture) — mvp_id is NULL
    - Phases 3-6: Per-MVP (Analyze, Design, Transform, Test) — mvp_id set
    """
    __tablename__ = "migration_phases"
    __table_args__ = (
        Index('idx_migration_phases_plan', 'plan_id'),
        Index('idx_migration_phases_mvp', 'mvp_id'),
        UniqueConstraint('plan_id', 'phase_number', 'mvp_id', name='uq_plan_phase_mvp'),
    )

    phase_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(), nullable=True)                       # Unique per execution attempt — distinguishes re-runs
    plan_id = Column(UUID(), ForeignKey("migration_plans.plan_id", ondelete="CASCADE"), nullable=False)
    mvp_id = Column(Integer, ForeignKey("functional_mvps.mvp_id", ondelete="CASCADE"), nullable=True)  # NULL for plan-level phases
    phase_number = Column(Integer, nullable=False)           # 1-6
    phase_type = Column(String(50), nullable=False)          # discovery, architecture, analyze, design, transform, testing
    status = Column(String(20), default='pending', nullable=False)
    input_summary = Column(Text)
    output = Column(Text)                                     # LLM-generated output (markdown)
    output_files = Column(JSONB, default=list)               # Generated code files
    approved = Column(Boolean, default=False, nullable=False)
    approved_at = Column(TIMESTAMP)
    phase_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    plan = relationship("MigrationPlan", back_populates="phases")
    mvp = relationship("FunctionalMVP", back_populates="phases")

    def __repr__(self):
        return f"<MigrationPhase(phase_id={self.phase_id}, phase={self.phase_number}, type='{self.phase_type}', mvp={self.mvp_id}, status='{self.status}')>"


class FunctionalMVP(Base):
    """Functional MVP — a vertical slice of related code units for incremental migration.

    Discovered by the clustering algorithm in Phase 1 (Discovery),
    then refined by the user on the graph UI before per-MVP migration begins.
    """
    __tablename__ = "functional_mvps"
    __table_args__ = (
        Index('idx_functional_mvps_plan', 'plan_id'),
    )

    mvp_id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(UUID(), ForeignKey("migration_plans.plan_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)               # "User Authentication", "Order Processing"
    description = Column(Text, nullable=True)
    status = Column(String(50), default="discovered")        # discovered, refined, in_progress, migrated, verified
    priority = Column(Integer, default=0)                     # Lower = higher priority (migration order)

    # What's in this MVP
    file_ids = Column(JSONB, default=list)                   # [file_id, ...]
    unit_ids = Column(JSONB, default=list)                   # [unit_id, ...] — more granular than files

    # Dependencies on other MVPs
    depends_on_mvp_ids = Column(JSONB, default=list)         # [mvp_id, ...] — must migrate first

    # Stored procedure references
    sp_references = Column(JSONB, default=list)              # [{"sp_name": "usp_GetUser", "file_id": "...", "call_sites": [...]}]

    # Clustering metrics
    metrics = Column(JSONB, default=dict)                    # {"cohesion": 0.85, "coupling": 0.2, "size": 42, "readiness": 0.7}

    # Per-MVP phase tracking
    current_phase = Column(Integer, default=0)               # 0=not started, 3-6=per-MVP phases

    # On-demand deep analysis (merges old Analyze + Design for V2 pipeline)
    analysis_output = Column(JSONB, nullable=True)           # Combined register + traceability from Deep Analyze
    analysis_at = Column(TIMESTAMP, nullable=True)           # When analysis was last run

    # Background analysis status tracking
    analysis_status = Column(String(20), default="pending")  # pending|analyzing|completed|failed
    analysis_error = Column(Text, nullable=True)             # Error message if failed

    # Cached UML diagrams (PlantUML + SVG) — behavioral diagrams are LLM-generated
    diagrams = Column(JSONB, nullable=True)                  # {"sequence": {"puml": "...", "svg": "...", "title": "...", "generated_at": "..."}, ...}

    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    plan = relationship("MigrationPlan", back_populates="mvps")
    phases = relationship("MigrationPhase", back_populates="mvp", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<FunctionalMVP(mvp_id={self.mvp_id}, name='{self.name}', status='{self.status}', priority={self.priority})>"


# =============================================================================
# Chat & Observability (Reused from DBProject, FK updated)
# =============================================================================

class Conversation(Base):
    """Persistent conversation history per project."""
    __tablename__ = "conversations"
    __table_args__ = (
        Index('idx_project_conversations', 'project_id', 'timestamp'),
        Index('idx_user_conversations', 'user_id', 'timestamp'),
        Index('idx_session_conversations', 'session_id', 'timestamp'),
    )

    conversation_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(), nullable=True)
    role = Column(String(20), nullable=False)       # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="conversations")
    user = relationship("User", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(conversation_id={self.conversation_id}, role='{self.role}', timestamp={self.timestamp})>"


class QueryLog(Base):
    """Query logs for observability and cost tracking."""
    __tablename__ = "query_logs"
    __table_args__ = (
        Index('idx_query_logs_timestamp', 'timestamp'),
        Index('idx_query_logs_project', 'project_id', 'timestamp'),
        Index('idx_query_logs_user', 'user_id', 'timestamp'),
    )

    log_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="SET NULL"))
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    query_text = Column(Text, nullable=False)
    model_name = Column(String(100))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    response_time_ms = Column(Integer)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="query_logs")
    user = relationship("User", back_populates="query_logs")

    def __repr__(self):
        return f"<QueryLog(log_id={self.log_id}, model='{self.model_name}', tokens={self.total_tokens}, time={self.response_time_ms}ms)>"


class EmbeddingConfig(Base):
    """Tracks active embedding model to prevent mixing incompatible embeddings."""
    __tablename__ = "embedding_config"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    provider = Column(String(50), nullable=False)
    dimensions = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<EmbeddingConfig(provider='{self.provider}', model='{self.model_name}', dim={self.dimensions})>"


# =============================================================================
# RBAC (Role-Based Access Control) Models
# =============================================================================

class Role(Base):
    """Role definitions for RBAC.

    Built-in roles:
    - admin: Full access to all features and user management
    - user: Standard access to own projects and assigned resources
    - viewer: Read-only access to assigned projects
    """
    __tablename__ = "roles"

    role_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=True)
    permissions = Column(JSONB, nullable=False, default=list)
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


class ProjectAccess(Base):
    """Grants users access to specific projects.

    Access levels:
    - owner: Full control including delete and share
    - editor: Can edit code and chat
    - viewer: Read-only access
    """
    __tablename__ = "project_access"
    __table_args__ = (
        Index("idx_project_access_project", "project_id"),
        Index("idx_project_access_user", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    access_level = Column(String(20), nullable=False, default="viewer")
    granted_by = Column(UUID(), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    granted_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", backref="access_grants")
    user = relationship("User", foreign_keys=[user_id], backref="project_access")
    granted_by_user = relationship("User", foreign_keys=[granted_by])

    def __repr__(self):
        return f"<ProjectAccess(project_id={self.project_id}, user_id={self.user_id}, level='{self.access_level}')>"


# =============================================================================
# Deep Understanding Models
# =============================================================================

class DeepAnalysisJob(Base):
    """Job queue for deep understanding analysis.

    Each row represents one analysis run for a project.
    Workers claim jobs via FOR UPDATE SKIP LOCKED with worker_id lease ownership.
    Exponential backoff via next_attempt_at column.
    """
    __tablename__ = "deep_analysis_jobs"
    __table_args__ = (
        Index('idx_deep_jobs_project_status', 'project_id', 'status'),
        Index('idx_deep_jobs_status_created', 'status', 'created_at'),
    )

    job_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default='pending', nullable=False)  # pending|running|completed|failed
    worker_id = Column(String(100), nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    next_attempt_at = Column(TIMESTAMP, nullable=True)

    # Progress tracking
    total_entry_points = Column(Integer, nullable=True)
    completed_entry_points = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    heartbeat_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

    # Error details (JSON array of {entry_point, error})
    error_details = Column(JSONB, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="deep_analysis_jobs")
    analyses = relationship("DeepAnalysis", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DeepAnalysisJob(job_id={self.job_id}, status='{self.status}', project={self.project_id})>"


class DeepAnalysis(Base):
    """Analysis result for a single entry point.

    Stores the complete DeepContextBundle as JSON plus a
    human-readable narrative for chat injection.
    Upsert on (project_id, entry_unit_id, schema_version).
    """
    __tablename__ = "deep_analyses"
    __table_args__ = (
        UniqueConstraint('project_id', 'entry_unit_id', 'schema_version',
                         name='uq_deep_analysis_entry'),
        Index('idx_deep_analyses_project', 'project_id'),
        Index('idx_deep_analyses_entry', 'entry_unit_id'),
    )

    analysis_id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(), ForeignKey("deep_analysis_jobs.job_id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    entry_unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"), nullable=False)

    entry_type = Column(String(50), nullable=False)     # EntryPointType value
    tier = Column(String(20), nullable=False)            # AnalysisTier value
    total_units = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=True)      # 0..1 extraction confidence
    coverage_pct = Column(Float, nullable=True)          # 0..100 chain coverage estimate

    # Full structured analysis output (DeepContextBundle shape)
    result_json = Column(JSONB, nullable=False)
    # Human-readable narrative (injected into chat context)
    narrative = Column(Text, nullable=True)

    # Versioning
    schema_version = Column(Integer, default=1, nullable=False)
    prompt_version = Column(String(20), default='v1.0', nullable=False)

    analyzed_at = Column(TIMESTAMP, default=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("DeepAnalysisJob", back_populates="analyses")
    project = relationship("Project", back_populates="deep_analyses")
    entry_unit = relationship("CodeUnit")
    analysis_units = relationship("AnalysisUnit", back_populates="analysis",
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DeepAnalysis(analysis_id={self.analysis_id}, entry={self.entry_unit_id}, tier='{self.tier}')>"


class AnalysisUnit(Base):
    """Junction table linking analyses to the code units they cover.

    Enables:
    1. "What analyses cover this unit?" (for chat enrichment)
    2. "What units does this analysis cover?" (for coverage calculation)
    """
    __tablename__ = "analysis_units"
    __table_args__ = (
        Index('idx_analysis_units_project_unit', 'project_id', 'unit_id'),
    )

    analysis_id = Column(UUID(), ForeignKey("deep_analyses.analysis_id", ondelete="CASCADE"),
                         primary_key=True, nullable=False)
    unit_id = Column(UUID(), ForeignKey("code_units.unit_id", ondelete="CASCADE"),
                     primary_key=True, nullable=False)
    project_id = Column(UUID(), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    min_depth = Column(Integer, nullable=False, default=0)
    path_count = Column(Integer, nullable=False, default=1)

    # Relationships
    analysis = relationship("DeepAnalysis", back_populates="analysis_units")
    unit = relationship("CodeUnit")

    def __repr__(self):
        return f"<AnalysisUnit(analysis={self.analysis_id}, unit={self.unit_id}, depth={self.min_depth})>"
