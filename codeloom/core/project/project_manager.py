"""Project Manager for CodeLoom.

Provides CRUD operations for projects, code files, and code units
with PostgreSQL persistence. Aligned with CodeLoom schema.
"""

import logging
import secrets
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from ..db import DatabaseManager
from ..db.models import Project, CodeFile, CodeUnit, User

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages projects and their code files with database persistence."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        logger.info("ProjectManager initialized")

    # =========================================================================
    # Project CRUD
    # =========================================================================

    def create_project(
        self,
        user_id: str,
        name: str,
        description: str = "",
    ) -> Dict:
        """Create a new project."""
        try:
            with self.db.get_session() as session:
                existing = session.query(Project).filter(
                    Project.user_id == UUID(user_id),
                    Project.name == name,
                ).first()

                if existing:
                    raise ValueError(f"Project '{name}' already exists for user {user_id}")

                project = Project(
                    project_id=uuid4(),
                    user_id=UUID(user_id),
                    name=name,
                    description=description,
                )

                session.add(project)
                session.flush()

                logger.info(f"Created project: {project.project_id} ({name})")
                return self._project_to_dict(project)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise

    def get_project(self, project_id: str) -> Optional[Dict]:
        """Retrieve project details by ID."""
        try:
            with self.db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()

                if not project:
                    return None

                return self._project_to_dict(project)

        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            raise

    def list_projects(self, user_id: str) -> List[Dict]:
        """List all projects for a user, newest first."""
        try:
            with self.db.get_session() as session:
                projects = session.query(Project).filter(
                    Project.user_id == UUID(user_id)
                ).order_by(Project.created_at.desc()).all()

                return [self._project_to_dict(p) for p in projects]

        except Exception as e:
            logger.error(f"Failed to list projects for user {user_id}: {e}")
            raise

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update project name/description."""
        try:
            with self.db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()

                if not project:
                    return False

                if name and name != project.name:
                    existing = session.query(Project).filter(
                        Project.user_id == project.user_id,
                        Project.name == name,
                        Project.project_id != UUID(project_id),
                    ).first()
                    if existing:
                        raise ValueError(f"Project name '{name}' already exists")
                    project.name = name

                if description is not None:
                    project.description = description

                project.updated_at = datetime.utcnow()
                return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update project {project_id}: {e}")
            raise

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data (CASCADE)."""
        try:
            with self.db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()

                if not project:
                    return False

                session.delete(project)
                logger.info(f"Deleted project: {project_id} ({project.name})")
                return True

        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            raise

    # =========================================================================
    # Project Stats
    # =========================================================================

    def update_project_stats(
        self,
        project_id: str,
        file_count: int,
        total_lines: int,
        primary_language: Optional[str],
        languages: Optional[list] = None,
        ast_status: str = "complete",
    ) -> bool:
        """Update project statistics after ingestion."""
        try:
            with self.db.get_session() as session:
                project = session.query(Project).filter(
                    Project.project_id == UUID(project_id)
                ).first()

                if not project:
                    return False

                project.file_count = file_count
                project.total_lines = total_lines
                project.primary_language = primary_language
                project.languages = languages or []
                project.ast_status = ast_status
                project.updated_at = datetime.utcnow()

                logger.info(f"Updated project stats: {project_id} ({file_count} files, {total_lines} lines)")
                return True

        except Exception as e:
            logger.error(f"Failed to update project stats {project_id}: {e}")
            return False

    # =========================================================================
    # Code Files
    # =========================================================================

    def get_project_files(self, project_id: str) -> List[Dict]:
        """List all code files in a project."""
        try:
            with self.db.get_session() as session:
                files = session.query(CodeFile).filter(
                    CodeFile.project_id == UUID(project_id)
                ).order_by(CodeFile.file_path).all()

                return [
                    {
                        "file_id": str(f.file_id),
                        "project_id": str(f.project_id),
                        "file_path": f.file_path,
                        "language": f.language,
                        "line_count": f.line_count,
                        "size_bytes": f.size_bytes,
                        "file_hash": f.file_hash,
                        "created_at": f.created_at.isoformat() if f.created_at else None,
                    }
                    for f in files
                ]

        except Exception as e:
            logger.error(f"Failed to get files for project {project_id}: {e}")
            raise

    def get_project_units(
        self,
        project_id: str,
        file_id: Optional[str] = None,
    ) -> List[Dict]:
        """List code units in a project, optionally filtered by file."""
        try:
            with self.db.get_session() as session:
                query = session.query(CodeUnit).filter(
                    CodeUnit.project_id == UUID(project_id)
                )
                if file_id:
                    query = query.filter(CodeUnit.file_id == UUID(file_id))

                units = query.order_by(CodeUnit.start_line).all()

                return [
                    {
                        "unit_id": str(u.unit_id),
                        "file_id": str(u.file_id),
                        "unit_type": u.unit_type,
                        "name": u.name,
                        "qualified_name": u.qualified_name,
                        "language": u.language,
                        "start_line": u.start_line,
                        "end_line": u.end_line,
                        "signature": u.signature,
                    }
                    for u in units
                ]

        except Exception as e:
            logger.error(f"Failed to get units for project {project_id}: {e}")
            raise

    def get_file_tree(self, project_id: str) -> Dict:
        """Build a nested dict representing the file tree for UI display.

        Returns:
            Nested dict like:
            {"name": "/", "children": [
                {"name": "src", "children": [
                    {"name": "main.py", "file_id": "...", "language": "python", "line_count": 42}
                ]}
            ]}
        """
        files = self.get_project_files(project_id)
        root: Dict = {"name": "/", "type": "directory", "children": []}

        for f in files:
            parts = f["file_path"].split("/")
            current = root

            for i, part in enumerate(parts):
                is_file = (i == len(parts) - 1)

                if is_file:
                    current["children"].append({
                        "name": part,
                        "type": "file",
                        "file_id": f["file_id"],
                        "file_path": f["file_path"],
                        "language": f["language"],
                        "line_count": f["line_count"],
                    })
                else:
                    # Find or create directory
                    existing = None
                    for child in current["children"]:
                        if child.get("type") == "directory" and child["name"] == part:
                            existing = child
                            break

                    if not existing:
                        existing = {"name": part, "type": "directory", "children": []}
                        current["children"].append(existing)

                    current = existing

        return root

    # =========================================================================
    # User Management
    # =========================================================================

    def ensure_default_user(
        self,
        user_id: str = "default",
        username: str = "admin",
    ) -> str:
        """Ensure a default user exists. Returns user_id."""
        try:
            if user_id == "default":
                user_uuid = UUID("00000000-0000-0000-0000-000000000001")
            else:
                user_uuid = UUID(user_id)

            with self.db.get_session() as session:
                user = session.query(User).filter(User.user_id == user_uuid).first()

                if not user:
                    from ..auth.auth_service import AuthService
                    password_hash = AuthService.hash_password("admin123")
                    user = User(
                        user_id=user_uuid,
                        username=username,
                        email=f"{username}@codeloom.local",
                        password_hash=password_hash,
                        api_key=f"cl_{secrets.token_hex(16)}",
                    )
                    session.add(user)
                    logger.info(f"Created default user: {user_uuid} ({username})")

                return str(user_uuid)

        except Exception as e:
            logger.error(f"Failed to ensure default user: {e}")
            raise

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _project_to_dict(project: Project) -> Dict:
        """Convert a Project ORM object to a dict."""
        return {
            "id": str(project.project_id),
            "project_id": str(project.project_id),
            "user_id": str(project.user_id),
            "name": project.name,
            "description": project.description or "",
            "primary_language": project.primary_language,
            "languages": project.languages or [],
            "file_count": project.file_count or 0,
            "total_lines": project.total_lines or 0,
            "ast_status": project.ast_status or "pending",
            "asg_status": project.asg_status or "pending",
            "deep_analysis_status": project.deep_analysis_status or "none",
            "source_type": project.source_type or "zip",
            "source_url": project.source_url,
            "repo_branch": project.repo_branch,
            "last_synced_at": project.last_synced_at.isoformat() if project.last_synced_at else None,
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None,
        }
