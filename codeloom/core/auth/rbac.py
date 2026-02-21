"""RBAC (Role-Based Access Control) Service.

Provides access control for all features:
- API endpoints (via API key or user authentication)
- Project-level access (code chat, migration)

Access Levels:
- owner: Full control including delete and share
- editor: Can edit code and chat
- viewer: Read-only access

Permissions:
- manage_users: Create, edit, delete users
- manage_roles: Create, edit, delete roles
- manage_projects: Create projects for any user
- view_all: View any project
- edit_all: Edit any project
- delete_all: Delete any project
- create_project: Create projects for self
- view_assigned: View assigned projects
- edit_assigned: Edit assigned projects
"""

import logging
from enum import Enum
from functools import wraps
from typing import List, Optional, Set
from uuid import UUID

from flask import request, jsonify, g, current_app
from sqlalchemy.orm import Session

from codeloom.core.db.models import (
    Role,
    UserRole,
    ProjectAccess,
    Project,
    User,
)

logger = logging.getLogger(__name__)


class AccessLevel(str, Enum):
    """Access levels for resources."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions."""
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_PROJECTS = "manage_projects"
    VIEW_ALL = "view_all"
    EDIT_ALL = "edit_all"
    DELETE_ALL = "delete_all"

    # User permissions
    CREATE_PROJECT = "create_project"
    VIEW_ASSIGNED = "view_assigned"
    EDIT_ASSIGNED = "edit_assigned"


class RBACService:
    """Role-Based Access Control service.

    Provides methods to check and enforce access control across all features.
    """

    def __init__(self, db_session: Session):
        self._session = db_session

    # ========== Role Management ==========

    def get_user_roles(self, user_id: str) -> List[Role]:
        user_roles = self._session.query(UserRole).filter(
            UserRole.user_id == UUID(user_id)
        ).all()
        return [ur.role for ur in user_roles]

    def get_user_permissions(self, user_id: str) -> Set[str]:
        roles = self.get_user_roles(user_id)
        permissions = set()
        for role in roles:
            if role.permissions:
                permissions.update(role.permissions)
        return permissions

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        permissions = self.get_user_permissions(user_id)
        return permission.value in permissions

    def has_role(self, user_id: str, role_name: str) -> bool:
        roles = self.get_user_roles(user_id)
        return any(role.name == role_name for role in roles)

    def is_admin(self, user_id: str) -> bool:
        return self.has_role(user_id, "admin")

    def assign_role(
        self,
        user_id: str,
        role_name: str,
        assigned_by: Optional[str] = None
    ) -> bool:
        role = self._session.query(Role).filter(Role.name == role_name).first()
        if not role:
            logger.warning(f"Role not found: {role_name}")
            return False

        existing = self._session.query(UserRole).filter(
            UserRole.user_id == UUID(user_id),
            UserRole.role_id == role.role_id
        ).first()

        if existing:
            return True

        user_role = UserRole(
            user_id=UUID(user_id),
            role_id=role.role_id,
            assigned_by=UUID(assigned_by) if assigned_by else None
        )
        self._session.add(user_role)
        self._session.commit()

        logger.info(f"Assigned role {role_name} to user {user_id}")
        return True

    def remove_role(self, user_id: str, role_name: str) -> bool:
        role = self._session.query(Role).filter(Role.name == role_name).first()
        if not role:
            return False

        user_role = self._session.query(UserRole).filter(
            UserRole.user_id == UUID(user_id),
            UserRole.role_id == role.role_id
        ).first()

        if user_role:
            self._session.delete(user_role)
            self._session.commit()
            logger.info(f"Removed role {role_name} from user {user_id}")

        return True

    # ========== Project Access ==========

    def get_project_access_level(
        self,
        user_id: str,
        project_id: str
    ) -> Optional[AccessLevel]:
        if self.has_permission(user_id, Permission.VIEW_ALL):
            return AccessLevel.OWNER

        project = self._session.query(Project).filter(
            Project.project_id == UUID(project_id)
        ).first()

        if project and str(project.user_id) == user_id:
            return AccessLevel.OWNER

        access = self._session.query(ProjectAccess).filter(
            ProjectAccess.project_id == UUID(project_id),
            ProjectAccess.user_id == UUID(user_id)
        ).first()

        if access:
            return AccessLevel(access.access_level)

        return None

    def can_view_project(self, user_id: str, project_id: str) -> bool:
        access_level = self.get_project_access_level(user_id, project_id)
        return access_level is not None

    def can_edit_project(self, user_id: str, project_id: str) -> bool:
        access_level = self.get_project_access_level(user_id, project_id)
        return access_level in (AccessLevel.OWNER, AccessLevel.EDITOR)

    def can_delete_project(self, user_id: str, project_id: str) -> bool:
        access_level = self.get_project_access_level(user_id, project_id)
        return access_level == AccessLevel.OWNER

    def grant_project_access(
        self,
        project_id: str,
        user_id: str,
        access_level: AccessLevel,
        granted_by: Optional[str] = None
    ) -> bool:
        existing = self._session.query(ProjectAccess).filter(
            ProjectAccess.project_id == UUID(project_id),
            ProjectAccess.user_id == UUID(user_id)
        ).first()

        if existing:
            existing.access_level = access_level.value
            existing.granted_by = UUID(granted_by) if granted_by else None
        else:
            access = ProjectAccess(
                project_id=UUID(project_id),
                user_id=UUID(user_id),
                access_level=access_level.value,
                granted_by=UUID(granted_by) if granted_by else None
            )
            self._session.add(access)

        self._session.commit()
        logger.info(f"Granted {access_level.value} access to project {project_id} for user {user_id}")
        return True

    def revoke_project_access(self, project_id: str, user_id: str) -> bool:
        access = self._session.query(ProjectAccess).filter(
            ProjectAccess.project_id == UUID(project_id),
            ProjectAccess.user_id == UUID(user_id)
        ).first()

        if access:
            self._session.delete(access)
            self._session.commit()
            logger.info(f"Revoked project access for user {user_id} from project {project_id}")

        return True

    def list_project_users(self, project_id: str) -> List[dict]:
        accesses = self._session.query(ProjectAccess).filter(
            ProjectAccess.project_id == UUID(project_id)
        ).all()

        result = []
        for access in accesses:
            user = self._session.query(User).filter(
                User.user_id == access.user_id
            ).first()
            if user:
                result.append({
                    "user_id": str(access.user_id),
                    "username": user.username,
                    "email": user.email,
                    "access_level": access.access_level,
                    "granted_at": access.granted_at.isoformat() if access.granted_at else None
                })

        return result


# ========== Flask Decorators ==========

def get_rbac_service():
    """Get RBAC service from Flask g object or create new one."""
    if not hasattr(g, 'rbac_service') or g.rbac_service is None:
        db_manager = current_app.extensions.get('db_manager')
        if not db_manager:
            raise RuntimeError("Database manager not available in app extensions")
        session = db_manager.SessionLocal()
        g.rbac_service = RBACService(session)
        g.rbac_session = session
    return g.rbac_service


def cleanup_rbac_session(exception=None):
    """Clean up RBAC session after request."""
    session = g.pop('rbac_session', None)
    if session:
        try:
            session.close()
        except Exception:
            pass
    g.pop('rbac_service', None)


def require_permission(permission: Permission):
    """Decorator to require a specific permission."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_id = _get_current_user_id()
            if not user_id:
                return jsonify({
                    "success": False,
                    "error": "Authentication required"
                }), 401

            rbac = get_rbac_service()
            if not rbac.has_permission(user_id, permission):
                return jsonify({
                    "success": False,
                    "error": f"Permission denied: {permission.value}"
                }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


def require_project_access(access_level: AccessLevel = AccessLevel.VIEWER):
    """Decorator to require project access.

    Expects project_id in route params or request body.
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_id = _get_current_user_id()
            if not user_id:
                return jsonify({
                    "success": False,
                    "error": "Authentication required"
                }), 401

            project_id = kwargs.get('project_id')
            if not project_id:
                data = request.get_json(silent=True) or {}
                project_id = data.get('project_id') or data.get('projectId')

            if not project_id:
                return jsonify({
                    "success": False,
                    "error": "project_id is required"
                }), 400

            rbac = get_rbac_service()
            user_access = rbac.get_project_access_level(user_id, project_id)

            if not user_access:
                return jsonify({
                    "success": False,
                    "error": "Access denied: no access to this project"
                }), 403

            access_hierarchy = {
                AccessLevel.VIEWER: 0,
                AccessLevel.EDITOR: 1,
                AccessLevel.OWNER: 2,
            }

            required_level = access_hierarchy.get(access_level, 0)
            user_level = access_hierarchy.get(user_access, 0)

            if user_level < required_level:
                return jsonify({
                    "success": False,
                    "error": f"Access denied: requires {access_level.value} access"
                }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


def _get_current_user_id() -> Optional[str]:
    """Get current user ID from request context."""
    from flask import session

    user_id = session.get('user_id')
    if user_id:
        return str(user_id)

    api_key = request.headers.get('X-API-Key')
    if api_key:
        db_manager = current_app.extensions.get('db_manager')
        if db_manager:
            from codeloom.core.db.models import User
            with db_manager.get_session() as db_session:
                user = db_session.query(User).filter(User.api_key == api_key).first()
                if user:
                    return str(user.user_id)

    user_id = request.headers.get('X-User-ID')
    if user_id:
        return user_id

    if request.method in ['POST', 'PUT', 'PATCH']:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id') or data.get('userId')
        if user_id:
            return str(user_id)

    user_id = request.args.get('user_id') or request.args.get('userId')
    if user_id:
        return user_id

    return None


# ========== Inline Access Check Helpers ==========

def check_project_access(
    user_id: str,
    project_id: str,
    access_level: AccessLevel = AccessLevel.VIEWER
) -> tuple[bool, Optional[str]]:
    """Check if user has access to a project."""
    import os
    if os.getenv("RBAC_STRICT_MODE", "false").lower() != "true":
        return True, None

    if not user_id:
        return False, "Authentication required"

    rbac = get_rbac_service()
    user_access = rbac.get_project_access_level(user_id, project_id)

    if not user_access:
        return False, "Access denied: no access to this project"

    access_hierarchy = {
        AccessLevel.VIEWER: 0,
        AccessLevel.EDITOR: 1,
        AccessLevel.OWNER: 2,
    }

    required_level = access_hierarchy.get(access_level, 0)
    user_level = access_hierarchy.get(user_access, 0)

    if user_level < required_level:
        return False, f"Access denied: requires {access_level.value} access"

    return True, None


def check_multi_project_access(
    user_id: str,
    project_ids: List[str],
    access_level: AccessLevel = AccessLevel.VIEWER
) -> tuple[bool, Optional[str], List[str]]:
    """Check if user has access to multiple projects."""
    import os
    if os.getenv("RBAC_STRICT_MODE", "false").lower() != "true":
        return True, None, project_ids

    if not user_id:
        return False, "Authentication required", []

    rbac = get_rbac_service()
    accessible = []
    denied = []

    access_hierarchy = {
        AccessLevel.VIEWER: 0,
        AccessLevel.EDITOR: 1,
        AccessLevel.OWNER: 2,
    }
    required_level = access_hierarchy.get(access_level, 0)

    for project_id in project_ids:
        user_access = rbac.get_project_access_level(user_id, project_id)
        if user_access:
            user_level = access_hierarchy.get(user_access, 0)
            if user_level >= required_level:
                accessible.append(project_id)
            else:
                denied.append(project_id)
        else:
            denied.append(project_id)

    if not accessible:
        return False, "Access denied: no access to any of the requested projects", []

    if denied:
        logger.warning(f"User {user_id} denied access to projects: {denied}")

    return True, None, accessible
