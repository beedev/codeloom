"""RBAC (Role-Based Access Control) Service.

Provides access control for all features:
- API endpoints (via API key or user authentication)
- Notebook Chat (notebook-level access)
- SQL Chat (connection-level access)

Access Levels:
- owner: Full control including delete and share
- editor: Can edit documents and chat
- viewer: Read-only access

Permissions:
- manage_users: Create, edit, delete users
- manage_roles: Create, edit, delete roles
- manage_notebooks: Create notebooks for any user
- manage_connections: Create connections for any user
- view_all: View any notebook/connection
- edit_all: Edit any notebook/connection
- delete_all: Delete any notebook/connection
- create_notebook: Create notebooks for self
- create_connection: Create connections for self
- view_assigned: View assigned notebooks/connections
- edit_assigned: Edit assigned notebooks/connections
"""

import logging
from enum import Enum
from functools import wraps
from typing import List, Optional, Set
from uuid import UUID

from flask import request, jsonify, g, current_app
from sqlalchemy.orm import Session

from dbnotebook.core.db.models import (
    Role,
    UserRole,
    NotebookAccess,
    SQLConnectionAccess,
    Notebook,
    DatabaseConnection,
    User,
)

logger = logging.getLogger(__name__)


class AccessLevel(str, Enum):
    """Access levels for resources."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    USER = "user"  # For SQL connections (same as editor)


class Permission(str, Enum):
    """System permissions."""
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_NOTEBOOKS = "manage_notebooks"
    MANAGE_CONNECTIONS = "manage_connections"
    VIEW_ALL = "view_all"
    EDIT_ALL = "edit_all"
    DELETE_ALL = "delete_all"

    # User permissions
    CREATE_NOTEBOOK = "create_notebook"
    CREATE_CONNECTION = "create_connection"
    VIEW_ASSIGNED = "view_assigned"
    EDIT_ASSIGNED = "edit_assigned"


class RBACService:
    """Role-Based Access Control service.

    Provides methods to check and enforce access control across all features.
    """

    def __init__(self, db_session: Session):
        """Initialize RBAC service.

        Args:
            db_session: SQLAlchemy database session
        """
        self._session = db_session

    # ========== Role Management ==========

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for a user.

        Args:
            user_id: User ID

        Returns:
            List of Role objects
        """
        user_roles = self._session.query(UserRole).filter(
            UserRole.user_id == UUID(user_id)
        ).all()
        return [ur.role for ur in user_roles]

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user (from all roles).

        Args:
            user_id: User ID

        Returns:
            Set of permission strings
        """
        roles = self.get_user_roles(user_id)
        permissions = set()
        for role in roles:
            if role.permissions:
                permissions.update(role.permissions)
        return permissions

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a specific permission.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if user has permission
        """
        permissions = self.get_user_permissions(user_id)
        return permission.value in permissions

    def is_admin(self, user_id: str) -> bool:
        """Check if user is an admin.

        Args:
            user_id: User ID

        Returns:
            True if user has admin role
        """
        roles = self.get_user_roles(user_id)
        return any(role.name == "admin" for role in roles)

    def assign_role(
        self,
        user_id: str,
        role_name: str,
        assigned_by: Optional[str] = None
    ) -> bool:
        """Assign a role to a user.

        Args:
            user_id: User ID
            role_name: Name of role to assign
            assigned_by: User ID of assigner

        Returns:
            True if successful
        """
        role = self._session.query(Role).filter(Role.name == role_name).first()
        if not role:
            logger.warning(f"Role not found: {role_name}")
            return False

        # Check if already assigned
        existing = self._session.query(UserRole).filter(
            UserRole.user_id == UUID(user_id),
            UserRole.role_id == role.role_id
        ).first()

        if existing:
            return True  # Already has role

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
        """Remove a role from a user.

        Args:
            user_id: User ID
            role_name: Name of role to remove

        Returns:
            True if successful
        """
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

    # ========== Notebook Access ==========

    def get_notebook_access_level(
        self,
        user_id: str,
        notebook_id: str
    ) -> Optional[AccessLevel]:
        """Get user's access level for a notebook.

        Args:
            user_id: User ID
            notebook_id: Notebook ID

        Returns:
            AccessLevel or None if no access
        """
        # Check if user is admin (has access to all)
        if self.has_permission(user_id, Permission.VIEW_ALL):
            return AccessLevel.OWNER

        # Check if user owns the notebook
        notebook = self._session.query(Notebook).filter(
            Notebook.notebook_id == UUID(notebook_id)
        ).first()

        if notebook and str(notebook.user_id) == user_id:
            return AccessLevel.OWNER

        # Check explicit access grant
        access = self._session.query(NotebookAccess).filter(
            NotebookAccess.notebook_id == UUID(notebook_id),
            NotebookAccess.user_id == UUID(user_id)
        ).first()

        if access:
            return AccessLevel(access.access_level)

        return None

    def can_view_notebook(self, user_id: str, notebook_id: str) -> bool:
        """Check if user can view a notebook.

        Args:
            user_id: User ID
            notebook_id: Notebook ID

        Returns:
            True if user can view
        """
        access_level = self.get_notebook_access_level(user_id, notebook_id)
        return access_level is not None

    def can_edit_notebook(self, user_id: str, notebook_id: str) -> bool:
        """Check if user can edit a notebook.

        Args:
            user_id: User ID
            notebook_id: Notebook ID

        Returns:
            True if user can edit
        """
        access_level = self.get_notebook_access_level(user_id, notebook_id)
        return access_level in (AccessLevel.OWNER, AccessLevel.EDITOR)

    def can_delete_notebook(self, user_id: str, notebook_id: str) -> bool:
        """Check if user can delete a notebook.

        Args:
            user_id: User ID
            notebook_id: Notebook ID

        Returns:
            True if user can delete
        """
        access_level = self.get_notebook_access_level(user_id, notebook_id)
        return access_level == AccessLevel.OWNER

    def grant_notebook_access(
        self,
        notebook_id: str,
        user_id: str,
        access_level: AccessLevel,
        granted_by: Optional[str] = None
    ) -> bool:
        """Grant a user access to a notebook.

        Args:
            notebook_id: Notebook ID
            user_id: User ID to grant access
            access_level: Level of access to grant
            granted_by: User ID of granter

        Returns:
            True if successful
        """
        # Check if already has access
        existing = self._session.query(NotebookAccess).filter(
            NotebookAccess.notebook_id == UUID(notebook_id),
            NotebookAccess.user_id == UUID(user_id)
        ).first()

        if existing:
            existing.access_level = access_level.value
            existing.granted_by = UUID(granted_by) if granted_by else None
        else:
            access = NotebookAccess(
                notebook_id=UUID(notebook_id),
                user_id=UUID(user_id),
                access_level=access_level.value,
                granted_by=UUID(granted_by) if granted_by else None
            )
            self._session.add(access)

        self._session.commit()
        logger.info(f"Granted {access_level.value} access to notebook {notebook_id} for user {user_id}")
        return True

    def revoke_notebook_access(self, notebook_id: str, user_id: str) -> bool:
        """Revoke a user's access to a notebook.

        Args:
            notebook_id: Notebook ID
            user_id: User ID to revoke access

        Returns:
            True if successful
        """
        access = self._session.query(NotebookAccess).filter(
            NotebookAccess.notebook_id == UUID(notebook_id),
            NotebookAccess.user_id == UUID(user_id)
        ).first()

        if access:
            self._session.delete(access)
            self._session.commit()
            logger.info(f"Revoked notebook access for user {user_id} from notebook {notebook_id}")

        return True

    def list_notebook_users(self, notebook_id: str) -> List[dict]:
        """List all users with access to a notebook.

        Args:
            notebook_id: Notebook ID

        Returns:
            List of user access info dicts
        """
        accesses = self._session.query(NotebookAccess).filter(
            NotebookAccess.notebook_id == UUID(notebook_id)
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

    # ========== SQL Connection Access ==========

    def get_sql_connection_access_level(
        self,
        user_id: str,
        connection_id: str
    ) -> Optional[AccessLevel]:
        """Get user's access level for a SQL connection.

        Args:
            user_id: User ID
            connection_id: Connection ID

        Returns:
            AccessLevel or None if no access
        """
        # Check if user is admin
        if self.has_permission(user_id, Permission.VIEW_ALL):
            return AccessLevel.OWNER

        # Check if user owns the connection
        connection = self._session.query(DatabaseConnection).filter(
            DatabaseConnection.id == UUID(connection_id)
        ).first()

        if connection and connection.user_id == user_id:
            return AccessLevel.OWNER

        # Check explicit access grant
        access = self._session.query(SQLConnectionAccess).filter(
            SQLConnectionAccess.connection_id == UUID(connection_id),
            SQLConnectionAccess.user_id == UUID(user_id)
        ).first()

        if access:
            return AccessLevel(access.access_level)

        return None

    def can_query_connection(self, user_id: str, connection_id: str) -> bool:
        """Check if user can query a SQL connection.

        Args:
            user_id: User ID
            connection_id: Connection ID

        Returns:
            True if user can query
        """
        access_level = self.get_sql_connection_access_level(user_id, connection_id)
        return access_level in (AccessLevel.OWNER, AccessLevel.USER, AccessLevel.EDITOR)

    def can_delete_connection(self, user_id: str, connection_id: str) -> bool:
        """Check if user can delete a SQL connection.

        Args:
            user_id: User ID
            connection_id: Connection ID

        Returns:
            True if user can delete
        """
        access_level = self.get_sql_connection_access_level(user_id, connection_id)
        return access_level == AccessLevel.OWNER

    def grant_sql_connection_access(
        self,
        connection_id: str,
        user_id: str,
        access_level: AccessLevel,
        granted_by: Optional[str] = None
    ) -> bool:
        """Grant a user access to a SQL connection.

        Args:
            connection_id: Connection ID
            user_id: User ID to grant access
            access_level: Level of access to grant
            granted_by: User ID of granter

        Returns:
            True if successful
        """
        # Check if already has access
        existing = self._session.query(SQLConnectionAccess).filter(
            SQLConnectionAccess.connection_id == UUID(connection_id),
            SQLConnectionAccess.user_id == UUID(user_id)
        ).first()

        if existing:
            existing.access_level = access_level.value
            existing.granted_by = UUID(granted_by) if granted_by else None
        else:
            access = SQLConnectionAccess(
                connection_id=UUID(connection_id),
                user_id=UUID(user_id),
                access_level=access_level.value,
                granted_by=UUID(granted_by) if granted_by else None
            )
            self._session.add(access)

        self._session.commit()
        logger.info(f"Granted {access_level.value} access to connection {connection_id} for user {user_id}")
        return True

    def revoke_sql_connection_access(self, connection_id: str, user_id: str) -> bool:
        """Revoke a user's access to a SQL connection.

        Args:
            connection_id: Connection ID
            user_id: User ID to revoke access

        Returns:
            True if successful
        """
        access = self._session.query(SQLConnectionAccess).filter(
            SQLConnectionAccess.connection_id == UUID(connection_id),
            SQLConnectionAccess.user_id == UUID(user_id)
        ).first()

        if access:
            self._session.delete(access)
            self._session.commit()
            logger.info(f"Revoked SQL connection access for user {user_id} from connection {connection_id}")

        return True


# ========== Flask Decorators ==========

def get_rbac_service():
    """Get RBAC service from Flask g object or create new one."""
    if not hasattr(g, 'rbac_service') or g.rbac_service is None:
        db_manager = current_app.extensions.get('db_manager')
        if not db_manager:
            raise RuntimeError("Database manager not available in app extensions")
        # Create a new session for this request context
        session = db_manager.SessionLocal()
        g.rbac_service = RBACService(session)
        g.rbac_session = session  # Store for cleanup in teardown
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
    """Decorator to require a specific permission.

    Usage:
        @app.route('/admin/users', methods=['GET'])
        @require_permission(Permission.MANAGE_USERS)
        def list_users():
            ...
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

            rbac = get_rbac_service()
            if not rbac.has_permission(user_id, permission):
                return jsonify({
                    "success": False,
                    "error": f"Permission denied: {permission.value}"
                }), 403

            return f(*args, **kwargs)
        return decorated
    return decorator


def require_notebook_access(access_level: AccessLevel = AccessLevel.VIEWER):
    """Decorator to require notebook access.

    Expects notebook_id in route params or request body.

    Usage:
        @app.route('/api/notebooks/<notebook_id>/chat', methods=['POST'])
        @require_notebook_access(AccessLevel.EDITOR)
        def chat_in_notebook(notebook_id):
            ...
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

            # Get notebook_id from route params or request body
            notebook_id = kwargs.get('notebook_id')
            if not notebook_id:
                data = request.get_json(silent=True) or {}
                notebook_id = data.get('notebook_id') or data.get('notebookId')

            if not notebook_id:
                return jsonify({
                    "success": False,
                    "error": "notebook_id is required"
                }), 400

            rbac = get_rbac_service()
            user_access = rbac.get_notebook_access_level(user_id, notebook_id)

            if not user_access:
                return jsonify({
                    "success": False,
                    "error": "Access denied: no access to this notebook"
                }), 403

            # Check access level hierarchy
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


def require_sql_connection_access(access_level: AccessLevel = AccessLevel.USER):
    """Decorator to require SQL connection access.

    Expects connection_id in route params or request body.

    Usage:
        @app.route('/api/sql-chat/query/<session_id>', methods=['POST'])
        @require_sql_connection_access(AccessLevel.USER)
        def execute_query(session_id):
            ...
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

            # Get connection_id from route params or request body
            connection_id = kwargs.get('connection_id')
            if not connection_id:
                data = request.get_json(silent=True) or {}
                connection_id = data.get('connection_id') or data.get('connectionId')

            if not connection_id:
                return jsonify({
                    "success": False,
                    "error": "connection_id is required"
                }), 400

            rbac = get_rbac_service()
            user_access = rbac.get_sql_connection_access_level(user_id, connection_id)

            if not user_access:
                return jsonify({
                    "success": False,
                    "error": "Access denied: no access to this connection"
                }), 403

            # Check access level hierarchy
            access_hierarchy = {
                AccessLevel.VIEWER: 0,
                AccessLevel.USER: 1,
                AccessLevel.EDITOR: 1,  # Same as USER for connections
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
    """Get current user ID from request context.

    Checks in order:
    1. Flask session (from login)
    2. X-API-Key header (API key authentication)
    3. X-User-ID header (explicit user override)
    4. Request body user_id/userId
    5. Query parameter user_id
    6. Default user ID (ONLY for backward compatibility with non-auth endpoints)

    Returns:
        User ID string or None
    """
    from flask import session

    # Check Flask session (from login)
    user_id = session.get('user_id')
    if user_id:
        return str(user_id)

    # Check API key authentication
    api_key = request.headers.get('X-API-Key')
    if api_key:
        # Look up user by API key
        db_manager = current_app.extensions.get('db_manager')
        if db_manager:
            from dbnotebook.core.db.models import User
            with db_manager.get_session() as db_session:
                user = db_session.query(User).filter(User.api_key == api_key).first()
                if user:
                    return str(user.user_id)

    # Check X-User-ID header (explicit override, useful for testing)
    user_id = request.headers.get('X-User-ID')
    if user_id:
        return user_id

    # Check request body
    if request.method in ['POST', 'PUT', 'PATCH']:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id') or data.get('userId')
        if user_id:
            return str(user_id)

    # Check query params
    user_id = request.args.get('user_id') or request.args.get('userId')
    if user_id:
        return user_id

    # No user identified - return None (will trigger 401)
    # NOTE: For backwards compatibility with non-auth endpoints, this could return DEFAULT_USER_ID
    # but for security, we now require explicit authentication
    return None


# ========== Inline Access Check Helpers ==========

def check_notebook_access(
    user_id: str,
    notebook_id: str,
    access_level: AccessLevel = AccessLevel.VIEWER
) -> tuple[bool, Optional[str]]:
    """Check if user has access to a notebook.

    Use this for inline access checking in routes.

    Args:
        user_id: User ID
        notebook_id: Notebook ID
        access_level: Required access level

    Returns:
        Tuple of (has_access: bool, error_message: Optional[str])
    """
    import os
    # Skip RBAC check if strict mode is disabled
    if os.getenv("RBAC_STRICT_MODE", "false").lower() != "true":
        return True, None

    if not user_id:
        return False, "Authentication required"

    rbac = get_rbac_service()
    user_access = rbac.get_notebook_access_level(user_id, notebook_id)

    if not user_access:
        return False, "Access denied: no access to this notebook"

    # Check access level hierarchy
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


def check_multi_notebook_access(
    user_id: str,
    notebook_ids: List[str],
    access_level: AccessLevel = AccessLevel.VIEWER
) -> tuple[bool, Optional[str], List[str]]:
    """Check if user has access to multiple notebooks.

    Use this for routes that query multiple notebooks.

    Args:
        user_id: User ID
        notebook_ids: List of Notebook IDs
        access_level: Required access level

    Returns:
        Tuple of (has_access: bool, error_message: Optional[str], accessible_notebooks: List[str])
    """
    import os
    # Skip RBAC check if strict mode is disabled
    if os.getenv("RBAC_STRICT_MODE", "false").lower() != "true":
        return True, None, notebook_ids

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

    for notebook_id in notebook_ids:
        user_access = rbac.get_notebook_access_level(user_id, notebook_id)
        if user_access:
            user_level = access_hierarchy.get(user_access, 0)
            if user_level >= required_level:
                accessible.append(notebook_id)
            else:
                denied.append(notebook_id)
        else:
            denied.append(notebook_id)

    if not accessible:
        return False, f"Access denied: no access to any of the requested notebooks", []

    if denied:
        logger.warning(f"User {user_id} denied access to notebooks: {denied}")

    return True, None, accessible


def check_sql_connection_access(
    user_id: str,
    connection_id: str,
    access_level: AccessLevel = AccessLevel.USER
) -> tuple[bool, Optional[str]]:
    """Check if user has access to a SQL connection.

    Use this for inline access checking in SQL Chat routes.

    Args:
        user_id: User ID
        connection_id: Connection ID
        access_level: Required access level

    Returns:
        Tuple of (has_access: bool, error_message: Optional[str])
    """
    import os
    # Skip RBAC check if strict mode is disabled
    if os.getenv("RBAC_STRICT_MODE", "false").lower() != "true":
        return True, None

    if not user_id:
        return False, "Authentication required"

    rbac = get_rbac_service()
    user_access = rbac.get_sql_connection_access_level(user_id, connection_id)

    if not user_access:
        return False, "Access denied: no access to this connection"

    # Check access level hierarchy
    access_hierarchy = {
        AccessLevel.VIEWER: 0,
        AccessLevel.USER: 1,
        AccessLevel.EDITOR: 1,
        AccessLevel.OWNER: 2,
    }

    required_level = access_hierarchy.get(access_level, 0)
    user_level = access_hierarchy.get(user_access, 0)

    if user_level < required_level:
        return False, f"Access denied: requires {access_level.value} access"

    return True, None
