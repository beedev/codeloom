"""API decorators for authentication, authorization, and request handling.

Provides reusable decorators for:
- API key authentication
- Session-based authentication
- Project access validation
- Request validation

Usage:
    from codeloom.api.core.decorators import require_api_key, require_session

    @app.route("/api/protected")
    @require_api_key
    def protected_endpoint():
        user_id = request.api_user_id  # Set by decorator
        ...

    @app.route("/api/projects/<project_id>")
    @require_project_access
    def project_endpoint(project_id):
        ...
"""

import logging
import os
from functools import wraps
from typing import Callable, Optional, TypeVar, cast

from flask import request, session

from .response import error_response, unauthorized, forbidden

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., any])

# Module-level database manager reference (set during app initialization)
_db_manager = None

# Fallback API key from environment (backward compatibility)
_env_api_key = os.getenv("API_KEY")


def set_db_manager(db_manager) -> None:
    """Set the database manager for decorator use.

    Should be called during app initialization.

    Args:
        db_manager: DatabaseManager instance
    """
    global _db_manager
    _db_manager = db_manager


def validate_api_key(provided_key: str) -> tuple[bool, Optional[str]]:
    """Validate API key against database or environment variable.

    Args:
        provided_key: The API key to validate

    Returns:
        Tuple of (is_valid, user_id or None)
    """
    if not provided_key:
        return False, None

    # Check database first (if db_manager is available)
    if _db_manager:
        try:
            from codeloom.core.db.models import User

            with _db_manager.get_session() as db_session:
                user = db_session.query(User).filter(User.api_key == provided_key).first()
                if user:
                    return True, str(user.user_id)
        except Exception as e:
            logger.warning(f"Database API key lookup failed: {e}")

    # Fallback to environment variable (backward compatibility)
    if _env_api_key and provided_key == _env_api_key:
        from codeloom.core.constants import DEFAULT_USER_ID
        return True, DEFAULT_USER_ID

    return False, None


def require_api_key(f: F) -> F:
    """Decorator to require valid API key for endpoint access.

    Validates the X-API-Key header against:
    1. Database: users.api_key column
    2. Environment: API_KEY variable (fallback)

    Sets request.api_user_id on successful validation.

    Usage:
        @app.route("/api/protected")
        @require_api_key
        def protected():
            user_id = request.api_user_id
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key")

        is_valid, user_id = validate_api_key(provided_key)
        if not is_valid:
            return error_response("Invalid or missing API key", 401)

        # Store validated user_id in request context
        request.api_user_id = user_id
        return f(*args, **kwargs)

    return cast(F, decorated)


def require_session(f: F) -> F:
    """Decorator to require valid Flask session for endpoint access.

    Checks for user_id in Flask session (set during login).

    Sets request.session_user_id on successful validation.

    Usage:
        @app.route("/api/user-data")
        @require_session
        def user_data():
            user_id = request.session_user_id
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = session.get("user_id")

        if not user_id:
            return unauthorized("Please log in to access this resource")

        # Store session user_id in request context
        request.session_user_id = user_id
        return f(*args, **kwargs)

    return cast(F, decorated)


def require_auth(f: F) -> F:
    """Decorator that accepts either API key or session auth.

    Checks API key first, then falls back to session.

    Sets request.auth_user_id on successful validation.

    Usage:
        @app.route("/api/flexible-auth")
        @require_auth
        def flexible():
            user_id = request.auth_user_id
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Try API key first
        provided_key = request.headers.get("X-API-Key")
        if provided_key:
            is_valid, user_id = validate_api_key(provided_key)
            if is_valid:
                request.auth_user_id = user_id
                return f(*args, **kwargs)

        # Fall back to session
        session_user_id = session.get("user_id")
        if session_user_id:
            request.auth_user_id = session_user_id
            return f(*args, **kwargs)

        return unauthorized("Authentication required (API key or session)")

    return cast(F, decorated)


def require_project_access(
    access_level: str = "viewer",
    project_id_param: str = "project_id"
) -> Callable[[F], F]:
    """Decorator factory to validate project access.

    Checks RBAC permissions for the authenticated user.

    Args:
        access_level: Required access level ("viewer", "editor", "owner")
        project_id_param: Name of the route/request param containing project_id

    Usage:
        @app.route("/api/projects/<project_id>")
        @require_auth
        @require_project_access(access_level="viewer")
        def get_project(project_id):
            ...

        @app.route("/api/projects/<project_id>/edit")
        @require_auth
        @require_project_access(access_level="editor")
        def edit_project(project_id):
            ...
    """
    def decorator(f: F) -> F:
        @wraps(f)
        def decorated(*args, **kwargs):
            from codeloom.core.auth import check_project_access, AccessLevel

            # Get project_id from route params or request body
            project_id = kwargs.get(project_id_param)
            if not project_id and request.is_json:
                project_id = request.json.get(project_id_param)
            if not project_id:
                project_id = request.args.get(project_id_param)

            if not project_id:
                return error_response(f"{project_id_param} is required", 400)

            # Get user_id from request context (set by auth decorator)
            user_id = getattr(request, 'auth_user_id', None) or \
                      getattr(request, 'api_user_id', None) or \
                      getattr(request, 'session_user_id', None) or \
                      session.get("user_id")

            if not user_id:
                return unauthorized("Authentication required")

            # Map string access level to enum
            level_map = {
                "viewer": AccessLevel.VIEWER,
                "editor": AccessLevel.EDITOR,
                "owner": AccessLevel.OWNER,
            }
            required_level = level_map.get(access_level, AccessLevel.VIEWER)

            # Check access
            has_access, error_msg = check_project_access(
                user_id=user_id,
                project_id=project_id,
                access_level=required_level
            )

            if not has_access:
                return forbidden(error_msg or "Access denied to this project")

            return f(*args, **kwargs)

        return cast(F, decorated)

    return decorator


def require_admin(f: F) -> F:
    """Decorator to require admin role.

    Checks if the authenticated user has admin role.

    Usage:
        @app.route("/api/admin/users")
        @require_session
        @require_admin
        def list_users():
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get user_id from various sources
        user_id = getattr(request, 'auth_user_id', None) or \
                  getattr(request, 'api_user_id', None) or \
                  getattr(request, 'session_user_id', None) or \
                  session.get("user_id")

        if not user_id:
            return unauthorized("Authentication required")

        # Check admin role
        if not _db_manager:
            logger.error("Database manager not set for admin check")
            return error_response("Server configuration error", 500)

        try:
            from codeloom.core.auth import RBACService

            with _db_manager.get_session() as db_session:
                rbac_service = RBACService(db_session)
                if not rbac_service.has_role(user_id, "admin"):
                    return forbidden("Admin access required")

        except Exception as e:
            logger.error(f"Admin role check failed: {e}")
            return error_response("Authorization check failed", 500)

        return f(*args, **kwargs)

    return cast(F, decorated)


def validate_json(*required_fields: str) -> Callable[[F], F]:
    """Decorator factory to validate required JSON fields.

    Args:
        *required_fields: Names of required fields in request JSON

    Usage:
        @app.route("/api/projects", methods=["POST"])
        @validate_json("name")
        def create_project():
            name = request.json["name"]  # Guaranteed to exist
            ...

        @app.route("/api/query", methods=["POST"])
        @validate_json("project_id", "query")
        def query():
            ...
    """
    def decorator(f: F) -> F:
        @wraps(f)
        def decorated(*args, **kwargs):
            if not request.is_json:
                return error_response("Content-Type must be application/json", 400)

            data = request.get_json() or {}
            missing = [field for field in required_fields if not data.get(field)]

            if missing:
                fields_str = ", ".join(missing)
                return error_response(f"Missing required fields: {fields_str}", 400)

            return f(*args, **kwargs)

        return cast(F, decorated)

    return decorator
