"""Authentication and Authorization module.

Provides:
- Authentication: Login, password management, API key generation
- RBAC (Role-Based Access Control) for all features

RBAC Strict Mode:
    Set RBAC_STRICT_MODE=true to enforce access control.
    When disabled (default), all users have access to all resources.
"""

from .auth_service import AuthService
from .rbac import (
    RBACService,
    AccessLevel,
    Permission,
    require_permission,
    require_notebook_access,
    require_sql_connection_access,
    get_rbac_service,
    check_notebook_access,
    check_multi_notebook_access,
    check_sql_connection_access,
)

__all__ = [
    # Authentication
    "AuthService",
    # RBAC
    "RBACService",
    "AccessLevel",
    "Permission",
    "require_permission",
    "require_notebook_access",
    "require_sql_connection_access",
    "get_rbac_service",
    "check_notebook_access",
    "check_multi_notebook_access",
    "check_sql_connection_access",
]
