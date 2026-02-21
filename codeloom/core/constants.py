"""Shared constants for DBProject.

This module contains constants that are used across multiple modules
to avoid duplication and ensure consistency.
"""

# =============================================================================
# Default User Configuration
# =============================================================================

# Default admin user ID (UUID format)
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"

# Default admin username
DEFAULT_USERNAME = "admin"

# Default admin email
DEFAULT_EMAIL = "admin@codeloom.local"

# Default admin password (bcrypt hashed in migrations)
DEFAULT_PASSWORD = "admin123"

# Default admin API key (fixed for predictable development/testing)
# Format: dbn_ + 32 hex chars
DEFAULT_API_KEY = "dbn_00000000000000000000000000000001"

# =============================================================================
# Default Project Configuration
# =============================================================================

# Default project ID for backward compatibility
DEFAULT_PROJECT_ID = "00000000-0000-0000-0000-000000000000"

# =============================================================================
# Role Names
# =============================================================================

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_VIEWER = "viewer"

# =============================================================================
# Role Permissions
# =============================================================================

ADMIN_PERMISSIONS = [
    "manage_users",
    "manage_roles",
    "manage_projects",
    "manage_connections",
    "view_all",
    "edit_all",
    "delete_all",
]

USER_PERMISSIONS = [
    "create_project",
    "create_connection",
    "view_assigned",
    "edit_assigned",
]

VIEWER_PERMISSIONS = [
    "view_assigned",
]
