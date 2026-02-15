"""API Core - Shared utilities for API routes.

This package provides:
- Unified response builders (success_response, error_response)
- Domain exceptions (ValidationError, NotFoundError, etc.)
- Auth decorators (@require_api_key, @require_session)
- Error handling middleware

Usage:
    from dbnotebook.api.core import success_response, error_response
    from dbnotebook.api.core.exceptions import ValidationError
    from dbnotebook.api.core.decorators import require_api_key
"""

from .response import (
    success_response,
    error_response,
    validation_error,
    not_found,
    unauthorized,
    forbidden,
    service_unavailable,
    rate_limited,
    APIResponse,
)

from .exceptions import (
    DBNotebookError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

__all__ = [
    # Response utilities
    "success_response",
    "error_response",
    "validation_error",
    "not_found",
    "unauthorized",
    "forbidden",
    "service_unavailable",
    "rate_limited",
    "APIResponse",
    # Exceptions
    "DBNotebookError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
]
