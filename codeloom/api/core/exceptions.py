"""Domain exceptions for API error handling.

These exceptions can be caught by the global error handler middleware
and converted to appropriate HTTP responses.

Usage:
    from dbnotebook.api.core.exceptions import ValidationError, NotFoundError

    def get_notebook(notebook_id):
        notebook = db.query(Notebook).get(notebook_id)
        if not notebook:
            raise NotFoundError("Notebook", notebook_id)
        return notebook

    def create_notebook(data):
        if not data.get("name"):
            raise ValidationError("Name is required")
"""

from typing import Any, Dict, List, Optional, Union


class DBNotebookError(Exception):
    """Base exception for all DBNotebook application errors.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code to return
        details: Additional error details (optional)
    """

    status_code: int = 500
    default_message: str = "An error occurred"

    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "success": False,
            "error": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(DBNotebookError):
    """Raised when request validation fails.

    HTTP Status: 400 Bad Request

    Examples:
        raise ValidationError("Email is required")
        raise ValidationError("Invalid format", details={"field": "email"})
        raise ValidationError(["Name is required", "Email is invalid"])
    """

    status_code = 400
    default_message = "Validation failed"

    def __init__(
        self,
        message: Optional[Union[str, List[str]]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if isinstance(message, list):
            message = "; ".join(message)
        super().__init__(message, details)


class AuthenticationError(DBNotebookError):
    """Raised when authentication fails.

    HTTP Status: 401 Unauthorized

    Examples:
        raise AuthenticationError()  # Uses default message
        raise AuthenticationError("Invalid API key")
        raise AuthenticationError("Token expired")
    """

    status_code = 401
    default_message = "Authentication required"


class AuthorizationError(DBNotebookError):
    """Raised when user lacks permission for an action.

    HTTP Status: 403 Forbidden

    Examples:
        raise AuthorizationError()
        raise AuthorizationError("Admin access required")
        raise AuthorizationError("You don't have access to this notebook")
    """

    status_code = 403
    default_message = "Access denied"


class NotFoundError(DBNotebookError):
    """Raised when a requested resource doesn't exist.

    HTTP Status: 404 Not Found

    Examples:
        raise NotFoundError("Notebook")
        raise NotFoundError("Notebook", "abc-123")
        raise NotFoundError("User with email user@example.com")
    """

    status_code = 404
    default_message = "Resource not found"

    def __init__(
        self,
        resource: str = "Resource",
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if resource_id:
            message = f"{resource} not found: {resource_id}"
        else:
            message = f"{resource} not found"
        super().__init__(message, details)


class ConflictError(DBNotebookError):
    """Raised when there's a conflict with existing data.

    HTTP Status: 409 Conflict

    Examples:
        raise ConflictError("Notebook with this name already exists")
        raise ConflictError("Username is already taken")
    """

    status_code = 409
    default_message = "Resource conflict"


class RateLimitError(DBNotebookError):
    """Raised when rate limit is exceeded.

    HTTP Status: 429 Too Many Requests

    Examples:
        raise RateLimitError()
        raise RateLimitError("API rate limit exceeded. Try again in 60 seconds.")
    """

    status_code = 429
    default_message = "Rate limit exceeded"

    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class ServiceUnavailableError(DBNotebookError):
    """Raised when a required service is unavailable.

    HTTP Status: 503 Service Unavailable

    Examples:
        raise ServiceUnavailableError("Database connection failed")
        raise ServiceUnavailableError("LLM service is temporarily unavailable")
    """

    status_code = 503
    default_message = "Service temporarily unavailable"


class ExternalServiceError(DBNotebookError):
    """Raised when an external service call fails.

    HTTP Status: 502 Bad Gateway

    Examples:
        raise ExternalServiceError("OpenAI API error")
        raise ExternalServiceError("Embedding service timeout")
    """

    status_code = 502
    default_message = "External service error"


class ConfigurationError(DBNotebookError):
    """Raised when there's a configuration issue.

    HTTP Status: 500 Internal Server Error

    Examples:
        raise ConfigurationError("Missing required environment variable: OPENAI_API_KEY")
    """

    status_code = 500
    default_message = "Configuration error"
