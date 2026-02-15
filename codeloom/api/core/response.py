"""Unified API response utilities.

Provides consistent response format across all API endpoints:
- success: boolean indicating success/failure
- data/error: response payload or error message
- Optional metadata

Usage:
    from dbnotebook.api.core import success_response, error_response

    @app.route("/api/example")
    def example():
        # Success
        return success_response({"items": [...]})

        # Error
        return error_response("Not found", 404)

        # Success with extra fields
        return success_response(data=result, execution_time_ms=123)
"""

from dataclasses import dataclass, field
from flask import jsonify
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class APIResponse:
    """Structured API response container.

    Attributes:
        success: Whether the operation succeeded
        data: Response payload (for successful responses)
        error: Error message (for failed responses)
        metadata: Additional response metadata
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"success": self.success}

        if self.data is not None:
            if isinstance(self.data, dict):
                result.update(self.data)
            else:
                result["data"] = self.data

        if self.error:
            result["error"] = self.error

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_response(self, status_code: int = 200) -> Tuple:
        """Convert to Flask JSON response tuple."""
        return jsonify(self.to_dict()), status_code


def success_response(
    data: Optional[Any] = None,
    status_code: int = 200,
    **kwargs
) -> Tuple:
    """Return a success JSON response.

    Args:
        data: Response payload (dict is merged, other types go under 'data' key)
        status_code: HTTP status code (default: 200)
        **kwargs: Additional fields to include in response

    Returns:
        Tuple of (Response, status_code)

    Examples:
        # Simple success
        return success_response()
        # {"success": true}

        # With data dict (merged)
        return success_response({"items": [1, 2, 3]})
        # {"success": true, "items": [1, 2, 3]}

        # With non-dict data
        return success_response(data=[1, 2, 3])
        # {"success": true, "data": [1, 2, 3]}

        # With extra fields
        return success_response(data=result, execution_time_ms=123)
        # {"success": true, "data": result, "execution_time_ms": 123}
    """
    response = {"success": True}

    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response["data"] = data

    response.update(kwargs)

    return jsonify(response), status_code


def error_response(
    message: str,
    code: int = 500,
    **kwargs
) -> Tuple:
    """Return an error JSON response.

    Args:
        message: Error message
        code: HTTP status code (default: 500)
        **kwargs: Additional fields to include in response

    Returns:
        Tuple of (Response, status_code)

    Examples:
        return error_response("Not found", 404)
        # {"success": false, "error": "Not found"}

        return error_response("Validation failed", 400, fields=["name"])
        # {"success": false, "error": "Validation failed", "fields": ["name"]}
    """
    response = {
        "success": False,
        "error": message
    }
    response.update(kwargs)

    return jsonify(response), code


def validation_error(
    errors: Union[str, List[str]],
    **kwargs
) -> Tuple:
    """Return a 400 validation error response.

    Args:
        errors: Single error message or list of error messages
        **kwargs: Additional fields to include in response

    Returns:
        Tuple of (Response, 400)

    Examples:
        return validation_error("Name is required")
        return validation_error(["Name is required", "Email is invalid"])
    """
    if isinstance(errors, list):
        message = "; ".join(errors)
    else:
        message = errors

    return error_response(message, 400, **kwargs)


def not_found(resource: str, resource_id: Optional[str] = None) -> Tuple:
    """Return a 404 not found response.

    Args:
        resource: Type of resource (e.g., "Notebook", "User")
        resource_id: Optional ID of the resource

    Returns:
        Tuple of (Response, 404)

    Examples:
        return not_found("Notebook")
        # {"success": false, "error": "Notebook not found"}

        return not_found("Notebook", "abc-123")
        # {"success": false, "error": "Notebook not found: abc-123"}
    """
    if resource_id:
        message = f"{resource} not found: {resource_id}"
    else:
        message = f"{resource} not found"

    return error_response(message, 404)


def unauthorized(message: str = "Authentication required") -> Tuple:
    """Return a 401 unauthorized response.

    Args:
        message: Error message (default: "Authentication required")

    Returns:
        Tuple of (Response, 401)
    """
    return error_response(message, 401)


def forbidden(message: str = "Access denied") -> Tuple:
    """Return a 403 forbidden response.

    Args:
        message: Error message (default: "Access denied")

    Returns:
        Tuple of (Response, 403)
    """
    return error_response(message, 403)


def service_unavailable(message: str = "Service temporarily unavailable") -> Tuple:
    """Return a 503 service unavailable response.

    Args:
        message: Error message

    Returns:
        Tuple of (Response, 503)
    """
    return error_response(message, 503)


def rate_limited(message: str = "Rate limit exceeded") -> Tuple:
    """Return a 429 rate limit response.

    Args:
        message: Error message

    Returns:
        Tuple of (Response, 429)
    """
    return error_response(message, 429)
