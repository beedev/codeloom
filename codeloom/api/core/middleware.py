"""API middleware for error handling, logging, and request processing.

Provides:
- Global error handlers for domain exceptions
- Request logging
- CORS handling

Usage:
    from dbnotebook.api.core.middleware import register_error_handlers

    app = Flask(__name__)
    register_error_handlers(app)
"""

import logging
import traceback
from flask import Flask, jsonify, request

from .exceptions import DBNotebookError

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    """Register global error handlers for the Flask application.

    Converts domain exceptions to appropriate HTTP responses.
    Logs errors for debugging.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(DBNotebookError)
    def handle_app_error(error: DBNotebookError):
        """Handle domain-specific application errors.

        Converts DBNotebookError subclasses to JSON responses with
        appropriate status codes.
        """
        # Log based on severity
        if error.status_code >= 500:
            logger.error(f"{error.__class__.__name__}: {error.message}")
        else:
            logger.warning(f"{error.__class__.__name__}: {error.message}")

        response = error.to_dict()
        return jsonify(response), error.status_code

    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({
            "success": False,
            "error": str(error.description) if hasattr(error, 'description') else "Bad request"
        }), 400

    @app.errorhandler(401)
    def handle_unauthorized(error):
        """Handle 401 Unauthorized errors."""
        return jsonify({
            "success": False,
            "error": str(error.description) if hasattr(error, 'description') else "Unauthorized"
        }), 401

    @app.errorhandler(403)
    def handle_forbidden(error):
        """Handle 403 Forbidden errors."""
        return jsonify({
            "success": False,
            "error": str(error.description) if hasattr(error, 'description') else "Forbidden"
        }), 403

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            "success": False,
            "error": str(error.description) if hasattr(error, 'description') else "Resource not found"
        }), 404

    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handle 429 Too Many Requests errors."""
        return jsonify({
            "success": False,
            "error": str(error.description) if hasattr(error, 'description') else "Rate limit exceeded"
        }), 429

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"Internal error: {error}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        """Handle uncaught exceptions.

        Logs the full traceback for debugging and returns
        a generic error response to the client.
        """
        logger.error(f"Unhandled error: {error}")
        logger.error(traceback.format_exc())

        # Don't expose internal error details in production
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred"
        }), 500


def register_request_logging(app: Flask) -> None:
    """Register request logging middleware.

    Logs incoming requests and response times.

    Args:
        app: Flask application instance
    """
    import time

    @app.before_request
    def log_request_start():
        """Log request start and store start time."""
        request.start_time = time.time()
        if request.endpoint and not request.endpoint.startswith('static'):
            logger.debug(f"→ {request.method} {request.path}")

    @app.after_request
    def log_request_end(response):
        """Log request completion with timing."""
        if hasattr(request, 'start_time'):
            duration_ms = int((time.time() - request.start_time) * 1000)
            if request.endpoint and not request.endpoint.startswith('static'):
                logger.info(
                    f"← {request.method} {request.path} "
                    f"→ {response.status_code} ({duration_ms}ms)"
                )
        return response


def setup_cors(app: Flask, origins: str = "*") -> None:
    """Set up CORS headers for the application.

    Args:
        app: Flask application instance
        origins: Allowed origins (default: "*" for all)
    """

    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to all responses."""
        response.headers["Access-Control-Allow-Origin"] = origins
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-API-Key, X-User-ID"
        )
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def handle_options(path):
        """Handle OPTIONS preflight requests."""
        return '', 204


def register_all_middleware(app: Flask, enable_logging: bool = True) -> None:
    """Register all middleware components.

    Convenience function to set up all middleware at once.

    Args:
        app: Flask application instance
        enable_logging: Whether to enable request logging (default: True)
    """
    register_error_handlers(app)
    if enable_logging:
        register_request_logging(app)
