"""
Custom middleware for the Solvix AI Engine.

Provides request tracing, error handling, and authentication capabilities.
"""

import logging
import time
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

logger = logging.getLogger(__name__)

# Context variable to store request ID for access anywhere in the request lifecycle
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique ID to each request.

    The request ID is:
    - Generated as a UUID4 if not provided by the client
    - Stored in a context variable for access throughout the request
    - Added to the response headers as X-Request-ID
    - Logged with each request for tracing

    Clients can optionally provide their own request ID via the
    X-Request-ID header for end-to-end tracing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Get request ID from header or generate new one
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid4())

        # Store in context variable for access anywhere
        token = request_id_var.set(request_id)

        # Add to request state for easy access in route handlers
        request.state.request_id = request_id

        # Track request timing
        start_time = time.perf_counter()

        try:
            # Log incoming request
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )

            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log completed request
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return response

        except Exception as e:
            # Calculate duration even for errors
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        finally:
            # Reset context variable
            request_id_var.reset(token)


# Paths that bypass authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces service-to-service authentication.

    When SERVICE_AUTH_TOKEN is set, all requests (except health/docs)
    must include a valid Authorization: Bearer <token> header.

    If SERVICE_AUTH_TOKEN is not configured, authentication is disabled
    (allows local development without a token).
    """

    def __init__(self, app, token: str | None = None):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth if no token configured (local dev)
        if not self.token:
            return await call_next(request)

        # Allow public paths without auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return StarletteJSONResponse(
                status_code=401,
                content={"error": "Missing or invalid Authorization header"},
            )

        provided_token = auth_header[7:]  # len("Bearer ") == 7
        if provided_token != self.token:
            return StarletteJSONResponse(
                status_code=401,
                content={"error": "Invalid service token"},
            )

        return await call_next(request)
