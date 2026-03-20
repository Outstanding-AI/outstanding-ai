"""
Structured error handling for the Outstanding AI Engine.

Provides custom exceptions and standardized error response models
for consistent API error responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_CLASSIFICATION = "INVALID_CLASSIFICATION"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"

    # LLM errors (5xx)
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_RESPONSE_INVALID = "LLM_RESPONSE_INVALID"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    All API errors return this structure for consistent client handling.
    """

    error: str = Field(..., description="Human-readable error message")
    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details (field errors, etc.)",
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the error occurred",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "LLM returned invalid classification",
                "error_code": "LLM_RESPONSE_INVALID",
                "details": {
                    "classification": "UNKNOWN",
                    "valid_values": ["PROMISE_TO_PAY", "DISPUTE"],
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }


# Custom Exceptions


class OutstandingAIBaseError(Exception):
    """Base exception for all Outstanding AI errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        self.status_code = status_code
        super().__init__(message)


class ValidationError(OutstandingAIBaseError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            status_code=400,
        )


class InvalidClassificationError(OutstandingAIBaseError):
    """Raised when LLM returns an invalid classification."""

    def __init__(self, classification: str, valid_values: list):
        super().__init__(
            message=f"LLM returned invalid classification: '{classification}'",
            error_code=ErrorCode.INVALID_CLASSIFICATION,
            details={"classification": classification, "valid_values": valid_values},
            status_code=500,
        )


class LLMProviderError(OutstandingAIBaseError):
    """Raised when LLM provider fails."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_PROVIDER_ERROR,
            details={"provider": provider} if provider else None,
            status_code=503,
        )


class LLMResponseInvalidError(OutstandingAIBaseError):
    """Raised when LLM response cannot be parsed or validated."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_RESPONSE_INVALID,
            details=details,
            status_code=500,
        )


class LLMTimeoutError(OutstandingAIBaseError):
    """Raised when LLM request times out."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.LLM_TIMEOUT,
            details={"timeout_seconds": timeout_seconds},
            status_code=504,
        )


class LLMRateLimitedError(OutstandingAIBaseError):
    """Raised when LLM provider rate limits the request."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limited by {provider}",
            error_code=ErrorCode.LLM_RATE_LIMITED,
            details={"provider": provider, "retry_after": retry_after},
            status_code=429,
        )
