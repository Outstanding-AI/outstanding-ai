"""
Solvix AI Engine - FastAPI Application

Main entry point for the AI Engine service providing:
- Email classification
- Draft generation
- Gate evaluation

Security:
- Rate limiting via slowapi (prevents DDoS and quota exhaustion)
- CORS configured via settings
- Structured error responses (no sensitive data leakage)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.errors import ErrorCode, ErrorResponse, SolvixBaseError
from src.api.middleware import RequestIDMiddleware, ServiceAuthMiddleware, get_request_id
from src.api.routes import classify, gates, generate, health, persona
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# =============================================================================
# RATE LIMITING CONFIGURATION
# =============================================================================
# Prevents DDoS, API quota exhaustion, and billing abuse
# Rates are per-IP address by default
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Solvix AI Engine")
    logger.info("=" * 60)
    model = settings.gemini_model if settings.llm_provider == "gemini" else settings.openai_model
    logger.info(f"Provider: {settings.llm_provider}, Model: {model}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Debug: {settings.debug}")
    logger.info("Rate limiting: ENABLED")
    yield


# Create app
app = FastAPI(
    title="Solvix AI Engine",
    description="AI-powered email classification and draft generation for debt collection",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach limiter to app state (required by slowapi)
app.state.limiter = limiter

# Add rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Service-to-service auth middleware (outermost — runs before request ID)
app.add_middleware(ServiceAuthMiddleware, token=settings.service_auth_token)

# Request ID middleware
app.add_middleware(RequestIDMiddleware)

# CORS middleware - configured via settings
cors_origins = settings.get_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {cors_origins}")
else:
    logger.warning("CORS disabled - no origins configured and not in debug mode")


# Global exception handler for structured error responses
@app.exception_handler(SolvixBaseError)
async def solvix_error_handler(request: Request, exc: SolvixBaseError) -> JSONResponse:
    """Handle all Solvix custom exceptions with structured response."""
    error_response = ErrorResponse(
        error=exc.message,
        error_code=exc.error_code,
        details=exc.details,
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with structured response."""
    logger.exception(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        error="An unexpected error occurred",
        error_code=ErrorCode.INTERNAL_ERROR,
        details={"exception_type": type(exc).__name__} if settings.debug else None,
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode="json"),
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(classify.router, tags=["Classification"])
app.include_router(generate.router, tags=["Generation"])
app.include_router(gates.router, tags=["Gates"])
app.include_router(persona.router, tags=["Persona"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug
    )
