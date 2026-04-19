"""
Outstanding AI Engine - FastAPI Application

Main entry point for the AI Engine service providing:
- Email classification
- Draft generation
- Gate evaluation

Security:
- Rate limiting via slowapi (prevents DDoS and quota exhaustion)
- CORS configured via settings
- Structured error responses (no sensitive data leakage)
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.errors import ErrorCode, ErrorResponse, OutstandingAIBaseError
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

# =============================================================================
# IDLE SHUTDOWN (ephemeral ECS Fargate)
# =============================================================================
# When IDLE_SHUTDOWN_SECONDS > 0, the engine will gracefully shut down after
# receiving no requests for the configured duration.  A background task checks
# every 30 seconds.  Set to 0 (default) to disable.
IDLE_SHUTDOWN_SECONDS = int(os.environ.get("IDLE_SHUTDOWN_SECONDS", "0"))
_last_request_time: float = time.monotonic()


async def _idle_shutdown_watchdog():
    """Background coroutine that terminates the process after sustained idle."""
    while True:
        await asyncio.sleep(30)
        idle_duration = time.monotonic() - _last_request_time
        if idle_duration >= IDLE_SHUTDOWN_SECONDS:
            logger.info(
                "Idle shutdown triggered: no requests for %.0f seconds (threshold: %d)",
                idle_duration,
                IDLE_SHUTDOWN_SECONDS,
            )
            os.kill(os.getpid(), signal.SIGTERM)
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Outstanding AI Engine")
    logger.info("=" * 60)
    model = settings.model_for_provider()
    logger.info(f"Provider: {settings.llm_provider}, Model: {model}")
    logger.info("Provider readiness: %s", settings.provider_status())
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Debug: {settings.debug}")
    logger.info("Rate limiting: ENABLED")

    # Start idle shutdown watchdog if enabled
    watchdog_task = None
    if IDLE_SHUTDOWN_SECONDS > 0:
        logger.info(f"Idle shutdown: ENABLED ({IDLE_SHUTDOWN_SECONDS}s)")
        watchdog_task = asyncio.create_task(_idle_shutdown_watchdog())
    else:
        logger.info("Idle shutdown: DISABLED")

    yield

    # Cancel watchdog on shutdown
    if watchdog_task is not None:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass


# Create app
app = FastAPI(
    title="Outstanding AI Engine",
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
@app.exception_handler(OutstandingAIBaseError)
async def app_error_handler(request: Request, exc: OutstandingAIBaseError) -> JSONResponse:
    """Handle all Outstanding AI custom exceptions with structured response."""
    logger.error(
        "Outstanding AI application error",
        extra={
            "request_id": get_request_id(),
            "path": request.url.path,
            "error_code": exc.error_code,
            "exception_type": type(exc).__name__,
            "provider": settings.llm_provider,
        },
    )
    error_response = ErrorResponse(
        error=exc.message,
        error_code=exc.error_code,
        details={
            **(exc.details or {}),
            "exception_type": type(exc).__name__,
            "provider": settings.llm_provider,
        },
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with structured response."""
    logger.exception(
        "Unhandled exception",
        extra={
            "request_id": get_request_id(),
            "path": request.url.path,
            "exception_type": type(exc).__name__,
            "provider": settings.llm_provider,
        },
    )
    error_response = ErrorResponse(
        error="An unexpected error occurred",
        error_code=ErrorCode.INTERNAL_ERROR,
        details={
            "exception_type": type(exc).__name__,
            "provider": settings.llm_provider,
            "path": request.url.path,
        },
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode="json"),
    )


# Idle shutdown request tracking middleware
if IDLE_SHUTDOWN_SECONDS > 0:

    @app.middleware("http")
    async def _track_last_request(request: Request, call_next):
        global _last_request_time
        _last_request_time = time.monotonic()
        return await call_next(request)


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
