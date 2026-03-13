"""
Health check API endpoints.

GET /ping   -- Simple liveness check for Docker / load balancer probes.
    Zero cost, returns immediately with uptime.
GET /health -- Full health check that makes actual LLM API calls to
    verify provider availability.  Expensive -- use sparingly
    (e.g., every 15 minutes from monitoring).
"""

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.models.responses import HealthResponse
from src.llm.factory import llm_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Track service start time for uptime calculation
_start_time = time.time()


class PingResponse(BaseModel):
    """Simple ping response for liveness checks.

    Attributes:
        status: Always "ok" if the service is running.
        uptime_seconds: Seconds since the FastAPI process started.
    """

    status: str = "ok"
    uptime_seconds: float


@router.get("/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    """
    Simple liveness check - does NOT call LLM APIs.

    Use this for Docker health checks to avoid expensive API calls.
    Returns immediately with basic service status.
    """
    uptime = time.time() - _start_time
    return PingResponse(status="ok", uptime_seconds=round(uptime, 2))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Full health check with LLM provider verification.

    WARNING: This endpoint makes actual API calls to Gemini and OpenAI
    to verify they are responding. Use sparingly (e.g., every 15 minutes)
    to avoid unnecessary API costs.

    Returns:
        - status: healthy/degraded/unhealthy
        - version: API version
        - provider: primary LLM provider (gemini/openai)
        - model: primary model name
        - fallback_provider: fallback LLM provider
        - fallback_model: fallback model name
        - fallback_count: number of times fallback was used
        - model_available: whether primary LLM is responding
        - fallback_available: whether fallback is healthy
        - uptime_seconds: API uptime
    """
    logger.debug("Running full health check (includes LLM API calls)")
    uptime = time.time() - _start_time

    # Check LLM health (makes actual API calls)
    llm_health = await llm_client.health_check()
    primary_healthy = llm_health["primary"]["status"] == "healthy"
    fallback_status = llm_health["fallback"].get("status", "disabled")

    logger.debug(f"Health check complete: primary={primary_healthy}, fallback={fallback_status}")

    return HealthResponse(
        status="healthy" if primary_healthy else "degraded",
        version="0.1.0",
        provider=llm_client.provider_name,
        model=llm_client.model_name,
        fallback_provider=llm_client.fallback.provider_name if llm_client.fallback else None,
        fallback_model=llm_client.fallback.model_name if llm_client.fallback else None,
        fallback_count=llm_client.fallback_count,
        model_available=primary_healthy,
        fallback_available=fallback_status == "healthy",
        uptime_seconds=round(uptime, 2),
    )
