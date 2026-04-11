"""
Health check API endpoints.

GET /ping       -- Simple liveness check for Docker / load balancer probes.
                   Zero cost, returns immediately with uptime. Public.
GET /health     -- Shallow service health: process up, settings parsed.
                   Zero cost, no LLM calls. Public.
GET /health/llm -- Deep LLM connectivity check. Makes real API calls to
                   Gemini / OpenAI to verify provider availability.
                   Requires service auth — ops-runbook use only.
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


class ShallowHealthResponse(BaseModel):
    """Shallow service health response. No LLM round-trip."""

    status: str = "ok"
    version: str
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


@router.get("/health", response_model=ShallowHealthResponse)
async def health_check() -> ShallowHealthResponse:
    """
    Shallow service health check. No LLM API calls — safe for ECS/LB probes.
    Use /health/llm for end-to-end LLM provider verification.
    """
    uptime = time.time() - _start_time
    return ShallowHealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(uptime, 2),
    )


@router.get("/health/llm", response_model=HealthResponse)
async def llm_health_check() -> HealthResponse:
    """
    Deep LLM connectivity check with real provider API calls.

    WARNING: Makes actual API calls to Gemini and OpenAI which consume
    quota. Requires service auth — not for monitoring probes, only
    ops-runbook use during incident response.
    """
    logger.debug("Running deep LLM health check (burns provider quota)")
    uptime = time.time() - _start_time

    llm_health = await llm_client.health_check()
    primary_healthy = llm_health["primary"]["status"] == "healthy"
    fallback_status = llm_health["fallback"].get("status", "disabled")

    logger.debug(
        f"LLM health check complete: primary={primary_healthy}, fallback={fallback_status}"
    )

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
