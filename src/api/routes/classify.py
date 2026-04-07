"""
Email classification API endpoint.

POST /classify -- Classify an inbound debtor email into one of 23
categories (COOPERATIVE, DISPUTE, PROMISE_TO_PAY, etc.) with confidence
score, extracted intent data, and optional guardrail validation.

Called by the Django backend's ``ai_engine/client.py`` during the
``ai.process_email_classification`` background job.

Security:
    - Rate limited via slowapi (default 100/minute, configurable).
    - Service-to-service auth via Bearer token when
      ``SERVICE_AUTH_TOKEN`` is set.
"""

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.errors import ErrorResponse
from src.api.models.requests import ClassifyRequest
from src.api.models.responses import ClassifyResponse
from src.config.settings import settings
from src.engine.classifier import classifier

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_tenant_key(request: Request) -> str:
    """Rate limit per tenant (X-Tenant-ID header). Falls back to IP for direct callers."""
    return request.headers.get("X-Tenant-ID") or get_remote_address(request)


# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=_get_tenant_key)


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_classify)
async def classify_email(request: Request, classify_request: ClassifyRequest) -> ClassifyResponse:
    """Classify an inbound email from a debtor.

    Accept a ``ClassifyRequest`` containing the email (subject, body,
    from_address) and case context (party, obligations, industry).
    Return the primary classification, confidence, secondary intents,
    extracted data (promise dates, amounts, redirect contacts), and
    guardrail validation results.

    The classification drives downstream side-effects in Django:
    draft discard/regeneration, verification task creation, and
    obligation collection status updates.
    """
    logger.info(f"Classifying email for party: {classify_request.context.party.party_id}")
    result = await classifier.classify(classify_request)
    logger.info(f"Classification: {result.classification} ({result.confidence:.2f})")
    return result
