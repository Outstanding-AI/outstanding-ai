"""
Draft generation API endpoint.

POST /generate-draft -- Generate a collection email draft with subject,
body (containing ``{INVOICE_TABLE}`` placeholder), tone metadata, and
guardrail validation results.

Called by the Django backend's ``ai_engine/client.py`` during the
``ai.generate_draft_for_party`` background job.

Security:
    - Rate limited via slowapi (default 100/minute, configurable).
    - Service-to-service auth via Bearer token when
      ``SERVICE_AUTH_TOKEN`` is set.
"""

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import get_request_id, tenant_rate_limit_key
from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import GenerateDraftResponse
from src.config.settings import settings
from src.engine.generator import generator

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/generate-draft",
    response_model=GenerateDraftResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_generate)
async def generate_draft(
    request: Request, generate_request: GenerateDraftRequest
) -> GenerateDraftResponse:
    """Generate a collection email draft.

    Accept a ``GenerateDraftRequest`` containing full case context
    (party, obligations, behaviour, escalation history, conversation
    history), tone, sender persona, and optional flags
    (``skip_invoice_table``, ``closure_mode``, ``trigger_classification``).

    Return subject, body (with ``{INVOICE_TABLE}`` placeholder for
    standard drafts), guardrail validation, token usage, and provider
    metadata.  The Django backend replaces the placeholder with a
    formatted HTML/plain-text invoice table before pushing to Outlook.
    """
    request_id = get_request_id()
    tenant_id = request.headers.get("X-Tenant-ID")
    party_id = generate_request.context.party.party_id
    lane_id = generate_request.context.collection_lane_id
    obligation_count = len(generate_request.context.obligations or [])
    provider = settings.llm_provider
    model = settings.model_for_provider(provider)

    logger.info(
        "Generating draft request",
        extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "party_id": party_id,
            "collection_lane_id": lane_id,
            "lane_mail_mode": generate_request.context.lane_mail_mode,
            "provider": provider,
            "model": model,
            "obligation_count": obligation_count,
        },
    )
    try:
        result = await generator.generate(generate_request)
    except Exception as exc:
        logger.exception(
            "Draft generation request failed",
            extra={
                "request_id": request_id,
                "tenant_id": tenant_id,
                "party_id": party_id,
                "collection_lane_id": lane_id,
                "lane_mail_mode": generate_request.context.lane_mail_mode,
                "provider": provider,
                "model": model,
                "obligation_count": obligation_count,
                "exception_type": type(exc).__name__,
            },
        )
        raise

    logger.info(
        "Generated draft response",
        extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "party_id": party_id,
            "collection_lane_id": lane_id,
            "lane_mail_mode": generate_request.context.lane_mail_mode,
            "provider": result.provider,
            "model": result.model,
            "obligation_count": obligation_count,
            "tone_used": result.tone_used,
        },
    )
    return result
