"""Sent-draft invoice-scope analysis endpoint."""

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import get_request_id, tenant_rate_limit_key
from src.api.models.requests import AnalyzeSentDraftScopeRequest
from src.api.models.responses import AnalyzeSentDraftScopeResponse
from src.config.settings import settings
from src.engine.sent_scope import sent_scope_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/analyze-sent-draft-scope",
    response_model=AnalyzeSentDraftScopeResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_classify)
async def analyze_sent_draft_scope(
    request: Request, analysis_request: AnalyzeSentDraftScopeRequest
) -> AnalyzeSentDraftScopeResponse:
    request_id = get_request_id()
    logger.info(
        "Analyzing sent draft scope",
        extra={
            "request_id": request_id,
            "tenant_id": analysis_request.tenant_id,
            "party_id": analysis_request.party_id,
            "draft_id": analysis_request.draft_id,
            "candidate_count": len(analysis_request.invoice_candidates),
        },
    )
    return await sent_scope_analyzer.analyze(analysis_request)
