"""Historical collection-thread protocol/adjudication classification route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests import HistoricalCollectionThreadRequest
from src.api.models.responses import HistoricalCollectionThreadResponse
from src.config.settings import settings
from src.engine.historical_collection_thread_classifier import (
    historical_collection_thread_classifier,
)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/classify-historical-collection-thread",
    response_model=HistoricalCollectionThreadResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_classify)
async def classify_historical_collection_thread(
    request: Request,
    classify_request: HistoricalCollectionThreadRequest,
) -> HistoricalCollectionThreadResponse:
    logger.info("Classifying historical collection thread evidence mode=%s", classify_request.mode)
    return await historical_collection_thread_classifier.classify(classify_request)
