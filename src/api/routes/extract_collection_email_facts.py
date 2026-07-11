"""Bounded collection-email fact extraction route."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests import CollectionEmailFactExtractionRequest
from src.api.models.responses import CollectionEmailFactExtractionResponse
from src.config.settings import settings
from src.engine.collection_email_fact_extractor import collection_email_fact_extractor

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/extract-collection-email-facts",
    response_model=CollectionEmailFactExtractionResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def extract_collection_email_facts(
    request: Request, extraction_request: CollectionEmailFactExtractionRequest
) -> CollectionEmailFactExtractionResponse:
    return await collection_email_fact_extractor.extract(extraction_request)
