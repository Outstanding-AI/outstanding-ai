"""Bounded post-reconciliation collection-chain identification route."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests import CollectionChainIdentificationRequest
from src.api.models.responses import CollectionChainIdentificationResponse
from src.config.settings import settings
from src.engine.collection_chain_identifier import collection_chain_identifier

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/identify-collection-chain",
    response_model=CollectionChainIdentificationResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def identify_collection_chain(
    request: Request, identification_request: CollectionChainIdentificationRequest
) -> CollectionChainIdentificationResponse:
    return await collection_chain_identifier.identify(identification_request)
