"""Select among multiple active collection chains."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests import CollectionChainRoutingRequest
from src.api.models.responses import CollectionChainRoutingResponse
from src.config.settings import settings
from src.engine.collection_chain_router import collection_chain_router

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post("/select-collection-chain", response_model=CollectionChainRoutingResponse)
@limiter.limit(settings.rate_limit_classify)
async def select_collection_chain(
    request: Request,
    routing_request: CollectionChainRoutingRequest,
) -> CollectionChainRoutingResponse:
    return await collection_chain_router.select(routing_request)
