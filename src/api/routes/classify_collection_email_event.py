"""Email-native collection-chain event classification route."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests import CollectionEmailEventRequest
from src.api.models.responses import CollectionEmailEventResponse
from src.config.settings import settings
from src.engine.collection_email_event_classifier import collection_email_event_classifier

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/classify-collection-email-event",
    response_model=CollectionEmailEventResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def classify_collection_email_event(
    request: Request, classify_request: CollectionEmailEventRequest
) -> CollectionEmailEventResponse:
    return await collection_email_event_classifier.classify(classify_request)
