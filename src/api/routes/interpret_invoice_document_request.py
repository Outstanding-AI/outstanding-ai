"""Bounded additive invoice-document request interpretation route."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.document_request_responses import (
    InvoiceDocumentRequestInterpretationResponse,
)
from src.api.models.requests.document_request import InvoiceDocumentRequestInterpretationRequest
from src.config.settings import settings
from src.engine.document_request_interpreter import document_request_interpreter

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/interpret-invoice-document-request",
    response_model=InvoiceDocumentRequestInterpretationResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def interpret_invoice_document_request(
    request: Request,
    interpretation_request: InvoiceDocumentRequestInterpretationRequest,
) -> InvoiceDocumentRequestInterpretationResponse:
    return await document_request_interpreter.interpret(interpretation_request)


__all__ = ["router", "interpret_invoice_document_request"]
