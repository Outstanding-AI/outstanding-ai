"""Versioned evidence-only collection-mail interpretation endpoint."""

from fastapi import APIRouter, Request
from slowapi import Limiter
from solvix_contracts.ai import MailSemanticEvidenceRequestV2, MailSemanticEvidenceResponseV2

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.config.settings import settings
from src.engine.mail_semantic_evidence_v2 import mail_semantic_evidence_interpreter_v2

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/interpret-collection-email-v2",
    response_model=MailSemanticEvidenceResponseV2,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def interpret_collection_email_v2(
    request: Request,
    interpretation_request: MailSemanticEvidenceRequestV2,
) -> MailSemanticEvidenceResponseV2:
    return await mail_semantic_evidence_interpreter_v2.interpret(interpretation_request)
