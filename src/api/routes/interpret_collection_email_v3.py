"""V3 sealed, evidence-only collection-mail interpretation endpoint."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.config.settings import settings

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)

try:  # V3 is not mounted until its independently versioned contract is installed.
    from solvix_contracts.ai import MailSemanticEvidenceRequestV3, MailSemanticEvidenceResponseV3
except ImportError:  # pragma: no cover - exercised by a V2-only installed contracts package
    V3_CONTRACT_AVAILABLE = False
else:
    from src.engine.mail_semantic_evidence_v3 import mail_semantic_evidence_interpreter_v3

    V3_CONTRACT_AVAILABLE = True

    @router.post(
        "/interpret-collection-email-v3",
        response_model=MailSemanticEvidenceResponseV3,
        responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    @limiter.limit(settings.rate_limit_classify)
    async def interpret_collection_email_v3(
        request: Request,
        interpretation_request: MailSemanticEvidenceRequestV3,
    ) -> MailSemanticEvidenceResponseV3:
        return await mail_semantic_evidence_interpreter_v3.interpret(interpretation_request)
