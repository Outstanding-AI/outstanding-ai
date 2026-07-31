"""Bounded manual-note interpretation route."""

from fastapi import APIRouter, Request
from slowapi import Limiter
from solvix_contracts.ai import (
    ManualNoteInterpretationRequestV1,
    ManualNoteInterpretationResponseV1,
)

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.config.settings import settings
from src.engine.manual_note_interpreter import manual_note_interpreter

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/interpret-manual-note",
    response_model=ManualNoteInterpretationResponseV1,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def interpret_manual_note(
    request: Request,
    interpretation_request: ManualNoteInterpretationRequestV1,
) -> ManualNoteInterpretationResponseV1:
    return await manual_note_interpreter.interpret(interpretation_request)
