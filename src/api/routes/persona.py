"""
Persona generation and refinement API endpoints.

POST /generate-persona - Generate initial personas for escalation contacts
POST /refine-persona - Refine a persona based on performance data
"""

import logging

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.errors import ErrorResponse
from src.api.models.requests import GeneratePersonaRequest, RefinePersonaRequest
from src.api.models.responses import (
    GeneratePersonaResponse,
    PersonaResult,
    RefinePersonaResponse,
)
from src.config.settings import settings
from src.engine.persona import persona_generator

logger = logging.getLogger(__name__)
router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/generate-persona",
    response_model=GeneratePersonaResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
    },
)
@limiter.limit(settings.rate_limit_generate)
async def generate_persona(
    request: Request, persona_request: GeneratePersonaRequest
) -> GeneratePersonaResponse:
    """
    Generate initial personas for escalation contacts (cold start).

    Called when admin saves the escalation hierarchy.
    """
    logger.info("Generating personas for %d contacts", len(persona_request.contacts))
    contacts = [c.model_dump() for c in persona_request.contacts]
    results = await persona_generator.generate_personas(contacts, persona_request.total_levels)

    personas = [
        PersonaResult(
            name=r.get("name", ""),
            level=r.get("level", 1),
            communication_style=r.get("communication_style"),
            formality_level=r.get("formality_level"),
            emphasis=r.get("emphasis"),
        )
        for r in results
    ]

    logger.info("Generated %d personas", len(personas))
    return GeneratePersonaResponse(personas=personas)


@router.post(
    "/refine-persona",
    response_model=RefinePersonaResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
    },
)
@limiter.limit(settings.rate_limit_generate)
async def refine_persona(
    request: Request, refine_request: RefinePersonaRequest
) -> RefinePersonaResponse:
    """
    Refine a sender persona based on performance data (LLM-driven).

    Called during sync cycle for senders with sufficient data.
    """
    logger.info(
        "Refining persona for %s (level %d)",
        refine_request.name,
        refine_request.level,
    )

    contact = {
        "name": refine_request.name,
        "title": refine_request.title,
        "level": refine_request.level,
    }
    current_persona = refine_request.current_persona.model_dump()
    performance = refine_request.performance.model_dump()

    result = await persona_generator.refine_persona(
        contact,
        current_persona,
        performance,
        persona_version=refine_request.persona_version,
    )

    logger.info("Persona refined for %s: %s", refine_request.name, result.get("reasoning", ""))
    return RefinePersonaResponse(
        communication_style=result["communication_style"],
        formality_level=result["formality_level"],
        emphasis=result["emphasis"],
        reasoning=result["reasoning"],
    )
