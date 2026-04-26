"""
Draft generation API endpoint.

POST /generate-draft -- Generate a collection email draft with subject,
body (containing ``{INVOICE_TABLE}`` placeholder), tone metadata, and
guardrail validation results.

Called by the Django backend's ``ai_engine/client.py`` during the
``ai.generate_draft_for_party`` background job.

Security:
    - Rate limited via slowapi (default 100/minute, configurable).
    - Service-to-service auth via Bearer token when
      ``SERVICE_AUTH_TOKEN`` is set.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import get_request_id, tenant_rate_limit_key
from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import (
    GenerateDraftFromManifestCandidateResult,
    GenerateDraftFromManifestResponse,
    GenerateDraftResponse,
)
from src.config.settings import settings
from src.engine.generator import generator
from src.lake import (
    CaseContextHydrator,
    DraftGenerationHandoff,
    ManifestLoadError,
    RegionalLakeClients,
    RegionalLakeReader,
    load_draft_candidate_manifest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=tenant_rate_limit_key)


def _tone_from_context(generate_request: GenerateDraftRequest) -> str:
    """Choose a generation tone from hydrated lane context when available."""
    lane = generate_request.context.lane or {}
    tone_ladder = lane.get("tone_ladder") if isinstance(lane, dict) else None
    if isinstance(tone_ladder, list) and tone_ladder:
        return str(tone_ladder[0])
    return generate_request.tone


@router.post(
    "/generate-draft",
    response_model=GenerateDraftResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_generate)
async def generate_draft(
    request: Request, generate_request: GenerateDraftRequest
) -> GenerateDraftResponse:
    """Generate a collection email draft.

    Accept a ``GenerateDraftRequest`` containing full case context
    (party, obligations, behaviour, escalation history, conversation
    history), tone, sender persona, and optional flags
    (``skip_invoice_table``, ``closure_mode``, ``trigger_classification``).

    Return subject, body (with ``{INVOICE_TABLE}`` placeholder for
    standard drafts), guardrail validation, token usage, and provider
    metadata.  The Django backend replaces the placeholder with a
    formatted HTML/plain-text invoice table before pushing to Outlook.
    """
    request_id = get_request_id()
    tenant_id = request.headers.get("X-Tenant-ID")
    party_id = generate_request.context.party.party_id
    lane_id = generate_request.context.collection_lane_id
    obligation_count = len(generate_request.context.obligations or [])
    provider = settings.llm_provider
    model = settings.model_for_provider(provider)

    logger.info(
        "Generating draft request",
        extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "party_id": party_id,
            "collection_lane_id": lane_id,
            "lane_mail_mode": generate_request.context.lane_mail_mode,
            "schema_version": generate_request.context.schema_version,
            "provider": provider,
            "model": model,
            "obligation_count": obligation_count,
        },
    )
    try:
        result = await generator.generate(generate_request)
    except Exception as exc:
        logger.exception(
            "Draft generation request failed",
            extra={
                "request_id": request_id,
                "tenant_id": tenant_id,
                "party_id": party_id,
                "collection_lane_id": lane_id,
                "lane_mail_mode": generate_request.context.lane_mail_mode,
                "schema_version": generate_request.context.schema_version,
                "provider": provider,
                "model": model,
                "obligation_count": obligation_count,
                "exception_type": type(exc).__name__,
            },
        )
        raise

    logger.info(
        "Generated draft response",
        extra={
            "request_id": request_id,
            "tenant_id": tenant_id,
            "party_id": party_id,
            "collection_lane_id": lane_id,
            "lane_mail_mode": generate_request.context.lane_mail_mode,
            "schema_version": generate_request.context.schema_version,
            "provider": result.provider,
            "model": result.model,
            "obligation_count": obligation_count,
            "tone_used": result.tone_used,
        },
    )
    return result


@router.post(
    "/generate-draft-from-manifest",
    response_model=GenerateDraftFromManifestResponse,
    responses={
        401: {"description": "Unauthorized — missing or invalid service token"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Regional lake or LLM error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_generate)
async def generate_draft_from_manifest(
    request: Request, handoff: DraftGenerationHandoff
) -> GenerateDraftFromManifestResponse:
    """Generate drafts from a regional lake handoff manifest.

    This is the additive Phase 4.4 handoff path. The existing
    ``/generate-draft`` endpoint still accepts backend-hydrated context.
    """
    request_id = get_request_id()
    logger.info(
        "Generating drafts from regional manifest",
        extra={
            "request_id": request_id,
            "tenant_id": handoff.tenant_id,
            "sync_run_id": handoff.sync_run_id,
            "manifest_uri": handoff.manifest_uri,
            "data_lake_region": handoff.data_lake_region,
        },
    )

    clients = RegionalLakeClients.from_handoff(handoff)
    try:
        candidates = load_draft_candidate_manifest(
            handoff.manifest_uri,
            region_name=handoff.data_lake_region,
            expected_tenant_id=handoff.tenant_id,
            expected_sync_run_id=handoff.sync_run_id,
            expected_data_lake_region=handoff.data_lake_region,
            s3_client=clients.s3(),
        )
    except ManifestLoadError as exc:
        logger.exception(
            "Failed to load regional draft candidate manifest",
            extra={
                "request_id": request_id,
                "tenant_id": handoff.tenant_id,
                "sync_run_id": handoff.sync_run_id,
                "data_lake_region": handoff.data_lake_region,
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not candidates:
        logger.error(
            "Regional draft candidate manifest was empty",
            extra={
                "request_id": request_id,
                "tenant_id": handoff.tenant_id,
                "sync_run_id": handoff.sync_run_id,
                "manifest_uri": handoff.manifest_uri,
                "data_lake_region": handoff.data_lake_region,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Regional draft candidate manifest contained no candidates",
        )

    reader = RegionalLakeReader.from_handoff(
        handoff,
        workgroup=settings.athena_workgroup,
        output_location=settings.athena_output_location,
        poll_interval_seconds=settings.regional_lake_poll_interval_seconds,
        timeout_seconds=settings.regional_lake_query_timeout_seconds,
    )
    hydrator = CaseContextHydrator(handoff.tenant_id, reader)
    results: list[GenerateDraftFromManifestCandidateResult] = []

    for candidate in candidates:
        try:
            context = hydrator.hydrate_candidate(candidate)
            generate_request = GenerateDraftRequest(context=context)
            generate_request = GenerateDraftRequest(
                context=context,
                tone=_tone_from_context(generate_request),
            )
            draft = await generator.generate(generate_request)
            results.append(
                GenerateDraftFromManifestCandidateResult(
                    candidate_id=candidate.candidate_id,
                    party_id=candidate.party_id,
                    lane_id=candidate.lane_id,
                    status="generated",
                    draft=draft,
                )
            )
        except Exception as exc:
            logger.exception(
                "Failed to generate draft from regional candidate",
                extra={
                    "request_id": request_id,
                    "tenant_id": handoff.tenant_id,
                    "sync_run_id": handoff.sync_run_id,
                    "candidate_id": candidate.candidate_id,
                    "party_id": candidate.party_id,
                    "lane_id": candidate.lane_id,
                    "exception_type": type(exc).__name__,
                },
            )
            results.append(
                GenerateDraftFromManifestCandidateResult(
                    candidate_id=candidate.candidate_id,
                    party_id=candidate.party_id,
                    lane_id=candidate.lane_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    generated_count = sum(1 for result in results if result.status == "generated")
    failed_count = len(results) - generated_count
    status = (
        "completed" if failed_count == 0 else "failed" if generated_count == 0 else "partial_failed"
    )

    logger.info(
        "Generated drafts from regional manifest",
        extra={
            "request_id": request_id,
            "tenant_id": handoff.tenant_id,
            "sync_run_id": handoff.sync_run_id,
            "data_lake_region": handoff.data_lake_region,
            "total": len(results),
            "generated_count": generated_count,
            "failed_count": failed_count,
            "status": status,
        },
    )

    return GenerateDraftFromManifestResponse(
        tenant_id=handoff.tenant_id,
        sync_run_id=handoff.sync_run_id,
        data_lake_region=handoff.data_lake_region,
        total=len(results),
        generated_count=generated_count,
        failed_count=failed_count,
        status=status,
        results=results,
    )
