"""Evidence-only interpretation for one canonical collection-mail version.

The request may contain retained source text in transit so this service can
ground model claims.  The returned V2 contract contains coordinates and hashes
only; callers must not copy request text into App DB bookkeeping.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from solvix_contracts.ai import (
    AIInvocationTelemetryV1,
    AIOperationSummaryV1,
    MailSemanticEventV2,
    MailSemanticEvidenceRequestV2,
    MailSemanticEvidenceResponseV2,
    MailSemanticSourceSpanV2,
)

from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback

from ._evidence_grounding import (
    amount_is_explicit_in_span,
    date_is_explicit_in_span,
    locate_evidence,
    reference_is_explicit_in_span,
)
from .collection_email_event_classifier import _parse_response_object

PROMPT_TEMPLATE_ID = "mail_semantic_evidence"
PROMPT_TEMPLATE_VERSION = "v2"
RESPONSE_SCHEMA_VERSION = "mail-semantic-evidence.v2"
OPERATION = "mail_semantic_evidence_v2"

_SYSTEM_PROMPT = """Interpret one canonical accounts-receivable mailbox message as source-grounded semantic evidence.
Return JSON only. This is an evidence-only endpoint: never select a recipient, route a message, create a draft, create
a control, request an effect, or claim accounting truth. Party and invoice candidates are optional enrichment. Empty,
unresolved, or ambiguous candidates are valid inputs and must never be treated as an API failure.

Use only current authored source segments for current-message meaning. Quoted, forwarded, attachment, and prior-message
segments may supply context or provenance but cannot be presented as newly authored intent. Every semantic event and
relevance conclusion must cite exact source text with one or more source spans. For every span, return source_id,
evidence_text, and supports. evidence_text must be a unique exact substring of that source. Never return source text
outside evidence_text. A message with no safe source text, unsupported/unsafe attachment evidence, or insufficient
meaning must return disposition=abstained with semantic_events=[]; a provider/schema failure is represented by
disposition=invalid with semantic_events=[]. Attachment evidence pending is not evidence that no attachment exists.

An accepted result is a valid semantic interpretation and may contain zero semantic events. It is never operational
acceptance. Unscoped observations are allowed; do not invent party or invoice scope. Return exactly:
{
  "relevance": "collection" | "non_collection" | "uncertain",
  "semantic_events": [{"event_id": string, "family": string, "transition": string,
    "polarity": "affirmed" | "negated" | "uncertain", "temporal_orientation": "past" | "current" | "future" | "unclear",
    "invoice_refs": [string], "account_wide": boolean, "amount": number|null, "currency": string|null,
    "asserted_date": "YYYY-MM-DD"|null, "reference": string|null,
    "evidence": [{"source_id": string, "evidence_text": string, "supports": [string]}],
    "confidence": number, "reason_codes": [string]}],
  "response_evidence": [{"source_id": string, "evidence_text": string, "supports": [string]}],
  "disposition": "accepted" | "abstained" | "deferred" | "invalid",
  "adjudication_status": "not_required" | "resolved" | "unresolved" | "failed",
  "confidence": number, "reason_codes": [string], "ambiguity_reason_codes": [string]
}"""


class _LLMSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    evidence_text: str
    supports: list[
        Literal[
            "relevance",
            "invoice_scope",
            "operational_state",
            "amount",
            "date",
            "reference",
            "reply_action",
        ]
    ] = Field(min_length=1, max_length=7)


class _LLMEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    family: Literal[
        "query",
        "commitment",
        "remittance",
        "already_paid",
        "payment_timing_dispute",
        "internal_processing_blocker",
        "request_information",
        "insolvency",
        "redirect",
        "email_bounce",
        "out_of_office",
        "hardship",
        "unsubscribe",
        "other",
    ]
    transition: Literal[
        "opened",
        "updated",
        "resolved",
        "cancelled",
        "made",
        "kept",
        "broken",
        "reported",
        "verified",
        "rejected",
        "observed",
        "requested",
        "redirected",
        "bounced",
        "active",
    ]
    polarity: Literal["affirmed", "negated", "uncertain"]
    temporal_orientation: Literal["past", "current", "future", "unclear"]
    invoice_refs: list[str] = Field(default_factory=list, max_length=100)
    account_wide: bool = False
    amount: Decimal | None = Field(default=None, gt=0)
    currency: str | None = Field(default=None, min_length=3, max_length=8)
    asserted_date: str | None = None
    reference: str | None = Field(default=None, max_length=200)
    evidence: list[_LLMSpan] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list, max_length=30)


class _LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: Literal["collection", "non_collection", "uncertain"]
    semantic_events: list[_LLMEvent] = Field(default_factory=list, max_length=50)
    response_evidence: list[_LLMSpan] = Field(default_factory=list, max_length=20)
    disposition: Literal["accepted", "abstained", "deferred", "invalid"]
    adjudication_status: Literal["not_required", "resolved", "unresolved", "failed"] = (
        "not_required"
    )
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list, max_length=30)
    ambiguity_reason_codes: list[str] = Field(default_factory=list, max_length=30)


def _summary_without_provider(request: MailSemanticEvidenceRequestV2) -> AIOperationSummaryV1:
    """Represent deterministic abstention without fabricating a model call."""

    return AIOperationSummaryV1(
        request_id=str(uuid4()),
        operation=OPERATION,
        cache_hit=True,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        normalized_input_hash=request.input_context_hash,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=0,
        total_cost_usd=Decimal("0"),
        provider_latency_ms=0,
        usage_complete=True,
        billing_complete=True,
        invocations=[],
    )


def _summary_from_provider(
    request: MailSemanticEvidenceRequestV2,
    *,
    response: object,
    latency_ms: int,
    success: bool,
    error_code: str | None = None,
    is_fallback: bool = False,
) -> AIOperationSummaryV1:
    usage = getattr(response, "usage", None) or {}
    usage_reported = all(
        key in usage and usage.get(key) is not None
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    prompt_tokens = int(usage.get("prompt_tokens") or 0) if usage_reported else None
    completion_tokens = int(usage.get("completion_tokens") or 0) if usage_reported else None
    total_tokens = int(usage.get("total_tokens") or 0) if usage_reported else None
    reasoning_tokens = int(usage.get("reasoning_tokens") or 0) if usage_reported else None
    cached_tokens = int(usage.get("cached_tokens") or 0) if usage_reported else None
    request_id = str(uuid4())
    invocation = AIInvocationTelemetryV1(
        invocation_id=str(uuid4()),
        parent_request_id=request_id,
        operation=OPERATION,
        suboperation="interpret",
        attempt_index=0,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        normalized_input_hash=request.input_context_hash,
        provider=str(getattr(response, "provider", "unknown") or "unknown"),
        model=str(getattr(response, "model", "unknown") or "unknown"),
        is_fallback=is_fallback,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        latency_ms=max(0, latency_ms),
        success=success,
        error_code=error_code,
        usage_reported=usage_reported,
        billing_status="not_reported",
    )
    return AIOperationSummaryV1(
        request_id=request_id,
        operation=OPERATION,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        response_schema_version=RESPONSE_SCHEMA_VERSION,
        normalized_input_hash=request.input_context_hash,
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        reasoning_tokens=int(reasoning_tokens or 0),
        cached_tokens=int(cached_tokens or 0),
        total_tokens=int(total_tokens or 0),
        total_cost_usd=Decimal("0"),
        provider_latency_ms=max(0, latency_ms),
        usage_complete=usage_reported,
        billing_complete=False,
        invocations=[invocation],
    )


def _source_map(request: MailSemanticEvidenceRequestV2) -> dict[str, tuple[str, str, str]]:
    """Map allowed source IDs to (text, kind, hash) for exact span validation."""

    attachment_states = {
        attachment.attachment_ref: attachment.state for attachment in request.attachment_evidence
    }
    sources = {}
    for segment in request.current_message.source_segments:
        if (
            segment.source_kind == "attachment"
            and attachment_states.get(segment.attachment_ref or "") != "complete"
        ):
            continue
        if segment.text:
            sources[segment.source_id] = (segment.text, segment.source_kind, segment.content_hash)
    for prior in request.prior_messages:
        if prior.authored_text:
            sources[prior.source_id] = (prior.authored_text, "prior_message", prior.content_hash)
    return sources


def _ground_span(
    span: _LLMSpan, sources: dict[str, tuple[str, str, str]]
) -> MailSemanticSourceSpanV2:
    source = sources.get(span.source_id)
    if source is None:
        raise ValueError("semantic_evidence_unknown_or_unavailable_source")
    text, source_kind, content_hash = source
    start, end, _ = locate_evidence(text, span.evidence_text, field="semantic_span")
    return MailSemanticSourceSpanV2(
        source_id=span.source_id,
        source_kind=source_kind,
        start=start,
        end=end,
        content_hash=content_hash,
        supports=span.supports,
    )


def _validate_event_field_grounding(event: _LLMEvent) -> None:
    """Require source text for every non-null material value before emitting it."""

    def supported_text(support: str) -> list[str]:
        return [span.evidence_text for span in event.evidence if support in span.supports]

    if event.amount is not None and not any(
        amount_is_explicit_in_span(event.amount, text) for text in supported_text("amount")
    ):
        raise ValueError("semantic_amount_not_grounded")
    if event.asserted_date is not None and not any(
        date_is_explicit_in_span(event.asserted_date, text) for text in supported_text("date")
    ):
        raise ValueError("semantic_date_not_grounded")
    if event.reference is not None and not any(
        reference_is_explicit_in_span(event.reference, text) for text in supported_text("reference")
    ):
        raise ValueError("semantic_reference_not_grounded")


class MailSemanticEvidenceInterpreterV2:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def interpret(
        self, request: MailSemanticEvidenceRequestV2
    ) -> MailSemanticEvidenceResponseV2:
        sources = _source_map(request)
        if not settings.enable_mail_semantic_evidence_v2:
            return MailSemanticEvidenceResponseV2(
                canonical_message_state_key=request.canonical_message_state_key,
                canonical_message_version_hash=request.canonical_message_version_hash,
                semantic_revision_key=request.semantic_revision_key,
                input_context_hash=request.input_context_hash,
                relevance="uncertain",
                attachment_evidence_state=request.attachment_evidence_state,
                disposition="deferred",
                confidence=0,
                reason_codes=["mail_semantic_evidence_v2_disabled"],
                operation_summary=_summary_without_provider(request),
            )
        if not sources:
            return MailSemanticEvidenceResponseV2(
                canonical_message_state_key=request.canonical_message_state_key,
                canonical_message_version_hash=request.canonical_message_version_hash,
                semantic_revision_key=request.semantic_revision_key,
                input_context_hash=request.input_context_hash,
                relevance="uncertain",
                attachment_evidence_state=request.attachment_evidence_state,
                disposition="abstained",
                confidence=0,
                reason_codes=["no_safe_semantic_source"],
                operation_summary=_summary_without_provider(request),
            )

        client = self._client
        if request.model_override:
            client = LLMProviderWithFallback(
                primary_provider="vertex",
                fallback_provider="openai",
                model_override=dict(request.model_override),
            )
        prompt_input = request.model_dump(
            mode="json", exclude_none=True, exclude={"model_override"}
        )
        started = time.monotonic()
        response = await client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
            temperature=settings.classification_temperature,
            json_mode=True,
            caller="mail_semantic_evidence_v2",
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            parsed = _LLMResponse.model_validate(_parse_response_object(response.content))
            if parsed.disposition != "accepted" and parsed.semantic_events:
                raise ValueError("non_accepted_response_contains_semantic_events")
            response_evidence = [_ground_span(span, sources) for span in parsed.response_evidence]
            if parsed.disposition == "accepted" and not any(
                "relevance" in span.supports for span in parsed.response_evidence
            ):
                raise ValueError("accepted_relevance_without_evidence")
            events = []
            for event in parsed.semantic_events:
                _validate_event_field_grounding(event)
                events.append(
                    MailSemanticEventV2(
                        event_id=event.event_id,
                        family=event.family,
                        transition=event.transition,
                        polarity=event.polarity,
                        temporal_orientation=event.temporal_orientation,
                        invoice_refs=event.invoice_refs,
                        account_wide=event.account_wide,
                        amount=event.amount,
                        currency=event.currency,
                        asserted_date=event.asserted_date,
                        reference=event.reference,
                        evidence=[_ground_span(span, sources) for span in event.evidence],
                        confidence=event.confidence,
                        reason_codes=event.reason_codes,
                    )
                )
            return MailSemanticEvidenceResponseV2(
                canonical_message_state_key=request.canonical_message_state_key,
                canonical_message_version_hash=request.canonical_message_version_hash,
                semantic_revision_key=request.semantic_revision_key,
                input_context_hash=request.input_context_hash,
                relevance=parsed.relevance,
                semantic_events=events,
                response_evidence=response_evidence,
                attachment_evidence_state=request.attachment_evidence_state,
                disposition=parsed.disposition,
                adjudication_status=parsed.adjudication_status,
                confidence=parsed.confidence,
                reason_codes=parsed.reason_codes,
                ambiguity_reason_codes=parsed.ambiguity_reason_codes,
                operation_summary=_summary_from_provider(
                    request,
                    response=response,
                    latency_ms=latency_ms,
                    success=True,
                    is_fallback=response.provider != client.primary_provider_name,
                ),
            )
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
            return MailSemanticEvidenceResponseV2(
                canonical_message_state_key=request.canonical_message_state_key,
                canonical_message_version_hash=request.canonical_message_version_hash,
                semantic_revision_key=request.semantic_revision_key,
                input_context_hash=request.input_context_hash,
                relevance="uncertain",
                attachment_evidence_state=request.attachment_evidence_state,
                disposition="invalid",
                confidence=0,
                reason_codes=["semantic_response_schema_or_grounding_invalid"],
                operation_summary=_summary_from_provider(
                    request,
                    response=response,
                    latency_ms=latency_ms,
                    success=False,
                    error_code="semantic_response_schema_or_grounding_invalid",
                    is_fallback=response.provider != client.primary_provider_name,
                ),
            )


mail_semantic_evidence_interpreter_v2 = MailSemanticEvidenceInterpreterV2()
