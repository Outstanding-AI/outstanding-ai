"""V3 evidence-only interpretation for one sealed canonical collection-mail version.

The request may contain retained source text in transit so this service can
ground model claims. The returned V3 contract contains coordinates and hashes
only; callers must not copy request text into App DB bookkeeping.
"""

from __future__ import annotations

import json
import re
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

# Keep the deployed V2 admission contract importable while the V3 contract is
# released independently.  V3 is feature-gated and its route is not mounted
# until the installed contracts package exports these types.
try:  # pragma: no cover - exercised by the V3 contract-release integration
    from solvix_contracts.ai import (
        MailSemanticEvidenceRequestV3,
        MailSemanticEvidenceResponseV3,
    )
except ImportError:  # pragma: no cover - V2-only deployed package compatibility
    MailSemanticEvidenceRequestV3 = None  # type: ignore[assignment,misc]
    MailSemanticEvidenceResponseV3 = None  # type: ignore[assignment,misc]

from src.config.settings import settings
from src.llm.base import BaseLLMProvider
from src.llm.factory import LLMProviderWithFallback
from src.llm.openrouter_provider import OpenRouterProvider

from ._evidence_grounding import (
    amount_is_explicit_in_span,
    date_is_explicit_in_span,
    locate_evidence,
    reference_is_explicit_in_span,
)
from .collection_email_event_classifier import _parse_response_object

PROMPT_TEMPLATE_ID = "mail_semantic_evidence"
PROMPT_TEMPLATE_VERSION = "v3"
RESPONSE_SCHEMA_VERSION = "mail-semantic-evidence.v3"
OPERATION = "mail_semantic_evidence_v3"

_V2_PROMPT_TEMPLATE_VERSION = "v2"
_V2_RESPONSE_SCHEMA_VERSION = "mail-semantic-evidence.v2"
_V2_OPERATION = "mail_semantic_evidence_v2"

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

`evidence_text` is a character-copy operation, never a paraphrase: do not omit articles, correct grammar, normalize
whitespace, change punctuation, or shorten a quoted table/body phrase. Before returning JSON, verify that every
evidence_text appears exactly once in the named source segment. If you cannot safely quote a narrow phrase verbatim,
copy the complete source segment verbatim and retain its source_id.

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

_EVENT_ENUM_GUIDANCE = """
Use only these `family` values: query, commitment, remittance, already_paid,
payment_timing_dispute, internal_processing_blocker, request_information,
insolvency, redirect, email_bounce, out_of_office, hardship, unsubscribe, other.
Use `request_information` for every document request, including an invoice PDF,
statement, packing list, POD or another supporting document; never emit
`document_request`. Use only these `transition` values: opened, updated,
resolved, cancelled, made, kept, broken, reported, verified, rejected, observed,
requested, redirected, bounced, active. Use only affirmed, negated or uncertain
for `polarity`, and past, current, future or unclear for `temporal_orientation`.
For every evidence `supports` item, use only relevance, invoice_scope,
operational_state, amount, date, reference or reply_action. A document request
is supported by operational_state; never emit document_request as a support.
For a document request, leave amount, currency, asserted_date and reference
null unless that exact value is part of the request and its evidence span marks
the matching support. A shipment, purchase-order or delivery date is not an
asserted_date for a document request.
Emit `other` only for a separate collection-relevant assertion that cannot fit a
listed family. Do not emit `other` for shipping, procurement, administrative or
logistics instructions that merely accompany a document request, query or other
already-emitted event.
Do not emit request_information merely because a message says “see attached”,
mentions an invoice/statement as an address preference, or contains shipping or
procurement instructions. A request_information event needs an explicit request
that the recipient provide or send a document. A query needs an explicit
challenge, correction request or unresolved issue; ordinary address, delivery
or account-administration traffic is not a collection query.
"""

_FAMILY_ALIASES = {"document_request": "request_information"}
_SUPPORT_ALIASES = {
    "document_request": "operational_state",
    "event_family": "operational_state",
    "event_type": "operational_state",
    "claim": "operational_state",
    "invoice_reference": "invoice_scope",
    "time": "date",
}


def _normalize_known_model_aliases(value: object) -> object:
    """Map only documented model vocabulary aliases before strict validation."""

    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    events = []
    for raw_event in value.get("semantic_events") or []:
        if not isinstance(raw_event, dict):
            events.append(raw_event)
            continue
        event = dict(raw_event)
        family = event.get("family")
        if isinstance(family, str):
            event["family"] = _FAMILY_ALIASES.get(family, family)
        event["evidence"] = _normalize_evidence_supports(event.get("evidence"))
        events.append(event)
    normalized["semantic_events"] = events
    normalized["response_evidence"] = _normalize_evidence_supports(value.get("response_evidence"))
    return normalized


def _normalize_evidence_supports(value: object) -> object:
    if not isinstance(value, list):
        return value
    spans = []
    for raw_span in value:
        if not isinstance(raw_span, dict):
            spans.append(raw_span)
            continue
        span = dict(raw_span)
        supports = span.get("supports")
        if isinstance(supports, list):
            span["supports"] = [
                _SUPPORT_ALIASES.get(item, item) if isinstance(item, str) else item
                for item in supports
            ]
        spans.append(span)
    return spans


def _ensure_event_grounded_relevance(parsed: _LLMResponse) -> _LLMResponse:
    """Reuse an accepted event's exact span for relevance when the model omitted only that tag."""

    if parsed.disposition != "accepted":
        return parsed
    if any("relevance" in span.supports for span in parsed.response_evidence):
        return parsed
    if not parsed.semantic_events or not parsed.semantic_events[0].evidence:
        return parsed

    source_span = parsed.semantic_events[0].evidence[0]
    relevance_span = source_span.model_copy(
        update={"supports": [*source_span.supports, "relevance"]}
    )
    return parsed.model_copy(
        update={"response_evidence": [*parsed.response_evidence, relevance_span]}
    )


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


def _summary_without_provider(request: MailSemanticEvidenceRequestV3) -> AIOperationSummaryV1:
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
    request: MailSemanticEvidenceRequestV3,
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


def _source_map(request: MailSemanticEvidenceRequestV3) -> dict[str, tuple[str, str, str]]:
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


def _drop_ungrounded_material_fields(event: _LLMEvent) -> _LLMEvent:
    """Retain a grounded event while removing a model-added value lacking its own source span."""

    def supported_text(support: str) -> list[str]:
        return [span.evidence_text for span in event.evidence if support in span.supports]

    updates: dict[str, object] = {}
    reason_codes = list(event.reason_codes)
    if event.amount is not None and not any(
        amount_is_explicit_in_span(event.amount, text) for text in supported_text("amount")
    ):
        updates.update({"amount": None, "currency": None})
        reason_codes.append("ungrounded_amount_dropped")
    if event.asserted_date is not None and not any(
        date_is_explicit_in_span(event.asserted_date, text) for text in supported_text("date")
    ):
        updates["asserted_date"] = None
        reason_codes.append("ungrounded_date_dropped")
    if event.reference is not None and not any(
        reference_is_explicit_in_span(event.reference, text) for text in supported_text("reference")
    ):
        updates["reference"] = None
        reason_codes.append("ungrounded_reference_dropped")
    if not updates:
        return event
    updates["reason_codes"] = list(dict.fromkeys(reason_codes))
    return event.model_copy(update=updates)


def _drop_unadmitted_invoice_refs(
    event: _LLMEvent, *, candidate_invoice_refs: set[str]
) -> _LLMEvent:
    """Do not let a semantic model promote arbitrary numbers into invoice scope."""

    admitted_refs = [
        reference for reference in event.invoice_refs if reference in candidate_invoice_refs
    ]
    if admitted_refs == event.invoice_refs:
        return event
    reason_codes = list(dict.fromkeys([*event.reason_codes, "unadmitted_invoice_refs_dropped"]))
    return event.model_copy(update={"invoice_refs": admitted_refs, "reason_codes": reason_codes})


def _drop_ungrounded_invoice_refs(event: _LLMEvent) -> _LLMEvent:
    """Retain only invoice-like strings explicitly copied from invoice-scope evidence."""

    supported_text = [
        span.evidence_text for span in event.evidence if "invoice_scope" in span.supports
    ]
    grounded_refs = [
        reference
        for reference in event.invoice_refs
        if any(reference_is_explicit_in_span(reference, text) for text in supported_text)
    ]
    if grounded_refs == event.invoice_refs:
        return event
    reason_codes = list(dict.fromkeys([*event.reason_codes, "ungrounded_invoice_refs_dropped"]))
    return event.model_copy(update={"invoice_refs": grounded_refs, "reason_codes": reason_codes})


def _safe_validation_reason(exc: Exception) -> str:
    """Return a bounded diagnostic code without preserving model/source content."""

    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0]
        error_type = str(first_error.get("type") or "unknown")
        location = "_".join(str(item) for item in first_error.get("loc") or ())
        suffix = f"{error_type}_{location}" if location else error_type
    else:
        suffix = str(exc).split(":", 1)[0]
    normalized = re.sub(r"[^a-z0-9]+", "_", suffix.lower()).strip("_")
    return f"semantic_validation_{normalized[:60] or 'unknown'}"


def _response_for(
    request: MailSemanticEvidenceRequestV3,
    **values: object,
) -> MailSemanticEvidenceResponseV3:
    """Build a V3 response that always echoes the sealed admission receipt."""

    return MailSemanticEvidenceResponseV3(
        canonical_message_state_key=request.canonical_message_state_key,
        canonical_message_version_hash=request.canonical_message_version_hash,
        semantic_revision_key=request.semantic_revision_key,
        input_context_hash=request.input_context_hash,
        attachment_evidence_state=request.attachment_evidence_state,
        admission=request.admission,
        **values,
    )


class MailSemanticEvidenceInterpreterV3:
    """Interpret mail evidence with one explicit, no-fallback primary.

    The provider is constructed lazily because this endpoint is deliberately
    disabled by default.  That keeps importing the AI service safe on hosts
    without an OpenRouter credential, while ensuring an enabled request cannot
    inherit the application-wide Vertex -> OpenAI fallback policy.
    """

    def __init__(self, *, primary_provider: BaseLLMProvider | None = None) -> None:
        self._primary_provider = primary_provider

    def _get_primary_provider(self) -> BaseLLMProvider:
        if self._primary_provider is None:
            self._primary_provider = OpenRouterProvider()
        return self._primary_provider

    async def interpret(
        self, request: MailSemanticEvidenceRequestV3
    ) -> MailSemanticEvidenceResponseV3:
        sources = _source_map(request)
        if not settings.enable_mail_semantic_evidence_v3:
            return _response_for(
                request,
                relevance="uncertain",
                disposition="deferred",
                confidence=0,
                reason_codes=["mail_semantic_evidence_v3_disabled"],
                operation_summary=_summary_without_provider(request),
            )
        if not sources:
            return _response_for(
                request,
                relevance="uncertain",
                disposition="abstained",
                confidence=0,
                reason_codes=["no_safe_semantic_source"],
                operation_summary=_summary_without_provider(request),
            )

        try:
            primary_provider = self._get_primary_provider()
        except ValueError:
            # A missing/invalid local OpenRouter configuration must not result
            # in a fallback provider receiving the request text.
            return _response_for(
                request,
                relevance="uncertain",
                disposition="deferred",
                confidence=0,
                reason_codes=["mail_semantic_evidence_v3_primary_provider_unavailable"],
                operation_summary=_summary_without_provider(request),
            )

        prompt_input = request.model_dump(mode="json", exclude_none=True)
        started = time.monotonic()
        response = await primary_provider.complete(
            system_prompt=f"{_SYSTEM_PROMPT}\n{_EVENT_ENUM_GUIDANCE}",
            user_prompt=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
            temperature=settings.classification_temperature,
            # DeepSeek's routed strict-schema path has returned HTTP 200 with
            # no choices in local boundary verification. Request JSON-object
            # mode and retain the existing strict Pydantic + exact-span
            # validation below until a provider-specific schema route proves
            # it can complete this contract reliably.
            json_mode=True,
            reasoning_effort=settings.openrouter_mail_semantic_primary_reasoning_effort,
            reasoning_enabled=settings.openrouter_mail_semantic_primary_reasoning_enabled,
            caller="mail_semantic_evidence_v3",
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            parsed = _ensure_event_grounded_relevance(
                _LLMResponse.model_validate(
                    _normalize_known_model_aliases(_parse_response_object(response.content))
                )
            )
            if parsed.disposition != "accepted" and parsed.semantic_events:
                raise ValueError("non_accepted_response_contains_semantic_events")
            response_evidence = [_ground_span(span, sources) for span in parsed.response_evidence]
            if parsed.disposition == "accepted" and not any(
                "relevance" in span.supports for span in parsed.response_evidence
            ):
                raise ValueError("accepted_relevance_without_evidence")
            events = []
            candidate_invoice_refs = {
                candidate.invoice_number for candidate in request.candidate_invoices
            }
            for raw_event in parsed.semantic_events:
                event = _drop_ungrounded_material_fields(raw_event)
                event = _drop_ungrounded_invoice_refs(event)
                event = _drop_unadmitted_invoice_refs(
                    event, candidate_invoice_refs=candidate_invoice_refs
                )
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
            return _response_for(
                request,
                relevance=parsed.relevance,
                semantic_events=events,
                response_evidence=response_evidence,
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
                    is_fallback=False,
                ),
            )
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return _response_for(
                request,
                relevance="uncertain",
                disposition="invalid",
                confidence=0,
                reason_codes=[
                    "semantic_response_schema_or_grounding_invalid",
                    _safe_validation_reason(exc),
                ],
                operation_summary=_summary_from_provider(
                    request,
                    response=response,
                    latency_ms=latency_ms,
                    success=False,
                    error_code="semantic_response_schema_or_grounding_invalid",
                    is_fallback=False,
                ),
            )


def _v2_summary_without_provider(request: MailSemanticEvidenceRequestV2) -> AIOperationSummaryV1:
    """Record a V2 deterministic outcome without inventing an LLM invocation."""

    return AIOperationSummaryV1(
        request_id=str(uuid4()),
        operation=_V2_OPERATION,
        cache_hit=True,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=_V2_PROMPT_TEMPLATE_VERSION,
        response_schema_version=_V2_RESPONSE_SCHEMA_VERSION,
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


def _v2_summary_from_provider(
    request: MailSemanticEvidenceRequestV2,
    *,
    response: object,
    latency_ms: int,
    success: bool,
    error_code: str | None = None,
    is_fallback: bool = False,
) -> AIOperationSummaryV1:
    """Build the original V2 audit summary without mixing it with V3 telemetry."""

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
        operation=_V2_OPERATION,
        suboperation="interpret",
        attempt_index=0,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=_V2_PROMPT_TEMPLATE_VERSION,
        response_schema_version=_V2_RESPONSE_SCHEMA_VERSION,
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
        operation=_V2_OPERATION,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_version=_V2_PROMPT_TEMPLATE_VERSION,
        response_schema_version=_V2_RESPONSE_SCHEMA_VERSION,
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


class MailSemanticEvidenceInterpreterV2:
    """Preserve the deployed V2 interpreter while V3 remains opt-in.

    V2 continues to use the existing Vertex-primary/OpenAI-fallback client and
    V2 request/response contracts.  It is intentionally not routed through
    the V3 OpenRouter-only provider or V3 admission receipt.
    """

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
                operation_summary=_v2_summary_without_provider(request),
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
                operation_summary=_v2_summary_without_provider(request),
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
            caller=_V2_OPERATION,
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
                operation_summary=_v2_summary_from_provider(
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
                operation_summary=_v2_summary_from_provider(
                    request,
                    response=response,
                    latency_ms=latency_ms,
                    success=False,
                    error_code="semantic_response_schema_or_grounding_invalid",
                    is_fallback=response.provider != client.primary_provider_name,
                ),
            )


mail_semantic_evidence_interpreter_v2 = MailSemanticEvidenceInterpreterV2()
mail_semantic_evidence_interpreter_v3 = MailSemanticEvidenceInterpreterV3()
