"""Vertex-first, fail-closed extraction of invoice control assertions from manual notes."""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from solvix_contracts.ai import (
    ManualNoteAssertionV1,
    ManualNoteInterpretationRequestV1,
    ManualNoteInterpretationResponseV1,
)

from src.api.errors import LLMResponseInvalidError
from src.llm.factory import LLMProviderWithFallback

from .audit import build_ai_audit
from .collection_email_event_classifier import _invalid_response_telemetry, _parse_response_object

PROMPT_TEMPLATE_ID = "manual_note_interpretation"
PROMPT_TEMPLATE_VERSION = "v4"
TAXONOMY_VERSION = "manual_note_controls.v1"


class _ManualNoteLLMAssertion(BaseModel):
    """Vertex-compatible shape; contract constraints are enforced after generation."""

    model_config = ConfigDict(extra="forbid")

    assertion_id: str
    assertion_type: Literal["query", "commitment", "remittance", "other"]
    transition: Literal[
        "raised",
        "active",
        "awaiting_response",
        "updated",
        "resolved",
        "reopened",
        "made",
        "revised",
        "kept",
        "partially_kept",
        "broken",
        "received",
        "expected",
        "partially_received",
        "not_received",
        "unmatched",
        "verified",
        "rejected",
        "cancelled",
        "superseded",
        "unclear",
        "no_operational_effect",
    ]
    polarity: Literal["affirmed", "negated", "uncertain"]
    temporal_orientation: Literal["past", "current", "future", "unclear"]
    invoice_refs: list[str]
    amount: float | None = None
    currency: str | None = None
    asserted_date: str | None = None
    date_evidence_text: str | None = None
    reference: str | None = None
    reference_evidence_text: str | None = None
    amount_evidence_text: str | None = None
    full_current_balance: bool = False
    evidence_text: str
    confidence: float
    reason_codes: list[str]


class _ManualNoteLLMResponse(BaseModel):
    """Provider-facing schema; transport/audit fields are added after validation."""

    model_config = ConfigDict(extra="forbid")

    extraction_status: Literal["accepted", "abstained", "invalid"]
    assertions: list[_ManualNoteLLMAssertion]
    reason_codes: list[str]


class _ManualNoteReviewResponse(BaseModel):
    """Narrow semantic verdict; the reviewer never regenerates extraction fields."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    reason_codes: list[str]


_SYSTEM_PROMPT = """You are a zero-shot semantic extractor for operator-authored accounts-receivable notes.
Return strict JSON matching the supplied schema. Do not add prose or undocumented keys.

SOURCE AUTHORITY
- The note is the only source for control type, transition, date, amount, currency, and reference.
- occurred_at, invoice due dates, balances, existing controls, and accounting watermarks are context;
  they are not evidence for a newly asserted fact.
- Never invent, calculate, copy from metadata, or complete a missing control fact.

OPERATOR PURPOSE
- query, commitment, and remittance are authoritative control-family selections. Emit assertions only
  from the selected family. The note determines the transition and extracted fields.
- chase is collection activity and must produce no query, commitment, or remittance assertion.
- general and internal_note do not constrain the family; extract any control directly supported by the note.
- If the purpose and prose conflict, or the required facts are absent, abstain rather than changing family.

INVOICE SCOPE
- invoice_facts is the complete maximum scope selected by the operator. Never add another invoice.
- For an actionable purpose, apply the supported control to all selected invoices even when invoice
  numbers are not repeated in the prose. Do not infer an amount allocation across multiple invoices.
- For general or internal_note, include only selected invoices to which the prose actually applies.

CONTROL ONTOLOGY
- query: a substantive invoice dispute, documentation/fulfilment blocker, incorrect charge, portal
  blocker, deduction, or other condition that pauses normal chasing. Routine outreach or a request by
  the collector for status is not a query.
- commitment: a debtor or counterparty commitment to pay on an explicit exact calendar date. Hopes,
  forecasts, conditional possibilities, invoice due dates, note timestamps, chase dates, and reminder
  dates are not commitment dates. An active/made/revised commitment without an exact payment date must
  abstain or be marked invalid.
- remittance: a claim that payment has actually been sent or received. It remains unverified accounting
  evidence. A completed statement that payment was sent or received uses transition=received; use
  partially_received only when the note says only part was paid. expected is limited to a future
  remittance document or advice and must never represent a future promise to pay. A date, amount,
  currency, or reference is optional unless explicitly present in the note.
- outreach activity alone has no operational invoice control.

TRANSITION CONTRACT
- query: raised, active, awaiting_response, updated, resolved, reopened, cancelled, or unclear.
- commitment: made, revised, active, kept, partially_kept, broken, cancelled, superseded, or unclear.
- remittance: received, expected, partially_received, not_received, unmatched, rejected, cancelled, or
  unclear. The verified transition is reserved for accounting and is forbidden for manual notes.
- other: no_operational_effect or unclear.

TEMPORAL AND CURRENT-STATE REASONING
- A note may contain multiple dated clauses. Bind each date only to the fact expressed in the same
  semantic clause; never choose a date merely because it appears earlier or later in the note.
- Prefer the final current state expressed by the note. Do not emit redundant historical and current
  assertions for the same control unless they describe distinct, simultaneously operative facts.
- Preserve negation, cancellation, uncertainty, resolution, and supersession.

GROUNDING
- evidence_text must be the smallest exact verbatim substring of the note supporting the assertion.
- For every non-null asserted_date, amount, or reference, return the corresponding field-specific
  evidence_text as an exact verbatim substring inside the assertion evidence. Use null field evidence
  when the field is null. Extend a quote only when needed to make its occurrence unique in the note.
- Normalize every non-null asserted_date as an ISO calendar date in YYYY-MM-DD form. Preserve the
  source's original date formatting only in date_evidence_text.
- Commitments without an explicit amount set full_current_balance=true. Every non-commitment sets it false.
- Never mark a remittance verified and never treat a manual note as accounting settlement.
- When no safe assertion exists, return extraction_status=abstained with no assertions.
"""

_REVIEW_PROMPT = """Act as an independent semantic guardrail for a proposed manual-note extraction.
Return only a strict accept/reject verdict matching the supplied schema; never regenerate extraction fields.
Use the original note as the only factual source and enforce the operator purpose and selected invoice
scope. Check control family, transition, negation, current state, invoice coverage, and semantic clause
ownership of every date, amount, and reference. Reject an unsupported assertion or an incorrect abstention.
Use short snake_case reason codes without quoting or reproducing note content. Do not add prose.
"""

_TRANSITIONS_BY_TYPE = {
    "query": {
        "raised",
        "active",
        "awaiting_response",
        "updated",
        "resolved",
        "reopened",
        "cancelled",
        "unclear",
    },
    "commitment": {
        "made",
        "revised",
        "active",
        "kept",
        "partially_kept",
        "broken",
        "cancelled",
        "superseded",
        "unclear",
    },
    "remittance": {
        "received",
        "expected",
        "partially_received",
        "not_received",
        "unmatched",
        "verified",
        "rejected",
        "cancelled",
        "unclear",
    },
    "other": {"no_operational_effect", "unclear"},
}


def _date_is_explicit_in_span(value: object, span: str) -> bool:
    """Verify a model-extracted date against its model-selected source span."""

    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    candidates = {
        parsed.isoformat(),
        parsed.strftime("%d/%m/%Y"),
        parsed.strftime("%d-%m-%Y"),
        parsed.strftime("%d.%m.%Y"),
        f"{parsed.day}/{parsed.month}/{parsed.year}",
        f"{parsed.day}-{parsed.month}-{parsed.year}",
        f"{parsed.day} {parsed.strftime('%B')} {parsed.year}",
        f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}",
    }
    source = str(span or "").casefold()
    return any(candidate.casefold() in source for candidate in candidates)


def _normalize_ref(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _locate_evidence(note: str, evidence: object, *, field: str) -> tuple[int, int, str]:
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"{field}_evidence_text_required")
    start = note.find(evidence)
    if start < 0:
        raise ValueError(f"{field}_evidence_not_verbatim")
    if note.find(evidence, start + 1) >= 0:
        raise ValueError(f"{field}_evidence_not_unique")
    return start, start + len(evidence), evidence


def _validate_optional_field_evidence(
    *,
    note: str,
    candidate: dict[str, object],
    field: str,
    assertion_evidence: str,
) -> str | None:
    value = candidate.get(field)
    evidence_key = f"{field.removeprefix('asserted_')}_evidence_text"
    evidence = candidate.get(evidence_key)
    if value is None:
        if evidence is not None:
            raise ValueError(f"null_{field}_must_not_have_evidence")
        return None
    _, _, span = _locate_evidence(note, evidence, field=field)
    if span not in assertion_evidence:
        raise ValueError(f"{field}_evidence_outside_assertion")
    return span


def _amount_is_explicit_in_span(value: object, span: str) -> bool:
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False
    for raw in re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?", span):
        try:
            if Decimal(raw.replace(",", "")) == expected:
                return True
        except InvalidOperation:
            continue
    return False


def _reference_is_explicit_in_span(value: object, span: str) -> bool:
    normalized_value = _normalize_ref(str(value or ""))
    return bool(normalized_value) and normalized_value in _normalize_ref(span)


def _validated_assertions(
    request: ManualNoteInterpretationRequestV1,
    raw_assertions: object,
) -> list[ManualNoteAssertionV1]:
    if not isinstance(raw_assertions, list):
        raise ValueError("assertions_must_be_list")
    allowed_refs = {
        _normalize_ref(row.invoice_number): row.invoice_number for row in request.invoice_facts
    }
    purpose = str(request.purpose)
    allowed_types_by_purpose: dict[str, set[str] | None] = {
        "query": {"query"},
        "commitment": {"commitment"},
        "remittance": {"remittance"},
        "chase": set(),
        "general": None,
        "internal_note": None,
    }
    if purpose not in allowed_types_by_purpose:
        raise ValueError("manual_note_purpose_unsupported")
    purpose_types = allowed_types_by_purpose[purpose]
    assertions: list[ManualNoteAssertionV1] = []
    duplicate_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw in raw_assertions:
        if not isinstance(raw, dict):
            raise ValueError("assertion_must_be_object")
        candidate = dict(raw)
        assertion_type = candidate.get("assertion_type")
        transition = candidate.get("transition")
        if assertion_type not in _TRANSITIONS_BY_TYPE:
            raise ValueError("assertion_type_unsupported")
        if purpose_types is not None and assertion_type not in purpose_types:
            raise ValueError("assertion_conflicts_with_operator_purpose")
        if transition not in _TRANSITIONS_BY_TYPE[assertion_type]:
            raise ValueError("assertion_transition_type_mismatch")
        if assertion_type == "remittance" and transition == "verified":
            raise ValueError("manual_note_cannot_verify_remittance")

        assertion_start, assertion_end, assertion_evidence = _locate_evidence(
            request.note,
            candidate.get("evidence_text"),
            field="assertion",
        )
        date_span = _validate_optional_field_evidence(
            note=request.note,
            candidate=candidate,
            field="asserted_date",
            assertion_evidence=assertion_evidence,
        )
        amount_span = _validate_optional_field_evidence(
            note=request.note,
            candidate=candidate,
            field="amount",
            assertion_evidence=assertion_evidence,
        )
        reference_span = _validate_optional_field_evidence(
            note=request.note,
            candidate=candidate,
            field="reference",
            assertion_evidence=assertion_evidence,
        )
        if date_span is not None:
            try:
                date.fromisoformat(str(candidate.get("asserted_date")))
            except (TypeError, ValueError) as exc:
                raise ValueError("asserted_date_must_be_iso_date") from exc
            if not _date_is_explicit_in_span(candidate.get("asserted_date"), date_span):
                raise ValueError("asserted_date_not_grounded_in_field_evidence")
        if amount_span is not None and not _amount_is_explicit_in_span(
            candidate.get("amount"), amount_span
        ):
            raise ValueError("amount_not_grounded_in_field_evidence")
        if reference_span is not None and not _reference_is_explicit_in_span(
            candidate.get("reference"), reference_span
        ):
            raise ValueError("reference_not_grounded_in_field_evidence")
        if candidate.get("currency") is not None and candidate.get("amount") is None:
            raise ValueError("currency_requires_amount")

        contract_candidate = {
            key: candidate.get(key)
            for key in (
                "assertion_id",
                "assertion_type",
                "transition",
                "polarity",
                "temporal_orientation",
                "invoice_refs",
                "amount",
                "currency",
                "asserted_date",
                "reference",
                "full_current_balance",
                "confidence",
                "reason_codes",
            )
        }
        contract_candidate["evidence_start"] = assertion_start
        contract_candidate["evidence_end"] = assertion_end
        assertion = ManualNoteAssertionV1.model_validate(contract_candidate)
        if assertion.transition not in _TRANSITIONS_BY_TYPE[assertion.assertion_type]:
            raise ValueError("assertion_transition_type_mismatch")
        resolved_refs: list[str] = []
        for invoice_ref in assertion.invoice_refs:
            normalized = _normalize_ref(invoice_ref)
            if normalized not in allowed_refs:
                raise ValueError("assertion_invoice_outside_candidate_scope")
            resolved_refs.append(allowed_refs[normalized])
        assertion.invoice_refs = list(dict.fromkeys(resolved_refs))
        if assertion.assertion_type != "other" and not assertion.invoice_refs:
            raise ValueError("operational_assertion_missing_invoice_scope")
        if len(assertion.invoice_refs) > 1 and assertion.amount is not None:
            raise ValueError("ambiguous_multi_invoice_amount")
        if assertion.asserted_date:
            try:
                date.fromisoformat(assertion.asserted_date)
            except ValueError as exc:
                raise ValueError("asserted_date_must_be_iso_date") from exc
        if assertion.assertion_type == "commitment":
            if (
                assertion.transition in {"made", "revised", "active"}
                and not assertion.asserted_date
            ):
                raise ValueError("commitment_date_required")
            if (
                assertion.amount is None
                and assertion.transition in {"made", "revised", "active"}
                and not assertion.full_current_balance
            ):
                raise ValueError("commitment_without_amount_requires_full_current_balance")
        duplicate_key = (
            assertion.assertion_type,
            assertion.transition,
            tuple(sorted(_normalize_ref(value) for value in assertion.invoice_refs)),
        )
        if duplicate_key in duplicate_keys:
            raise ValueError("redundant_current_assertion")
        duplicate_keys.add(duplicate_key)
        assertions.append(assertion)
    if purpose_types is not None and purpose != "chase" and assertions:
        covered_refs = {
            _normalize_ref(invoice_ref)
            for assertion in assertions
            for invoice_ref in assertion.invoice_refs
        }
        if covered_refs != set(allowed_refs):
            raise ValueError("actionable_purpose_must_cover_selected_invoices")
    return assertions


def _validation_code(exc: Exception) -> str:
    if isinstance(exc, ValueError) and str(exc):
        code = str(exc)
        if re.fullmatch(r"[a-z0-9_]+", code):
            return code
    return "manual_note_response_schema_invalid"


def _decode_response(
    request: ManualNoteInterpretationRequestV1,
    response: object,
) -> tuple[dict[str, object], str, list[ManualNoteAssertionV1], list[str]]:
    raw = _parse_response_object(response.content)
    if set(raw) - {"extraction_status", "assertions", "reason_codes"}:
        raise ValueError("manual_note_response_unknown_fields")
    status = raw.get("extraction_status", "invalid")
    if status not in {"accepted", "abstained", "invalid"}:
        raise ValueError("manual_note_extraction_status_invalid")
    assertions = _validated_assertions(request, raw.get("assertions") or [])
    if status == "accepted" and not assertions:
        raise ValueError("accepted_response_requires_assertions")
    if status != "accepted" and assertions:
        raise ValueError("non_accepted_response_contains_assertions")
    reason_codes = raw.get("reason_codes") or []
    if not isinstance(reason_codes, list) or not all(
        isinstance(value, str) for value in reason_codes
    ):
        raise ValueError("manual_note_reason_codes_invalid")
    return raw, status, assertions, list(dict.fromkeys(reason_codes))


def _prompt_input(request: ManualNoteInterpretationRequestV1) -> dict[str, object]:
    """Return the source-minimal LLM payload.

    Balance, currency, due date, occurrence time, and accounting metadata are
    intentionally excluded. They are useful to reconciliation after extraction
    but are not note evidence and previously invited the model to copy metadata
    into asserted fields.
    """

    return {
        "note": request.note,
        "purpose": str(request.purpose),
        "channel": request.channel,
        "direction": request.direction,
        "tenant_timezone": request.tenant_timezone,
        "selected_invoices": [
            {
                "obligation_id": row.obligation_id,
                "invoice_number": row.invoice_number,
            }
            for row in request.invoice_facts
        ],
        "existing_control_states": [
            {
                "obligation_id": row.obligation_id,
                "collection_status": row.collection_status,
                "promise_status": row.promise_status,
                "has_remittance_claim": bool(
                    row.remittance_received_at
                    or row.remittance_amount is not None
                    or row.remittance_reference
                ),
            }
            for row in request.existing_controls
        ],
    }


class ManualNoteInterpreter:
    def __init__(self) -> None:
        # Manual controls can suppress legally and commercially significant
        # collection activity. Use the stronger context model as primary.
        self._client = LLMProviderWithFallback(
            primary_provider="openai", fallback_provider="vertex"
        )

    async def _complete(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        response_schema: type[BaseModel] = _ManualNoteLLMResponse,
    ):
        user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        response = await self._client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            json_mode=True,
            response_schema=response_schema,
            reasoning_effort="low",
            caller="manual_note_interpretation",
        )
        return response, user_prompt

    async def _review(
        self,
        *,
        request_payload: dict[str, object],
        proposed_extraction: dict[str, object],
    ):
        payload = {
            "request": request_payload,
            "proposed_extraction": proposed_extraction,
        }
        response, user_prompt = await self._complete(
            system_prompt=_REVIEW_PROMPT,
            payload=payload,
            response_schema=_ManualNoteReviewResponse,
        )
        try:
            raw = _parse_response_object(response.content)
            review = _ManualNoteReviewResponse.model_validate(raw)
            if not all(re.fullmatch(r"[a-z0-9_]+", value) for value in review.reason_codes):
                raise ValueError("manual_note_review_reason_codes_invalid")
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid manual-note semantic review",
                details={
                    "operation": "manual_note_interpretation_review",
                    "validation_code": _validation_code(exc),
                    "telemetry": _invalid_response_telemetry(response),
                },
            ) from exc
        return review, response, user_prompt, payload

    async def _generate_validated(
        self,
        *,
        request: ManualNoteInterpretationRequestV1,
        system_prompt: str,
        initial_payload: dict[str, object],
        source_request_payload: dict[str, object],
        max_attempts: int,
    ):
        payload = initial_payload
        responses: list[object] = []
        last_exc: Exception | None = None
        response = None
        user_prompt = ""
        for _ in range(max_attempts):
            response, user_prompt = await self._complete(
                system_prompt=system_prompt,
                payload=payload,
            )
            responses.append(response)
            try:
                raw, status, assertions, reason_codes = _decode_response(request, response)
                return (
                    raw,
                    status,
                    assertions,
                    reason_codes,
                    response,
                    user_prompt,
                    payload,
                    responses,
                )
            except (ValidationError, ValueError, TypeError) as exc:
                last_exc = exc
                try:
                    proposed = _parse_response_object(response.content)
                except (ValidationError, ValueError, TypeError):
                    proposed = None
                payload = {
                    "request": source_request_payload,
                    "proposed_extraction": proposed,
                    "validation_feedback": _validation_code(exc),
                }
        assert response is not None and last_exc is not None
        raise LLMResponseInvalidError(
            message="LLM returned invalid manual-note interpretation",
            details={
                "operation": "manual_note_interpretation",
                "validation_code": _validation_code(last_exc),
                "attempt_count": len(responses),
                "telemetry": _invalid_response_telemetry(response),
            },
        ) from last_exc

    async def interpret(
        self,
        request: ManualNoteInterpretationRequestV1,
    ) -> ManualNoteInterpretationResponseV1:
        prompt_input = _prompt_input(request)
        (
            raw,
            status,
            assertions,
            reason_codes,
            response,
            user_prompt,
            final_prompt_input,
            responses,
        ) = await self._generate_validated(
            request=request,
            system_prompt=_SYSTEM_PROMPT,
            initial_payload=prompt_input,
            source_request_payload=prompt_input,
            max_attempts=3,
        )
        review, response, user_prompt, final_prompt_input = await self._review(
            request_payload=prompt_input,
            proposed_extraction=raw,
        )
        responses.append(response)
        review_code = "semantic_guardrail_reviewed"
        if review.verdict == "reject":
            correction_payload = {
                "request": prompt_input,
                "proposed_extraction": raw,
                "semantic_review_feedback": review.reason_codes,
            }
            (
                raw,
                status,
                assertions,
                reason_codes,
                _,
                _,
                _,
                correction_responses,
            ) = await self._generate_validated(
                request=request,
                system_prompt=_SYSTEM_PROMPT,
                initial_payload=correction_payload,
                source_request_payload=prompt_input,
                max_attempts=3,
            )
            responses.extend(correction_responses)
            review, response, user_prompt, final_prompt_input = await self._review(
                request_payload=prompt_input,
                proposed_extraction=raw,
            )
            responses.append(response)
            if review.verdict == "reject":
                raise LLMResponseInvalidError(
                    message="LLM semantic guardrail rejected manual-note interpretation",
                    details={
                        "operation": "manual_note_interpretation_review",
                        "validation_code": "semantic_guardrail_rejected",
                        "review_reason_codes": review.reason_codes,
                        "telemetry": _invalid_response_telemetry(response),
                    },
                )
            review_code = "semantic_guardrail_corrected"
        final_system_prompt = _REVIEW_PROMPT
        reason_codes = list(dict.fromkeys([*reason_codes, review_code]))

        usage = {
            key: sum(int((item.usage or {}).get(key, 0) or 0) for item in responses)
            for key in ("total_tokens", "prompt_tokens", "completion_tokens", "reasoning_tokens")
        }
        return ManualNoteInterpretationResponseV1(
            extraction_status=status,
            assertions=assertions,
            reason_codes=reason_codes,
            tokens_used=usage["total_tokens"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            provider=response.provider,
            model=response.model,
            is_fallback=bool(getattr(response, "is_fallback", False)),
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                system_prompt=final_system_prompt,
                user_prompt=user_prompt,
                prompt_input=final_prompt_input,
                guardrail_pipeline_version="manual-note-semantic-grounding.v1",
                token_count=usage["total_tokens"],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                inference_profile="manual_note_interpretation",
            ).model_dump(mode="json"),
        )


manual_note_interpreter = ManualNoteInterpreter()
