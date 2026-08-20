"""Vertex-first, fail-closed extraction of invoice control assertions from manual notes."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from solvix_contracts.ai import (
    ManualNoteAssertionV1,
    ManualNoteInterpretationRequestV1,
    ManualNoteInterpretationResponseV1,
)

from src.api.errors import LLMResponseInvalidError
from src.llm.factory import LLMProviderWithFallback

from ._evidence_grounding import (
    amount_is_explicit_in_span,
    date_is_explicit_in_span,
    locate_evidence,
    normalize_ref,
    reference_is_explicit_in_span,
    validate_optional_field_evidence,
)
from .audit import build_ai_audit
from .collection_email_event_classifier import _invalid_response_telemetry, _parse_response_object

PROMPT_TEMPLATE_ID = "manual_note_interpretation"
PROMPT_TEMPLATE_VERSION = "v5"
TAXONOMY_VERSION = "manual_note_controls.v2"


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
    resolves_manual_query_history_ref: str | None = None
    evidence_text: str
    confidence: float
    reason_codes: list[str]


class _ManualNoteLLMResponse(BaseModel):
    """Provider-facing schema; transport/audit fields are added after validation."""

    model_config = ConfigDict(extra="forbid")

    extraction_status: Literal["accepted", "abstained", "invalid"]
    assertions: list[_ManualNoteLLMAssertion]
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
- chase is collection activity and normally produces no query, commitment, or remittance assertion.
  The sole exception is a query transition=resolved that is explicitly supported by
  the new note and names exactly one matching opaque history_ref from
  active_manual_queries for that same selected invoice. Never raise, update, or
  reopen a control from a chase note.
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
  currency, or reference is optional unless explicitly present in the note. A collector saying that
  payment has not been made, is overdue, or is still awaited is not a remittance claim; abstain unless
  the note actually reports a remittance, payment transfer, or remittance advice.
- outreach activity alone has no operational invoice control.

TRANSITION CONTRACT
- query: raised, active, awaiting_response, updated, resolved, reopened, cancelled, or unclear.
- commitment: made, revised, active, kept, partially_kept, broken, cancelled, superseded, or unclear.
- remittance: received, expected, partially_received, not_received, unmatched, rejected, cancelled, or
  unclear. The verified transition is reserved for accounting and is forbidden for manual notes.
- other: no_operational_effect or unclear.

ACTIVE MANUAL QUERY HISTORY
- active_manual_queries is the complete, invoice-scoped set of active manual
  query controls that this note may resolve. history_ref is an opaque identifier:
  never invent, transform, reuse across invoices, or treat it as note evidence.
- A query transition=resolved must have exactly one invoice and must return the
  exact history_ref of the active manual query it resolves. Do not resolve a
  Sage query or a query not present in this list.
- A status update such as "POD received" or "invoice submitted" is not by
  itself a financial control. It becomes query resolution only when it clearly
  resolves the active manual query for that same invoice. If that relationship
  is uncertain, abstain.

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
- Never convert relative time language into asserted_date. A phrase such as next week, end of month,
  or in a few days is not an exact calendar date. For query and remittance assertions, leave
  asserted_date and date_evidence_text null unless the note contains an exact calendar date. For a
  commitment requiring a date, abstain when the note contains only relative time language.
- Normalize every non-null asserted_date as an ISO calendar date in YYYY-MM-DD form. Preserve the
  source's original date formatting only in date_evidence_text.
- Commitments without an explicit amount set full_current_balance=true. Every non-commitment sets it false.
- Never mark a remittance verified and never treat a manual note as accounting settlement.
- When no safe assertion exists, return extraction_status=abstained with no assertions.
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

_VALIDATION_REMEDIATION = {
    "asserted_date_not_grounded_in_field_evidence": (
        "Do not infer a calendar date from relative language or metadata. Set asserted_date and "
        "date_evidence_text to null unless an exact calendar date is present in the note's quoted "
        "field evidence. If the selected control requires an exact date, abstain."
    ),
    "amount_not_grounded_in_field_evidence": (
        "Set amount, currency, and amount_evidence_text to null unless the exact amount occurs in "
        "the note's quoted field evidence."
    ),
    "reference_not_grounded_in_field_evidence": (
        "Set reference and reference_evidence_text to null unless the exact reference occurs in "
        "the note's quoted field evidence."
    ),
}


_date_is_explicit_in_span = date_is_explicit_in_span
_normalize_ref = normalize_ref
_locate_evidence = locate_evidence
_amount_is_explicit_in_span = amount_is_explicit_in_span
_reference_is_explicit_in_span = reference_is_explicit_in_span


def _active_manual_query_history_by_ref(
    request: ManualNoteInterpretationRequestV1,
) -> dict[str, tuple[str, str]]:
    """Return the opaque active-manual-query history bound to each invoice.

    The backend supplies only an opaque history reference to the AI.  The ETL
    reducer independently rechecks it against live OCS provenance before it
    performs an append-only resolution, so this helper is a response-shape
    guard rather than trust in a model-provided control identity.
    """

    history: dict[str, tuple[str, str]] = {}
    for row in getattr(request, "active_manual_queries", ()) or ():
        history_ref = str(getattr(row, "history_ref", "") or "")
        obligation_id = str(getattr(row, "obligation_id", "") or "")
        invoice_number = _normalize_ref(getattr(row, "invoice_number", ""))
        if history_ref and obligation_id and invoice_number:
            history[history_ref] = (obligation_id, invoice_number)
    return history


def _validate_optional_field_evidence(
    *,
    note: str,
    candidate: dict[str, object],
    field: str,
    assertion_evidence: str,
) -> str | None:
    return validate_optional_field_evidence(
        source_text=note,
        candidate=candidate,
        field=field,
        assertion_evidence=assertion_evidence,
        evidence_key=f"{field.removeprefix('asserted_')}_evidence_text",
    )


def _validated_assertions(
    request: ManualNoteInterpretationRequestV1,
    raw_assertions: object,
) -> list[ManualNoteAssertionV1]:
    if not isinstance(raw_assertions, list):
        raise ValueError("assertions_must_be_list")
    allowed_refs = {
        _normalize_ref(row.invoice_number): row.invoice_number for row in request.invoice_facts
    }
    obligation_id_by_ref = {
        _normalize_ref(row.invoice_number): row.obligation_id for row in request.invoice_facts
    }
    remittance_claim_by_obligation = {
        row.obligation_id: bool(
            row.remittance_received_at
            or row.remittance_amount is not None
            or row.remittance_reference
        )
        for row in request.existing_controls
    }
    active_manual_query_history = _active_manual_query_history_by_ref(request)
    purpose = str(request.purpose)
    allowed_types_by_purpose: dict[str, set[str] | None] = {
        "query": {"query"},
        "commitment": {"commitment"},
        "remittance": {"remittance"},
        # A chase note can only release a specific, currently active manual
        # query.  The transition/reference guard below rejects every other
        # control mutation from this operator purpose.
        "chase": {"query"},
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
        try:
            date_span = _validate_optional_field_evidence(
                note=request.note,
                candidate=candidate,
                field="asserted_date",
                assertion_evidence=assertion_evidence,
            )
            if date_span is not None:
                date.fromisoformat(str(candidate.get("asserted_date")))
                if not _date_is_explicit_in_span(candidate.get("asserted_date"), date_span):
                    raise ValueError("asserted_date_not_grounded_in_field_evidence")
        except ValueError:
            if assertion_type not in {"query", "remittance"}:
                raise
            candidate["asserted_date"] = None
            candidate["date_evidence_text"] = None
            candidate["reason_codes"] = list(
                dict.fromkeys(
                    [*(candidate.get("reason_codes") or []), "ungrounded_optional_date_removed"]
                )
            )
            date_span = None
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
                "resolves_manual_query_history_ref",
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
        query_history_ref = str(getattr(assertion, "resolves_manual_query_history_ref", "") or "")
        if assertion.assertion_type == "query" and assertion.transition == "resolved":
            if len(assertion.invoice_refs) != 1:
                raise ValueError("query_resolution_requires_single_invoice_scope")
            expected_history = active_manual_query_history.get(query_history_ref)
            invoice_ref = _normalize_ref(assertion.invoice_refs[0])
            obligation_id = obligation_id_by_ref[invoice_ref]
            if not expected_history or expected_history != (obligation_id, invoice_ref):
                raise ValueError("query_resolution_history_ref_mismatch")
        elif query_history_ref:
            raise ValueError("manual_query_history_ref_only_for_query_resolution")
        if str(request.purpose) == "chase" and not (
            assertion.assertion_type == "query" and assertion.transition == "resolved"
        ):
            raise ValueError("chase_note_cannot_create_or_update_control")
        if (
            assertion.assertion_type == "remittance"
            and assertion.transition in {"not_received", "rejected", "cancelled"}
            and not all(
                remittance_claim_by_obligation.get(
                    obligation_id_by_ref[_normalize_ref(invoice_ref)],
                    False,
                )
                for invoice_ref in assertion.invoice_refs
            )
        ):
            raise ValueError("remittance_clear_without_active_claim")
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
        "active_manual_queries": [
            {
                "history_ref": str(getattr(row, "history_ref", "") or ""),
                "obligation_id": str(getattr(row, "obligation_id", "") or ""),
                "invoice_number": str(getattr(row, "invoice_number", "") or ""),
                "transition": str(getattr(row, "transition", "") or ""),
                "reason_code": str(getattr(row, "reason_code", "") or ""),
                "opened_at": str(getattr(row, "opened_at", "") or ""),
            }
            for row in getattr(request, "active_manual_queries", ()) or ()
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
        active_system_prompt = system_prompt
        responses: list[object] = []
        last_exc: Exception | None = None
        response = None
        user_prompt = ""
        for _ in range(max_attempts):
            response, user_prompt = await self._complete(
                system_prompt=active_system_prompt,
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
                validation_code = _validation_code(exc)
                required_correction = _VALIDATION_REMEDIATION.get(
                    validation_code,
                    "Correct the proposed extraction to satisfy the system contract exactly.",
                )
                try:
                    proposed = _parse_response_object(response.content)
                except (ValidationError, ValueError, TypeError):
                    proposed = None
                payload = {
                    "request": source_request_payload,
                    "proposed_extraction": proposed,
                    "validation_feedback": validation_code,
                    "required_correction": required_correction,
                }
                active_system_prompt = (
                    f"{system_prompt}\n\nVALIDATION CORRECTION MODE\n"
                    f"The previous output failed {validation_code}. {required_correction} "
                    "Return a fresh corrected extraction and do not repeat the invalid field value."
                )
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
        if str(request.purpose) == "chase" and not (
            getattr(request, "active_manual_queries", ()) or ()
        ):
            return ManualNoteInterpretationResponseV1(
                extraction_status="abstained",
                assertions=[],
                reason_codes=["authoritative_chase_no_control"],
            )
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
        reason_codes = list(dict.fromkeys([*reason_codes, "structural_validation_passed"]))

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
                system_prompt=_SYSTEM_PROMPT,
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
