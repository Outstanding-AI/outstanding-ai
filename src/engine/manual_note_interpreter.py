"""Vertex-first, fail-closed extraction of invoice control assertions from manual notes."""

from __future__ import annotations

import json
import re
import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from solvix_contracts.ai import (
    ManualNoteAssertionV1,
    ManualNoteInterpretationRequestV1,
    ManualNoteInterpretationResponseV1,
)

from src.api.errors import LLMResponseInvalidError
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback

from .audit import build_ai_audit
from .collection_email_event_classifier import _invalid_response_telemetry, _parse_response_object

PROMPT_TEMPLATE_ID = "manual_note_interpretation"
PROMPT_TEMPLATE_VERSION = "v2"
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
    reference: str | None = None
    full_current_balance: bool = False
    evidence_start: int
    evidence_end: int
    confidence: float
    reason_codes: list[str]


class _ManualNoteLLMResponse(BaseModel):
    """Provider-facing schema; transport/audit fields are added after validation."""

    model_config = ConfigDict(extra="forbid")

    extraction_status: Literal["accepted", "abstained", "invalid"]
    assertions: list[_ManualNoteLLMAssertion]
    reason_codes: list[str]


_SYSTEM_PROMPT = """Extract source-grounded invoice control assertions from one operator manual note.
Return strict JSON with exactly: extraction_status, assertions, reason_codes.
The note is evidence of what the operator reported; it is never proof of accounting settlement.
Allowed assertion_type values: query, commitment, remittance, other.
invoice_facts is the exact invoice scope already linked or uniquely resolved for this note. Use only
invoice numbers present in invoice_facts; never choose an invoice from debtor identity alone. When
invoice_facts contains exactly one invoice, a clear control statement applies to that invoice even
when the note does not repeat its invoice number.
For every assertion return: assertion_id, assertion_type, transition, polarity, temporal_orientation,
invoice_refs, amount, currency, asserted_date, reference, full_current_balance, evidence_start,
evidence_end, confidence, reason_codes. evidence_start/end are zero-based offsets into the exact note
and must cover non-empty supporting text.

Commitments require an explicit date. If a dated commitment has no amount, set
full_current_balance=true. For every non-commitment assertion, full_current_balance must be false.
Never divide one total among several invoices. If one explicit total is ambiguous across multiple
invoices, abstain. Remittance received is only an operator claim; do not mark it verified. Query
raised/active can be asserted without an accounting-system flag. Treat negation, uncertainty,
cancellation, and resolution explicitly. Do not invent dates, amounts, currencies, references, or
invoice numbers. When nothing safe is asserted, return
{"extraction_status":"abstained","assertions":[],"reason_codes":["no_safe_operational_assertion"]}.
Example: for the exact note "REMIT RECEIVED" with one invoice_fact, return one remittance assertion
with transition="received", polarity="affirmed", temporal_orientation="current", that invoice number
in invoice_refs, no amount/date/reference, and the full note as evidence. This remains an unverified
operator claim and must never use transition="verified".
Do not add prose or any other key."""

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

_EXPLICIT_SINGLE_INVOICE_REMITTANCE_RECEIVED = re.compile(
    r"^\s*(?:remit|remittance)(?:\s+(?:has\s+been|was))?\s+received\s*[.!]?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_REMITTANCE_EVIDENCE = re.compile(
    r"\bremit(?:tance)?\b|\bpayment\s+(?:advice|confirmation|reference)\b",
    re.IGNORECASE,
)


def _remittance_date_is_explicit_in_note(value: object, note: str) -> bool:
    """Return whether a normalised remittance date is explicitly in the note.

    ``occurred_at`` is provided as request metadata, not source evidence.  A
    model must therefore never turn it into a claimed payment date.  The
    bounded variants cover the calendar representations we can safely recover
    without interpreting relative language such as "tomorrow".
    """

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
    source = str(note or "").casefold()
    return any(candidate.casefold() in source for candidate in candidates)


def _normalize_ref(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _recover_explicit_single_invoice_remittance(
    request: ManualNoteInterpretationRequestV1,
    *,
    status: str,
    assertions: list[ManualNoteAssertionV1],
) -> tuple[str, list[ManualNoteAssertionV1], list[str]] | None:
    """Recover one narrow, source-explicit claim when the model abstains."""

    if status != "abstained" or assertions or len(request.invoice_facts) != 1:
        return None
    note = str(request.note or "")
    if not _EXPLICIT_SINGLE_INVOICE_REMITTANCE_RECEIVED.fullmatch(note):
        return None
    invoice_number = str(request.invoice_facts[0].invoice_number)
    assertion = ManualNoteAssertionV1(
        assertion_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{request.touch_id}:explicit-single-invoice-remittance-received",
            )
        ),
        assertion_type="remittance",
        transition="received",
        polarity="affirmed",
        temporal_orientation="current",
        invoice_refs=[invoice_number],
        amount=None,
        currency=None,
        asserted_date=None,
        reference=None,
        full_current_balance=False,
        evidence_start=0,
        evidence_end=len(note),
        confidence=1.0,
        reason_codes=["explicit_single_invoice_remittance_received"],
    )
    return "accepted", [assertion], ["deterministic_explicit_claim_recovery"]


def _validated_assertions(
    request: ManualNoteInterpretationRequestV1,
    raw_assertions: object,
) -> tuple[list[ManualNoteAssertionV1], list[str]]:
    if not isinstance(raw_assertions, list):
        raise ValueError("assertions_must_be_list")
    allowed_refs = {
        _normalize_ref(row.invoice_number): row.invoice_number for row in request.invoice_facts
    }
    assertions: list[ManualNoteAssertionV1] = []
    normalization_reason_codes: list[str] = []
    has_existing_remittance = any(
        control.remittance_received_at
        or control.remittance_amount is not None
        or control.remittance_reference
        for control in request.existing_controls
    )
    for index, raw in enumerate(raw_assertions):
        if not isinstance(raw, dict):
            raise ValueError("assertion_must_be_object")
        candidate = dict(raw)
        candidate.setdefault(
            "assertion_id",
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{request.touch_id}:{index}:{json.dumps(raw, sort_keys=True)}",
                )
            ),
        )
        if candidate.get("assertion_type") != "commitment":
            candidate["full_current_balance"] = False
        assertion_type = candidate.get("assertion_type")
        transition = candidate.get("transition")
        # A remittance date is optional.  Do not reject an otherwise safe
        # claim merely because a model copied request metadata (for example
        # ``occurred_at``) as a timestamp.  Drop only that ungrounded optional
        # detail; the accounting reconciler remains the verification authority.
        if (
            assertion_type == "remittance"
            and candidate.get("asserted_date")
            and not _remittance_date_is_explicit_in_note(
                candidate.get("asserted_date"), request.note
            )
        ):
            candidate["asserted_date"] = None
            normalization_reason_codes.append("ungrounded_remittance_date_dropped")
        if (
            assertion_type in _TRANSITIONS_BY_TYPE
            and transition not in _TRANSITIONS_BY_TYPE[assertion_type]
        ):
            normalization_reason_codes.append("unsupported_assertion_transition_dropped")
            continue
        if (
            assertion_type == "remittance"
            and transition in {"not_received", "unmatched", "rejected", "cancelled"}
            and not has_existing_remittance
            and not _EXPLICIT_REMITTANCE_EVIDENCE.search(request.note)
        ):
            normalization_reason_codes.append("unscoped_negative_remittance_dropped")
            continue
        assertion = ManualNoteAssertionV1.model_validate(candidate)
        if assertion.transition not in _TRANSITIONS_BY_TYPE[assertion.assertion_type]:
            raise ValueError("assertion_transition_type_mismatch")
        if assertion.evidence_end > len(request.note):
            raise ValueError("assertion_evidence_span_out_of_bounds")
        if not request.note[assertion.evidence_start : assertion.evidence_end].strip():
            raise ValueError("assertion_evidence_span_empty")
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
            if assertion.amount is None and assertion.transition in {"made", "revised", "active"}:
                assertion.full_current_balance = True
        assertions.append(assertion)
    return assertions, normalization_reason_codes


class ManualNoteInterpreter:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def interpret(
        self,
        request: ManualNoteInterpretationRequestV1,
    ) -> ManualNoteInterpretationResponseV1:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            json_mode=True,
            response_schema=_ManualNoteLLMResponse,
            caller="manual_note_interpretation",
        )
        try:
            raw = _parse_response_object(response.content)
            if set(raw) - {"extraction_status", "assertions", "reason_codes"}:
                raise ValueError("manual_note_response_unknown_fields")
            status = raw.get("extraction_status", "invalid")
            if status not in {"accepted", "abstained", "invalid"}:
                raise ValueError("manual_note_extraction_status_invalid")
            assertions, normalization_reason_codes = _validated_assertions(
                request,
                raw.get("assertions") or [],
            )
            if status == "accepted" and not assertions:
                status = "abstained"
            if status != "accepted" and assertions:
                raise ValueError("non_accepted_response_contains_assertions")
            reason_codes = raw.get("reason_codes") or []
            if not isinstance(reason_codes, list) or not all(
                isinstance(value, str) for value in reason_codes
            ):
                raise ValueError("manual_note_reason_codes_invalid")
            reason_codes = list(dict.fromkeys([*reason_codes, *normalization_reason_codes]))
            recovered = _recover_explicit_single_invoice_remittance(
                request,
                status=status,
                assertions=assertions,
            )
            if recovered is not None:
                status, assertions, reason_codes = recovered
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid manual-note interpretation",
                details={
                    "operation": "manual_note_interpretation",
                    "telemetry": _invalid_response_telemetry(response),
                },
            ) from exc

        usage = response.usage or {}
        return ManualNoteInterpretationResponseV1(
            extraction_status=status,
            assertions=assertions,
            reason_codes=reason_codes,
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            provider=response.provider,
            model=response.model,
            is_fallback=response.provider != self._client.primary_provider_name,
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_input=prompt_input,
                token_count=usage.get("total_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                inference_profile="manual_note_interpretation",
            ).model_dump(mode="json"),
        )


manual_note_interpreter = ManualNoteInterpreter()
