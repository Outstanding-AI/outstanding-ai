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
PROMPT_TEMPLATE_VERSION = "v1"
TAXONOMY_VERSION = "manual_note_controls.v1"


class _ManualNoteLLMResponse(BaseModel):
    """Provider-facing schema; transport/audit fields are added after validation."""

    model_config = ConfigDict(extra="forbid")

    extraction_status: Literal["accepted", "abstained", "invalid"]
    assertions: list[ManualNoteAssertionV1]
    reason_codes: list[str]


_SYSTEM_PROMPT = """Extract source-grounded invoice control assertions from one operator manual note.
Return strict JSON with exactly: extraction_status, assertions, reason_codes.
The note is evidence of what the operator reported; it is never proof of accounting settlement.
Allowed assertion_type values: query, commitment, remittance, other.
Use only invoice numbers present in invoice_facts. Do not choose an invoice from debtor identity alone.
For every assertion return: assertion_id, assertion_type, transition, polarity, temporal_orientation,
invoice_refs, amount, currency, asserted_date, reference, full_current_balance, evidence_start,
evidence_end, confidence, reason_codes. evidence_start/end are zero-based offsets into the exact note
and must cover non-empty supporting text.

Commitments require an explicit date. If a dated commitment has no amount, set
full_current_balance=true. Never divide one total among several invoices. If one explicit total is
ambiguous across multiple invoices, abstain. Remittance received is only an operator claim; do not
mark it verified. Query raised/active can be asserted without an accounting-system flag. Treat
negation, uncertainty, cancellation, and resolution explicitly. Do not invent dates, amounts,
currencies, references, or invoice numbers. When nothing safe is asserted, return
{"extraction_status":"abstained","assertions":[],"reason_codes":["no_safe_operational_assertion"]}.
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


def _normalize_ref(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _validated_assertions(
    request: ManualNoteInterpretationRequestV1,
    raw_assertions: object,
) -> list[ManualNoteAssertionV1]:
    if not isinstance(raw_assertions, list):
        raise ValueError("assertions_must_be_list")
    allowed_refs = {
        _normalize_ref(row.invoice_number): row.invoice_number for row in request.invoice_facts
    }
    assertions: list[ManualNoteAssertionV1] = []
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
    return assertions


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
            assertions = _validated_assertions(request, raw.get("assertions") or [])
            if status == "accepted" and not assertions:
                status = "abstained"
            if status != "accepted" and assertions:
                raise ValueError("non_accepted_response_contains_assertions")
            reason_codes = raw.get("reason_codes") or []
            if not isinstance(reason_codes, list) or not all(
                isinstance(value, str) for value in reason_codes
            ):
                raise ValueError("manual_note_reason_codes_invalid")
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
                inference_profile="manual_note_interpretation",
            ).model_dump(mode="json"),
        )


manual_note_interpreter = ManualNoteInterpreter()
