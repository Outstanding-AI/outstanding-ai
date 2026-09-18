"""Synthetic fixture coverage for the disabled V2 semantic interpreter."""

from __future__ import annotations

import json

import pytest
from solvix_contracts.ai import (
    MailSemanticEvidenceRequestV3,
    mail_semantic_candidate_set_hash,
    mail_semantic_context_manifest_hash,
)

from src.config.settings import settings
from src.engine.mail_semantic_evidence_v3 import MailSemanticEvidenceInterpreterV3
from src.llm.base import LLMResponse

_HASH = "a" * 64


class _FakePrimaryProvider:
    provider_name = "openrouter"
    model_name = "synthetic-primary"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    async def complete(self, system_prompt: str, user_prompt: str, **kwargs: object) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        return LLMResponse(
            content=json.dumps(self.payload),
            provider=self.provider_name,
            model=self.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def _request(
    *,
    source_text: str = "Synthetic statement for INV-SYN-1.",
):
    source_segments = []
    body_state = "missing"
    if source_text:
        body_state = "complete"
        source_segments = [
            {
                "source_id": "authored-1",
                "source_kind": "authored_body",
                "text": source_text,
                "content_hash": _HASH,
            }
        ]
    candidate_hash = mail_semantic_candidate_set_hash(
        party_resolution="unresolved",
        candidate_parties=[],
        invoice_resolution="unresolved",
        candidate_invoices=[],
    )
    return MailSemanticEvidenceRequestV3(
        tenant_ref="synthetic-tenant",
        canonical_message_state_key="synthetic-message-state",
        canonical_message_version_hash=_HASH,
        semantic_revision_key="synthetic-revision-1",
        input_context_hash=_HASH,
        mode="known_collection_inbound",
        current_message={
            "source_id": "message-1",
            "timestamp": "2026-09-17T10:00:00Z",
            "direction": "inbound",
            "subject": "Synthetic only",
            "envelope": {
                "sender_role": "debtor",
                "recipient_roles": ["shared_mailbox"],
                "mailbox_direction": "inbound",
            },
            "body_state": body_state,
            "source_segments": source_segments,
        },
        admission={
            "identity_revision": "identity-terminal-1",
            "accounting_cut_at": "2026-09-17T09:59:00Z",
            "identity_candidate_set_hash": candidate_hash,
            "context_manifest_revision": "context-terminal-1",
            "context_manifest_hash": mail_semantic_context_manifest_hash([]),
            "attachment_evidence_state": "not_present",
        },
    )


def _accepted_payload(*, source_id: str = "authored-1") -> dict[str, object]:
    span = {
        "source_id": source_id,
        "evidence_text": "Synthetic statement for INV-SYN-1.",
        "supports": ["relevance", "invoice_scope", "operational_state"],
    }
    return {
        "relevance": "collection",
        "semantic_events": [
            {
                "event_id": "synthetic-event-1",
                "family": "commitment",
                "transition": "made",
                "polarity": "affirmed",
                "temporal_orientation": "future",
                "invoice_refs": ["INV-SYN-1"],
                "account_wide": False,
                "amount": None,
                "currency": None,
                "asserted_date": None,
                "reference": None,
                "evidence": [span],
                "confidence": 0.9,
                "reason_codes": [],
            }
        ],
        "response_evidence": [span],
        "disposition": "accepted",
        "adjudication_status": "not_required",
        "confidence": 0.9,
        "reason_codes": [],
        "ambiguity_reason_codes": [],
    }


@pytest.mark.asyncio
async def test_disabled_interpreter_defers_without_constructing_or_calling_provider(
    monkeypatch,
) -> None:
    fake = _FakePrimaryProvider(_accepted_payload())
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", False)

    result = await interpreter.interpret(_request())

    assert result.disposition == "deferred"
    assert result.reason_codes == ["mail_semantic_evidence_v3_disabled"]
    assert fake.calls == []


@pytest.mark.asyncio
async def test_no_safe_source_abstains_without_calling_primary(monkeypatch) -> None:
    fake = _FakePrimaryProvider(_accepted_payload())
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request(source_text=""))

    assert result.disposition == "abstained"
    assert result.reason_codes == ["no_safe_semantic_source"]
    assert fake.calls == []


@pytest.mark.asyncio
async def test_enabled_interpreter_uses_injected_no_fallback_primary_and_returns_grounded_output(
    monkeypatch,
) -> None:
    fake = _FakePrimaryProvider(_accepted_payload())
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert result.semantic_events[0].evidence[0].source_id == "authored-1"
    assert result.operation_summary.invocations[0].provider == "openrouter"
    assert result.operation_summary.invocations[0].is_fallback is False
    assert result.admission.identity_revision == "identity-terminal-1"
    assert len(fake.calls) == 1
    assert "model_override" not in fake.calls[0]["user_prompt"]
    assert fake.calls[0].get("json_mode", False) is False
    assert fake.calls[0]["response_schema"].__name__ == "_LLMResponse"


@pytest.mark.asyncio
async def test_unknown_provider_span_is_rejected_as_invalid(monkeypatch) -> None:
    fake = _FakePrimaryProvider(_accepted_payload(source_id="unknown-source"))
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "invalid"
    assert result.semantic_events == []
    assert result.reason_codes == [
        "semantic_response_schema_or_grounding_invalid",
        "semantic_validation_semantic_evidence_unknown_or_unavailable_source",
    ]
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_document_request_aliases_are_normalized_before_strict_validation(
    monkeypatch,
) -> None:
    payload = _accepted_payload()
    payload["semantic_events"][0]["family"] = "document_request"
    payload["semantic_events"][0]["transition"] = "requested"
    payload["semantic_events"][0]["evidence"] = [dict(payload["semantic_events"][0]["evidence"][0])]
    payload["semantic_events"][0]["evidence"][0]["supports"] = ["document_request"]
    fake = _FakePrimaryProvider(payload)
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert result.semantic_events[0].family == "request_information"
    assert result.semantic_events[0].evidence[0].supports == ["operational_state"]


@pytest.mark.asyncio
async def test_single_response_evidence_object_is_normalized_to_a_list(monkeypatch) -> None:
    payload = _accepted_payload()
    payload["response_evidence"] = payload["response_evidence"][0]
    fake = _FakePrimaryProvider(payload)
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert len(result.response_evidence) == 1


@pytest.mark.asyncio
async def test_accepted_event_can_ground_omitted_response_relevance(monkeypatch) -> None:
    payload = _accepted_payload()
    payload["response_evidence"] = [dict(payload["response_evidence"][0])]
    payload["response_evidence"][0]["supports"] = ["operational_state"]
    fake = _FakePrimaryProvider(payload)
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert any("relevance" in span.supports for span in result.response_evidence)


@pytest.mark.asyncio
async def test_ungrounded_material_value_is_dropped_without_discarding_safe_event(
    monkeypatch,
) -> None:
    payload = _accepted_payload()
    payload["semantic_events"][0]["asserted_date"] = "2026-09-30"
    fake = _FakePrimaryProvider(payload)
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert result.semantic_events[0].asserted_date is None
    assert "ungrounded_date_dropped" in result.semantic_events[0].reason_codes


@pytest.mark.asyncio
async def test_unadmitted_invoice_references_are_dropped_without_losing_event(monkeypatch) -> None:
    payload = _accepted_payload()
    payload["semantic_events"][0]["invoice_refs"] = ["UNTRUSTED-REFERENCE"]
    payload["semantic_events"][0]["evidence"][0]["evidence_text"] = (
        "Synthetic statement for UNTRUSTED-REFERENCE."
    )
    fake = _FakePrimaryProvider(payload)
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=fake)
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v3", True)

    result = await interpreter.interpret(
        _request(source_text="Synthetic statement for UNTRUSTED-REFERENCE.")
    )

    assert result.disposition == "accepted"
    assert result.semantic_events[0].invoice_refs == []
    assert "unadmitted_invoice_refs_dropped" in result.semantic_events[0].reason_codes
