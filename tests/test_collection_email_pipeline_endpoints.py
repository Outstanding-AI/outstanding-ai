from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from solvix_contracts.ai import ManualNoteInterpretationRequestV1

from src.api.models.requests import (
    CollectionChainIdentificationRequest,
    CollectionEmailFactExtractionRequest,
)
from src.llm.base import LLMResponse
from src.llm.schemas import (
    CollectionChainIdentificationLLMResponse,
    CollectionEmailFactExtractionLLMResponse,
)


def test_fact_extraction_contract_contains_no_relevance_or_route_fields():
    request = CollectionEmailFactExtractionRequest(
        current_message={"body": "Invoice INV-1 is overdue"},
        prior_chain_invoice_context={
            "invoice_candidates": [{"invoice_ref": "INV-1"}],
            "candidate_count": 1,
            "is_truncated": False,
        },
    )
    assert request.prior_messages == []
    assert request.prior_chain_invoice_context["candidate_count"] == 1
    parsed = CollectionEmailFactExtractionLLMResponse(
        invoice_assertions=["INV-1"],
        amount_assertions=[],
        date_assertions=[],
        confidence=0.9,
        reason_codes=["explicit_invoice"],
    )
    assert parsed.invoice_assertions == ["INV-1"]


def test_chain_identifier_contract_is_bounded_and_strict():
    request = CollectionChainIdentificationRequest(
        current_message={"body": "Please settle the invoice"},
        prior_messages=[{"ordinal": index} for index in range(6)],
        prior_chain_invoice_context={"invoice_candidates": [{"invoice_ref": "INV-1"}]},
        reconciled_scope=[{"mapping_status": "exact"}],
    )
    assert len(request.prior_messages) == 6
    parsed = CollectionChainIdentificationLLMResponse(
        collection_status="collection",
        event_effect="confirmed",
        confidence=0.9,
        reason_codes=["payment_request"],
        evidence_message_ordinals=[1],
    )
    assert parsed.collection_status == "collection"
    with pytest.raises(Exception):
        CollectionChainIdentificationLLMResponse(
            collection_status="collection",
            event_effect="route_to_this_thread",
            confidence=0.9,
        )


@pytest.mark.asyncio
async def test_fact_extractor_uses_json_mode_and_a_closed_post_provider_normalizer():
    from src.engine.collection_email_fact_extractor import CollectionEmailFactExtractor

    extractor = CollectionEmailFactExtractor()
    extractor._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "invoice_assertions": [],
                    "amount_assertions": [],
                    "date_assertions": [],
                    "confidence": 0.0,
                    "reason_codes": ["no_explicit_invoice_fact"],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )

    await extractor.extract(
        CollectionEmailFactExtractionRequest(current_message={"body": "synthetic"})
    )

    assert extractor._client.complete.await_args.kwargs["json_mode"] is True
    assert "response_schema" not in extractor._client.complete.await_args.kwargs


@pytest.mark.asyncio
async def test_chain_identifier_uses_json_mode_and_a_closed_post_provider_normalizer():
    from src.engine.collection_chain_identifier import CollectionChainIdentifier

    identifier = CollectionChainIdentifier()
    identifier._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "collection_status": "uncertain",
                    "event_effect": "no_change",
                    "confidence": 0.0,
                    "reason_codes": ["insufficient_email_evidence"],
                    "evidence_message_ordinals": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )

    await identifier.identify(
        CollectionChainIdentificationRequest(current_message={"body": "synthetic"})
    )

    assert identifier._client.complete.await_args.kwargs["json_mode"] is True
    assert "response_schema" not in identifier._client.complete.await_args.kwargs


def test_fact_normalizer_accepts_only_documented_aliases_and_conservative_defaults():
    from src.engine.collection_email_fact_extractor import _canonical_fact_response_object

    normalized = _canonical_fact_response_object(
        json.dumps(
            {
                "invoice_refs": ["INV-1"],
                "amounts": [],
                "dates": [],
                "reasons": ["explicit_invoice"],
            }
        )
    )

    assert normalized["invoice_assertions"] == ["INV-1"]
    assert normalized["confidence"] == 0.0
    with pytest.raises(ValueError, match="unknown_fields"):
        _canonical_fact_response_object(json.dumps({"summary": "not an allowed fact field"}))


def test_chain_normalizer_abstains_when_the_lifecycle_effect_is_missing():
    from src.engine.collection_chain_identifier import _canonical_chain_response_object

    normalized = _canonical_chain_response_object(
        json.dumps({"relevance_label": "collection_related", "reason_codes": []})
    )

    assert normalized["collection_status"] == "uncertain"
    assert normalized["event_effect"] == "no_change"
    assert "invalid_event_effect_abstention" in normalized["reason_codes"]


def test_chain_normalizer_safely_canonicalizes_common_json_mode_variants():
    from src.engine.collection_chain_identifier import _canonical_chain_response_object

    normalized = _canonical_chain_response_object(
        json.dumps(
            {
                "relevance_label": "collection_related",
                "event_effect": "ongoing",
                "confidence": 85,
                "reason_codes": "explicit_payment_request",
                "evidence_message_ordinals": ["1", "invalid"],
                "explanation": "transport-only text is discarded",
            }
        )
    )

    assert normalized == {
        "collection_status": "collection",
        "event_effect": "confirmed",
        "confidence": 0.85,
        "reason_codes": ["explicit_payment_request"],
        "evidence_message_ordinals": [1],
    }

    unsafe = _canonical_chain_response_object(
        json.dumps({"collection_status": "collection", "event_effect": "route_to_this_thread"})
    )
    assert unsafe["collection_status"] == "uncertain"
    assert unsafe["event_effect"] == "no_change"
    assert "invalid_event_effect_abstention" in unsafe["reason_codes"]


@pytest.mark.asyncio
async def test_manual_note_interpreter_is_source_grounded_and_defaults_commitment_balance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "Invoice INV-1 will be paid on 2026-08-03."
    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "accepted",
                    "assertions": [
                        {
                            "assertion_id": "assertion-1",
                            "assertion_type": "commitment",
                            "transition": "made",
                            "polarity": "affirmed",
                            "temporal_orientation": "future",
                            "invoice_refs": ["INV-1"],
                            "amount": None,
                            "currency": None,
                            "asserted_date": "2026-08-03",
                            "reference": None,
                            "full_current_balance": False,
                            "evidence_start": 0,
                            "evidence_end": len(note),
                            "confidence": 0.99,
                            "reason_codes": ["explicit_commitment_date"],
                        }
                    ],
                    "reason_codes": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
    )
    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-1",
            note=note,
            occurred_at="2026-07-31T10:00:00Z",
            tenant_timezone="Europe/London",
            invoice_facts=[
                {
                    "obligation_id": "obligation-1",
                    "invoice_number": "INV-1",
                    "amount_due": 100,
                    "currency": "GBP",
                }
            ],
        )
    )

    assert result.extraction_status == "accepted"
    assert result.assertions[0].full_current_balance is True
    assert interpreter._client.complete.await_args.kwargs["json_mode"] is True
    assert (
        interpreter._client.complete.await_args.kwargs["response_schema"].__name__
        == "_ManualNoteLLMResponse"
    )
    assert interpreter._client.complete.await_args.kwargs["caller"] == "manual_note_interpretation"


@pytest.mark.asyncio
async def test_manual_note_interpreter_recovers_explicit_single_invoice_remittance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "REMIT RECEIVED"
    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "abstained",
                    "assertions": [],
                    "reason_codes": ["no_safe_operational_assertion"],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-remit",
            note=note,
            occurred_at="2026-07-31T10:00:00Z",
            tenant_timezone="Europe/London",
            invoice_facts=[
                {
                    "obligation_id": "obligation-1",
                    "invoice_number": "INV-1",
                    "amount_due": 100,
                    "currency": "GBP",
                }
            ],
        )
    )

    assert result.extraction_status == "accepted"
    assert len(result.assertions) == 1
    assertion = result.assertions[0]
    assert assertion.assertion_type == "remittance"
    assert assertion.transition == "received"
    assert assertion.invoice_refs == ["INV-1"]
    assert assertion.amount is None
    assert assertion.asserted_date is None
    assert assertion.reference is None
    assert "deterministic_explicit_claim_recovery" in result.reason_codes


@pytest.mark.asyncio
async def test_manual_note_interpreter_discards_commitment_only_flag_from_remittance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "Remittance received."
    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "accepted",
                    "assertions": [
                        {
                            "assertion_id": "assertion-remittance",
                            "assertion_type": "remittance",
                            "transition": "received",
                            "polarity": "affirmed",
                            "temporal_orientation": "past",
                            "invoice_refs": ["INV-1"],
                            "amount": None,
                            "currency": None,
                            "asserted_date": None,
                            "reference": None,
                            "full_current_balance": True,
                            "evidence_start": 0,
                            "evidence_end": len(note),
                            "confidence": 0.9,
                            "reason_codes": [],
                        }
                    ],
                    "reason_codes": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-remittance-not-received",
            note=note,
            occurred_at="2026-07-31T10:00:00Z",
            tenant_timezone="Europe/London",
            invoice_facts=[
                {
                    "obligation_id": "obligation-1",
                    "invoice_number": "INV-1",
                    "amount_due": 100,
                    "currency": "GBP",
                }
            ],
        )
    )

    assert result.extraction_status == "accepted"
    assert result.assertions[0].assertion_type == "remittance"
    assert result.assertions[0].transition == "received"
    assert result.assertions[0].full_current_balance is False


@pytest.mark.asyncio
async def test_manual_note_interpreter_drops_unscoped_negative_remittance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "NEW STATEMENTS TO BE SENT WITH FIRM EMAIL. CUSTOMER DID NOT MAKE PAYMENT"
    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "accepted",
                    "assertions": [
                        {
                            "assertion_id": "assertion-remittance",
                            "assertion_type": "remittance",
                            "transition": "not_received",
                            "polarity": "affirmed",
                            "temporal_orientation": "past",
                            "invoice_refs": ["INV-1"],
                            "amount": None,
                            "currency": None,
                            "asserted_date": None,
                            "reference": None,
                            "full_current_balance": True,
                            "evidence_start": 45,
                            "evidence_end": len(note),
                            "confidence": 1.0,
                            "reason_codes": [],
                        }
                    ],
                    "reason_codes": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-no-payment",
            note=note,
            occurred_at="2026-07-31T10:00:00Z",
            tenant_timezone="Europe/London",
            invoice_facts=[
                {
                    "obligation_id": "obligation-1",
                    "invoice_number": "INV-1",
                    "amount_due": 100,
                    "currency": "GBP",
                }
            ],
        )
    )

    assert result.extraction_status == "abstained"
    assert result.assertions == []
    assert "unscoped_negative_remittance_dropped" in result.reason_codes


@pytest.mark.asyncio
async def test_manual_note_interpreter_rejects_cross_invoice_and_ambiguous_amount_output():
    from src.api.errors import LLMResponseInvalidError
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "accepted",
                    "assertions": [
                        {
                            "assertion_id": "assertion-1",
                            "assertion_type": "remittance",
                            "transition": "received",
                            "polarity": "affirmed",
                            "temporal_orientation": "past",
                            "invoice_refs": ["INV-1", "INV-OUTSIDE"],
                            "amount": 100,
                            "currency": "GBP",
                            "asserted_date": None,
                            "reference": None,
                            "full_current_balance": False,
                            "evidence_start": 0,
                            "evidence_end": 4,
                            "confidence": 0.99,
                            "reason_codes": [],
                        }
                    ],
                    "reason_codes": [],
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )

    with pytest.raises(LLMResponseInvalidError):
        await interpreter.interpret(
            ManualNoteInterpretationRequestV1(
                touch_id="touch-1",
                note="paid",
                occurred_at="2026-07-31T10:00:00Z",
                tenant_timezone="Europe/London",
                invoice_facts=[
                    {
                        "obligation_id": "obligation-1",
                        "invoice_number": "INV-1",
                        "amount_due": 100,
                        "currency": "GBP",
                    }
                ],
            )
        )
