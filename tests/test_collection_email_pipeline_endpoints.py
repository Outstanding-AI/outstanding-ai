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


def _manual_note_review_response(
    verdict: str = "accept",
    reason_codes: list[str] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(
            {
                "verdict": verdict,
                "reason_codes": reason_codes or [],
            }
        ),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )


@pytest.mark.asyncio
async def test_manual_note_semantic_review_corrects_commitment_clause_date():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "2026-09-01 contact confirmed payment will be made on 2026-09-12."
    interpreter = ManualNoteInterpreter()
    proposed_response = LLMResponse(
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
                        "asserted_date": "2026-09-01",
                        "date_evidence_text": "2026-09-01",
                        "reference": None,
                        "full_current_balance": True,
                        "evidence_text": note,
                        "confidence": 0.8,
                        "reason_codes": ["proposed_commitment"],
                    }
                ],
                "reason_codes": [],
            }
        ),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    reviewed_response = LLMResponse(
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
                        "asserted_date": "2026-09-12",
                        "date_evidence_text": "2026-09-12",
                        "reference": None,
                        "full_current_balance": True,
                        "evidence_text": note,
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
    interpreter._client.complete = AsyncMock(
        side_effect=[
            proposed_response,
            _manual_note_review_response("reject", ["commitment_date_bound_to_activity_date"]),
            reviewed_response,
            _manual_note_review_response(),
        ]
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
    assert result.assertions[0].asserted_date == "2026-09-12"
    assert result.assertions[0].full_current_balance is True
    assert interpreter._client.complete.await_args.kwargs["json_mode"] is True
    assert (
        interpreter._client.complete.await_args_list[0].kwargs["response_schema"].__name__
        == "_ManualNoteLLMResponse"
    )
    assert interpreter._client.complete.await_args.kwargs["caller"] == "manual_note_interpretation"


@pytest.mark.asyncio
async def test_manual_note_interpreter_requires_llm_to_extract_single_invoice_remittance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "REMIT RECEIVED"
    interpreter = ManualNoteInterpreter()
    extraction_response = LLMResponse(
        content=json.dumps(
            {
                "extraction_status": "accepted",
                "assertions": [
                    {
                        "assertion_id": "assertion-remittance",
                        "assertion_type": "remittance",
                        "transition": "received",
                        "polarity": "affirmed",
                        "temporal_orientation": "current",
                        "invoice_refs": ["INV-1"],
                        "amount": None,
                        "currency": None,
                        "asserted_date": None,
                        "reference": None,
                        "full_current_balance": False,
                        "evidence_text": note,
                        "confidence": 0.99,
                        "reason_codes": ["source_supported_remittance"],
                    }
                ],
                "reason_codes": [],
            }
        ),
        provider="vertex",
        model="gemini-2.5-flash",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    interpreter._client.complete = AsyncMock(
        side_effect=[extraction_response, _manual_note_review_response()]
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
    assert "deterministic_explicit_claim_recovery" not in result.reason_codes
    assert "semantic_guardrail_reviewed" in result.reason_codes


@pytest.mark.asyncio
async def test_manual_note_interpreter_retries_ungrounded_remittance_timestamp():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "REMIT RECEIVED"
    interpreter = ManualNoteInterpreter()
    invalid_response = LLMResponse(
        content=json.dumps(
            {
                "extraction_status": "accepted",
                "assertions": [
                    {
                        "assertion_id": "remit-1",
                        "assertion_type": "remittance",
                        "transition": "received",
                        "polarity": "affirmed",
                        "temporal_orientation": "current",
                        "invoice_refs": ["INV-1"],
                        "amount": None,
                        "currency": None,
                        "asserted_date": "2026-07-30T12:38:00Z",
                        "reference": None,
                        "full_current_balance": False,
                        "evidence_text": note,
                        "confidence": 0.99,
                        "reason_codes": [],
                    }
                ],
                "reason_codes": [],
            }
        ),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    interpreter._client.complete = AsyncMock(
        side_effect=[invalid_response, _manual_note_review_response()]
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-remit-timestamp",
            note=note,
            occurred_at="2026-07-30T12:38:00Z",
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
    assert result.assertions[0].asserted_date is None
    assert "ungrounded_optional_date_removed" in result.assertions[0].reason_codes
    assert "semantic_guardrail_reviewed" in result.reason_codes
    assert interpreter._client.complete.await_count == 2


@pytest.mark.asyncio
async def test_manual_note_interpreter_accepts_authoritative_chase_abstention():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "extraction_status": "abstained",
                    "assertions": [],
                    "reason_codes": ["operator_purpose_chase"],
                }
            ),
            provider="openai",
            model="gpt-5.6-luna",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-chase",
            note="Customer expects to pay next Friday; chased for an update.",
            purpose="chase",
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
    assert "authoritative_chase_abstention" in result.reason_codes
    assert interpreter._client.complete.await_count == 1


@pytest.mark.asyncio
async def test_manual_note_interpreter_removes_inferred_relative_query_date():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "Delivery evidence is missing and should be available next week."
    invalid_response = LLMResponse(
        content=json.dumps(
            {
                "extraction_status": "accepted",
                "assertions": [
                    {
                        "assertion_id": "query-1",
                        "assertion_type": "query",
                        "transition": "active",
                        "polarity": "affirmed",
                        "temporal_orientation": "current",
                        "invoice_refs": ["INV-1"],
                        "amount": None,
                        "currency": None,
                        "asserted_date": "2026-08-07",
                        "date_evidence_text": "next week",
                        "reference": None,
                        "full_current_balance": False,
                        "evidence_text": note,
                        "confidence": 0.9,
                        "reason_codes": [],
                    }
                ],
                "reason_codes": [],
            }
        ),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    interpreter = ManualNoteInterpreter()
    interpreter._client.complete = AsyncMock(
        side_effect=[invalid_response, _manual_note_review_response()]
    )

    result = await interpreter.interpret(
        ManualNoteInterpretationRequestV1(
            touch_id="touch-query-relative-date",
            note=note,
            purpose="query",
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
    assert result.assertions[0].asserted_date is None
    assert "ungrounded_optional_date_removed" in result.assertions[0].reason_codes
    assert interpreter._client.complete.await_count == 2


@pytest.mark.asyncio
async def test_manual_note_interpreter_corrects_commitment_only_flag_from_remittance():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "Remittance received."
    interpreter = ManualNoteInterpreter()
    invalid_response = LLMResponse(
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
                        "evidence_text": note,
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
    corrected_payload = {
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
                "full_current_balance": False,
                "evidence_text": note,
                "confidence": 0.9,
                "reason_codes": ["source_supported_remittance"],
            }
        ],
        "reason_codes": [],
    }
    corrected_response = LLMResponse(
        content=json.dumps(corrected_payload),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    interpreter._client.complete = AsyncMock(
        side_effect=[invalid_response, corrected_response, _manual_note_review_response()]
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
async def test_manual_note_interpreter_semantic_review_can_abstain():
    from src.engine.manual_note_interpreter import ManualNoteInterpreter

    note = "NEW STATEMENTS TO BE SENT WITH FIRM EMAIL. CUSTOMER DID NOT MAKE PAYMENT"
    interpreter = ManualNoteInterpreter()
    proposed_response = LLMResponse(
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
                        "full_current_balance": False,
                        "evidence_text": note[45:],
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
    abstained_response = LLMResponse(
        content=json.dumps(
            {
                "extraction_status": "abstained",
                "assertions": [],
                "reason_codes": ["no_safe_operational_assertion"],
            }
        ),
        provider="openai",
        model="gpt-5.6-luna",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    interpreter._client.complete = AsyncMock(
        side_effect=[
            proposed_response,
            _manual_note_review_response("reject", ["routine_outreach_not_remittance"]),
            abstained_response,
            _manual_note_review_response(),
        ]
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
    assert "semantic_guardrail_corrected" in result.reason_codes


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
                            "evidence_text": "paid",
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
