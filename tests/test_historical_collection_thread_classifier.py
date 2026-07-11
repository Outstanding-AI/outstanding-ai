import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.requests import HistoricalCollectionThreadRequest
from src.engine import historical_collection_thread_classifier as historical_module
from src.engine.historical_collection_thread_classifier import HistoricalCollectionThreadClassifier
from src.llm.base import LLMResponse
from src.llm.schemas import HistoricalCollectionThreadLLMResponse


def _llm_response(payload: dict) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(payload),
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    )


def test_historical_collection_schema_is_openai_strict_compatible():
    schema = HistoricalCollectionThreadLLMResponse.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["HistoricalThreadActionLLM"]["additionalProperties"] is False
    assert schema["$defs"]["HistoricalIntentDetailLLM"]["additionalProperties"] is False
    assert schema["properties"]["thread_actions"]["items"] == {
        "$ref": "#/$defs/HistoricalThreadActionLLM"
    }
    assert schema["properties"]["intent_details"]["items"] == {
        "$ref": "#/$defs/HistoricalIntentDetailLLM"
    }


def test_historical_collection_schema_accepts_legacy_thread_action_dict():
    parsed = HistoricalCollectionThreadLLMResponse(
        classification="needs_review",
        thread_actions={"conv-a": "active", "conv-b": "needs_review"},
        intent_details=[
            {
                "intent": "remittance",
                "invoice_refs": ["0000001234"],
                "details": "Debtor attached remittance advice.",
            }
        ],
    )

    assert parsed.thread_actions_dict() == {"conv-a": "active", "conv-b": "needs_review"}
    assert parsed.intent_details_payload() == [
        {
            "intent": "remittance",
            "invoice_refs": ["0000001234"],
            "evidence_message_ids": [],
            "summary": "Debtor attached remittance advice.",
        }
    ]


def test_historical_collection_routes_vertex_primary_with_openai_fallback():
    assert historical_module.historical_llm_client.primary_provider_name == "vertex"
    assert historical_module.historical_llm_client.fallback_provider_name == "openai"


@pytest.mark.asyncio
async def test_message_protocol_classifies_reply_response_not_escalation():
    request = HistoricalCollectionThreadRequest(
        mode="message_protocol",
        message={
            "mail_message_id": "msg-3",
            "message_role": "outbound_operator",
            "subject": "Re: Invoice 0000007324",
            "body": "Thanks for the update. Please confirm once payment is released.",
        },
        deterministic_facts={
            "prior_debtor_reply_message_id": "msg-2",
            "contact_transition_fact": "none",
        },
    )
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {
                "classification": "debtor_reply_response",
                "protocol_touch_type": "debtor_reply_response",
                "is_escalation": False,
                "escalation_kind": "none",
                "debtor_reply_response": True,
                "commitment_acknowledgement_type": "none",
                "confidence": 0.91,
                "reason": "Outbound message responds to debtor update without level or escalation intent.",
                "evidence_message_ids": ["msg-2", "msg-3"],
            }
        )
        result = await HistoricalCollectionThreadClassifier().classify(request)

    assert result.protocol_touch_type == "debtor_reply_response"
    assert result.is_escalation is False
    assert result.escalation_kind == "none"
    assert result.tokens_used == 20
    assert result.ai_audit is not None


@pytest.mark.asyncio
async def test_debtor_thread_adjudication_can_return_needs_review():
    request = HistoricalCollectionThreadRequest(
        mode="debtor_thread_adjudication",
        party_id="party-1",
        candidate_threads=[
            {"conversation_id": "conv-a", "current_open_overdue_invoice_numbers": ["0001"]},
            {"conversation_id": "conv-b", "current_open_overdue_invoice_numbers": ["0001"]},
        ],
    )
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {
                "classification": "needs_review",
                "confidence": 0.42,
                "reason": "Two chains plausibly compete for the same open exposure.",
                "thread_actions": {"conv-a": "needs_review", "conv-b": "needs_review"},
                "guardrail_warnings": ["multiple_competing_threads"],
            }
        )
        result = await HistoricalCollectionThreadClassifier().classify(request)

    assert result.recommended_active_thread_id is None
    assert result.thread_actions == {"conv-a": "needs_review", "conv-b": "needs_review"}
    assert "multiple_competing_threads" in result.guardrail_warnings


@pytest.mark.asyncio
async def test_thread_relevance_returns_only_thread_gate_fields():
    request = HistoricalCollectionThreadRequest(
        mode="thread_collection_relevance",
        prior_messages_summary=[
            {
                "ordinal": 1,
                "direction": "outbound",
                "unique_body_plain": "Please confirm payment date.",
            },
            {"ordinal": 2, "direction": "inbound", "unique_body_plain": "We will pay next week."},
        ],
        deterministic_facts={"visible_party_match": True, "invoice_mention_count": 1},
    )
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {
                "relevance_label": "collection_related",
                "confidence": 0.94,
                "signal_codes": ["explicit_collection_request", "debtor_payment_response"],
                "evidence_message_ordinals": [1, 2],
                "reason": "The authored chronology is a payment follow-up and debtor response.",
            }
        )
        result = await HistoricalCollectionThreadClassifier().classify(request)

    assert result.relevance_label == "collection_related"
    assert result.signal_codes == ["explicit_collection_request", "debtor_payment_response"]
    assert result.evidence_message_ordinals == [1, 2]
    assert result.classification is None


@pytest.mark.asyncio
async def test_thread_relevance_missing_label_fails_closed():
    request = HistoricalCollectionThreadRequest(mode="thread_collection_relevance")
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {"confidence": 0.4, "reason": "insufficient evidence"}
        )
        with pytest.raises(Exception, match="no thread relevance label"):
            await HistoricalCollectionThreadClassifier().classify(request)


@pytest.mark.asyncio
async def test_thread_relevance_rejects_message_intent_fields():
    request = HistoricalCollectionThreadRequest(mode="thread_collection_relevance")
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {
                "relevance_label": "collection_related",
                "classification": "PROMISE_TO_PAY",
                "confidence": 0.9,
            }
        )
        with pytest.raises(Exception, match="message/adjudication fields"):
            await HistoricalCollectionThreadClassifier().classify(request)


@pytest.mark.asyncio
async def test_chain_selection_tiebreak_returns_only_supplied_candidate():
    request = HistoricalCollectionThreadRequest(
        mode="chain_selection_tiebreak",
        candidate_threads=[
            {
                "candidate_key": "route-a",
                "invoice_scope_hash": "scope-a",
                "evidence_ordinals": [1, 2],
            },
            {"candidate_key": "route-b", "invoice_scope_hash": "scope-a", "evidence_ordinals": [3]},
        ],
    )
    with patch(
        "src.engine.historical_collection_thread_classifier.historical_llm_client.complete",
        new_callable=AsyncMock,
    ) as mock_complete:
        mock_complete.return_value = _llm_response(
            {
                "selected_candidate_key": "route-a",
                "action": "continue_existing_chain",
                "confidence": 0.88,
                "reason_codes": ["latest_exact_anchor"],
                "evidence_message_ordinals": [1, 2],
                "reason": "The supplied route has the strongest exact continuity evidence.",
            }
        )
        result = await HistoricalCollectionThreadClassifier().classify(request)

    assert result.selected_candidate_key == "route-a"
    assert result.selection_action == "continue_existing_chain"
    assert result.tokens_used == 20
