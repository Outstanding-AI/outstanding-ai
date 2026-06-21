import json
from unittest.mock import AsyncMock, patch

import pytest

from src.api.models.requests import HistoricalCollectionThreadRequest
from src.engine.historical_collection_thread_classifier import HistoricalCollectionThreadClassifier
from src.llm.base import LLMResponse


def _llm_response(payload: dict) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(payload),
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    )


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
        "src.engine.historical_collection_thread_classifier.llm_client.complete",
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
        "src.engine.historical_collection_thread_classifier.llm_client.complete",
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
