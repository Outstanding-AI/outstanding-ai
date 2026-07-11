import json
from unittest.mock import AsyncMock

import pytest

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailEventRequest
from src.engine.collection_email_event_classifier import (
    _SYSTEM_PROMPT,
    CollectionEmailEventClassifier,
)
from src.llm.base import LLMResponse
from src.llm.schemas import CollectionEmailEventLLMResponse


def test_collection_email_event_schema_is_strict_for_post_provider_validation():
    """Do not reintroduce open-ended ``dict`` items into validated output."""
    schema = CollectionEmailEventLLMResponse.model_json_schema()
    amount_ref = schema["properties"]["amount_assertions"]["items"]["$ref"]
    date_ref = schema["properties"]["date_assertions"]["items"]["$ref"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"][amount_ref.rsplit("/", 1)[-1]]["additionalProperties"] is False
    assert schema["$defs"][date_ref.rsplit("/", 1)[-1]]["additionalProperties"] is False


def test_collection_email_event_reuses_per_intent_debtor_response_scope():
    parsed = CollectionEmailEventLLMResponse(
        relevance_status="collection",
        lifecycle_status="pending_financial_confirmation",
        semantic_classification="ALREADY_PAID",
        secondary_intents=["PROMISE_TO_PAY"],
        intent_details=[
            {"intent": "ALREADY_PAID", "extracted_data": {"invoice_refs": ["INV-A"]}},
            {"intent": "PROMISE_TO_PAY", "extracted_data": {"invoice_refs": ["INV-B"]}},
        ],
        confidence=0.9,
    )
    assert parsed.intent_details[0].extracted_data.invoice_refs == ["INV-A"]
    assert parsed.intent_details[1].extracted_data.invoice_refs == ["INV-B"]


def test_collection_email_event_schema_rejects_unrecognised_output_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CollectionEmailEventLLMResponse(
            relevance_status="collection",
            lifecycle_status="active",
            confidence=0.9,
            unexpected_provider_field="must_fail_closed",
        )


@pytest.mark.asyncio
async def test_collection_email_event_invalid_json_reports_only_sanitized_locations():
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "relevance_status": "collection",
                    "lifecycle_status": "not_a_lifecycle",
                    "confidence": 0.9,
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )

    with pytest.raises(LLMResponseInvalidError) as raised:
        await classifier.classify(
            CollectionEmailEventRequest(
                mode="initial_chain",
                current_message={"body": "synthetic body must not appear in error details"},
            )
        )

    assert raised.value.details == {
        "mode": "initial_chain",
        "validation_errors": [
            {"location": "lifecycle_status", "type": "literal_error"},
        ],
    }
    assert "synthetic body" not in json.dumps(raised.value.details)
    assert "date_value" in _SYSTEM_PROMPT
    assert "pending_financial_confirmation" in _SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_collection_email_event_accepts_only_a_fenced_json_transport_wrapper():
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=LLMResponse(
            content="""```json
{"relevance_status":"collection","lifecycle_status":"active","confidence":0.9}
```""",
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )

    result = await classifier.classify(
        CollectionEmailEventRequest(mode="initial_chain", current_message={"body": "synthetic"})
    )

    assert result.relevance_status == "collection"
    assert result.lifecycle_status == "active"


@pytest.mark.asyncio
async def test_collection_email_event_uses_vertex_primary_and_strict_schema():
    classifier = CollectionEmailEventClassifier()
    assert classifier._client.primary_provider_name == "vertex"
    assert classifier._client.fallback_provider_name == "openai"
    classifier._client.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "relevance_status": "collection",
                    "lifecycle_status": "pending_financial_confirmation",
                    "semantic_classification": "PROMISE_TO_PAY",
                    "secondary_intents": [],
                    "intent_details": [
                        {
                            "intent": "PROMISE_TO_PAY",
                            "extracted_data": {
                                "invoice_refs": ["INV-1"],
                                "promise_amount": 100.0,
                                "promise_date": "2026-07-15",
                            },
                        }
                    ],
                    "invoice_assertions": ["INV-1"],
                    "amount_assertions": [
                        {
                            "invoice_ref": "INV-1",
                            "amount": 100.0,
                            "currency": "GBP",
                            "assertion_type": "promised_payment",
                        }
                    ],
                    "date_assertions": [
                        {
                            "invoice_ref": "INV-1",
                            "date_value": "2026-07-15",
                            "assertion_type": "promise_date",
                        }
                    ],
                    "reason_codes": ["debtor_payment_commitment"],
                    "confidence": 0.91,
                }
            ),
            provider="vertex",
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
    )
    result = await classifier.classify(
        CollectionEmailEventRequest(
            mode="known_collection_inbound",
            current_message={"body": "We will pay INV-1 tomorrow."},
        )
    )

    assert result.semantic_classification == "PROMISE_TO_PAY"
    assert result.lifecycle_status == "pending_financial_confirmation"
    assert result.intent_details[0].extracted_data.invoice_refs == ["INV-1"]
    assert classifier._client.complete.await_args.kwargs["json_mode"] is True
    assert "response_schema" not in classifier._client.complete.await_args.kwargs
    assert result.amount_assertions == [
        {
            "invoice_ref": "INV-1",
            "amount": 100.0,
            "currency": "GBP",
            "assertion_type": "promised_payment",
        }
    ]
