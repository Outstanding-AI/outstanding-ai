import json
from unittest.mock import AsyncMock

import pytest

from src.api.models.requests import CollectionEmailEventRequest
from src.engine.collection_email_event_classifier import CollectionEmailEventClassifier
from src.llm.base import LLMResponse
from src.llm.schemas import CollectionEmailEventLLMResponse


def test_collection_email_event_schema_is_strict_and_provider_compatible():
    """Do not reintroduce open-ended ``dict`` items into provider schemas."""
    schema = CollectionEmailEventLLMResponse.model_json_schema()
    amount_ref = schema["properties"]["amount_assertions"]["items"]["$ref"]
    date_ref = schema["properties"]["date_assertions"]["items"]["$ref"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"][amount_ref.rsplit("/", 1)[-1]]["additionalProperties"] is False
    assert schema["$defs"][date_ref.rsplit("/", 1)[-1]]["additionalProperties"] is False


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
    assert result.amount_assertions == [
        {
            "invoice_ref": "INV-1",
            "amount": 100.0,
            "currency": "GBP",
            "assertion_type": "promised_payment",
        }
    ]
