import json
from unittest.mock import AsyncMock

import pytest

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailEventRequest
from src.engine.collection_email_event_classifier import (
    _SYSTEM_PROMPT,
    PROMPT_TEMPLATE_VERSION,
    CollectionEmailEventClassifier,
)
from src.llm.base import LLMResponse
from src.llm.schemas import CollectionEmailEventLLMResponse


def _llm_response(payload: dict) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(payload),
        provider="vertex",
        model="gemini-2.5-flash",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


_BASE_PAYLOAD = {
    "relevance_status": "collection",
    "lifecycle_status": "pending_financial_confirmation",
    "semantic_classification": "PROMISE_TO_PAY",
    "secondary_intents": [],
    "intent_details": [
        {"intent": "PROMISE_TO_PAY", "extracted_data": {"invoice_refs": ["INV-1"]}},
    ],
    "invoice_assertions": ["INV-1"],
    "date_assertions": [],
    "reason_codes": [],
    "confidence": 0.9,
}


@pytest.mark.asyncio
async def test_amount_without_evidence_text_is_nulled_not_trusted():
    """A model that omits amount_evidence_text is treated as abstention, not a hallucination risk."""
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=_llm_response(
            {
                **_BASE_PAYLOAD,
                "amount_assertions": [
                    {
                        "invoice_ref": "INV-1",
                        "amount": 100.0,
                        "currency": "GBP",
                        "assertion_type": "promised_payment",
                    }
                ],
            }
        )
    )

    result = await classifier.classify(
        CollectionEmailEventRequest(
            mode="known_collection_inbound",
            current_message={"body": "We will pay £100.00 for INV-1."},
        )
    )

    assert result.amount_assertions[0].amount is None
    assert result.amount_assertions[0].currency is None
    assert classifier._client.complete.await_count == 1


@pytest.mark.asyncio
async def test_amount_evidence_that_does_not_verify_exhausts_retries_and_raises():
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=_llm_response(
            {
                **_BASE_PAYLOAD,
                "amount_assertions": [
                    {
                        "invoice_ref": "INV-1",
                        "amount": 100.0,
                        "currency": "GBP",
                        "assertion_type": "promised_payment",
                        "amount_evidence_text": "£999.00",
                    }
                ],
            }
        )
    )

    with pytest.raises(LLMResponseInvalidError) as raised:
        await classifier.classify(
            CollectionEmailEventRequest(
                mode="known_collection_inbound",
                current_message={"body": "We will pay £100.00 for INV-1."},
            )
        )

    assert raised.value.details["validation_errors"] == [
        {"location": "grounding", "type": "amount_evidence_not_verbatim"}
    ]
    assert raised.value.details["attempt_count"] == 3
    assert classifier._client.complete.await_count == 3
    last_call_system_prompt = classifier._client.complete.await_args_list[-1].kwargs[
        "system_prompt"
    ]
    assert "VALIDATION CORRECTION MODE" in last_call_system_prompt
    assert "amount_evidence_not_verbatim" in last_call_system_prompt


@pytest.mark.asyncio
async def test_retry_succeeds_once_the_model_supplies_a_verifiable_amount_span():
    classifier = CollectionEmailEventClassifier()
    bad_response = _llm_response(
        {
            **_BASE_PAYLOAD,
            "amount_assertions": [
                {
                    "invoice_ref": "INV-1",
                    "amount": 100.0,
                    "currency": "GBP",
                    "assertion_type": "promised_payment",
                    "amount_evidence_text": "£999.00",
                }
            ],
        }
    )
    good_response = _llm_response(
        {
            **_BASE_PAYLOAD,
            "amount_assertions": [
                {
                    "invoice_ref": "INV-1",
                    "amount": 100.0,
                    "currency": "GBP",
                    "assertion_type": "promised_payment",
                    "amount_evidence_text": "£100.00",
                }
            ],
        }
    )
    classifier._client.complete = AsyncMock(side_effect=[bad_response, good_response])

    result = await classifier.classify(
        CollectionEmailEventRequest(
            mode="known_collection_inbound",
            current_message={"body": "We will pay £100.00 for INV-1."},
        )
    )

    assert result.amount_assertions[0].amount == 100.0
    assert classifier._client.complete.await_count == 2


@pytest.mark.asyncio
async def test_per_intent_amount_and_date_without_evidence_are_nulled_not_trusted():
    """Per-intent controls require their own verbatim evidence, not just the ledger's."""
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=_llm_response(
            {
                **_BASE_PAYLOAD,
                "intent_details": [
                    {
                        "intent": "PROMISE_TO_PAY",
                        "extracted_data": {
                            "invoice_refs": ["INV-1"],
                            "promise_amount": 100.0,
                            "promise_date": "2026-08-14",
                        },
                    }
                ],
            }
        )
    )

    result = await classifier.classify(
        CollectionEmailEventRequest(
            mode="known_collection_inbound",
            current_message={"body": "We will pay £100.00 for INV-1 on 2026-08-14."},
        )
    )

    extracted = result.intent_details[0].extracted_data
    assert extracted is not None
    assert extracted.promise_amount is None
    assert extracted.promise_date is None
    assert classifier._client.complete.await_count == 1


@pytest.mark.asyncio
async def test_per_intent_evidence_span_that_does_not_support_value_exhausts_retries():
    """A verbatim per-intent span cannot support a different asserted amount."""
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(
        return_value=_llm_response(
            {
                **_BASE_PAYLOAD,
                "intent_details": [
                    {
                        "intent": "PROMISE_TO_PAY",
                        "extracted_data": {
                            "invoice_refs": ["INV-1"],
                            "promise_amount": 999.0,
                            "promise_amount_evidence_text": "£100.00",
                            "promise_date": "2026-08-14",
                            "promise_date_evidence_text": "2026-08-14",
                        },
                    }
                ],
            }
        )
    )

    with pytest.raises(LLMResponseInvalidError) as raised:
        await classifier.classify(
            CollectionEmailEventRequest(
                mode="known_collection_inbound",
                current_message={"body": "We will pay £100.00 for INV-1 on 2026-08-14."},
            )
        )

    assert raised.value.details["validation_errors"] == [
        {"location": "grounding", "type": "amount_evidence_does_not_support_value"}
    ]
    assert raised.value.details["attempt_count"] == 3
    assert classifier._client.complete.await_count == 3
    last_call_system_prompt = classifier._client.complete.await_args_list[-1].kwargs[
        "system_prompt"
    ]
    assert "VALIDATION CORRECTION MODE" in last_call_system_prompt
    assert "amount_evidence_does_not_support_value" in last_call_system_prompt


@pytest.mark.asyncio
async def test_manual_outbound_mode_uses_its_dedicated_client():
    """Manual outbound uses its separately configured model client by default."""
    classifier = CollectionEmailEventClassifier()
    classifier._client.complete = AsyncMock(return_value=_llm_response(_BASE_PAYLOAD))
    classifier._manual_outbound_client.complete = AsyncMock(
        return_value=_llm_response(_BASE_PAYLOAD)
    )

    result = await classifier.classify(
        CollectionEmailEventRequest(
            mode="manual_outbound",
            current_message={"body": "Please arrange payment for INV-1."},
        )
    )

    assert result.semantic_classification == "PROMISE_TO_PAY"
    assert classifier._manual_outbound_client.complete.await_count == 1
    assert classifier._client.complete.await_count == 0


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
    assert "candidate_count is 1" in _SYSTEM_PROMPT
    assert "Never assign one promise, dispute, or" in _SYSTEM_PROMPT
    assert PROMPT_TEMPLATE_VERSION == "v9"
    assert "only earlier retained events" in _SYSTEM_PROMPT
    assert "GROUNDING" in _SYSTEM_PROMPT
    assert "amount_evidence_text" in _SYSTEM_PROMPT


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

    assert raised.value.details["mode"] == "initial_chain"
    assert raised.value.details["validation_errors"] == [
        {"location": "lifecycle_status", "type": "literal_error"},
    ]
    assert raised.value.details["telemetry"] == {
        "provider": "vertex",
        "model": "gemini-2.5-flash",
        "is_fallback": False,
        "tokens_used": 15,
        "prompt_tokens": 10,
        "completion_tokens": 5,
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
                            "amount_evidence_text": "£100.00",
                        }
                    ],
                    "date_assertions": [
                        {
                            "invoice_ref": "INV-1",
                            "date_value": "2026-07-15",
                            "assertion_type": "promise_date",
                            "date_evidence_text": "2026-07-15",
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
            current_message={"body": "We will pay £100.00 for INV-1 on 2026-07-15."},
        )
    )

    assert result.semantic_classification == "PROMISE_TO_PAY"
    assert result.lifecycle_status == "pending_financial_confirmation"
    assert result.intent_details[0].extracted_data.invoice_refs == ["INV-1"]
    assert classifier._client.complete.await_args.kwargs["json_mode"] is True
    assert "response_schema" not in classifier._client.complete.await_args.kwargs
    assert [item.model_dump() for item in result.amount_assertions] == [
        {
            "invoice_ref": "INV-1",
            "amount": 100.0,
            "currency": "GBP",
            "assertion_type": "promised_payment",
            "amount_evidence_text": "£100.00",
        }
    ]
