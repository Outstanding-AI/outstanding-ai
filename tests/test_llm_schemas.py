import pytest

from src.llm.schemas import ClassificationLLMResponse


def test_intent_details_reject_secondary_material_intent_without_extraction():
    with pytest.raises(ValueError, match="extracted_data is required"):
        ClassificationLLMResponse(
            classification="ALREADY_PAID",
            confidence=0.9,
            intent_details=[
                {
                    "intent": "ALREADY_PAID",
                    "extracted_data": {"invoice_refs": ["INV-1"], "claimed_reference": "PAY-1"},
                },
                {"intent": "PROMISE_TO_PAY", "extracted_data": None},
            ],
        )


def test_intent_details_drops_secondary_shared_invoice_refs():
    response = ClassificationLLMResponse(
        classification="ALREADY_PAID",
        confidence=0.9,
        secondary_intents=["PROMISE_TO_PAY"],
        intent_details=[
            {
                "intent": "ALREADY_PAID",
                "extracted_data": {"invoice_refs": ["INV-1"], "claimed_reference": "PAY-1"},
            },
            {
                "intent": "PROMISE_TO_PAY",
                "extracted_data": {"invoice_refs": ["INV-1"], "promise_date": "2026-05-20"},
            },
        ],
    )

    assert response.secondary_intents == []
    assert len(response.intent_details or []) == 1
    assert response.intent_details[0].intent == "ALREADY_PAID"
    assert response.intent_details[0].extracted_data.invoice_refs == ["INV-1"]


def test_intent_details_keeps_distinct_secondary_refs_after_normalization():
    response = ClassificationLLMResponse(
        classification="DEBTOR_INTERNAL_PROCESSING_BLOCKER",
        confidence=0.9,
        secondary_intents=["PROMISE_TO_PAY"],
        intent_details=[
            {
                "intent": "DEBTOR_INTERNAL_PROCESSING_BLOCKER",
                "extracted_data": {
                    "invoice_refs": ["INV-001"],
                    "internal_blocker_type": "payment_run_pending",
                },
            },
            {
                "intent": "PROMISE_TO_PAY",
                "extracted_data": {
                    "invoice_refs": ["INV 001", "INV-002"],
                    "promise_date": "2026-05-20",
                },
            },
        ],
    )

    assert response.secondary_intents == ["PROMISE_TO_PAY"]
    assert len(response.intent_details or []) == 2
    assert response.intent_details[1].extracted_data.invoice_refs == ["INV-002"]


def test_forbidden_content_findings_are_strict_for_structured_outputs():
    schema = ClassificationLLMResponse.model_json_schema()
    finding_ref = schema["properties"]["forbidden_content_detected"]["items"]["$ref"]
    finding_name = finding_ref.rsplit("/", 1)[-1]

    assert schema["$defs"][finding_name]["additionalProperties"] is False
