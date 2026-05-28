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


def test_intent_details_reject_shared_invoice_refs_across_intents():
    with pytest.raises(ValueError, match="appears in multiple intent_details"):
        ClassificationLLMResponse(
            classification="ALREADY_PAID",
            confidence=0.9,
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


def test_intent_details_reject_shared_invoice_refs_after_normalization():
    with pytest.raises(ValueError, match="appears in multiple intent_details"):
        ClassificationLLMResponse(
            classification="ALREADY_PAID",
            confidence=0.9,
            intent_details=[
                {
                    "intent": "ALREADY_PAID",
                    "extracted_data": {"invoice_refs": ["INV-001"], "claimed_reference": "PAY-1"},
                },
                {
                    "intent": "PROMISE_TO_PAY",
                    "extracted_data": {"invoice_refs": ["INV 001"], "promise_date": "2026-05-20"},
                },
            ],
        )


def test_forbidden_content_findings_are_strict_for_structured_outputs():
    schema = ClassificationLLMResponse.model_json_schema()
    finding_ref = schema["properties"]["forbidden_content_detected"]["items"]["$ref"]
    finding_name = finding_ref.rsplit("/", 1)[-1]

    assert schema["$defs"][finding_name]["additionalProperties"] is False
