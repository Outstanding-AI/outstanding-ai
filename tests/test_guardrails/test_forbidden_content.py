from src.guardrails.forbidden_content import ForbiddenContentDetector


def test_forbidden_content_detector_flags_review_findings(sample_case_context):
    guardrail = ForbiddenContentDetector()

    results = guardrail.validate(
        "Please pay via IBAN GB29NWBK60161331926819 or visit https://example.com/pay",
        sample_case_context,
    )

    assert len(results) == 1
    assert results[0].is_review_finding is True
    assert results[0].severity.value == "review"
    findings = results[0].details["findings"]
    assert any(item["category"] == "bank_payment_details" for item in findings)
    assert any(item["category"] == "external_url" for item in findings)
