from src.guardrails.placeholder import PlaceholderValidationGuardrail


def test_sender_placeholders_are_blocked(sample_case_context):
    guardrail = PlaceholderValidationGuardrail()

    results = guardrail.validate(
        "Best regards,\nAccounts USA\n[SENDER_TITLE]\n[SENDER_COMPANY]",
        sample_case_context,
        skip_invoice_table=True,
    )

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].details["hallucinated_placeholders"] == [
        "[SENDER_COMPANY]",
        "[SENDER_TITLE]",
    ]


def test_invoice_table_remains_allowed_for_collection_drafts(sample_case_context):
    guardrail = PlaceholderValidationGuardrail()

    results = guardrail.validate(
        "Please see the table below.\n{INVOICE_TABLE}\nRegards,\nAccounts USA",
        sample_case_context,
    )

    assert len(results) == 1
    assert results[0].passed
