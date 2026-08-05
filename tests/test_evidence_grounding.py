from __future__ import annotations

import pytest

from src.engine._evidence_grounding import (
    amount_is_explicit_in_span,
    date_is_explicit_in_span,
    locate_evidence,
    normalize_ref,
    reference_is_explicit_in_span,
    validate_optional_field_evidence,
)


def test_locate_evidence_finds_unique_verbatim_span():
    start, end, span = locate_evidence("We will pay £100.00 on Friday.", "£100.00", field="amount")
    assert span == "£100.00"
    assert "We will pay £100.00 on Friday."[start:end] == "£100.00"


def test_locate_evidence_rejects_missing_or_empty_evidence():
    with pytest.raises(ValueError, match="amount_evidence_text_required"):
        locate_evidence("body", None, field="amount")
    with pytest.raises(ValueError, match="amount_evidence_text_required"):
        locate_evidence("body", "   ", field="amount")


def test_locate_evidence_rejects_non_verbatim_span():
    with pytest.raises(ValueError, match="amount_evidence_not_verbatim"):
        locate_evidence("We will pay £100.00 on Friday.", "£200.00", field="amount")


def test_locate_evidence_rejects_non_unique_span():
    with pytest.raises(ValueError, match="amount_evidence_not_unique"):
        locate_evidence("pay 100 now, or pay 100 later.", "pay 100", field="amount")


def test_date_is_explicit_in_span_accepts_multiple_formats():
    assert date_is_explicit_in_span("2026-07-15", "on 2026-07-15 we will pay")
    assert date_is_explicit_in_span("2026-07-15", "on 15/07/2026 we will pay")
    assert date_is_explicit_in_span("2026-07-15", "on 15 July 2026 we will pay")
    assert not date_is_explicit_in_span("2026-07-15", "next week sometime")
    assert not date_is_explicit_in_span("not-a-date", "on 2026-07-15")


def test_amount_is_explicit_in_span_handles_thousands_separators():
    assert amount_is_explicit_in_span(1234.5, "totalling 1,234.50 due")
    assert amount_is_explicit_in_span(100.0, "£100.00")
    assert not amount_is_explicit_in_span(100.0, "£200.00")
    assert not amount_is_explicit_in_span(None, "£100.00")


def test_reference_is_explicit_in_span_normalizes_punctuation():
    assert reference_is_explicit_in_span("INV-0001", "see invoice inv 0001 attached")
    assert not reference_is_explicit_in_span("INV-0002", "see invoice inv 0001 attached")


def test_normalize_ref_strips_non_alnum():
    assert normalize_ref("inv-0001/A") == "INV0001A"


def test_validate_optional_field_evidence_null_value_requires_no_evidence():
    assert (
        validate_optional_field_evidence(
            source_text="body",
            candidate={"amount": None},
            field="amount",
            assertion_evidence="body",
        )
        is None
    )
    with pytest.raises(ValueError, match="null_amount_must_not_have_evidence"):
        validate_optional_field_evidence(
            source_text="body",
            candidate={"amount": None, "amount_evidence_text": "100"},
            field="amount",
            assertion_evidence="body",
        )


def test_validate_optional_field_evidence_span_must_be_inside_assertion_evidence():
    with pytest.raises(ValueError, match="amount_evidence_outside_assertion"):
        validate_optional_field_evidence(
            source_text="We will pay £100.00 for INV-1, unrelated to the £50.00 fee.",
            candidate={"amount": 50.0, "amount_evidence_text": "£50.00"},
            field="amount",
            assertion_evidence="We will pay £100.00 for INV-1",
        )


def test_validate_optional_field_evidence_custom_evidence_key():
    span = validate_optional_field_evidence(
        source_text="We will pay on 2026-07-15.",
        candidate={"asserted_date": "2026-07-15", "date_evidence_text": "2026-07-15"},
        field="asserted_date",
        assertion_evidence="We will pay on 2026-07-15.",
        evidence_key="date_evidence_text",
    )
    assert span == "2026-07-15"
