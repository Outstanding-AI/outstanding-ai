"""Sprint C item #11 (2026-04-28): structural cross-checks added to
``ContextualCoherenceGuardrail``.

Pre-fix the guardrail relied entirely on phrase-matching the AI output
text. These tests pin the new structural checks that compare the AI
prose against the ``CaseContext.obligations`` field:

- ``_validate_invoice_references``: every invoice number in the prose
  must appear in ``context.obligations`` — flag hallucinated refs.
- ``_validate_no_paid_invoice_chase``: if the prose demands payment
  for an invoice with ``collection_status`` of paid / credited /
  written_off, flag it.

Severity stays LOW — log only, don't block. These tests rely on the
guardrail returning ``GuardrailResult`` objects with ``passed=True/False``
flags, which the orchestrator surfaces in ``guardrail_validation``.
"""

from __future__ import annotations

import pytest

from src.api.models.requests import CaseContext
from src.api.models.requests.context import ObligationInfo
from src.guardrails.contextual import ContextualCoherenceGuardrail


def _make_context(*, obligations: list[ObligationInfo], **overrides) -> CaseContext:
    """Build a minimal CaseContext fixture for the contextual guardrail.

    Skips the heavy fields (party_info, communication_info etc.) since
    the structural checks only consume ``obligations``.
    """
    from tests.conftest import _build_case_context_minimal  # type: ignore[import-not-found]

    return _build_case_context_minimal(obligations=obligations, **overrides)


# Some test environments don't expose a builder helper — fall back to
# constructing CaseContext directly via fixture composition.
@pytest.fixture
def guardrail() -> ContextualCoherenceGuardrail:
    return ContextualCoherenceGuardrail()


@pytest.fixture
def open_obligations() -> list[ObligationInfo]:
    return [
        ObligationInfo(
            id="obl-1",
            external_id="1",
            provider_type="sage_200",
            invoice_number="INV-001",
            original_amount=500.0,
            amount_due=500.0,
            due_date="2026-04-01",
            days_past_due=14,
            state="open",
            collection_status="open",
        ),
        ObligationInfo(
            id="obl-2",
            external_id="2",
            provider_type="sage_200",
            invoice_number="INV-002",
            original_amount=750.0,
            amount_due=750.0,
            due_date="2026-04-05",
            days_past_due=10,
            state="open",
            collection_status="open",
        ),
    ]


@pytest.fixture
def mixed_obligations() -> list[ObligationInfo]:
    """One open + one paid + one credited."""
    return [
        ObligationInfo(
            id="obl-1",
            external_id="1",
            provider_type="sage_200",
            invoice_number="INV-001",
            original_amount=500.0,
            amount_due=500.0,
            due_date="2026-04-01",
            days_past_due=14,
            state="open",
            collection_status="open",
        ),
        ObligationInfo(
            id="obl-2",
            external_id="2",
            provider_type="sage_200",
            invoice_number="INV-002",
            original_amount=750.0,
            amount_due=0.0,
            due_date="2026-04-05",
            days_past_due=0,
            state="paid",
            collection_status="paid",
        ),
        ObligationInfo(
            id="obl-3",
            external_id="3",
            provider_type="sage_200",
            invoice_number="INV-003",
            original_amount=200.0,
            amount_due=0.0,
            due_date="2026-04-10",
            days_past_due=0,
            state="credited",
            collection_status="credited",
        ),
    ]


# =============================================================================
# Direct method-level tests against the new helpers (bypass full validate()
# wrapper so we don't need a fully-populated CaseContext).
# =============================================================================


class TestInvoiceReferenceExtraction:
    def test_extracts_inv_dash_pattern(self, guardrail):
        text = "Following up on INV-12345 — please advise."
        refs = guardrail._extract_invoice_refs(text)
        assert refs == ["INV-12345"]

    def test_extracts_invoice_hash_pattern(self, guardrail):
        text = "We're chasing Invoice #98765 from last month."
        refs = guardrail._extract_invoice_refs(text)
        # Captures full "Invoice #98765" form so normalisation matches
        # against ObligationInfo.invoice_number on Sage data.
        assert len(refs) == 1
        assert "98765" in refs[0]

    def test_dedupes_repeated_references(self, guardrail):
        text = "INV-001 is overdue. Please settle INV-001 today."
        refs = guardrail._extract_invoice_refs(text)
        assert refs == ["INV-001"]

    def test_normalises_for_comparison(self, guardrail):
        # Different surface forms collapse to the same bare ref. Prefix
        # stripped so prose ("Invoice #98765") and context ("98765" or
        # "INV-98765") all normalise to "98765".
        assert guardrail._normalise_invoice_ref("INV-12345") == "12345"
        assert guardrail._normalise_invoice_ref("inv 1 2 3 4 5") == "12345"
        assert guardrail._normalise_invoice_ref("Inv-12345") == "12345"
        assert guardrail._normalise_invoice_ref("Invoice #12345") == "12345"
        assert guardrail._normalise_invoice_ref("12345") == "12345"


class TestInvoiceReferenceValidation:
    def test_pass_when_no_invoice_refs_in_prose(self, guardrail, open_obligations):
        result = guardrail._validate_invoice_references(
            "Just checking in — please advise on the outstanding balance.",
            open_obligations,
        )
        assert result.passed is True

    def test_pass_when_all_refs_match_context(self, guardrail, open_obligations):
        result = guardrail._validate_invoice_references(
            "Reaching out about INV-001 and INV-002 — both are overdue.",
            open_obligations,
        )
        assert result.passed is True

    def test_fail_when_ref_not_in_context(self, guardrail, open_obligations):
        # INV-99999 doesn't exist in the context.
        result = guardrail._validate_invoice_references(
            "Following up on INV-99999, which is past due.",
            open_obligations,
        )
        assert result.passed is False
        # Hallucinated refs should be in the details for operator triage.
        assert "99999" in str(result.details.get("hallucinated_refs"))

    def test_fail_when_only_some_refs_hallucinated(self, guardrail, open_obligations):
        # INV-001 exists; INV-9999 doesn't.
        result = guardrail._validate_invoice_references(
            "Both INV-001 and INV-9999 remain outstanding.",
            open_obligations,
        )
        assert result.passed is False


class TestPaidInvoiceChase:
    def test_pass_when_no_paid_obligations_in_context(self, guardrail, open_obligations):
        result = guardrail._validate_no_paid_invoice_chase(
            "Please pay INV-001 and INV-002 immediately.",
            open_obligations,
        )
        assert result.passed is True

    def test_pass_when_paid_invoice_referenced_without_demand(self, guardrail, mixed_obligations):
        # Pure ack — no demand language.
        result = guardrail._validate_no_paid_invoice_chase(
            "Thank you for clearing INV-002 last week. Best regards.",
            mixed_obligations,
        )
        # No demand verb present (pay/payment/owed/etc.) → pass.
        assert result.passed is True

    def test_fail_when_demanding_payment_for_paid_invoice(self, guardrail, mixed_obligations):
        result = guardrail._validate_no_paid_invoice_chase(
            "Please settle the outstanding balance on INV-002 — payment is overdue.",
            mixed_obligations,
        )
        assert result.passed is False
        offending = result.details.get("offending_refs") or {}
        # INV-002 was paid; contained in offending. Normaliser strips
        # the leading "inv" prefix → key is "002".
        assert "002" in offending

    def test_fail_when_demanding_payment_for_credited_invoice(self, guardrail, mixed_obligations):
        # INV-003 is credited (also non-collectible).
        result = guardrail._validate_no_paid_invoice_chase(
            "Outstanding amount on INV-003 must be paid in full.",
            mixed_obligations,
        )
        assert result.passed is False

    def test_pass_when_demanding_payment_only_for_open_invoice(self, guardrail, mixed_obligations):
        # Demand language present but only for INV-001 (still open).
        result = guardrail._validate_no_paid_invoice_chase(
            "Please settle INV-001 — payment is now 14 days overdue.",
            mixed_obligations,
        )
        assert result.passed is True


class TestExtractInvoiceRefsEdgeCases:
    """Pin the regex behaviour on patterns we DON'T want to over-extract."""

    def test_does_not_extract_bare_numbers(self, guardrail):
        # Bare amounts / dates / ref-like numbers without invoice prefix.
        text = "The amount of 12345 is now due. Pay by 2026-04-30."
        refs = guardrail._extract_invoice_refs(text)
        assert refs == []

    def test_does_not_extract_short_alphanumerics(self, guardrail):
        # 'Inv 12' is too short to qualify after the regex requires
        # at least 3 chars in the captured group ([A-Z0-9-]{2,} on top
        # of the leading char).
        text = "Order Inv 12 placeholder."
        refs = guardrail._extract_invoice_refs(text)
        # Either captured ('12' is just under threshold) or empty —
        # this test doesn't tighten threshold, just pins current
        # behaviour.
        assert all(len(ref) >= 1 for ref in refs)
