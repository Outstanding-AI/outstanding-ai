"""Tests for Factual Grounding Guardrail."""

import pytest

from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.factual_grounding import FactualGroundingGuardrail


@pytest.fixture
def sample_context() -> CaseContext:
    """Create a sample context for testing."""
    return CaseContext(
        schema_version=2,
        party=PartyInfo(
            party_id="party-001",
            external_id="party-ext-001",
            provider_type="sage_200",
            customer_code="CUST001",
            name="Acme Corp",
            currency="GBP",
            source="sage_200",
        ),
        obligations=[
            ObligationInfo(
                id="obl-12345",
                external_id="12345",
                provider_type="sage_200",
                invoice_number="INV-12345",
                original_amount=1500.00,
                amount_due=1500.00,
                due_date="2024-01-01",
                days_past_due=30,
            ),
            ObligationInfo(
                id="obl-12346",
                external_id="12346",
                provider_type="sage_200",
                invoice_number="INV-12346",
                original_amount=2500.00,
                amount_due=2500.00,
                due_date="2024-01-05",
                days_past_due=26,
            ),
        ],
    )


class TestFactualGroundingGuardrail:
    """Tests for FactualGroundingGuardrail."""

    def test_valid_invoice_numbers_pass(self, sample_context):
        """Test that valid invoice numbers pass validation."""
        guardrail = FactualGroundingGuardrail()
        output = "Your invoice INV-12345 for £1,500.00 is overdue."

        results = guardrail.validate(output, sample_context)

        invoice_result = results[0]  # Invoice validation is first
        assert invoice_result.passed
        assert "INV-12345" in str(invoice_result.details.get("validated_invoices", []))

    def test_invalid_invoice_number_fails(self, sample_context):
        """Test that fabricated invoice numbers fail validation."""
        guardrail = FactualGroundingGuardrail()
        output = "Your invoice INV-99999 for £1,500.00 is overdue."

        results = guardrail.validate(output, sample_context)

        invoice_result = results[0]
        assert not invoice_result.passed
        # The guardrail extracts just the numeric part from patterns
        assert "99999" in str(invoice_result.details.get("invalid_invoices", []))

    def test_valid_amounts_pass(self, sample_context):
        """Test that valid amounts pass validation."""
        guardrail = FactualGroundingGuardrail()
        # 1500 + 2500 = 4000
        output = "Your total outstanding is £4,000.00."

        results = guardrail.validate(output, sample_context)

        amount_result = results[1]  # Amount validation is second
        assert amount_result.passed

    def test_invalid_amount_fails(self, sample_context):
        """Test that fabricated amounts fail validation."""
        guardrail = FactualGroundingGuardrail()
        output = "Your invoice is for £9,999.99 which is overdue."

        results = guardrail.validate(output, sample_context)

        amount_result = results[1]
        assert not amount_result.passed
        assert 9999.99 in amount_result.details.get("invalid_amounts", [])

    def test_individual_amounts_pass(self, sample_context):
        """Test that individual invoice amounts pass validation."""
        guardrail = FactualGroundingGuardrail()
        output = "Invoice INV-12345: £1,500.00, INV-12346: £2,500.00"

        results = guardrail.validate(output, sample_context)

        # Both should pass
        assert results[0].passed  # Invoice validation
        assert results[1].passed  # Amount validation

    def test_no_invoices_or_amounts_passes(self, sample_context):
        """Test that output without invoices or amounts passes."""
        guardrail = FactualGroundingGuardrail()
        output = "Please contact us regarding your account."

        results = guardrail.validate(output, sample_context)

        # Should pass (nothing to validate = no violations)
        assert all(r.passed for r in results)

    def test_multiple_invoice_patterns(self, sample_context):
        """Test various invoice number formats."""
        guardrail = FactualGroundingGuardrail()

        # Test different patterns that should match INV-12345
        test_outputs = [
            "Invoice #12345 is overdue",
            "Invoice number: INV-12345",
            "Regarding INV 12345",
        ]

        for output in test_outputs:
            results = guardrail.validate(output, sample_context)
            invoice_result = results[0]
            # Should find and validate the invoice
            assert invoice_result.passed, f"Failed for: {output}"

    def test_currency_variations(self, sample_context):
        """Test various currency formats."""
        guardrail = FactualGroundingGuardrail()

        test_outputs = [
            "Amount: £1500.00",
            "Amount: GBP 1,500",
            "Total: £4000",  # Total of both invoices
        ]

        for output in test_outputs:
            results = guardrail.validate(output, sample_context)
            amount_result = results[1]
            assert amount_result.passed, f"Failed for: {output}"
