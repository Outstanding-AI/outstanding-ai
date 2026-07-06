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

    def test_invoice_digit_prefix_does_not_validate(self, sample_context):
        """Invoice 1234 is not grounded by INV-12345."""
        guardrail = FactualGroundingGuardrail()
        sample_context.obligations = [sample_context.obligations[0]]

        results = guardrail.validate("Please pay invoice 1234 today.", sample_context)

        assert not results[0].passed
        assert "1234" in str(results[0].details.get("invalid_invoices", []))

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

    def test_temporal_evidence_amount_does_not_validate_current_demand(self, sample_context):
        """Historical message-time amounts are continuity context, not demand authority."""
        sample_context.collection_thread_invoice_evidence = [
            {
                "invoice_number": "INV-OLD",
                "current_state": "paid",
                "message_states": [{"as_of_amount_due": 9999.99, "as_of_state": "open"}],
            }
        ]

        results = FactualGroundingGuardrail().validate(
            "Please pay the current overdue amount of £9,999.99.",
            sample_context,
        )

        amount_result = results[1]
        assert not amount_result.passed
        assert 9999.99 in amount_result.details.get("invalid_amounts", [])

    def test_closed_current_obligation_amount_does_not_validate_current_demand(self):
        """Paid/closed current obligations are historical context, not demand authority."""
        context = CaseContext(
            schema_version=4,
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
                    id="obl-open",
                    external_id="12345",
                    provider_type="sage_200",
                    invoice_number="INV-12345",
                    original_amount=1500.00,
                    amount_due=1500.00,
                    due_date="2024-01-01",
                    days_past_due=30,
                    state="open",
                    collection_status="open",
                    is_overdue=True,
                    is_sendable=True,
                    is_chase_eligible=True,
                ),
                ObligationInfo(
                    id="obl-paid",
                    external_id="12346",
                    provider_type="sage_200",
                    invoice_number="INV-12346",
                    original_amount=2500.00,
                    amount_due=0.00,
                    due_date="2024-01-05",
                    days_past_due=26,
                    state="paid",
                    collection_status="paid",
                    is_overdue=True,
                    is_sendable=False,
                    is_chase_eligible=False,
                ),
            ],
            sendable_obligation_ids=["obl-open"],
            blocked_obligation_ids=["obl-paid"],
            collection_basis="overdue",
            chase_basis="overdue",
        )

        results = FactualGroundingGuardrail().validate(
            "Please pay the current overdue balance of £2,500.00.",
            context,
        )

        amount_result = results[1]
        assert not amount_result.passed
        assert 2500.0 in amount_result.details.get("invalid_amounts", [])

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

    def test_source_disputed_invoice_payment_ask_fails(self, sample_context):
        """Sage query/source-disputed obligations cannot be chased."""
        sample_context.obligations[0].is_source_disputed = True
        sample_context.obligations[0].source_query_raw = "Queried in Sage"

        results = FactualGroundingGuardrail().validate(
            "Please pay invoice INV-12345 today.", sample_context
        )

        source_result = results[2]
        assert not source_result.passed
        assert "source-disputed" in source_result.message

    def test_source_disputed_invoice_can_be_labelled_excluded(self, sample_context):
        """Mentioning a source-disputed invoice as excluded is allowed."""
        sample_context.obligations[0].is_source_disputed = True
        sample_context.obligations[0].source_query_raw = "Queried in Sage"

        results = FactualGroundingGuardrail().validate(
            "Invoice INV-12345 is excluded due to an invoice dispute.", sample_context
        )

        assert results[2].passed

    def test_source_disputed_exclusion_can_coexist_with_other_invoice_chase(self, sample_context):
        """A correct exclusion note should not block chasing a different invoice."""
        sample_context.obligations[0].is_source_disputed = True
        sample_context.obligations[0].source_query_raw = "Queried in Sage"

        results = FactualGroundingGuardrail().validate(
            "Invoice INV-12345 is excluded due to an invoice query. Please pay invoice INV-12346.",
            sample_context,
        )

        assert results[2].passed

    def test_source_disputed_invoice_digit_prefix_does_not_block_clean_invoice(
        self, sample_context
    ):
        """A source-disputed INV-1234 must not match a chase for INV-12345."""
        sample_context.obligations[0].invoice_number = "INV-1234"
        sample_context.obligations[0].is_source_disputed = True
        sample_context.obligations[0].source_query_raw = "Queried in Sage"
        sample_context.obligations[1].invoice_number = "INV-12345"

        results = FactualGroundingGuardrail().validate(
            "Please pay invoice INV-12345 today.",
            sample_context,
        )

        assert results[2].passed

    def test_unverified_procurement_claim_fails(self, sample_context):
        """PO/POD claims require verified procurement flags."""
        results = FactualGroundingGuardrail().validate(
            "This invoice is backed by PO number PO-1 and proof of delivery.",
            sample_context,
        )

        procurement_result = results[3]
        assert not procurement_result.passed
        assert "unverified procurement" in procurement_result.message

    def test_unverified_procurement_workflow_claim_fails(self, sample_context):
        """Generic procurement/approval workflow wording is also evidence-gated."""
        results = FactualGroundingGuardrail().validate(
            "Please let us know if your procurement process or GRN approval is holding up payment.",
            sample_context,
        )

        procurement_result = results[3]
        assert not procurement_result.passed
        assert "procurement_workflow" in procurement_result.found
        assert "proof_of_delivery" in procurement_result.found

    def test_po_box_address_does_not_count_as_procurement_claim(self, sample_context):
        """Sender/footer PO Box text is not a purchase-order claim."""
        results = FactualGroundingGuardrail().validate(
            "Please contact us about your account.\nAccounts Team, PO Box 123, London",
            sample_context,
        )

        assert results[3].passed

    def test_verified_procurement_claim_passes(self, sample_context):
        """Verified PO/POD context allows procurement wording."""
        sample_context.obligations[0].has_verified_purchase_order = True
        sample_context.obligations[0].has_verified_pod = True

        results = FactualGroundingGuardrail().validate(
            "This invoice is backed by PO number PO-1 and proof of delivery.",
            sample_context,
        )

        assert results[3].passed

    def test_source_query_context_allows_procurement_workflow_ack(self, sample_context):
        """A real Sage/query flag can ground neutral workflow wording."""
        sample_context.obligations[0].source_query_raw = "Awaiting approval"
        sample_context.obligations[0].has_source_query_flag = True

        results = FactualGroundingGuardrail().validate(
            "If your approval workflow is still preventing payment, please let us know.",
            sample_context,
        )

        assert results[3].passed
