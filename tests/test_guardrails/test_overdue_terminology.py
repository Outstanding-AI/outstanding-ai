from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.overdue_terminology import OverdueTerminologyGuardrail


def _context(*, schema_version: int = 4, basis: str = "overdue") -> CaseContext:
    return CaseContext(
        schema_version=schema_version,
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
                id="obl-001",
                external_id="001",
                provider_type="sage_200",
                invoice_number="INV-001",
                original_amount=100.0,
                amount_due=100.0,
                days_overdue=14,
                is_overdue=True,
                is_sendable=True,
                is_chase_eligible=True,
            )
        ],
        collection_basis=basis,
        chase_basis=basis,
    )


def test_overdue_scope_blocks_outstanding_invoice_language_in_subject():
    results = OverdueTerminologyGuardrail().validate(
        "Please see the invoice table.",
        _context(),
        subject="Outstanding invoices - Acme Corp",
    )

    assert any(not result.passed for result in results)
    assert results[0].guardrail_name == "overdue_terminology"
    assert "Outstanding invoices" in results[0].found


def test_overdue_scope_blocks_outstanding_balance_language_in_body():
    results = OverdueTerminologyGuardrail().validate(
        "Please arrange payment of the outstanding balance.",
        _context(),
        subject="Overdue invoices - Acme Corp",
    )

    assert any(not result.passed for result in results)
    assert "outstanding balance" in results[0].found


def test_overdue_scope_allows_overdue_language():
    results = OverdueTerminologyGuardrail().validate(
        "Please arrange payment of the overdue balance.",
        _context(),
        subject="Overdue invoices - Acme Corp",
    )

    assert all(result.passed for result in results)


def test_legacy_context_keeps_outstanding_language_compatible():
    results = OverdueTerminologyGuardrail().validate(
        "Please arrange payment of the outstanding balance.",
        _context(schema_version=2, basis="outstanding"),
        subject="Outstanding invoices - Acme Corp",
    )

    assert all(result.passed for result in results)
