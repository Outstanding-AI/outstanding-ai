from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.identity_scope import IdentityScopeGuardrail


def _context() -> CaseContext:
    return CaseContext(
        schema_version=2,
        party=PartyInfo(
            party_id="party-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="CUST1",
            name="Acme Corp",
            source="sage_200",
        ),
        obligations=[
            ObligationInfo(
                id="obl-100",
                external_id="INV-100",
                provider_type="sage_200",
                invoice_number="INV-100",
                original_amount=100,
                amount_due=100,
                due_date="2026-01-01",
                days_past_due=10,
            )
        ],
        debtor_contact={"name": "Edward Smith", "email": "edward@example.com"},
        party_contacts=[
            {"name": "Edward Smith", "email": "edward@example.com"},
        ],
    )


def test_identity_scope_allows_authorized_emails():
    guardrail = IdentityScopeGuardrail()
    results = guardrail.validate(
        "Hi Edward,\nPlease reply to collections@example.com.\nRegards, Sarah",
        _context(),
        recipient_name="Edward Smith",
        sender_name="Sarah Jones",
        sender_email="sarah@example.com",
        reply_anchor_email="collections@example.com",
        cc_emails=["teamlead@example.com"],
    )
    assert all(result.passed for result in results)


def test_identity_scope_blocks_unknown_email():
    guardrail = IdentityScopeGuardrail()
    results = guardrail.validate(
        "Hi Edward,\nPlease send confirmation to random@example.net.\nRegards, Sarah",
        _context(),
        recipient_name="Edward Smith",
        sender_name="Sarah Jones",
        sender_email="sarah@example.com",
        reply_anchor_email="collections@example.com",
        cc_emails=["teamlead@example.com"],
    )
    assert any(not result.passed for result in results)
