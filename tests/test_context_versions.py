import pytest
from pydantic import ValidationError

from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo


def test_case_context_requires_explicit_schema_version():
    with pytest.raises(ValidationError, match="schema_version"):
        CaseContext(
            party=PartyInfo(party_id="party-1", customer_code="C001", name="Acme Ltd"),
            obligations=[
                ObligationInfo(
                    invoice_number="INV-12345",
                    original_amount=100.0,
                    amount_due=75.0,
                )
            ],
        )


def test_case_context_v2_requires_canonical_identity_fields():
    with pytest.raises(ValueError, match="party.external_id is required"):
        CaseContext(
            schema_version=2,
            party=PartyInfo(party_id="party-1", customer_code="C001", name="Acme Ltd"),
            obligations=[],
        )


def test_case_context_v2_accepts_canonical_identity_fields():
    context = CaseContext(
        schema_version=2,
        party=PartyInfo(
            party_id="party-uuid-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="C001",
            name="Acme Ltd",
        ),
        obligations=[
            ObligationInfo(
                id="obl-uuid-1",
                external_id="12345",
                provider_type="sage_200",
                provider_ref="9001",
                invoice_number="INV-12345",
                original_amount=100.0,
                amount_due=75.0,
            )
        ],
        blocked_obligation_ids=["obl-uuid-1"],
    )

    assert context.schema_version == 2
    assert context.obligations[0].external_id == "12345"
