from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.guardrails.lane_scope import LaneScopeGuardrail


def test_lane_scope_blocks_v2_obligation_ids():
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
                invoice_number="12345",
                original_amount=100.0,
                amount_due=75.0,
            )
        ],
        blocked_obligation_ids=["obl-uuid-1"],
        lane={
            "invoice_refs": ["12345"],
            "outstanding_amount": 75.0,
        },
    )

    results = LaneScopeGuardrail().validate("Please pay Invoice 12345 today.", context)

    assert not results[0].passed
    assert "blocked obligation" in results[0].message.lower()
