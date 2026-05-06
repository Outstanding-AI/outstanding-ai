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
            source="sage_200",
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


def test_lane_scope_uses_candidate_refs_over_open_lane_refs():
    context = CaseContext(
        schema_version=2,
        party=PartyInfo(
            party_id="party-uuid-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="C001",
            name="Acme Ltd",
            source="sage_200",
        ),
        obligations=[
            ObligationInfo(
                id="obl-uuid-1",
                external_id="12345",
                provider_type="sage_200",
                invoice_number="INV-12345",
                original_amount=100.0,
                amount_due=75.0,
            ),
            ObligationInfo(
                id="obl-uuid-2",
                external_id="12346",
                provider_type="sage_200",
                invoice_number="INV-12346",
                original_amount=100.0,
                amount_due=75.0,
            ),
        ],
        lane={
            "invoice_refs": ["INV-12345", "INV-12346"],
            "outstanding_amount": 150.0,
        },
    )

    results = LaneScopeGuardrail().validate(
        "Please pay Invoice 12346 today.",
        context,
        candidate_invoice_refs=["INV-12345"],
    )

    assert not results[0].passed
    assert "outside lane cohort" in results[0].message.lower()
