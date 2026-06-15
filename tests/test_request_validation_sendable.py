from src.api.models.requests import CaseContext, ObligationInfo, PartyInfo
from src.api.models.requests.validation import _is_sendable_candidate


def test_is_sendable_candidate_honors_sendable_obligation_ids():
    obligation = ObligationInfo(
        id="obl-1",
        external_id="ext-1",
        provider_type="sage_200",
        invoice_number="INV-1",
        original_amount=100.0,
        amount_due=100.0,
        due_date="2026-04-01",
        is_overdue=True,
    )
    context = CaseContext(
        schema_version=4,
        source_sync_run_id="sync-1",
        application_run_id="app-1",
        core_snapshot_watermark="core",
        application_snapshot_watermark="app",
        application_decision_cutoff="2026-05-06T00:00:00Z",
        policy_snapshot_id="policy-1",
        draft_candidate_id="draft-candidate-1",
        collection_basis="overdue",
        sendable_obligation_ids=["obl-2"],
        party=PartyInfo(
            party_id="party-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="CUST-1",
            name="Customer",
            currency="GBP",
            source="sage_200",
        ),
        obligations=[obligation],
    )

    assert _is_sendable_candidate(obligation, context) is False


def test_is_sendable_candidate_rejects_zero_balance_current_obligation():
    obligation = ObligationInfo(
        id="obl-1",
        external_id="ext-1",
        provider_type="sage_200",
        invoice_number="INV-1",
        original_amount=100.0,
        amount_due=0.0,
        due_date="2026-04-01",
        is_overdue=True,
        is_sendable=True,
        is_chase_eligible=True,
    )
    context = CaseContext(
        schema_version=4,
        source_sync_run_id="sync-1",
        application_run_id="app-1",
        core_snapshot_watermark="core",
        application_snapshot_watermark="app",
        application_decision_cutoff="2026-05-06T00:00:00Z",
        policy_snapshot_id="policy-1",
        draft_candidate_id="draft-candidate-1",
        collection_basis="overdue",
        sendable_obligation_ids=["obl-1"],
        party=PartyInfo(
            party_id="party-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="CUST-1",
            name="Customer",
            currency="GBP",
            source="sage_200",
        ),
        obligations=[obligation],
    )

    assert _is_sendable_candidate(obligation, context) is False
