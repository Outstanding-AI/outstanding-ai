import pytest
from pydantic import ValidationError
from solvix_contracts.ai.context.v2 import CaseContextV2, ObligationInfoV2, PartyInfoV2

from src.api.models.requests import CaseContext, GenerateDraftRequest, ObligationInfo, PartyInfo


def _party(**overrides) -> PartyInfo:
    data = {
        "party_id": "party-uuid-1",
        "external_id": "party-ext-1",
        "provider_type": "sage_200",
        "customer_code": "C001",
        "name": "Acme Ltd",
        "source": "sage_200",
    }
    data.update(overrides)
    return PartyInfo(**data)


def _obligation(**overrides) -> ObligationInfo:
    data = {
        "id": "obl-uuid-1",
        "external_id": "12345",
        "provider_type": "sage_200",
        "provider_ref": "9001",
        "invoice_number": "INV-12345",
        "original_amount": 100.0,
        "amount_due": 75.0,
    }
    data.update(overrides)
    return ObligationInfo(**data)


def test_case_context_defaults_schema_version_to_shared_v2():
    context = CaseContext(
        party=_party(),
        obligations=[_obligation()],
    )

    assert context.schema_version == 2


def test_case_context_v2_requires_canonical_identity_fields():
    with pytest.raises(ValidationError, match="external_id"):
        CaseContext(
            schema_version=2,
            party=_party(external_id=None),
            obligations=[],
        )


def test_case_context_v2_accepts_canonical_identity_fields():
    context = CaseContext(
        schema_version=2,
        party=_party(),
        obligations=[_obligation()],
        blocked_obligation_ids=["obl-uuid-1"],
    )

    assert context.schema_version == 2
    assert context.obligations[0].external_id == "12345"


def test_case_context_rejects_v1_payloads():
    with pytest.raises(ValidationError):
        CaseContext(
            schema_version=1,
            party=_party(),
            obligations=[_obligation()],
        )


def test_case_context_ignores_top_level_extras_but_forbids_nested_identity_extras():
    context = CaseContext(
        party=_party(),
        obligations=[_obligation()],
        legacy_rollout_flag=True,
    )
    assert "legacy_rollout_flag" not in context.model_dump()

    with pytest.raises(ValidationError):
        ObligationInfo(
            id="obl-uuid-1",
            external_id="12345",
            provider_type="sage_200",
            invoice_number="INV-12345",
            original_amount=100.0,
            amount_due=75.0,
            sage_id="retired",
        )


def test_party_source_must_equal_provider_type():
    with pytest.raises(ValidationError, match="source must equal provider_type"):
        _party(source="sage")


def test_local_context_models_track_shared_contract_core():
    assert (
        CaseContext.model_fields["schema_version"].default
        == CaseContextV2.model_fields["schema_version"].default
        == 2
    )
    assert set(CaseContext.model_fields["schema_version"].annotation.__args__) == {2, 3, 4}
    assert set(CaseContextV2.model_fields["schema_version"].annotation.__args__) <= set(
        CaseContext.model_fields["schema_version"].annotation.__args__
    )
    assert PartyInfo.model_config["extra"] == PartyInfoV2.model_config["extra"] == "forbid"
    assert (
        ObligationInfo.model_config["extra"] == ObligationInfoV2.model_config["extra"] == "forbid"
    )
    for field_name in ("id", "external_id", "provider_type"):
        assert ObligationInfo.model_fields[field_name].is_required()


def test_current_datalake_context_accepts_additive_obligation_fields():
    context = CaseContext(
        schema_version=4,
        party=_party(),
        obligations=[
            _obligation(
                silver_version_id="silver-core-obligation-v1",
                document_no="INV-12345",
                sage_transaction_urn="urn:sage:txn:12345",
                is_outstanding=True,
                is_overdue=True,
                days_overdue=17,
                effective_grace_days=3,
                is_chase_eligible=True,
                source_query_raw=None,
                is_source_disputed=False,
                procurement_context_status="verified",
                has_verified_purchase_order=True,
                purchase_order_reference="PO-55",
            )
        ],
    )

    assert context.schema_version == 4
    assert context.collection_basis == "overdue"
    assert context.obligations[0].silver_version_id == "silver-core-obligation-v1"
    assert context.obligations[0].procurement_context_status == "verified"


def test_current_datalake_detection_requires_schema_v4():
    context = CaseContext(
        schema_version=3,
        party=_party(),
        obligations=[_obligation()],
        source_sync_run_id="sync-1",
        application_run_id="app-1",
        policy_snapshot_id="policy-1",
    )

    assert context.uses_current_datalake_contract() is False


def test_current_datalake_generate_request_requires_lineage_and_recipient():
    context = CaseContext(
        schema_version=4,
        party=_party(),
        obligations=[
            _obligation(
                is_overdue=True,
                is_sendable=True,
                is_chase_eligible=True,
            )
        ],
    )

    with pytest.raises(ValidationError, match="required lineage"):
        GenerateDraftRequest(context=context)


def test_current_datalake_generate_request_rejects_all_blocked_candidates():
    context = CaseContext(
        schema_version=4,
        party=_party(),
        obligations=[
            _obligation(
                is_overdue=True,
                is_sendable=True,
                is_chase_eligible=True,
                is_source_disputed=True,
                source_query_raw="Query",
            )
        ],
        debtor_contact={"email": "ap@example.com"},
        source_sync_run_id="sync-1",
        application_run_id="app-1",
        core_snapshot_watermark="2026-05-01T00:00:00Z",
        application_snapshot_watermark="2026-05-01T00:10:00Z",
        application_decision_cutoff="2026-05-01T00:15:00Z",
        policy_snapshot_id="policy-1",
        draft_candidate_id="cand-1",
    )

    with pytest.raises(ValidationError, match="no eligible/sendable obligations"):
        GenerateDraftRequest(context=context)
