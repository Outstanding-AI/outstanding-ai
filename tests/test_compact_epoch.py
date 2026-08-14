from __future__ import annotations

import pytest
from pydantic import ValidationError

try:
    from solvix_contracts.datalake import (
        compact_current_manifest_digest,
        compact_current_relation_for,
        compact_current_relations,
    )
except ImportError:
    compact_current_manifest_digest = None
    compact_current_relation_for = None
    compact_current_relations = None

from src.lake.compact_epoch import (
    COMPACT_CURRENT_EPOCH_SCHEMA_VERSION,
    CompactCurrentEpochV1,
    compact_epoch_source,
    unresolved_virtual_current_relations,
)
from src.lake.materialized_current_registry import (
    AI_CONTEXT_CURRENT_VIEWS,
    compact_epoch_table_name,
    logical_relation_for_current_view,
)
from src.lake.models import DraftGenerationHandoff


def _epoch_payload(*, tenant_id: str = "tenant-1") -> dict:
    epoch_id = "epoch-1"
    digest = compact_current_manifest_digest() if compact_current_manifest_digest else "a" * 64
    view_names = (
        tuple(relation.canonical_view for relation in compact_current_relations())
        if compact_current_relations
        else AI_CONTEXT_CURRENT_VIEWS
    )
    members = []
    for view_name in view_names:
        relation = compact_current_relation_for(view_name) if compact_current_relation_for else None
        members.append(
            {
                "logical_relation": logical_relation_for_current_view(view_name),
                "materialized_table": compact_epoch_table_name(view_name),
                "compact_epoch_id": epoch_id,
                "canonical_columns": list(relation.canonical_columns)
                if relation is not None
                else ["tenant_id", "id"],
            }
        )
    return {
        "schema_version": COMPACT_CURRENT_EPOCH_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "epoch_id": epoch_id,
        "manifest_digest": digest,
        "committed_at": "2026-08-14T10:00:00Z",
        "members": members,
    }


def test_compact_only_handoff_requires_complete_tenant_epoch() -> None:
    base = {
        "tenant_id": "tenant-1",
        "sync_run_id": "sync-1",
        "manifest_uri": "s3://bucket/manifest.json",
        "data_lake_region": "eu-west-2",
        "current_read_mode": "compact_only",
    }

    with pytest.raises(ValidationError, match="compact_current_epoch is required"):
        DraftGenerationHandoff.model_validate(base)

    wrong_tenant = {**base, "compact_current_epoch": _epoch_payload(tenant_id="tenant-2")}
    with pytest.raises(ValidationError, match="tenant_id does not match"):
        DraftGenerationHandoff.model_validate(wrong_tenant)

    incomplete = _epoch_payload()
    incomplete["members"] = incomplete["members"][:-1]
    # The released shared model rejects an incomplete member set while it is
    # being parsed; the pre-release fallback reports the missing relation
    # during compact-only validation.  Both are fail-closed behaviour.
    with pytest.raises(ValidationError):
        DraftGenerationHandoff.model_validate({**base, "compact_current_epoch": incomplete})


def test_compact_only_handoff_rejects_legacy_map() -> None:
    with pytest.raises(ValidationError, match="must not include the legacy"):
        DraftGenerationHandoff(
            tenant_id="tenant-1",
            sync_run_id="sync-1",
            manifest_uri="s3://bucket/manifest.json",
            data_lake_region="eu-west-2",
            current_read_mode="compact_only",
            compact_current_epoch=_epoch_payload(),
            current_source_map={
                "silver_core_parties_current": "silver_current_silver_core_parties_current"
            },
        )


def test_compact_source_is_epoch_filtered_and_preserves_canonical_columns() -> None:
    epoch = CompactCurrentEpochV1.model_validate(_epoch_payload())

    source = compact_epoch_source(
        epoch,
        tenant_id="tenant-1",
        current_view="silver_core_parties_current",
    )

    assert source.startswith("(SELECT ")
    assert '"tenant_id"' in source.split(" FROM ", 1)[0]
    assert 'FROM "silver_current_silver_core_parties_current_v2"' in source
    assert "\"tenant_id\" = 'tenant-1'" in source
    assert "\"compact_epoch_id\" = 'epoch-1'" in source
    assert unresolved_virtual_current_relations(f"SELECT * FROM {source}") == ()


def test_compact_source_uses_reused_member_partition_not_pointer_epoch() -> None:
    payload = _epoch_payload()
    payload["members"][0]["compact_epoch_id"] = "prior-epoch"
    epoch = CompactCurrentEpochV1.model_validate(payload)

    source = compact_epoch_source(
        epoch,
        tenant_id="tenant-1",
        current_view="silver_core_parties_current",
    )

    assert "\"compact_epoch_id\" = 'prior-epoch'" in source


def test_unresolved_current_guard_checks_from_and_join_relations_only() -> None:
    sql = """
        SELECT p.id
        FROM silver_core_parties_current p
        JOIN outstandingai_eu_west_2.silver_app_drafts_current d ON d.party_id = p.id
    """
    assert unresolved_virtual_current_relations(sql) == (
        "silver_app_drafts_current",
        "silver_core_parties_current",
    )
    assert (
        unresolved_virtual_current_relations("SELECT * FROM gold_dashboard_summary_current") == ()
    )
