"""Static validation for backend-issued compact-current sources.

AI never chooses or proves a compact publication.  This module is the local
consumer projection of the shared relation contract: it normalizes deployed
compatibility aliases and exposes the exact relations used by AI hydration.
The shared contracts package remains the long-term authority; keeping this
small adapter local lets a dark AI image be built before the corresponding
immutable contracts release is published.
"""

from __future__ import annotations

from solvix_contracts.datalake import load_manifest_v2
from solvix_contracts.silver_application import (
    EMPTY_HOLD_TABLES,
    SILVER_APPLICATION_V1_COLLISION_TABLES,
)

try:
    from solvix_contracts.datalake import (
        canonical_compact_current_view_name as _shared_canonical_view_name,
    )
    from solvix_contracts.datalake import compact_current_relation_for as _shared_relation_for
except ImportError:  # Contracts release is deployed backend-first.
    _shared_canonical_view_name = None
    _shared_relation_for = None

CORE_MATERIALIZED_CURRENT_VIEWS: tuple[str, ...] = (
    "silver_core_obligations_current",
    "silver_core_parties_current",
    "silver_core_party_contacts_current",
    "silver_core_evidence_current",
    "silver_core_ar_allocation_events_current",
    "silver_core_mail_messages_current",
    "silver_core_mail_party_links_current",
)

# Time-bounded rollout policy refreshed from the one-tenant production
# inventory on 13 August 2026. These remain valid canonical contracts; a fresh
# first-write/reactivation proof must remove the hold before compact routing is
# accepted.
ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES: frozenset[str] = frozenset(
    {
        "app_reconciliation_obs",
        "collection_chain_invoice_evidence",
        "collection_chain_selection_candidate_evidence",
        "collection_chain_selection_evidence",
        "collection_email_invoice_reconciliation_evidence",
    }
)
# These were temporarily classified as dormant from an earlier bounded
# inventory. The live signal rollup proved they remain active
# ingress/observability contracts, so this is a reactivation record, not a
# compact hold.
REACTIVATED_APPLICATION_TABLES: frozenset[str] = frozenset(
    {
        "application_event_scopes",
        "application_events",
        "application_phase_stats",
        "application_run_stats",
    }
)
INACTIVE_APPLICATION_TABLES: frozenset[str] = frozenset(
    set(EMPTY_HOLD_TABLES) | set(ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES)
)


def _application_current_view(table: str) -> str:
    prefix = "silver_app_" if table in SILVER_APPLICATION_V1_COLLISION_TABLES else ""
    return f"{prefix}{table}_current"


_manifest = load_manifest_v2()
USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS: tuple[str, ...] = tuple(
    _application_current_view(table)
    for table in sorted(_manifest.silver_application)
    if table not in {"silver_application_delta_manifest", "silver_application_work_manifest"}
    and table not in INACTIVE_APPLICATION_TABLES
)
USEFUL_MATERIALIZED_CURRENT_VIEWS: frozenset[str] = frozenset(
    (*CORE_MATERIALIZED_CURRENT_VIEWS, *USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS)
)

# Deployed compatibility aliases are never independently materialized.  They
# resolve to the primary projection and therefore share one epoch member.
MATERIALIZED_CURRENT_VIEW_ALIASES: dict[str, str] = {
    "silver_app_collection_cases_current": "collection_cases_current",
    "silver_app_dso_dimension_snapshot_current": "dso_dimension_snapshot_current",
    "silver_app_dso_monthly_snapshot_current": "dso_monthly_snapshot_current",
    "message_workflow_facts_current": "silver_app_message_workflow_facts_current",
    "silver_app_overdue_monthly_snapshot_current": "overdue_monthly_snapshot_current",
    "silver_app_party_dso_monthly_snapshot_current": "party_dso_monthly_snapshot_current",
    "silver_app_query_monthly_snapshot_current": "query_monthly_snapshot_current",
    "silver_app_tracked_threads_current": "tracked_threads_current",
    "silver_app_dispute_obligations_current": "dispute_obligations_current",
}

# Every relation that ``ContextReadRepository`` may touch, including optional
# historical evidence.  Compact-only handoffs must contain all of them so an
# optional exception handler cannot accidentally turn a missing publication
# into an empty evidence result.
AI_CONTEXT_CURRENT_VIEWS: tuple[str, ...] = (
    "silver_core_parties_current",
    "silver_core_party_contacts_current",
    "silver_core_obligations_current",
    "silver_app_collection_lanes_current",
    "silver_app_collection_lane_invoices_current",
    "silver_app_collection_lane_history_current",
    "collection_case_threads_current",
    "collection_thread_message_invoice_evidence_current",
    "sent_draft_analysis_events_current",
    "draft_provider_lifecycle_events_current",
    "silver_app_drafts_current",
    "silver_app_collection_mail_invoices_current",
    "silver_core_mail_messages_current",
    "party_collection_state_events_current",
    "party_operator_override_versions_current",
    "party_comm_state_events_current",
    "party_behavior_profile_versions_current",
)


def canonical_materialized_current_view_name(view_name: str) -> str:
    """Resolve a compatibility alias to its single canonical projection."""

    if _shared_canonical_view_name is not None:
        shared = _shared_canonical_view_name(str(view_name))
        if shared is not None:
            return str(shared)
    return MATERIALIZED_CURRENT_VIEW_ALIASES.get(str(view_name), str(view_name))


def logical_relation_for_current_view(view_name: str) -> str:
    """Return the suffix-free logical relation used in epoch manifests."""

    canonical = canonical_materialized_current_view_name(view_name)
    if not canonical.endswith("_current"):
        raise ValueError(f"not a current relation: {view_name!r}")
    return canonical.removesuffix("_current")


def compact_epoch_table_name(view_name: str) -> str:
    """Return the additive atomic-epoch table for one canonical projection."""

    if _shared_relation_for is not None:
        relation = _shared_relation_for(str(view_name))
        if relation is not None:
            return str(relation.materialized_table)
    canonical = canonical_materialized_current_view_name(view_name)
    return f"silver_current_{canonical}_v2"


def approved_materialized_current_source(canonical_view: str, candidate: str | None) -> str:
    """Return an exact approved compact identifier or the canonical fallback."""

    canonical = canonical_materialized_current_view_name(canonical_view)
    expected = f"silver_current_{canonical}"
    if canonical in USEFUL_MATERIALIZED_CURRENT_VIEWS and candidate == expected:
        return expected
    return canonical_view


__all__ = [
    "AI_CONTEXT_CURRENT_VIEWS",
    "CORE_MATERIALIZED_CURRENT_VIEWS",
    "INACTIVE_APPLICATION_TABLES",
    "MATERIALIZED_CURRENT_VIEW_ALIASES",
    "REACTIVATED_APPLICATION_TABLES",
    "ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES",
    "USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS",
    "USEFUL_MATERIALIZED_CURRENT_VIEWS",
    "approved_materialized_current_source",
    "canonical_materialized_current_view_name",
    "compact_epoch_table_name",
    "logical_relation_for_current_view",
]
