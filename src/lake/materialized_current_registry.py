"""Static validation for backend-issued compact-current source maps.

AI never chooses or proves a compact publication. It accepts only exact
canonical-to-compact pairs from the backend handoff; absent or invalid entries
leave reads on canonical ``*_current`` views.
"""

from __future__ import annotations

from solvix_contracts.datalake import load_manifest_v2
from solvix_contracts.silver_application import (
    EMPTY_HOLD_TABLES,
    SILVER_APPLICATION_V1_COLLISION_TABLES,
)

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
ROLLOUT_DORMANT_HOLD_APPLICATION_TABLES: frozenset[str] = frozenset(
    {
        "application_event_scopes",
        "application_events",
        "application_phase_stats",
        "application_run_stats",
    }
)
INACTIVE_APPLICATION_TABLES: frozenset[str] = frozenset(
    set(EMPTY_HOLD_TABLES)
    | set(ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES)
    | set(ROLLOUT_DORMANT_HOLD_APPLICATION_TABLES)
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


def approved_materialized_current_source(canonical_view: str, candidate: str | None) -> str:
    """Return an exact approved compact identifier or the canonical fallback."""

    expected = f"silver_current_{canonical_view}"
    if canonical_view in USEFUL_MATERIALIZED_CURRENT_VIEWS and candidate == expected:
        return expected
    return canonical_view


__all__ = [
    "CORE_MATERIALIZED_CURRENT_VIEWS",
    "INACTIVE_APPLICATION_TABLES",
    "ROLLOUT_DORMANT_HOLD_APPLICATION_TABLES",
    "ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES",
    "USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS",
    "USEFUL_MATERIALIZED_CURRENT_VIEWS",
    "approved_materialized_current_source",
]
