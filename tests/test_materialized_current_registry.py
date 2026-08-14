from __future__ import annotations

from src.lake.materialized_current_registry import (
    AI_CONTEXT_CURRENT_VIEWS,
    CORE_MATERIALIZED_CURRENT_VIEWS,
    INACTIVE_APPLICATION_TABLES,
    REACTIVATED_APPLICATION_TABLES,
    ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES,
    USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS,
    USEFUL_MATERIALIZED_CURRENT_VIEWS,
    approved_materialized_current_source,
    canonical_materialized_current_view_name,
    logical_relation_for_current_view,
)


def test_ai_compact_registry_matches_the_61_plus_7_rollout_inventory() -> None:
    assert len(CORE_MATERIALIZED_CURRENT_VIEWS) == 7
    assert len(ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES) == 5
    assert len(REACTIVATED_APPLICATION_TABLES) == 4
    assert len(INACTIVE_APPLICATION_TABLES) == 25
    assert len(USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS) == 61
    assert len(USEFUL_MATERIALIZED_CURRENT_VIEWS) == 68
    assert len(AI_CONTEXT_CURRENT_VIEWS) == 17


def test_ai_accepts_only_exact_useful_canonical_to_compact_pairs() -> None:
    view = "silver_app_drafts_current"
    compact = "silver_current_silver_app_drafts_current"
    assert approved_materialized_current_source(view, compact) == compact
    assert (
        approved_materialized_current_source(view, "silver_current_silver_core_obligations_current")
        == view
    )
    reactivated = "application_events_current"
    assert (
        approved_materialized_current_source(reactivated, f"silver_current_{reactivated}")
        == f"silver_current_{reactivated}"
    )


def test_ai_registry_normalizes_aliases_to_one_suffix_free_relation() -> None:
    alias = "silver_app_collection_cases_current"
    assert canonical_materialized_current_view_name(alias) == "collection_cases_current"
    assert logical_relation_for_current_view(alias) == "collection_cases"
