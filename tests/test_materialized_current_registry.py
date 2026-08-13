from __future__ import annotations

from src.lake.materialized_current_registry import (
    CORE_MATERIALIZED_CURRENT_VIEWS,
    INACTIVE_APPLICATION_TABLES,
    ROLLOUT_DORMANT_HOLD_APPLICATION_TABLES,
    ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES,
    USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS,
    USEFUL_MATERIALIZED_CURRENT_VIEWS,
    approved_materialized_current_source,
)


def test_ai_compact_registry_matches_the_57_plus_7_rollout_inventory() -> None:
    assert len(CORE_MATERIALIZED_CURRENT_VIEWS) == 7
    assert len(ROLLOUT_EMPTY_HOLD_APPLICATION_TABLES) == 5
    assert len(ROLLOUT_DORMANT_HOLD_APPLICATION_TABLES) == 4
    assert len(INACTIVE_APPLICATION_TABLES) == 29
    assert len(USEFUL_APPLICATION_MATERIALIZED_CURRENT_VIEWS) == 57
    assert len(USEFUL_MATERIALIZED_CURRENT_VIEWS) == 64


def test_ai_accepts_only_exact_useful_canonical_to_compact_pairs() -> None:
    view = "silver_app_drafts_current"
    compact = "silver_current_silver_app_drafts_current"
    assert approved_materialized_current_source(view, compact) == compact
    assert (
        approved_materialized_current_source(view, "silver_current_silver_core_obligations_current")
        == view
    )
    inactive = "application_events_current"
    assert approved_materialized_current_source(inactive, f"silver_current_{inactive}") == inactive
