from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from src.lake import CaseContextHydrator, ContextHydrationError, DraftCandidate


class _FakeReader:
    def __init__(self) -> None:
        self.execute_one_calls: list[tuple[str, list[Any]]] = []
        self.execute_calls: list[tuple[str, list[Any]]] = []
        self.party: dict[str, Any] | None = {
            "id": "party-1",
            "external_id": "CUST-1",
            "provider_type": "sage_200",
            "customer_code": "CUST-1",
            "name": "Acme Ltd",
            "country_code": "GB",
            "currency": "GBP",
            "base_currency": "GBP",
            "relationship_tier": "standard",
            "is_verified": True,
            "source": "sage_200",
            "touch_count": 2,
            "last_touch_at": datetime(2026, 4, 1, 9, 30, tzinfo=timezone.utc),
            "last_touch_channel": "email",
            "last_sender_level": 1,
            "last_tone_used": "firm",
            "broken_promises_count": 1,
            "hardship_indicated": False,
            "monthly_touch_count": 2,
            "behaviour_segment": "reliable_late_payer",
        }
        self.lane: dict[str, Any] | None = {
            "id": "lane-1",
            "entry_level": 1,
            "current_level": 2,
            "status": "open",
            "suppression_state": None,
            "outstanding_amount": 250.0,
            "outstanding_amount_base": 250.0,
            "tone_ladder_snapshot_json": '["firm", "final_notice"]',
        }
        self.obligations = [
            {
                "id": "obl-1",
                "external_id": "INV-1",
                "provider_type": "sage_200",
                "provider_ref": "INV-1",
                "invoice_number": "INV-1",
                "original_amount": 250.0,
                "original_amount_base": 250.0,
                "amount_due": 250.0,
                "amount_due_base": 250.0,
                "currency": "GBP",
                "base_currency": "GBP",
                "due_date": date(2026, 3, 1),
                "days_past_due": 55,
                "state": "open",
            }
        ]
        self.history = [
            {
                "event_type": "level_started",
                "from_status": "open",
                "to_status": "open",
                "from_level": 1,
                "to_level": 2,
                "draft_id": "draft-1",
                "touch_id": "touch-1",
                "thread_id": "thread-1",
                "detail_json": '{"reason": "cadence"}',
                "created_at": datetime(2026, 4, 2, 9, 30, tzinfo=timezone.utc),
            }
        ]

    def execute_one(self, sql: str, params: list[Any]) -> dict[str, Any] | None:
        self.execute_one_calls.append((sql, params))
        if "FROM (SELECT" in sql and "parties" in sql:
            return self.party
        if "collection_lanes" in sql:
            return self.lane
        return None

    def execute(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        self.execute_calls.append((sql, params))
        if "collection_lane_invoices" in sql:
            return self.obligations
        if "collection_lane_history" in sql:
            return self.history
        return []


def _candidate() -> DraftCandidate:
    return DraftCandidate(
        party_id="party-1",
        lane_id="lane-1",
        sync_run_id="sync-1",
        candidate_id="candidate-1",
    )


def test_hydrate_candidate_builds_existing_case_context_shape() -> None:
    reader = _FakeReader()
    context = CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())

    assert context.party.party_id == "party-1"
    assert context.party.external_id == "CUST-1"
    assert context.party.provider_type == "sage_200"
    assert context.party.source == "sage_200"
    assert context.behavior.behaviour_segment == "reliable_late_payer"
    assert context.obligations[0].invoice_number == "INV-1"
    assert context.obligations[0].due_date == "2026-03-01"
    assert context.collection_lane_id == "lane-1"
    assert context.lane["invoice_refs"] == ["INV-1"]
    assert context.lane["tone_ladder"] == ["firm", "final_notice"]
    assert context.lane_contexts[0].lane_id == "lane-1"
    assert context.lane_history[0]["detail"] == {"reason": "cadence"}
    assert context.sendable_obligation_ids == ["obl-1"]

    assert reader.execute_one_calls[0][1] == ["tenant-1", "party-1"]
    assert reader.execute_one_calls[1][1] == ["tenant-1", "lane-1"]
    assert reader.execute_calls[0][1] == ["tenant-1", "tenant-1", "lane-1", "open"]
    assert reader.execute_calls[1][1] == ["tenant-1", "lane-1"]


def test_hydrate_candidate_fails_closed_when_party_missing() -> None:
    reader = _FakeReader()
    reader.party = None

    with pytest.raises(ContextHydrationError, match="Party not found"):
        CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())


def test_hydrate_candidate_fails_closed_when_lane_missing() -> None:
    reader = _FakeReader()
    reader.lane = None

    with pytest.raises(ContextHydrationError, match="Collection lane not found"):
        CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())
