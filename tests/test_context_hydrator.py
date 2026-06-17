from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from src.lake import (
    BatchHydrationResult,
    CaseContextHydrator,
    ContextHydrationError,
    DraftCandidate,
)


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
            "collection_case_id": "case-1",
            "threading_strategy": "single_active_debtor_thread",
            "threading_mode": "case_continuation",
            "active_thread_id": "thread-1",
            "active_conversation_id": "conv-1",
            "entry_level": 1,
            "current_level": 2,
            "status": "open",
            "suppression_state": None,
            "outstanding_amount": 250.0,
            "outstanding_amount_base": 250.0,
            "tone_ladder_snapshot_json": '["firm", "final_notice"]',
            "policy_snapshot_id": "policy-1",
            "application_run_id": "app-run-1",
            "updated_at": datetime(2026, 4, 3, 9, 30, tzinfo=timezone.utc),
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
        self.actual_sent_scope = [
            {
                "party_id": "party-1",
                "sent_draft_analysis_event_id": "analysis-event-1",
                "application_content_hash": "analysis-hash-1",
                "draft_id": "draft-1",
                "touch_id": "touch-1",
                "provider_message_id": "msg-1",
                "lane_id": "lane-1",
                "sent_at": datetime(2026, 4, 2, 9, 45, tzinfo=timezone.utc),
                "invoice_refs_generated_json": '["INV-OLD"]',
                "invoice_refs_sent_json": '["INV-1"]',
                "invoice_refs_added_json": '["INV-1"]',
                "invoice_refs_removed_json": '["INV-OLD"]',
                "invoice_scope_changed": True,
                "edit_severity": "critical",
                "payment_expectation_added": True,
                "payment_expectation_kind": "promise_to_pay",
                "payment_expectation_date": date(2026, 4, 10),
                "payment_expectation_amount": 250.0,
                "review_reason_codes_json": '["invoice_scope_changed"]',
            }
        ]
        self.contacts = [
            {
                "id": "contact-1",
                "name": "AP Team",
                "email": "ap@example.com",
                "is_default": True,
                "is_send_statement_to": True,
                "is_preferred_send_statement_to": True,
                "recipient_selection_source": "send_statement_to_preferred",
                "is_active": True,
                "email_valid": True,
                "source": "sage",
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
        # Bulk hydration path (P3-2) routes parties + lanes through
        # ``execute`` with ``IN %s`` filters; legacy per-id paths still
        # use ``execute_one`` above. The fake reader emulates both so
        # tests written against either path keep working.
        if "silver_core_parties_current" in sql:
            return [self.party] if self.party is not None else []
        if "silver_app_collection_lanes_current" in sql:
            return [self.lane] if self.lane is not None else []
        if "collection_lane_invoices" in sql:
            return self.obligations
        if "party_contacts" in sql:
            return self.contacts
        if "collection_lane_history" in sql:
            return self.history
        if "sent_draft_analysis_events_current" in sql:
            return self.actual_sent_scope
        return []


def _candidate() -> DraftCandidate:
    return DraftCandidate(
        party_id="party-1",
        lane_id="lane-1",
        collection_case_id="case-1",
        threading_strategy="single_active_debtor_thread",
        threading_mode="case_continuation",
        active_thread_id="thread-1",
        active_conversation_id="conv-1",
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
    assert context.schema_version == 4
    assert context.source_sync_run_id == "sync-1"
    assert context.application_run_id == "app-run-1"
    assert context.policy_snapshot_id == "policy-1"
    assert context.draft_candidate_id == "candidate-1"
    assert context.debtor_contact["email"] == "ap@example.com"
    assert context.debtor_contact["is_preferred_send_statement_to"] is True
    assert context.debtor_contact["recipient_selection_source"] == "send_statement_to_preferred"
    assert context.behavior.behaviour_segment == "reliable_late_payer"
    assert context.obligations[0].invoice_number == "INV-1"
    assert context.obligations[0].due_date == "2026-03-01"
    assert context.obligations[0].is_sendable is True
    assert context.obligations[0].is_overdue is True
    assert context.collection_lane_id == "lane-1"
    assert context.collection_case_id == "case-1"
    assert context.threading_strategy == "single_active_debtor_thread"
    assert context.threading_mode == "case_continuation"
    assert context.case_lane_contexts
    assert context.case_lane_contexts[0]["lane_id"] == "lane-1"
    assert context.active_thread_subject is None
    assert context.lane["invoice_refs"] == ["INV-1"]
    assert context.lane["tone_ladder"] == ["firm", "final_notice"]
    assert context.lane_contexts[0].lane_id == "lane-1"
    assert context.lane_history[0]["detail"] == {"reason": "cadence"}
    assert context.actual_sent_scope_history[0].invoice_refs_sent == ["INV-1"]
    assert context.actual_sent_scope_history[0].invoice_refs_removed == ["INV-OLD"]
    assert context.actual_sent_scope_history[0].payment_expectation_added is True
    assert context.actual_sent_scope_history[0].sent_draft_analysis_event_id == "analysis-event-1"
    assert context.actual_sent_scope_history[0].application_content_hash == "analysis-hash-1"
    assert "sent_draft_analysis_event:analysis-event-1" in context.input_silver_version_ids
    assert "sent_draft_analysis_hash:analysis-hash-1" in context.input_silver_version_ids
    assert context.sendable_obligation_ids == ["obl-1"]

    # P3-2: per-id loaders now route through the bulk SELECTs with a
    # single-element ``IN %s`` tuple so the same code paths cover both
    # ``hydrate_candidate`` and ``hydrate_batch``.
    parties_call = reader.execute_calls[0]
    assert parties_call[1][0] == "tenant-1"
    assert parties_call[1][1] == ("party-1",)
    lanes_call = reader.execute_calls[1]
    assert lanes_call[1] == ["tenant-1", ("lane-1",)]
    obligations_call = reader.execute_calls[2]
    assert obligations_call[1] == ["tenant-1", ("lane-1",), "tenant-1", "open"]
    contacts_call = reader.execute_calls[3]
    assert contacts_call[1] == ["tenant-1", ("party-1",)]
    history_call = reader.execute_calls[4]
    assert history_call[1] == ["tenant-1", ("lane-1",)]
    actual_sent_call = reader.execute_calls[5]
    assert actual_sent_call[1] == ["tenant-1", ("party-1",), "tenant-1", "tenant-1"]

    all_sql = "\n".join(
        [sql for sql, _ in reader.execute_one_calls] + [sql for sql, _ in reader.execute_calls]
    )
    assert "ROW_NUMBER()" not in all_sql
    assert "silver_core_parties_current" in all_sql
    assert "party_collection_state_events_current" in all_sql
    assert "party_comm_state_events_current" in all_sql
    assert "party_behavior_profile_versions_current" in all_sql
    assert "silver_core_party_contacts_current" in all_sql
    assert "is_preferred_send_statement_to" in all_sql
    assert "recipient_selection_source" in all_sql
    assert "silver_core_obligations_current" in all_sql
    assert "silver_app_collection_lanes_current" in all_sql
    assert "COALESCE(lane_id, id) IN %s" in all_sql
    assert "silver_app_collection_lane_invoices_current" in all_sql
    assert "silver_app_collection_lane_history_current" in all_sql
    assert "sent_draft_analysis_events_current" in all_sql
    assert "source_query_raw" in all_sql
    assert "is_source_disputed" in all_sql
    assert "NULLIF(o.currency_code, '')" in all_sql
    assert all_sql.index("NULLIF(o.currency_code, '')") < all_sql.index("NULLIF(o.currency, '')")


def test_hydrate_candidate_truncates_lane_history_to_latest_25() -> None:
    """Single-candidate hydration must keep the same history cap as batch mode."""
    reader = _FakeReader()
    reader.history = [
        {
            "event_type": f"event-{idx}",
            "detail_json": "{}",
            "created_at": datetime(2026, 4, 2, 9, 30, tzinfo=timezone.utc),
        }
        for idx in range(30)
    ]

    context = CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())
    event_types = [event["event_type"] for event in context.lane_history]

    assert len(event_types) == 25
    assert "event-29" not in event_types
    assert event_types[0] == "event-24"
    assert event_types[-1] == "event-0"


def test_hydrate_candidate_preserves_source_query_blocks() -> None:
    reader = _FakeReader()
    reader.obligations[0].update(
        {
            "is_source_disputed": True,
            "has_source_query_flag": True,
            "source_query_raw": "Queried in Sage",
            "source_dispute_type": "invoice_query",
            "source_dispute_observed_from": "sales_posted_transactions",
        }
    )

    context = CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())
    obligation = context.obligations[0]

    assert obligation.is_source_disputed is True
    assert obligation.has_source_query_flag is True
    assert obligation.source_query_raw == "Queried in Sage"
    assert obligation.source_dispute_type == "invoice_query"
    assert obligation.source_dispute_observed_from == "sales_posted_transactions"
    assert obligation.is_sendable is False
    assert obligation.is_chase_eligible is False
    assert context.sendable_obligation_ids == []


def test_hydrate_candidate_excludes_stale_lane_snapshot_when_obligation_is_closed() -> None:
    reader = _FakeReader()
    reader.obligations[0].update(
        {
            "amount_due": 0.0,
            "amount_due_base": 0.0,
            "obligation_is_open": False,
            "is_outstanding": True,
            "is_overdue": True,
            "days_overdue": 73,
            "state": "closed",
        }
    )

    context = CaseContextHydrator("tenant-1", reader).hydrate_candidate(_candidate())

    assert context.obligations == []
    assert context.sendable_obligation_ids == []
    assert context.lane["invoice_refs"] == []

    obligation_sql = reader.execute_calls[2][0]
    assert "COALESCE(o.amount_due, 0) > 0" in obligation_sql
    assert "COALESCE(o.is_open, o.amount_due > 0, FALSE) = TRUE" in obligation_sql
    assert "COALESCE(li.is_outstanding" not in obligation_sql
    assert "COALESCE(\n                    li.is_overdue" not in obligation_sql
    assert "COALESCE(\n                    li.days_overdue" not in obligation_sql


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


# ---------------------------------------------------------------------------
# Batch hydration (P3-2)
# ---------------------------------------------------------------------------


class _BatchFakeReader:
    """Multi-party / multi-lane fake reader for ``hydrate_batch`` tests.

    Tracks every ``execute`` call so tests can assert exactly six
    bulk SELECTs were issued -- one per shape -- regardless of how
    many candidates were passed in.
    """

    def __init__(
        self,
        *,
        parties: dict[str, dict[str, Any]],
        lanes: dict[str, dict[str, Any]],
        obligations_by_lane: dict[str, list[dict[str, Any]]] | None = None,
        contacts_by_party: dict[str, list[dict[str, Any]]] | None = None,
        history_by_lane: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.parties = parties
        self.lanes = lanes
        self.obligations_by_lane = obligations_by_lane or {}
        self.contacts_by_party = contacts_by_party or {}
        self.history_by_lane = history_by_lane or {}
        self.execute_one_calls: list[tuple[str, list[Any]]] = []
        self.execute_calls: list[tuple[str, list[Any]]] = []

    def execute_one(self, sql: str, params: list[Any]) -> dict[str, Any] | None:
        self.execute_one_calls.append((sql, params))
        return None  # batch path goes through execute()

    def execute(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        self.execute_calls.append((sql, params))
        if "silver_core_parties_current" in sql:
            ids = params[1] if len(params) > 1 else ()
            return [self.parties[pid] for pid in ids if pid in self.parties]
        if "silver_app_collection_lanes_current" in sql:
            ids = params[1] if len(params) > 1 else ()
            return [self.lanes[lid] for lid in ids if lid in self.lanes]
        if "collection_lane_invoices" in sql:
            ids = params[1] if len(params) > 1 else ()
            rows: list[dict[str, Any]] = []
            for lid in ids:
                for r in self.obligations_by_lane.get(lid, []):
                    rows.append({**r, "lane_id": lid})
            return rows
        if "party_contacts" in sql:
            ids = params[1] if len(params) > 1 else ()
            rows = []
            for pid in ids:
                for c in self.contacts_by_party.get(pid, []):
                    rows.append({**c, "party_id": pid})
            return rows
        if "collection_lane_history" in sql:
            ids = params[1] if len(params) > 1 else ()
            rows = []
            for lid in ids:
                for h in self.history_by_lane.get(lid, []):
                    rows.append({**h, "lane_id": lid})
            return rows
        if "sent_draft_analysis_events_current" in sql:
            return []
        return []


def _party_row(party_id: str, customer_code: str) -> dict[str, Any]:
    return {
        "id": party_id,
        "external_id": customer_code,
        "provider_type": "sage_200",
        "customer_code": customer_code,
        "name": f"Customer {customer_code}",
        "country_code": "GB",
        "currency": "GBP",
        "base_currency": "GBP",
        "relationship_tier": "standard",
        "is_verified": True,
        "source": "sage_200",
        "broken_promises_count": 0,
        "hardship_indicated": False,
        "monthly_touch_count": 0,
        "behaviour_segment": "reliable_late_payer",
    }


def _lane_row(lane_id: str) -> dict[str, Any]:
    return {
        "id": lane_id,
        "lane_id": lane_id,
        "entry_level": 1,
        "current_level": 1,
        "status": "open",
        "suppression_state": None,
        "outstanding_amount": 100.0,
        "outstanding_amount_base": 100.0,
        "tone_ladder_snapshot_json": '["firm"]',
        "policy_snapshot_id": "policy-1",
        "application_run_id": "app-run-1",
        "updated_at": datetime(2026, 4, 3, 9, 30, tzinfo=timezone.utc),
    }


def _obligation_row(obligation_id: str, *, is_source_disputed: bool = False) -> dict[str, Any]:
    return {
        "id": obligation_id,
        "external_id": obligation_id,
        "provider_type": "sage_200",
        "provider_ref": obligation_id,
        "invoice_number": obligation_id,
        "original_amount": 100.0,
        "original_amount_base": 100.0,
        "amount_due": 100.0,
        "amount_due_base": 100.0,
        "currency": "GBP",
        "base_currency": "GBP",
        "due_date": date(2026, 3, 1),
        "days_past_due": 30,
        "state": "disputed" if is_source_disputed else "open",
        "is_source_disputed": is_source_disputed,
        "source_query_raw": "queried" if is_source_disputed else None,
        "source_dispute_type": "sage_query" if is_source_disputed else None,
    }


def _contact_row() -> dict[str, Any]:
    return {
        "id": "contact-1",
        "name": "AP Team",
        "email": "ap@example.com",
        "is_default": True,
        "is_send_statement_to": False,
        "is_preferred_send_statement_to": False,
        "recipient_selection_source": "default_email",
        "is_active": True,
        "email_valid": True,
        "source": "sage",
    }


def test_hydrate_batch_issues_one_bulk_select_per_shape() -> None:
    """Six Athena SELECTs for N candidates, not 6 * N. The whole point
    of batch hydration is to collapse per-candidate fan-out.
    """
    reader = _BatchFakeReader(
        parties={
            "party-A": _party_row("party-A", "CUST-A"),
            "party-B": _party_row("party-B", "CUST-B"),
            "party-C": _party_row("party-C", "CUST-C"),
        },
        lanes={
            "lane-A": _lane_row("lane-A"),
            "lane-B": _lane_row("lane-B"),
            "lane-C": _lane_row("lane-C"),
        },
        obligations_by_lane={
            "lane-A": [_obligation_row("inv-A1")],
            "lane-B": [_obligation_row("inv-B1")],
            "lane-C": [_obligation_row("inv-C1")],
        },
        contacts_by_party={
            "party-A": [_contact_row()],
            "party-B": [_contact_row()],
            "party-C": [_contact_row()],
        },
    )
    candidates = [
        DraftCandidate(
            party_id="party-A", lane_id="lane-A", sync_run_id="sync-1", candidate_id="c-A"
        ),
        DraftCandidate(
            party_id="party-B", lane_id="lane-B", sync_run_id="sync-1", candidate_id="c-B"
        ),
        DraftCandidate(
            party_id="party-C", lane_id="lane-C", sync_run_id="sync-1", candidate_id="c-C"
        ),
    ]

    results = CaseContextHydrator("tenant-1", reader).hydrate_batch(candidates)

    assert len(results) == 3
    assert all(r.context is not None and r.error is None for r in results)
    assert reader.execute_one_calls == []
    assert len(reader.execute_calls) == 6, (
        "hydrate_batch must issue exactly 6 bulk SELECTs (parties / lanes / "
        "lane obligations / party contacts / lane history / actual sent scope) regardless of "
        "candidate count."
    )

    sql_blob = "\n".join(sql for sql, _ in reader.execute_calls)
    assert "silver_core_parties_current" in sql_blob
    assert "silver_app_collection_lanes_current" in sql_blob
    assert "collection_lane_invoices" in sql_blob
    assert "party_contacts" in sql_blob
    assert "collection_lane_history" in sql_blob
    assert "sent_draft_analysis_events_current" in sql_blob

    # Each batch query passes the IDs as a tuple param so ``IN %s``
    # renders to ``('id-1', 'id-2', ...)``.
    parties_call = reader.execute_calls[0]
    assert isinstance(parties_call[1][1], tuple)
    assert set(parties_call[1][1]) == {"party-A", "party-B", "party-C"}


def test_hydrate_batch_partial_failure_per_candidate() -> None:
    """A candidate whose party is missing from regional Silver yields
    a per-candidate ``ContextHydrationError``; the rest of the batch
    keeps building. Manifest mode treats hydration as best-effort
    per-candidate, not all-or-nothing.
    """
    reader = _BatchFakeReader(
        parties={
            "party-A": _party_row("party-A", "CUST-A"),
            # party-B intentionally absent -- simulates a Silver row
            # not yet visible after a fresh sync partition repair.
        },
        lanes={
            "lane-A": _lane_row("lane-A"),
            "lane-B": _lane_row("lane-B"),
        },
        obligations_by_lane={"lane-A": [_obligation_row("inv-A1")]},
        contacts_by_party={"party-A": [_contact_row()]},
    )
    candidates = [
        DraftCandidate(
            party_id="party-A", lane_id="lane-A", sync_run_id="sync-1", candidate_id="c-A"
        ),
        DraftCandidate(
            party_id="party-B", lane_id="lane-B", sync_run_id="sync-1", candidate_id="c-B"
        ),
    ]

    results = CaseContextHydrator("tenant-1", reader).hydrate_batch(candidates)

    assert len(results) == 2
    by_id = {r.candidate.candidate_id: r for r in results}
    assert isinstance(by_id["c-A"], BatchHydrationResult)
    assert by_id["c-A"].context is not None
    assert by_id["c-A"].error is None

    failed = by_id["c-B"]
    assert failed.context is None
    assert isinstance(failed.error, ContextHydrationError)
    assert "party-B" in str(failed.error)


def test_hydrate_batch_partial_failure_when_lane_missing() -> None:
    """Same partial-failure invariant when the LANE is the missing piece."""
    reader = _BatchFakeReader(
        parties={
            "party-A": _party_row("party-A", "CUST-A"),
            "party-B": _party_row("party-B", "CUST-B"),
        },
        lanes={
            "lane-A": _lane_row("lane-A"),
            # lane-B missing.
        },
        obligations_by_lane={"lane-A": [_obligation_row("inv-A1")]},
        contacts_by_party={
            "party-A": [_contact_row()],
            "party-B": [_contact_row()],
        },
    )
    candidates = [
        DraftCandidate(
            party_id="party-A", lane_id="lane-A", sync_run_id="sync-1", candidate_id="c-A"
        ),
        DraftCandidate(
            party_id="party-B", lane_id="lane-B", sync_run_id="sync-1", candidate_id="c-B"
        ),
    ]

    results = CaseContextHydrator("tenant-1", reader).hydrate_batch(candidates)
    by_id = {r.candidate.candidate_id: r for r in results}
    assert by_id["c-A"].context is not None
    assert by_id["c-B"].context is None
    assert "lane-B" in str(by_id["c-B"].error)


def test_hydrate_batch_preserves_source_query_metadata() -> None:
    """Source-query / source-dispute metadata must survive the bulk
    join + per-candidate assembly so downstream guardrails still see
    the queried-invoice signal.
    """
    reader = _BatchFakeReader(
        parties={"party-A": _party_row("party-A", "CUST-A")},
        lanes={"lane-A": _lane_row("lane-A")},
        obligations_by_lane={
            "lane-A": [
                _obligation_row("inv-A1"),
                _obligation_row("inv-A2", is_source_disputed=True),
            ],
        },
        contacts_by_party={"party-A": [_contact_row()]},
    )
    candidates = [
        DraftCandidate(
            party_id="party-A", lane_id="lane-A", sync_run_id="sync-1", candidate_id="c-A"
        ),
    ]

    [result] = CaseContextHydrator("tenant-1", reader).hydrate_batch(candidates)
    assert result.context is not None
    by_id = {o.id: o for o in result.context.obligations}
    assert by_id["inv-A1"].is_source_disputed is False
    assert by_id["inv-A1"].is_sendable is True
    assert by_id["inv-A2"].is_source_disputed is True
    assert by_id["inv-A2"].source_query_raw == "queried"
    # Source-disputed obligations are never sendable.
    assert by_id["inv-A2"].is_sendable is False


def test_hydrate_batch_preserves_manifest_grouped_lane_scope() -> None:
    reader = _BatchFakeReader(
        parties={"party-A": _party_row("party-A", "CUST-A")},
        lanes={
            "lane-A": _lane_row("lane-A"),
            "lane-B": _lane_row("lane-B"),
        },
        obligations_by_lane={
            "lane-A": [_obligation_row("inv-A1")],
            "lane-B": [
                _obligation_row("inv-B1"),
                _obligation_row("inv-B2", is_source_disputed=True),
            ],
        },
        contacts_by_party={"party-A": [_contact_row()]},
    )
    candidates = [
        DraftCandidate(
            party_id="party-A",
            lane_id="lane-A",
            sync_run_id="sync-1",
            candidate_id="c-A",
            mode="multi_lane",
            obligation_ids=["inv-A1", "inv-B1"],
            lane_contexts=[
                {
                    "lane_id": "lane-A",
                    "collection_lane_id": "lane-A",
                    "current_level": 1,
                    "invoice_refs": ["inv-A1"],
                    "obligation_ids": ["inv-A1"],
                },
                {
                    "lane_id": "lane-B",
                    "collection_lane_id": "lane-B",
                    "current_level": 1,
                    "invoice_refs": ["inv-B1"],
                    "obligation_ids": ["inv-B1"],
                },
            ],
        ),
    ]

    [result] = CaseContextHydrator("tenant-1", reader).hydrate_batch(candidates)

    assert result.context is not None
    assert result.context.mode == "multi_lane"
    assert [obligation.id for obligation in result.context.obligations] == ["inv-A1", "inv-B1"]
    assert result.context.sendable_obligation_ids == ["inv-A1", "inv-B1"]
    assert [context.lane_id for context in result.context.lane_contexts] == ["lane-A", "lane-B"]
    obligation_call = reader.execute_calls[2]
    assert set(obligation_call[1][1]) == {"lane-A", "lane-B"}


def test_hydrate_batch_returns_empty_for_empty_input() -> None:
    reader = _BatchFakeReader(parties={}, lanes={})
    assert CaseContextHydrator("tenant-1", reader).hydrate_batch([]) == []
    # No queries issued when there's nothing to hydrate.
    assert reader.execute_calls == []
    assert reader.execute_one_calls == []
