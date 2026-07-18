"""Hydrate AI case context from regional Silver tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from src.api.models.requests import (
    CaseContext,
    ObligationInfo,
)

from . import context_evidence as evidence
from .models import DraftCandidate


class ContextHydrationError(RuntimeError):
    """Raised when a candidate cannot be hydrated from regional Silver."""


@dataclass
class BatchHydrationResult:
    """Per-candidate outcome from ``CaseContextHydrator.hydrate_batch``.

    Exactly one of ``context`` / ``error`` is populated. The batch
    method returns one of these per input candidate so callers can
    iterate without having to filter raises -- partial failure stays
    partial: a missing party / lane in the bulk read raises
    ``ContextHydrationError`` for that candidate only, not the whole
    batch.
    """

    candidate: DraftCandidate
    context: CaseContext | None = None
    error: ContextHydrationError | None = None


class LakeReader(Protocol):
    def execute_one(
        self, sql: str, params: list[Any] | tuple[Any, ...] | None = None
    ) -> dict[str, Any] | None: ...

    def execute(
        self, sql: str, params: list[Any] | tuple[Any, ...] | None = None
    ) -> list[dict[str, Any]]: ...


PARTIES_CURRENT = "silver_core_parties_current"
PARTY_CONTACTS_CURRENT = "silver_core_party_contacts_current"
OBLIGATIONS_CURRENT = "silver_core_obligations_current"
COLLECTION_LANES_CURRENT = "silver_app_collection_lanes_current"
COLLECTION_LANE_INVOICES_CURRENT = "silver_app_collection_lane_invoices_current"
COLLECTION_LANE_HISTORY_CURRENT = "silver_app_collection_lane_history_current"
COLLECTION_CASES_CURRENT = "collection_cases_current"
COLLECTION_CASE_THREADS_CURRENT = "collection_case_threads_current"
COLLECTION_THREAD_MESSAGE_INVOICE_EVIDENCE_CURRENT = (
    "collection_thread_message_invoice_evidence_current"
)
SENT_DRAFT_ANALYSIS_EVENTS_CURRENT = "sent_draft_analysis_events_current"
DRAFT_PROVIDER_LIFECYCLE_EVENTS_CURRENT = "draft_provider_lifecycle_events_current"
DRAFTS_CURRENT = "silver_app_drafts_current"
PARTY_COLLECTION_STATE_CURRENT = "party_collection_state_events_current"
PARTY_COMM_STATE_CURRENT = "party_comm_state_events_current"
PARTY_BEHAVIOR_PROFILE_CURRENT = "party_behavior_profile_versions_current"


def _current_projection(projection: str, alias: str) -> str:
    return f"(SELECT * FROM {projection} WHERE tenant_id = %s) {alias}"


def _current_projection_in(projection: str, alias: str, *, id_column: str) -> str:
    """Tenant-scoped projection sub-select pre-filtered by an ``IN %s`` id list.

    The ``IN %s`` literal renders to ``('id1', 'id2', ...)`` via
    ``solvix_contracts.datalake.athena_dialect.render_params`` -- the
    list parameter must be passed as a tuple/list to the reader.
    Pre-filtering inside the projection sub-select keeps the bulk
    queries pruned so the row scan doesn't fan out across the entire
    tenant.
    """
    return f"(SELECT * FROM {projection} WHERE tenant_id = %s AND {id_column} IN %s) {alias}"


_json_value = evidence.json_value
_int_or_default = evidence.int_or_default
_date_string = evidence.date_string


class CaseContextHydrator:
    """Build existing GenerateDraftRequest context from regional Silver reads."""

    def __init__(
        self,
        tenant_id: str,
        reader: LakeReader,
        *,
        current_source_map: dict[str, str] | None = None,
    ) -> None:
        self.tenant_id = str(tenant_id)
        self.reader = reader
        self.current_source_map = dict(current_source_map or {})

    def _source(self, canonical_view: str) -> str:
        """Use only backend-issued, identifier-safe materialized sources."""

        source = str(self.current_source_map.get(canonical_view) or canonical_view)
        return source if source.replace("_", "").isalnum() else canonical_view

    def hydrate_candidate(self, candidate: DraftCandidate) -> CaseContext:
        """Hydrate a single candidate via per-id reads (legacy single-shot path).

        Manifest-mode draft generation should prefer ``hydrate_batch``
        for N candidates -- it issues one bulk SELECT per shape (5
        SELECTs total) instead of 5 SELECTs per candidate.
        """
        party = self._load_party(candidate.party_id)
        lane = self._load_lane(candidate.lane_id)
        lane_ids = candidate.lane_ids()
        obligations_by_lane = self._load_lane_obligations_batch(lane_ids)
        obligations = self._candidate_obligations(candidate, obligations_by_lane, lane_ids)
        party_contacts = self._load_party_contacts(candidate.party_id)
        history = self._load_lane_history(candidate.lane_id)
        actual_sent_scope_history = self._load_actual_sent_scope_history(candidate.party_id)
        case_ids = [str(candidate.collection_case_id)] if candidate.collection_case_id else []
        case_threads = self._load_case_threads_batch(case_ids)
        temporal_evidence = self._load_case_temporal_invoice_evidence_batch(case_ids)
        commitment_evidence = self._load_case_commitment_evidence_batch(case_ids)
        return self._assemble_case_context(
            candidate=candidate,
            party=party,
            lane=lane,
            obligations=obligations,
            party_contacts=party_contacts,
            history=history,
            actual_sent_scope_history=actual_sent_scope_history,
            case_thread=case_threads.get(str(candidate.collection_case_id or "")),
            case_temporal_evidence=temporal_evidence.get(
                str(candidate.collection_case_id or ""), []
            ),
            case_commitment_evidence=commitment_evidence.get(
                str(candidate.collection_case_id or ""), []
            ),
        )

    def hydrate_batch(self, candidates: list[DraftCandidate]) -> list[BatchHydrationResult]:
        """Hydrate N candidates with bulk current-projection reads.

        Issues one batch SELECT per shape (parties, lanes, lane
        obligations, party contacts, lane history) using ``IN %s``
        literals, then assembles each candidate's ``CaseContext`` from
        the cached maps. Missing required party / lane data surfaces
        as a per-candidate ``ContextHydrationError`` -- the batch keeps
        going so the caller can decide whether to fail the whole
        manifest or partial-fail the affected candidates only.

        Returns a list of ``BatchHydrationResult`` in the input order.
        """
        if not candidates:
            return []

        party_ids = sorted({str(c.party_id) for c in candidates})
        lane_ids = sorted({lane_id for candidate in candidates for lane_id in candidate.lane_ids()})

        parties_by_id = self._load_parties_batch(party_ids)
        lanes_by_id = self._load_lanes_batch(lane_ids)
        obligations_by_lane = self._load_lane_obligations_batch(lane_ids)
        contacts_by_party = self._load_party_contacts_batch(party_ids)
        history_by_lane = self._load_lane_history_batch(lane_ids)
        actual_sent_scope_by_party = self._load_actual_sent_scope_history_batch(party_ids)
        case_ids = sorted({str(c.collection_case_id) for c in candidates if c.collection_case_id})
        case_threads_by_id = self._load_case_threads_batch(case_ids)
        case_temporal_evidence_by_id = self._load_case_temporal_invoice_evidence_batch(case_ids)
        case_commitment_evidence_by_id = self._load_case_commitment_evidence_batch(case_ids)

        results: list[BatchHydrationResult] = []
        for candidate in candidates:
            party_id = str(candidate.party_id)
            lane_id = str(candidate.lane_id)
            candidate_lane_ids = candidate.lane_ids()
            try:
                party = parties_by_id.get(party_id)
                if party is None:
                    raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
                lane = lanes_by_id.get(lane_id)
                if lane is None:
                    raise ContextHydrationError(
                        f"Collection lane not found in regional Silver: {lane_id}"
                    )
                ctx = self._assemble_case_context(
                    candidate=candidate,
                    party=party,
                    lane=lane,
                    obligations=self._candidate_obligations(
                        candidate, obligations_by_lane, candidate_lane_ids
                    ),
                    party_contacts=contacts_by_party.get(party_id, []),
                    history=history_by_lane.get(lane_id, []),
                    actual_sent_scope_history=actual_sent_scope_by_party.get(party_id, []),
                    case_thread=case_threads_by_id.get(str(candidate.collection_case_id or "")),
                    case_temporal_evidence=case_temporal_evidence_by_id.get(
                        str(candidate.collection_case_id or ""), []
                    ),
                    case_commitment_evidence=case_commitment_evidence_by_id.get(
                        str(candidate.collection_case_id or ""), []
                    ),
                )
            except ContextHydrationError as exc:
                results.append(BatchHydrationResult(candidate=candidate, error=exc))
            else:
                results.append(BatchHydrationResult(candidate=candidate, context=ctx))
        return results

    @staticmethod
    def _candidate_obligations(
        candidate: DraftCandidate,
        obligations_by_lane: dict[str, list[ObligationInfo]],
        lane_ids: list[str],
    ) -> list[ObligationInfo]:
        return evidence.candidate_obligations(candidate, obligations_by_lane, lane_ids)

    def _assemble_case_context(
        self,
        *,
        candidate: DraftCandidate,
        party: dict[str, Any],
        lane: dict[str, Any],
        obligations: list[ObligationInfo],
        party_contacts: list[dict[str, Any]],
        history: list[dict[str, Any]],
        actual_sent_scope_history: list[dict[str, Any]],
        case_thread: dict[str, Any] | None = None,
        case_temporal_evidence: list[dict[str, Any]] | None = None,
        case_commitment_evidence: list[dict[str, Any]] | None = None,
    ) -> CaseContext:
        """Compose a V4 ``CaseContext`` from pre-loaded row data.

        Shared by both ``hydrate_candidate`` (per-id path) and
        ``hydrate_batch`` (bulk path) so the assembly logic stays in
        one place.
        """
        lane_invoice_refs = [str(row.invoice_number) for row in obligations if row.invoice_number]
        base_currency = party.get("base_currency") or party.get("currency") or "GBP"
        debtor_contact = party_contacts[0] if party_contacts else None
        decision_time = (
            lane.get("valid_from")
            or lane.get("observed_at")
            or party.get("silver_valid_from")
            or party.get("silver_observed_at")
            or datetime.now(timezone.utc)
        )
        sendable_obligation_ids = [
            obligation.id for obligation in obligations if obligation.is_sendable is not False
        ]
        if candidate.obligation_ids:
            expected_ids = {str(value) for value in candidate.obligation_ids if str(value)}
            sendable_obligation_ids = [
                obligation_id
                for obligation_id in sendable_obligation_ids
                if obligation_id in expected_ids
            ]
        input_silver_version_ids = [
            str(value)
            for value in (
                [party.get("silver_version_id"), lane.get("application_version_id")]
                + [getattr(obligation, "silver_version_id", None) for obligation in obligations]
                + self._actual_sent_scope_version_ids(actual_sent_scope_history)
            )
            if value
        ]
        total_outstanding = sum(float(obligation.amount_due or 0) for obligation in obligations)
        total_overdue = sum(
            float(obligation.amount_due or 0)
            for obligation in obligations
            if getattr(obligation, "is_overdue", False)
        )

        current_level = int(lane.get("current_level") or lane.get("entry_level") or 0)
        lane_context = {
            "collection_lane_id": str(lane["id"]),
            "lane_id": str(lane["id"]),
            "collection_case_id": candidate.collection_case_id or lane.get("collection_case_id"),
            "threading_strategy": candidate.threading_strategy
            or lane.get("threading_strategy")
            or (
                "single_active_debtor_thread"
                if (candidate.collection_case_id or lane.get("collection_case_id"))
                else None
            ),
            "threading_mode": candidate.threading_mode,
            "entry_level": lane.get("entry_level"),
            "current_level": current_level,
            "status": lane.get("status"),
            "suppression_state": lane.get("suppression_state"),
            "outstanding_amount": lane.get("outstanding_amount"),
            "invoice_refs": lane_invoice_refs,
            "tone_ladder": _json_value(lane.get("tone_ladder_snapshot_json"), fallback=[]),
        }
        sparse_lane_context = {
            "lane_id": str(lane["id"]),
            "current_level": current_level,
            "entry_level": lane.get("entry_level"),
            "tone_ladder": lane_context["tone_ladder"],
        }
        lane_contexts = self._candidate_lane_contexts(
            candidate=candidate,
            fallback_context=sparse_lane_context,
            obligations=obligations,
        )
        mode = candidate.mode or ("multi_lane" if len(lane_contexts) > 1 else "single_lane")
        held_commitments, broken_commitments = self._commitments_from_lane_contexts(lane_contexts)
        temporal_evidence = case_temporal_evidence or []

        return CaseContext(
            schema_version=4,
            party=self._party_info(party),
            behavior=self._behavior_info(party),
            obligations=obligations,
            communication=self._communication_info(party),
            case_state=party.get("case_state"),
            base_currency=base_currency,
            total_outstanding_base=lane.get("outstanding_amount_base")
            or lane.get("outstanding_amount"),
            broken_promises_count=int(party.get("broken_promises_count") or 0),
            active_dispute=bool(party.get("dispute_type")),
            hardship_indicated=bool(party.get("hardship_indicated")),
            brand_tone=party.get("tone_override") or "professional",
            touch_cap=int(party.get("touch_cap_override") or 10),
            grace_days=_int_or_default(party.get("grace_days_override"), 0),
            do_not_contact_until=_date_string(party.get("do_not_contact_until")),
            monthly_touch_count=int(party.get("monthly_touch_count") or 0),
            relationship_tier=party.get("relationship_tier") or "standard",
            unsubscribe_requested=bool(party.get("unsubscribe_requested")),
            collection_lane_id=str(lane["id"]),
            collection_case_id=candidate.collection_case_id or lane.get("collection_case_id"),
            threading_strategy=candidate.threading_strategy
            or lane_context.get("threading_strategy")
            or "invoice_cohort_thread",
            threading_mode=candidate.threading_mode
            or lane_context.get("threading_mode")
            or (
                "case_continuation"
                if (candidate.collection_case_id or lane.get("collection_case_id"))
                else "cohort_thread"
            ),
            case_lane_contexts=lane_contexts,
            active_thread_subject=(case_thread or {}).get("latest_subject"),
            collection_thread_messages=self._format_case_thread_messages(temporal_evidence),
            collection_thread_invoice_evidence=self._format_case_invoice_evidence(
                temporal_evidence
            ),
            collection_thread_commitment_evidence=case_commitment_evidence or [],
            held_commitments=held_commitments,
            broken_commitments=broken_commitments,
            manual_intervention_summary=None,
            lane=lane_context,
            lane_history=history,
            actual_sent_scope_history=actual_sent_scope_history,
            lane_mail_mode="single_lane",
            sendable_obligation_ids=sendable_obligation_ids,
            lane_broken_promises_count=int(party.get("broken_promises_count") or 0),
            lane_last_tone_used=party.get("last_tone_used"),
            lane_contexts=lane_contexts,
            mode=mode,
            debtor_contact=debtor_contact,
            party_contacts=party_contacts,
            context_version="v4",
            source_sync_run_id=str(candidate.sync_run_id),
            application_run_id=str(
                lane.get("application_run_id") or f"app:{candidate.sync_run_id}"
            ),
            core_snapshot_watermark=party.get("silver_valid_from") or decision_time,
            application_snapshot_watermark=lane.get("valid_from") or decision_time,
            application_decision_cutoff=decision_time,
            input_silver_version_ids=input_silver_version_ids,
            policy_snapshot_id=str(lane.get("policy_snapshot_id") or ""),
            draft_candidate_id=str(candidate.candidate_id),
            collection_basis=str(
                lane.get("collection_basis") or lane.get("chase_basis") or "overdue"
            ),
            chase_basis=str(lane.get("chase_basis") or lane.get("collection_basis") or "overdue"),
            total_outstanding_amount=total_outstanding,
            total_overdue_amount=total_overdue,
            outstanding_invoice_count=sum(
                1 for obligation in obligations if getattr(obligation, "is_outstanding", True)
            ),
            overdue_invoice_count=sum(
                1 for obligation in obligations if getattr(obligation, "is_overdue", False)
            ),
        )

    @staticmethod
    def _candidate_lane_contexts(
        *,
        candidate: DraftCandidate,
        fallback_context: dict[str, Any],
        obligations: list[ObligationInfo],
    ) -> list[dict[str, Any]]:
        return evidence.candidate_lane_contexts(
            candidate=candidate,
            fallback_context=fallback_context,
            obligations=obligations,
        )

    # ------------------------------------------------------------------
    # Per-id loaders (legacy single-shot path, used by hydrate_candidate)
    # ------------------------------------------------------------------

    def _load_party(self, party_id: str) -> dict[str, Any]:
        rows = self._fetch_parties([party_id])
        if not rows:
            raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
        return rows[0]

    # ------------------------------------------------------------------
    # Bulk loaders (used by hydrate_batch)
    # ------------------------------------------------------------------

    def _load_parties_batch(self, party_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return ``{party_id: party_row}`` for the requested ids.

        One Athena SELECT against ``silver_core_parties_current`` plus
        the same three left-join projections the per-id loader uses.
        Tenant scoping is pushed into every projection sub-select so
        the joins stay pruned.
        """
        if not party_ids:
            return {}
        rows = self._fetch_parties(party_ids)
        return {str(row["id"]): row for row in rows if row.get("id") is not None}

    def _fetch_parties(self, party_ids: list[str]) -> list[dict[str, Any]]:
        """Shared SELECT used by both per-id and batch party loaders."""
        sql = f"""
            SELECT
                p.id,
                p.provider_type,
                COALESCE(p.source_id, p.sage_customer_reference, p.customer_code, p.id) AS external_id,
                COALESCE(p.customer_code, p.sage_customer_reference, p.source_id, p.id) AS customer_code,
                COALESCE(p.customer_name, p.name, p.customer_code, p.id) AS name,
                CAST(NULL AS VARCHAR) AS country_code,
                COALESCE(
                    NULLIF(p.customer_currency_code, ''),
                    NULLIF(p.currency_code, ''),
                    NULLIF(p.currency, ''),
                    'GBP'
                ) AS currency,
                COALESCE(
                    NULLIF(p.base_currency, ''),
                    NULLIF(p.customer_currency_code, ''),
                    NULLIF(p.currency_code, ''),
                    NULLIF(p.currency, ''),
                    'GBP'
                ) AS base_currency,
                p.credit_limit,
                CAST(FALSE AS BOOLEAN) AS on_hold,
                p.silver_version_id,
                p.silver_valid_from,
                p.silver_observed_at,
                cs.case_state,
                cs.pause_reason,
                cs.do_not_contact_until,
                cs.active_dispute_id AS dispute_type,
                comm.touch_count,
                comm.last_outbound_at AS last_touch_at,
                CASE
                    WHEN comm.last_outbound_at IS NOT NULL THEN 'email'
                    ELSE NULL
                END AS last_touch_channel,
                comm.last_inbound_at AS last_response_at,
                CASE
                    WHEN comm.last_inbound_at IS NOT NULL THEN 'inbound_email'
                    ELSE NULL
                END AS last_response_type,
                bp.avg_days_to_pay,
                bp.on_time_rate,
                bp.partial_payment_rate,
                CAST(NULL AS DOUBLE) AS lifetime_value,
                CAST(NULL AS DOUBLE) AS total_collected,
                CAST(NULL AS VARCHAR) AS behaviour_profile,
                CAST(NULL AS VARCHAR) AS behaviour_segment,
                CAST(TRUE AS BOOLEAN) AS is_verified,
                CAST(NULL AS VARCHAR) AS relationship_tier,
                CAST(NULL AS VARCHAR) AS tone_override,
                CAST(NULL AS INTEGER) AS grace_days_override,
                CAST(NULL AS INTEGER) AS touch_cap_override,
                CAST(0 AS INTEGER) AS monthly_touch_count,
                CAST(FALSE AS BOOLEAN) AS hardship_indicated,
                CAST(FALSE AS BOOLEAN) AS unsubscribe_requested,
                CAST(NULL AS VARCHAR) AS customer_type,
                CAST(NULL AS VARCHAR) AS size_bucket
            FROM {_current_projection_in(self._source(PARTIES_CURRENT), "p", id_column="id")}
            LEFT JOIN {_current_projection(self._source(PARTY_COLLECTION_STATE_CURRENT), "cs")}
              ON cs.party_id = p.id
             AND cs.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(self._source(PARTY_COMM_STATE_CURRENT), "comm")}
              ON comm.party_id = p.id
             AND comm.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(self._source(PARTY_BEHAVIOR_PROFILE_CURRENT), "bp")}
              ON bp.party_id = p.id
             AND bp.tenant_id = p.tenant_id
            """
        return self.reader.execute(
            sql,
            [
                self.tenant_id,
                tuple(party_ids),
                self.tenant_id,
                self.tenant_id,
                self.tenant_id,
            ],
        )

    def _load_lane(self, lane_id: str) -> dict[str, Any]:
        rows = self._fetch_lanes([lane_id])
        if not rows:
            raise ContextHydrationError(f"Collection lane not found in regional Silver: {lane_id}")
        return rows[0]

    def _load_lanes_batch(self, lane_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return ``{lane_id: lane_row}`` keyed by ``COALESCE(lane_id, id)``."""
        if not lane_ids:
            return {}
        rows = self._fetch_lanes(lane_ids)
        return {
            str(row.get("lane_id") or row.get("id")): row
            for row in rows
            if (row.get("lane_id") or row.get("id")) is not None
        }

    def _fetch_lanes(self, lane_ids: list[str]) -> list[dict[str, Any]]:
        """Shared SELECT used by both per-id and batch lane loaders.

        Lanes can be addressed via either ``lane_id`` or ``id`` so the
        ``IN`` filter targets ``COALESCE(lane_id, id)`` to match whichever
        the candidate was emitted with.
        """
        sql = f"""
            SELECT lane.*
            FROM (
                SELECT *
                FROM {self._source(COLLECTION_LANES_CURRENT)}
                WHERE tenant_id = %s
                  AND COALESCE(lane_id, id) IN %s
            ) lane
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids)])

    def _load_lane_obligations(self, lane_id: str) -> list[ObligationInfo]:
        rows = self._fetch_lane_obligations([lane_id])
        return [self._obligation_info(row) for row in rows]

    def _load_lane_obligations_batch(self, lane_ids: list[str]) -> dict[str, list[ObligationInfo]]:
        """Return ``{lane_id: [ObligationInfo, ...]}`` for the requested lanes."""
        if not lane_ids:
            return {}
        rows = self._fetch_lane_obligations(lane_ids)
        grouped: dict[str, list[ObligationInfo]] = defaultdict(list)
        for row in rows:
            if not self._row_has_current_open_balance(row):
                continue
            lane_key = str(row.get("lane_id") or (lane_ids[0] if len(lane_ids) == 1 else "") or "")
            if not lane_key:
                continue
            grouped[lane_key].append(self._obligation_info(row))
        # Order each lane's obligations the same way the per-id query
        # did (days_overdue DESC, invoice_number ASC). The bulk SQL
        # already applies that ORDER BY, but ``defaultdict`` insertion
        # order preserves it per group on Athena's result set.
        return dict(grouped)

    def _fetch_lane_obligations(self, lane_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                li.lane_id,
                o.id,
                COALESCE(o.source_id, o.sage_sales_transaction_id, o.id) AS external_id,
                o.provider_type,
                COALESCE(
                    o.document_no,
                    o.reference,
                    o.second_reference,
                    o.invoice_number,
                    o.urn,
                    o.source_id,
                    o.id
                ) AS provider_ref,
                COALESCE(o.invoice_number, o.document_no, o.reference, o.source_id, o.id) AS invoice_number,
                o.document_gross_value AS original_amount,
                o.document_gross_value * COALESCE(o.exchange_rate, 1.0) AS original_amount_base,
                o.document_allocated_value AS allocated_amount,
                o.document_allocated_value * COALESCE(o.exchange_rate, 1.0) AS allocated_amount_base,
                o.amount_due,
                o.amount_due * COALESCE(o.exchange_rate, 1.0) AS amount_due_base,
                COALESCE(o.is_open, o.amount_due > 0, FALSE) AS obligation_is_open,
                COALESCE(
                    NULLIF(o.currency_code, ''),
                    NULLIF(o.currency, ''),
                    NULLIF(o.document_currency_code, ''),
                    NULLIF(o.base_currency, ''),
                    'GBP'
                ) AS currency,
                COALESCE(
                    NULLIF(o.base_currency, ''),
                    NULLIF(o.currency_code, ''),
                    NULLIF(o.currency, ''),
                    NULLIF(o.document_currency_code, ''),
                    'GBP'
                ) AS base_currency,
                o.exchange_rate AS document_to_base_rate,
                o.due_date,
                CASE
                    WHEN o.due_date IS NULL THEN 0
                    ELSE GREATEST(date_diff('day', CAST(o.due_date AS DATE), CURRENT_DATE), 0)
                END AS days_past_due,
                CASE
                    WHEN COALESCE(o.is_source_disputed, o.has_source_query_flag, FALSE) THEN 'disputed'
                    WHEN COALESCE(o.is_open, o.amount_due > 0) THEN 'open'
                    ELSE 'closed'
                END AS state,
                o.silver_version_id,
                o.document_no,
                COALESCE(
                    NULLIF(o.document_currency_code, ''),
                    NULLIF(o.currency_code, ''),
                    NULLIF(o.currency, ''),
                    'GBP'
                ) AS document_currency_code,
                (
                    COALESCE(o.is_open, o.amount_due > 0, FALSE)
                    AND COALESCE(o.amount_due, 0) > 0
                ) AS is_outstanding,
                (
                    COALESCE(o.is_open, o.amount_due > 0, FALSE)
                    AND COALESCE(o.amount_due, 0) > 0
                    AND o.due_date IS NOT NULL
                    AND date_diff('day', CAST(o.due_date AS DATE), CURRENT_DATE) > 0
                ) AS is_overdue,
                CASE
                    WHEN o.due_date IS NULL THEN 0
                    ELSE GREATEST(date_diff('day', CAST(o.due_date AS DATE), CURRENT_DATE), 0)
                END AS days_overdue,
                COALESCE(li.effective_grace_days, 0) AS effective_grace_days,
                COALESCE(
                    li.is_source_disputed,
                    o.is_source_disputed,
                    o.has_source_query_flag,
                    FALSE
                ) AS is_source_disputed,
                COALESCE(o.has_source_query_flag, li.is_source_disputed, FALSE) AS has_source_query_flag,
                COALESCE(li.source_query_raw, o.source_query_raw) AS source_query_raw,
                COALESCE(li.source_dispute_type, o.source_dispute_type) AS source_dispute_type,
                COALESCE(
                    li.source_dispute_observed_from,
                    o.source_dispute_observed_from
                ) AS source_dispute_observed_from
            FROM {_current_projection_in(self._source(COLLECTION_LANE_INVOICES_CURRENT), "li", id_column="lane_id")}
            JOIN {_current_projection(self._source(OBLIGATIONS_CURRENT), "o")}
              ON li.obligation_id = o.id
             AND li.tenant_id = o.tenant_id
            WHERE COALESCE(li.lane_invoice_status, 'open') = %s
              AND COALESCE(o.amount_due, 0) > 0
              AND COALESCE(o.is_open, o.amount_due > 0, FALSE) = TRUE
            ORDER BY li.lane_id, days_overdue DESC NULLS LAST, invoice_number
            """
        return self.reader.execute(
            sql,
            [self.tenant_id, tuple(lane_ids), self.tenant_id, "open"],
        )

    @staticmethod
    def _row_has_current_open_balance(row: dict[str, Any]) -> bool:
        return evidence.row_has_current_open_balance(row)

    def _load_lane_history(self, lane_id: str) -> list[dict[str, Any]]:
        rows = self._fetch_lane_history([lane_id])
        return self._format_lane_history_rows(rows[:25])

    def _load_lane_history_batch(self, lane_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return ``{lane_id: [history_event, ...]}`` for the requested lanes.

        Per-lane LIMIT 25 ordering is preserved by sorting in Python --
        Athena's window-based ``RANK ... PARTITION BY lane_id`` works
        but is more expensive than a single sorted scan + Python-side
        per-lane truncation for typical batch sizes.
        """
        if not lane_ids:
            return {}
        rows = self._fetch_lane_history(lane_ids)
        per_lane_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            lane_key = str(row.get("lane_id") or "")
            if not lane_key:
                continue
            per_lane_raw[lane_key].append(row)
        return {
            lane_key: self._format_lane_history_rows(group[:25])
            for lane_key, group in per_lane_raw.items()
        }

    def _fetch_lane_history(self, lane_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                h.lane_id,
                h.event_type,
                h.from_status,
                h.to_status,
                h.from_level,
                h.to_level,
                h.draft_id,
                h.touch_id,
                h.thread_id,
                h.event_payload_json AS detail_json,
                h.created_at,
                h.event_time,
                h.valid_from
            FROM {_current_projection_in(self._source(COLLECTION_LANE_HISTORY_CURRENT), "h", id_column="lane_id")}
            ORDER BY h.lane_id, COALESCE(h.event_time, h.created_at, h.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids)])

    @staticmethod
    def _format_lane_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Shape lane history rows for the case-context payload.

        Reverses to chronological order (oldest first), drops the
        ``lane_id`` join key, and parses the ``detail_json`` payload.
        """
        return evidence.format_lane_history_rows(rows)

    def _load_case_threads_batch(self, case_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return the active/current thread row keyed by collection_case_id."""
        if not case_ids:
            return {}
        try:
            rows = self._fetch_case_threads(case_ids)
        except Exception:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("collection_case_id") or "")
            if case_id and case_id not in result:
                result[case_id] = row
        return result

    def _fetch_case_threads(self, case_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                ct.collection_case_id,
                ct.collection_case_thread_id,
                ct.conversation_id,
                CAST(NULL AS VARCHAR) AS mailbox_email,
                CAST(NULL AS VARCHAR) AS latest_subject,
                ct.thread_status,
                ct.adopted_at AS first_message_at,
                COALESCE(ct.superseded_at, ct.adopted_at, ct.valid_from, ct.observed_at) AS last_message_at
            FROM {_current_projection_in(self._source(COLLECTION_CASE_THREADS_CURRENT), "ct", id_column="collection_case_id")}
            ORDER BY
                CASE WHEN ct.thread_status = 'active' THEN 0 ELSE 1 END,
                ct.last_message_at DESC NULLS LAST,
                ct.collection_case_thread_id
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(case_ids)])

    def _load_case_temporal_invoice_evidence_batch(
        self,
        case_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return message-time invoice evidence keyed by collection_case_id.

        These rows are context only. They must never widen the current
        candidate obligations selected from Silver Core obligations.
        """
        if not case_ids:
            return {}
        try:
            rows = self._fetch_case_temporal_invoice_evidence(case_ids)
        except Exception:
            return {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            case_id = str(row.get("collection_case_id") or "")
            message_id = str(row.get("mail_message_id") or "")
            invoice_key = str(row.get("invoice_ref_normalized") or row.get("invoice_number") or "")
            dedupe_key = (case_id, message_id, invoice_key)
            if not case_id or not message_id or not invoice_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            grouped[case_id].append(self._format_temporal_evidence_row(row))
        return dict(grouped)

    def _fetch_case_temporal_invoice_evidence(self, case_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                ev.collection_case_id,
                ev.collection_case_thread_id,
                ev.mail_message_id,
                ev.message_time,
                ev.invoice_ref_raw,
                ev.invoice_ref_normalized,
                ev.invoice_number,
                ev.obligation_id,
                ev.current_amount_due,
                ev.current_amount_due_base,
                ev.current_state,
                ev.current_state_reason,
                ev.as_of_amount_due,
                ev.as_of_amount_due_base,
                ev.as_of_state,
                ev.as_of_source,
                ev.as_of_confidence,
                ev.commitment_event_ids_json,
                ev.warnings_json
            FROM {_current_projection_in(self._source(COLLECTION_THREAD_MESSAGE_INVOICE_EVIDENCE_CURRENT), "ev", id_column="collection_case_id")}
            ORDER BY ev.collection_case_id,
                     ev.message_time DESC NULLS LAST,
                     ev.mail_message_id,
                     ev.invoice_ref_normalized
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(case_ids)])

    @staticmethod
    def _format_temporal_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
        return evidence.format_temporal_evidence_row(row)

    @staticmethod
    def _format_case_thread_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return evidence.format_case_thread_messages(rows)

    @staticmethod
    def _format_case_invoice_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return evidence.format_case_invoice_evidence(rows)

    @staticmethod
    def _commitments_from_lane_contexts(
        lane_contexts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return evidence.commitments_from_lane_contexts(lane_contexts)

    def _load_case_commitment_evidence_batch(
        self,
        case_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return explicit case commitment evidence when available.

        V1 keeps commitment selection from current lane/case scope. The
        temporal evidence table carries durable commitment event ids, so this
        helper exposes those ids without querying by ambiguous thread keys.
        """
        temporal = self._load_case_temporal_invoice_evidence_batch(case_ids)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str, str]] = set()
        for case_id, rows in temporal.items():
            for row in rows:
                event_ids = row.get("commitment_event_ids") or []
                if not event_ids:
                    continue
                key = (
                    case_id,
                    str(row.get("mail_message_id") or ""),
                    str(row.get("invoice_number") or row.get("invoice_ref_normalized") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                grouped[case_id].append(
                    {
                        "type": "commitment",
                        "invoice_number": row.get("invoice_number"),
                        "source_message_id": row.get("mail_message_id"),
                        "event_time": row.get("message_time"),
                        "status_at_event": row.get("as_of_state"),
                        "current_outcome": row.get("current_state"),
                        "blocks_chasing": str(row.get("current_state") or "").lower()
                        in {"promised", "remittance_pending", "payment_plan"},
                        "commitment_event_ids": event_ids,
                    }
                )
        return dict(grouped)

    def _load_actual_sent_scope_history(self, party_id: str) -> list[dict[str, Any]]:
        rows_by_party = self._load_actual_sent_scope_history_batch([party_id])
        return rows_by_party.get(str(party_id), [])

    def _load_actual_sent_scope_history_batch(
        self,
        party_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return post-send invoice-scope evidence keyed by party.

        Missing schema columns should not block draft generation during a
        rolling deployment; the history block simply stays empty until schema
        evolution and the ETL writer are both live.
        """
        if not party_ids:
            return {}
        try:
            rows = self._fetch_actual_sent_scope_history(party_ids)
        except Exception:
            return {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            party_key = str(row.get("party_id") or "")
            if not party_key:
                continue
            if len(grouped[party_key]) >= 10:
                continue
            grouped[party_key].append(self._format_actual_sent_scope_row(row))
        return dict(grouped)

    def _fetch_actual_sent_scope_history(self, party_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                d.party_id,
                a.sent_draft_analysis_event_id,
                a.application_content_hash,
                a.draft_id,
                a.touch_id,
                a.provider_message_id,
                d.lane_id,
                COALESCE(d.sent_at, a.event_time, a.valid_from) AS sent_at,
                a.invoice_refs_generated_json,
                a.invoice_refs_sent_json,
                a.invoice_refs_added_json,
                a.invoice_refs_removed_json,
                COALESCE(a.invoice_scope_changed, FALSE) AS invoice_scope_changed,
                a.edit_severity,
                COALESCE(a.payment_expectation_added, FALSE) AS payment_expectation_added,
                a.payment_expectation_kind,
                a.payment_expectation_date,
                a.payment_expectation_amount,
                a.review_reason_codes_json
            FROM {_current_projection_in(self._source(DRAFTS_CURRENT), "d", id_column="party_id")}
            JOIN {_current_projection(self._source(SENT_DRAFT_ANALYSIS_EVENTS_CURRENT), "a")}
              ON a.tenant_id = d.tenant_id
             AND a.draft_id = d.draft_id
            JOIN {_current_projection(self._source(DRAFT_PROVIDER_LIFECYCLE_EVENTS_CURRENT), "dle")}
              ON dle.tenant_id = d.tenant_id
             AND dle.draft_id = d.draft_id
             AND dle.event_type = 'sent_confirmed'
             AND dle.proof_type IN ('graph_sent_items_exact_oai', 'message_trace', 'purview_send_as')
            ORDER BY d.party_id, COALESCE(d.sent_at, a.event_time, a.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(
            sql, [self.tenant_id, tuple(party_ids), self.tenant_id, self.tenant_id]
        )

    @staticmethod
    def _format_actual_sent_scope_row(row: dict[str, Any]) -> dict[str, Any]:
        return evidence.format_actual_sent_scope_row(row)

    @staticmethod
    def _actual_sent_scope_version_ids(rows: list[dict[str, Any]]) -> list[str]:
        return evidence.actual_sent_scope_version_ids(rows)

    def _load_party_contacts(self, party_id: str) -> list[dict[str, Any]]:
        rows = self._fetch_party_contacts([party_id])
        return self._format_party_contact_rows(rows)

    def _load_party_contacts_batch(self, party_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return ``{party_id: [contact, ...]}`` for the requested parties."""
        if not party_ids:
            return {}
        rows = self._fetch_party_contacts(party_ids)
        per_party_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            party_key = str(row.get("party_id") or "")
            if not party_key:
                continue
            per_party_raw[party_key].append(row)
        return {
            party_key: self._format_party_contact_rows(group[:10])
            for party_key, group in per_party_raw.items()
        }

    def _fetch_party_contacts(self, party_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                c.party_id,
                c.id,
                COALESCE(c.contact_name, c.name, c.email_normalized, c.email, c.email_address) AS name,
                COALESCE(c.email_normalized, c.email, c.email_address) AS email,
                COALESCE(c.is_default_email, c.is_default_contact, c.is_default, FALSE) AS is_default,
                COALESCE(c.is_send_statement_to, FALSE) AS is_send_statement_to,
                COALESCE(c.is_preferred_send_statement_to, FALSE) AS is_preferred_send_statement_to,
                c.recipient_selection_source,
                c.is_active,
                c.email_valid,
                COALESCE(c.source_contact_key, c.source) AS source
            FROM {_current_projection_in(self._source(PARTY_CONTACTS_CURRENT), "c", id_column="party_id")}
            WHERE COALESCE(c.is_active, TRUE) = TRUE
              AND COALESCE(c.email_valid, TRUE) = TRUE
              AND COALESCE(c.email_normalized, c.email, c.email_address) IS NOT NULL
            ORDER BY
                c.party_id,
                COALESCE(c.is_preferred_send_statement_to, FALSE) DESC,
                COALESCE(c.is_send_statement_to, FALSE) DESC,
                COALESCE(c.is_default_email, c.is_default_contact, c.is_default, FALSE) DESC,
                c.silver_observed_at DESC NULLS LAST
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(party_ids)])

    @staticmethod
    def _format_party_contact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return evidence.format_party_contact_rows(rows)

    def _party_info(self, row: dict[str, Any]):
        return evidence.party_info(row)

    @staticmethod
    def _behavior_info(row: dict[str, Any]):
        return evidence.behavior_info(row)

    @staticmethod
    def _communication_info(row: dict[str, Any]):
        return evidence.communication_info(row)

    @staticmethod
    def _obligation_info(row: dict[str, Any]) -> ObligationInfo:
        return evidence.obligation_info(row)
