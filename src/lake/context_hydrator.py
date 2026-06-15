"""Hydrate AI case context from regional Silver tables."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Protocol

from src.api.models.requests import (
    BehaviorInfo,
    CaseContext,
    CommunicationInfo,
    ObligationInfo,
    PartyInfo,
)

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
SENT_DRAFT_ANALYSIS_EVENTS_CURRENT = "sent_draft_analysis_events_current"
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


def _json_value(value: Any, *, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _int_or_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return str(value)


class CaseContextHydrator:
    """Build existing GenerateDraftRequest context from regional Silver reads."""

    def __init__(self, tenant_id: str, reader: LakeReader) -> None:
        self.tenant_id = str(tenant_id)
        self.reader = reader

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
        return self._assemble_case_context(
            candidate=candidate,
            party=party,
            lane=lane,
            obligations=obligations,
            party_contacts=party_contacts,
            history=history,
            actual_sent_scope_history=actual_sent_scope_history,
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
        obligations = [
            obligation
            for lane_id in lane_ids
            for obligation in obligations_by_lane.get(lane_id, [])
        ]
        expected_ids = {str(value) for value in (candidate.obligation_ids or []) if str(value)}
        if not expected_ids:
            return obligations

        seen: set[str] = set()
        filtered: list[ObligationInfo] = []
        for obligation in obligations:
            obligation_id = str(getattr(obligation, "id", "") or "")
            if obligation_id not in expected_ids or obligation_id in seen:
                continue
            seen.add(obligation_id)
            filtered.append(obligation)
        return filtered

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
        contexts = [
            dict(context)
            for context in (candidate.lane_contexts or [])
            if isinstance(context, dict)
            and (context.get("lane_id") or context.get("collection_lane_id"))
        ]
        if not contexts:
            return [fallback_context]

        refs_by_obligation_id = {
            str(getattr(obligation, "id", "") or ""): str(
                getattr(obligation, "invoice_number", "") or ""
            )
            for obligation in obligations
            if getattr(obligation, "id", None)
        }
        for context in contexts:
            if not context.get("invoice_refs") and context.get("obligation_ids"):
                context["invoice_refs"] = [
                    refs_by_obligation_id.get(str(obligation_id), "")
                    for obligation_id in context.get("obligation_ids") or []
                    if refs_by_obligation_id.get(str(obligation_id), "")
                ]
            context["lane_id"] = str(context.get("lane_id") or context.get("collection_lane_id"))
            context.setdefault("role", "single" if len(contexts) == 1 else "guest")
        return contexts

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
            FROM {_current_projection_in(PARTIES_CURRENT, "p", id_column="id")}
            LEFT JOIN {_current_projection(PARTY_COLLECTION_STATE_CURRENT, "cs")}
              ON cs.party_id = p.id
             AND cs.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(PARTY_COMM_STATE_CURRENT, "comm")}
              ON comm.party_id = p.id
             AND comm.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(PARTY_BEHAVIOR_PROFILE_CURRENT, "bp")}
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
                FROM {COLLECTION_LANES_CURRENT}
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
            FROM {_current_projection_in(COLLECTION_LANE_INVOICES_CURRENT, "li", id_column="lane_id")}
            JOIN {_current_projection(OBLIGATIONS_CURRENT, "o")}
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
        try:
            amount_due = float(row.get("amount_due") or 0)
        except (TypeError, ValueError):
            return False
        obligation_is_open = (
            bool(row.get("obligation_is_open"))
            if row.get("obligation_is_open") is not None
            else amount_due > 0
        )
        return amount_due > 0 and obligation_is_open

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
            FROM {_current_projection_in(COLLECTION_LANE_HISTORY_CURRENT, "h", id_column="lane_id")}
            ORDER BY h.lane_id, COALESCE(h.event_time, h.created_at, h.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids)])

    @staticmethod
    def _format_lane_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Shape lane history rows for the case-context payload.

        Reverses to chronological order (oldest first), drops the
        ``lane_id`` join key, and parses the ``detail_json`` payload.
        """
        return [
            {
                "event_type": row.get("event_type"),
                "from_status": row.get("from_status"),
                "to_status": row.get("to_status"),
                "from_level": row.get("from_level"),
                "to_level": row.get("to_level"),
                "draft_id": row.get("draft_id"),
                "touch_id": row.get("touch_id"),
                "thread_id": row.get("thread_id"),
                "detail": _json_value(row.get("detail_json"), fallback={}),
                "created_at": row.get("created_at"),
            }
            for row in reversed(rows)
        ]

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
            FROM {_current_projection_in(DRAFTS_CURRENT, "d", id_column="party_id")}
            JOIN {_current_projection(SENT_DRAFT_ANALYSIS_EVENTS_CURRENT, "a")}
              ON a.tenant_id = d.tenant_id
             AND a.draft_id = d.draft_id
            ORDER BY d.party_id, COALESCE(d.sent_at, a.event_time, a.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(party_ids), self.tenant_id])

    @staticmethod
    def _format_actual_sent_scope_row(row: dict[str, Any]) -> dict[str, Any]:
        def _json_list(value: Any) -> list[str]:
            parsed = _json_value(value, fallback=[])
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
            return []

        return {
            "sent_draft_analysis_event_id": str(row.get("sent_draft_analysis_event_id"))
            if row.get("sent_draft_analysis_event_id")
            else None,
            "application_content_hash": row.get("application_content_hash"),
            "draft_id": str(row.get("draft_id")) if row.get("draft_id") else None,
            "touch_id": str(row.get("touch_id")) if row.get("touch_id") else None,
            "provider_message_id": row.get("provider_message_id"),
            "lane_id": str(row.get("lane_id")) if row.get("lane_id") else None,
            "sent_at": row.get("sent_at"),
            "invoice_refs_generated": _json_list(row.get("invoice_refs_generated_json")),
            "invoice_refs_sent": _json_list(row.get("invoice_refs_sent_json")),
            "invoice_refs_added": _json_list(row.get("invoice_refs_added_json")),
            "invoice_refs_removed": _json_list(row.get("invoice_refs_removed_json")),
            "invoice_scope_changed": bool(row.get("invoice_scope_changed")),
            "edit_severity": row.get("edit_severity"),
            "payment_expectation_added": bool(row.get("payment_expectation_added")),
            "payment_expectation_kind": row.get("payment_expectation_kind"),
            "payment_expectation_date": _date_string(row.get("payment_expectation_date")),
            "payment_expectation_amount": row.get("payment_expectation_amount"),
            "review_reason_codes": _json_list(row.get("review_reason_codes_json")),
        }

    @staticmethod
    def _actual_sent_scope_version_ids(rows: list[dict[str, Any]]) -> list[str]:
        version_ids: list[str] = []
        for row in rows:
            event_id = row.get("sent_draft_analysis_event_id")
            content_hash = row.get("application_content_hash")
            if event_id:
                version_ids.append(f"sent_draft_analysis_event:{event_id}")
            if content_hash:
                version_ids.append(f"sent_draft_analysis_hash:{content_hash}")
        return version_ids

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
            FROM {_current_projection_in(PARTY_CONTACTS_CURRENT, "c", id_column="party_id")}
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
        return [
            {
                "party_contact_id": str(row.get("id")) if row.get("id") else None,
                "name": row.get("name"),
                "email": row.get("email"),
                "is_default": bool(row.get("is_default")),
                "is_send_statement_to": bool(row.get("is_send_statement_to")),
                "is_preferred_send_statement_to": bool(row.get("is_preferred_send_statement_to")),
                "recipient_selection_source": row.get("recipient_selection_source"),
                "source": row.get("source"),
            }
            for row in rows
            if row.get("email")
        ]

    def _party_info(self, row: dict[str, Any]) -> PartyInfo:
        provider_type = str(row.get("provider_type") or row.get("source") or "").strip()
        return PartyInfo(
            party_id=str(row["id"]),
            external_id=str(row.get("external_id") or row["id"]),
            provider_type=provider_type,
            customer_code=str(row.get("customer_code") or row.get("external_id") or row["id"]),
            name=str(row.get("name") or row.get("customer_code") or row["id"]),
            country_code=row.get("country_code"),
            currency=row.get("currency") or "GBP",
            base_currency=row.get("base_currency") or row.get("currency") or "GBP",
            credit_limit=row.get("credit_limit"),
            on_hold=bool(row.get("on_hold")),
            relationship_tier=row.get("relationship_tier") or "standard",
            tone_override=row.get("tone_override"),
            grace_days_override=row.get("grace_days_override"),
            touch_cap_override=row.get("touch_cap_override"),
            do_not_contact_until=_date_string(row.get("do_not_contact_until")),
            monthly_touch_count=int(row.get("monthly_touch_count") or 0),
            is_verified=bool(row.get("is_verified", True)),
            source=provider_type,
            customer_type=row.get("customer_type"),
            size_bucket=row.get("size_bucket"),
        )

    @staticmethod
    def _behavior_info(row: dict[str, Any]) -> BehaviorInfo:
        return BehaviorInfo(
            lifetime_value=row.get("lifetime_value"),
            total_collected=row.get("total_collected"),
            avg_days_to_pay=row.get("avg_days_to_pay"),
            on_time_rate=row.get("on_time_rate"),
            partial_payment_rate=row.get("partial_payment_rate"),
            behaviour_profile=_json_value(row.get("behaviour_profile"), fallback=None),
            behaviour_segment=row.get("behaviour_segment") or row.get("segment"),
        )

    @staticmethod
    def _communication_info(row: dict[str, Any]) -> CommunicationInfo:
        return CommunicationInfo(
            touch_count=int(row.get("touch_count") or 0),
            last_touch_at=row.get("last_touch_at"),
            last_touch_channel=row.get("last_touch_channel"),
            last_sender_level=row.get("last_sender_level"),
            last_tone_used=row.get("last_tone_used"),
            last_response_at=row.get("last_response_at"),
            last_response_type=row.get("last_response_type"),
        )

    @staticmethod
    def _obligation_info(row: dict[str, Any]) -> ObligationInfo:
        amount_due = float(row.get("amount_due") or 0)
        days_overdue = int(row.get("days_overdue") or row.get("days_past_due") or 0)
        is_source_disputed = bool(row.get("is_source_disputed")) or bool(
            row.get("source_query_raw")
        )
        obligation_is_open = (
            bool(row.get("obligation_is_open"))
            if row.get("obligation_is_open") is not None
            else amount_due > 0
        )
        has_current_balance = amount_due > 0 and obligation_is_open
        is_outstanding = has_current_balance
        is_overdue = has_current_balance and (
            bool(row.get("is_overdue")) if row.get("is_overdue") is not None else days_overdue > 0
        )
        is_chase_eligible = is_outstanding and is_overdue and not is_source_disputed
        return ObligationInfo(
            id=str(row["id"]),
            external_id=str(row.get("external_id") or row["id"]),
            provider_type=str(row["provider_type"]),
            provider_ref=row.get("provider_ref"),
            invoice_number=str(row.get("invoice_number") or row.get("external_id") or row["id"]),
            original_amount=float(row.get("original_amount") or 0),
            original_amount_base=row.get("original_amount_base"),
            allocated_amount=row.get("allocated_amount"),
            allocated_amount_base=row.get("allocated_amount_base"),
            amount_due=amount_due,
            amount_due_base=row.get("amount_due_base"),
            currency=row.get("currency"),
            base_currency=row.get("base_currency"),
            document_to_base_rate=row.get("document_to_base_rate"),
            due_date=_date_string(row.get("due_date")),
            days_past_due=int(row.get("days_past_due") or days_overdue),
            state=row.get("state") or "open",
            silver_version_id=row.get("silver_version_id"),
            document_no=row.get("document_no"),
            document_currency_code=row.get("document_currency_code") or row.get("currency"),
            is_outstanding=is_outstanding,
            is_overdue=is_overdue,
            days_overdue=days_overdue,
            effective_grace_days=int(row.get("effective_grace_days") or 0),
            is_sendable=is_chase_eligible,
            is_chase_eligible=is_chase_eligible,
            source_query_raw=row.get("source_query_raw"),
            has_source_query_flag=bool(row.get("has_source_query_flag")),
            is_source_disputed=is_source_disputed,
            source_dispute_type=row.get("source_dispute_type"),
            source_dispute_observed_from=row.get("source_dispute_observed_from"),
        )
