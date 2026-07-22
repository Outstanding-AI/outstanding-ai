"""Tenant-scoped current-projection reads for AI draft context hydration.

This module deliberately owns only read concerns: identifier-safe source
selection, tenant predicates, batch grouping, and graceful handling of
optional rolling-deployment projections.  It never constructs ``CaseContext``
or decides an invoice/lane scope; those are evidence/projection concerns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from src.api.models.requests import ObligationInfo

from . import context_evidence as evidence


class ContextHydrationError(RuntimeError):
    """Raised when required current-projection context is unavailable."""


class LakeReader(Protocol):
    """Minimal Athena-compatible reader required by context hydration."""

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


def current_projection(projection: str, alias: str) -> str:
    """Render a tenant-scoped current projection subselect."""

    return f"(SELECT * FROM {projection} WHERE tenant_id = %s) {alias}"


def current_projection_in(projection: str, alias: str, *, id_column: str) -> str:
    """Render a tenant- and ID-scoped current projection subselect.

    ``IN %s`` is rendered by the shared Athena dialect from a list/tuple
    parameter.  Keeping it inside the projection limits reads before joins.
    """

    return f"(SELECT * FROM {projection} WHERE tenant_id = %s AND {id_column} IN %s) {alias}"


class ContextReadRepository:
    """Read the normalized rows needed to assemble one or many case contexts."""

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

    def source(self, canonical_view: str) -> str:
        """Return a backend-issued identifier-safe serving source or canonical view."""

        source = str(self.current_source_map.get(canonical_view) or canonical_view)
        return source if source.replace("_", "").isalnum() else canonical_view

    def load_party(self, party_id: str) -> dict[str, Any]:
        rows = self._fetch_parties([party_id])
        if not rows:
            raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
        return rows[0]

    def load_parties_batch(self, party_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return the requested party rows, keyed by canonical party ID."""

        if not party_ids:
            return {}
        rows = self._fetch_parties(party_ids)
        return {str(row["id"]): row for row in rows if row.get("id") is not None}

    def _fetch_parties(self, party_ids: list[str]) -> list[dict[str, Any]]:
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
            FROM {current_projection_in(self.source(PARTIES_CURRENT), "p", id_column="id")}
            LEFT JOIN {current_projection(self.source(PARTY_COLLECTION_STATE_CURRENT), "cs")}
              ON cs.party_id = p.id
             AND cs.tenant_id = p.tenant_id
            LEFT JOIN {current_projection(self.source(PARTY_COMM_STATE_CURRENT), "comm")}
              ON comm.party_id = p.id
             AND comm.tenant_id = p.tenant_id
            LEFT JOIN {current_projection(self.source(PARTY_BEHAVIOR_PROFILE_CURRENT), "bp")}
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

    def load_lane(self, lane_id: str) -> dict[str, Any]:
        rows = self._fetch_lanes([lane_id])
        if not rows:
            raise ContextHydrationError(f"Collection lane not found in regional Silver: {lane_id}")
        return rows[0]

    def load_lanes_batch(self, lane_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not lane_ids:
            return {}
        rows = self._fetch_lanes(lane_ids)
        return {
            str(row.get("lane_id") or row.get("id")): row
            for row in rows
            if (row.get("lane_id") or row.get("id")) is not None
        }

    def _fetch_lanes(self, lane_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT lane.*
            FROM (
                SELECT *
                FROM {self.source(COLLECTION_LANES_CURRENT)}
                WHERE tenant_id = %s
                  AND COALESCE(lane_id, id) IN %s
            ) lane
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids)])

    def load_lane_obligations(self, lane_id: str) -> list[ObligationInfo]:
        return [evidence.obligation_info(row) for row in self._fetch_lane_obligations([lane_id])]

    def load_lane_obligations_batch(self, lane_ids: list[str]) -> dict[str, list[ObligationInfo]]:
        if not lane_ids:
            return {}
        grouped: dict[str, list[ObligationInfo]] = defaultdict(list)
        for row in self._fetch_lane_obligations(lane_ids):
            if not evidence.row_has_current_open_balance(row):
                continue
            lane_key = str(row.get("lane_id") or (lane_ids[0] if len(lane_ids) == 1 else "") or "")
            if lane_key:
                grouped[lane_key].append(evidence.obligation_info(row))
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
            FROM {current_projection_in(self.source(COLLECTION_LANE_INVOICES_CURRENT), "li", id_column="lane_id")}
            JOIN {current_projection(self.source(OBLIGATIONS_CURRENT), "o")}
              ON li.obligation_id = o.id
             AND li.tenant_id = o.tenant_id
            WHERE COALESCE(li.lane_invoice_status, 'open') = %s
              AND COALESCE(o.amount_due, 0) > 0
              AND COALESCE(o.is_open, o.amount_due > 0, FALSE) = TRUE
            ORDER BY li.lane_id, days_overdue DESC NULLS LAST, invoice_number
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids), self.tenant_id, "open"])

    def load_lane_history(self, lane_id: str) -> list[dict[str, Any]]:
        return evidence.format_lane_history_rows(self._fetch_lane_history([lane_id])[:25])

    def load_lane_history_batch(self, lane_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not lane_ids:
            return {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._fetch_lane_history(lane_ids):
            lane_key = str(row.get("lane_id") or "")
            if lane_key:
                grouped[lane_key].append(row)
        return {
            lane_key: evidence.format_lane_history_rows(rows[:25])
            for lane_key, rows in grouped.items()
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
            FROM {current_projection_in(self.source(COLLECTION_LANE_HISTORY_CURRENT), "h", id_column="lane_id")}
            ORDER BY h.lane_id, COALESCE(h.event_time, h.created_at, h.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(lane_ids)])

    def load_case_threads_batch(self, case_ids: list[str]) -> dict[str, dict[str, Any]]:
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
            FROM {current_projection_in(self.source(COLLECTION_CASE_THREADS_CURRENT), "ct", id_column="collection_case_id")}
            ORDER BY
                CASE WHEN ct.thread_status = 'active' THEN 0 ELSE 1 END,
                ct.last_message_at DESC NULLS LAST,
                ct.collection_case_thread_id
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(case_ids)])

    def load_case_temporal_invoice_evidence_batch(
        self,
        case_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Read message-time evidence without widening current candidate scope."""

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
            grouped[case_id].append(evidence.format_temporal_evidence_row(row))
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
            FROM {current_projection_in(self.source(COLLECTION_THREAD_MESSAGE_INVOICE_EVIDENCE_CURRENT), "ev", id_column="collection_case_id")}
            ORDER BY ev.collection_case_id,
                     ev.message_time DESC NULLS LAST,
                     ev.mail_message_id,
                     ev.invoice_ref_normalized
            """
        return self.reader.execute(sql, [self.tenant_id, tuple(case_ids)])

    def load_case_commitment_evidence_batch(
        self,
        case_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Derive durable commitment references from temporal current evidence."""

        temporal = self.load_case_temporal_invoice_evidence_batch(case_ids)
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

    def load_actual_sent_scope_history(self, party_id: str) -> list[dict[str, Any]]:
        return self.load_actual_sent_scope_history_batch([party_id]).get(str(party_id), [])

    def load_actual_sent_scope_history_batch(
        self,
        party_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return bounded post-send invoice-scope evidence per party.

        This is intentionally optional during a rolling additive schema release;
        a missing projection cannot block a draft but returns no historical scope.
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
            if party_key and len(grouped[party_key]) < 10:
                grouped[party_key].append(evidence.format_actual_sent_scope_row(row))
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
            FROM {current_projection_in(self.source(DRAFTS_CURRENT), "d", id_column="party_id")}
            JOIN {current_projection(self.source(SENT_DRAFT_ANALYSIS_EVENTS_CURRENT), "a")}
              ON a.tenant_id = d.tenant_id
             AND a.draft_id = d.draft_id
            JOIN {current_projection(self.source(DRAFT_PROVIDER_LIFECYCLE_EVENTS_CURRENT), "dle")}
              ON dle.tenant_id = d.tenant_id
             AND dle.draft_id = d.draft_id
             AND dle.event_type = 'sent_confirmed'
             AND dle.proof_type IN (
                 'graph_sent_items_exact_oai',
                 'graph_create_reply_sent_items_match',
                 'captured_sent_copy_exact_oai',
                 'message_trace',
                 'purview_send_as'
             )
            ORDER BY d.party_id, COALESCE(d.sent_at, a.event_time, a.valid_from) DESC NULLS LAST
            """
        return self.reader.execute(
            sql, [self.tenant_id, tuple(party_ids), self.tenant_id, self.tenant_id]
        )

    def load_party_contacts(self, party_id: str) -> list[dict[str, Any]]:
        return evidence.format_party_contact_rows(self._fetch_party_contacts([party_id]))

    def load_party_contacts_batch(self, party_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not party_ids:
            return {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._fetch_party_contacts(party_ids):
            party_key = str(row.get("party_id") or "")
            if party_key:
                grouped[party_key].append(row)
        return {
            party_key: evidence.format_party_contact_rows(rows[:10])
            for party_key, rows in grouped.items()
        }

    def _fetch_party_contacts(self, party_ids: list[str]) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                c.party_id,
                c.id,
                NULLIF(TRIM(COALESCE(c.contact_name, c.name)), '') AS name,
                COALESCE(c.email_normalized, c.email, c.email_address) AS email,
                COALESCE(c.is_default_email, c.is_default_contact, c.is_default, FALSE) AS is_default,
                COALESCE(c.is_send_statement_to, FALSE) AS is_send_statement_to,
                COALESCE(c.is_preferred_send_statement_to, FALSE) AS is_preferred_send_statement_to,
                c.recipient_selection_source,
                c.is_active,
                c.email_valid,
                COALESCE(c.source_contact_key, c.source) AS source
            FROM {current_projection_in(self.source(PARTY_CONTACTS_CURRENT), "c", id_column="party_id")}
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


__all__ = [
    "ContextHydrationError",
    "ContextReadRepository",
    "LakeReader",
]
