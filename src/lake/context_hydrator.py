"""Hydrate AI case context from regional Silver tables."""

from __future__ import annotations

import json
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
PARTY_COLLECTION_STATE_CURRENT = "party_collection_state_events_current"
PARTY_COMM_STATE_CURRENT = "party_comm_state_events_current"
PARTY_BEHAVIOR_PROFILE_CURRENT = "party_behavior_profile_versions_current"


def _current_projection(projection: str, alias: str) -> str:
    return f"(SELECT * FROM {projection} WHERE tenant_id = %s) {alias}"


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
        party = self._load_party(candidate.party_id)
        lane = self._load_lane(candidate.lane_id)
        obligations = self._load_lane_obligations(candidate.lane_id)
        lane_invoice_refs = [str(row.invoice_number) for row in obligations if row.invoice_number]
        base_currency = party.get("base_currency") or party.get("currency") or "GBP"
        party_contacts = self._load_party_contacts(candidate.party_id)
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
        input_silver_version_ids = [
            str(value)
            for value in (
                [party.get("silver_version_id"), lane.get("application_version_id")]
                + [getattr(obligation, "silver_version_id", None) for obligation in obligations]
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
            lane_history=self._load_lane_history(candidate.lane_id),
            lane_mail_mode="single_lane",
            sendable_obligation_ids=sendable_obligation_ids,
            lane_broken_promises_count=int(party.get("broken_promises_count") or 0),
            lane_last_tone_used=party.get("last_tone_used"),
            lane_contexts=[sparse_lane_context],
            mode="single_lane",
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

    def _load_party(self, party_id: str) -> dict[str, Any]:
        row = self.reader.execute_one(
            f"""
            SELECT
                p.id,
                p.provider_type,
                COALESCE(p.source_id, p.sage_customer_reference, p.customer_code, p.id) AS external_id,
                COALESCE(p.customer_code, p.sage_customer_reference, p.source_id, p.id) AS customer_code,
                COALESCE(p.customer_name, p.name, p.customer_code, p.id) AS name,
                CAST(NULL AS VARCHAR) AS country_code,
                COALESCE(p.customer_currency_code, p.currency_code, p.currency, 'GBP') AS currency,
                COALESCE(
                    p.base_currency,
                    p.customer_currency_code,
                    p.currency_code,
                    p.currency,
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
            FROM {_current_projection(PARTIES_CURRENT, "p")}
            LEFT JOIN {_current_projection(PARTY_COLLECTION_STATE_CURRENT, "cs")}
              ON cs.party_id = p.id
             AND cs.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(PARTY_COMM_STATE_CURRENT, "comm")}
              ON comm.party_id = p.id
             AND comm.tenant_id = p.tenant_id
            LEFT JOIN {_current_projection(PARTY_BEHAVIOR_PROFILE_CURRENT, "bp")}
              ON bp.party_id = p.id
             AND bp.tenant_id = p.tenant_id
            WHERE p.id = %s
            """,
            [
                self.tenant_id,
                self.tenant_id,
                self.tenant_id,
                self.tenant_id,
                party_id,
            ],
        )
        if row is None:
            raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
        return row

    def _load_lane(self, lane_id: str) -> dict[str, Any]:
        row = self.reader.execute_one(
            f"""
            SELECT lane.*
            FROM {_current_projection(COLLECTION_LANES_CURRENT, "lane")}
            WHERE COALESCE(lane.lane_id, lane.id) = %s
            """,
            [self.tenant_id, lane_id],
        )
        if row is None:
            raise ContextHydrationError(f"Collection lane not found in regional Silver: {lane_id}")
        return row

    def _load_lane_obligations(self, lane_id: str) -> list[ObligationInfo]:
        rows = self.reader.execute(
            f"""
            SELECT
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
                o.currency_code AS currency,
                COALESCE(o.currency_code, 'GBP') AS base_currency,
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
                o.currency_code AS document_currency_code,
                COALESCE(li.is_outstanding, o.amount_due > 0) AS is_outstanding,
                COALESCE(
                    li.is_overdue,
                    CASE
                        WHEN o.due_date IS NULL THEN FALSE
                        ELSE date_diff('day', CAST(o.due_date AS DATE), CURRENT_DATE) > 0
                    END
                ) AS is_overdue,
                COALESCE(
                    li.days_overdue,
                    CASE
                        WHEN o.due_date IS NULL THEN 0
                        ELSE GREATEST(date_diff('day', CAST(o.due_date AS DATE), CURRENT_DATE), 0)
                    END,
                    0
                ) AS days_overdue,
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
            FROM {_current_projection(COLLECTION_LANE_INVOICES_CURRENT, "li")}
            JOIN {_current_projection(OBLIGATIONS_CURRENT, "o")}
              ON li.obligation_id = o.id
             AND li.tenant_id = o.tenant_id
            WHERE li.lane_id = %s
              AND COALESCE(li.lane_invoice_status, 'open') = %s
            ORDER BY days_overdue DESC NULLS LAST, invoice_number
            """,
            [self.tenant_id, self.tenant_id, lane_id, "open"],
        )
        return [self._obligation_info(row) for row in rows]

    def _load_lane_history(self, lane_id: str) -> list[dict[str, Any]]:
        rows = self.reader.execute(
            f"""
            SELECT
                h.event_type,
                h.from_status,
                h.to_status,
                h.from_level,
                h.to_level,
                h.draft_id,
                h.touch_id,
                h.thread_id,
                h.event_payload_json AS detail_json,
                h.created_at
            FROM {_current_projection(COLLECTION_LANE_HISTORY_CURRENT, "h")}
            WHERE h.lane_id = %s
            ORDER BY COALESCE(h.event_time, h.created_at, h.valid_from) DESC NULLS LAST
            LIMIT 25
            """,
            [self.tenant_id, lane_id],
        )
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

    def _load_party_contacts(self, party_id: str) -> list[dict[str, Any]]:
        rows = self.reader.execute(
            f"""
            SELECT
                c.id,
                COALESCE(c.contact_name, c.name, c.email_normalized, c.email, c.email_address) AS name,
                COALESCE(c.email_normalized, c.email, c.email_address) AS email,
                COALESCE(c.is_default_email, c.is_default_contact, c.is_default, FALSE) AS is_default,
                c.is_active,
                c.email_valid,
                COALESCE(c.source_contact_key, c.source) AS source
            FROM {_current_projection(PARTY_CONTACTS_CURRENT, "c")}
            WHERE c.party_id = %s
              AND COALESCE(c.is_active, TRUE) = TRUE
              AND COALESCE(c.email_valid, TRUE) = TRUE
              AND COALESCE(c.email_normalized, c.email, c.email_address) IS NOT NULL
            ORDER BY
                COALESCE(c.is_default_email, c.is_default_contact, c.is_default, FALSE) DESC,
                c.silver_observed_at DESC NULLS LAST
            LIMIT 10
            """,
            [self.tenant_id, party_id],
        )
        return [
            {
                "party_contact_id": str(row.get("id")) if row.get("id") else None,
                "name": row.get("name"),
                "email": row.get("email"),
                "is_default": bool(row.get("is_default")),
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
        is_outstanding = (
            bool(row.get("is_outstanding"))
            if row.get("is_outstanding") is not None
            else amount_due > 0
        )
        is_overdue = (
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
