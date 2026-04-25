"""Hydrate AI case context from regional Silver tables."""

from __future__ import annotations

import json
from datetime import date, datetime
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


def _dedup_cte(table: str, alias: str, *, partition_by: str = "id") -> str:
    return (
        f"(SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_by} "
        f"ORDER BY updated_at DESC NULLS LAST) AS _rn "
        f"FROM {table} WHERE tenant_id = %s) {alias}"
    )


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
            schema_version=2,
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
            grace_days=int(party.get("grace_days_override") or 14),
            do_not_contact_until=_date_string(party.get("do_not_contact_until")),
            monthly_touch_count=int(party.get("monthly_touch_count") or 0),
            relationship_tier=party.get("relationship_tier") or "standard",
            unsubscribe_requested=bool(party.get("unsubscribe_requested")),
            collection_lane_id=str(lane["id"]),
            lane=lane_context,
            lane_history=self._load_lane_history(candidate.lane_id),
            lane_mail_mode="single_lane",
            sendable_obligation_ids=[obligation.id for obligation in obligations],
            lane_broken_promises_count=int(party.get("broken_promises_count") or 0),
            lane_last_tone_used=party.get("last_tone_used"),
            lane_contexts=[sparse_lane_context],
            mode="single_lane",
        )

    def _load_party(self, party_id: str) -> dict[str, Any]:
        row = self.reader.execute_one(
            f"""
            SELECT p.*
            FROM {_dedup_cte("parties", "p")}
            WHERE p._rn = 1
              AND p.id = %s
            """,
            [self.tenant_id, party_id],
        )
        if row is None:
            raise ContextHydrationError(f"Party not found in regional Silver: {party_id}")
        return row

    def _load_lane(self, lane_id: str) -> dict[str, Any]:
        row = self.reader.execute_one(
            f"""
            SELECT lane.*
            FROM {_dedup_cte("collection_lanes", "lane")}
            WHERE lane._rn = 1
              AND lane.id = %s
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
                o.external_id,
                o.provider_type,
                o.provider_ref,
                o.invoice_number,
                o.original_amount,
                o.original_amount_base,
                o.allocated_amount,
                o.allocated_amount_base,
                o.amount_due,
                o.amount_due_base,
                o.currency,
                o.base_currency,
                o.document_to_base_rate,
                o.due_date,
                o.days_past_due,
                o.state
            FROM {_dedup_cte("collection_lane_invoices", "li")}
            JOIN {_dedup_cte("obligations", "o")}
              ON o._rn = 1
             AND li.obligation_id = o.id
             AND li.tenant_id = o.tenant_id
            WHERE li._rn = 1
              AND li.collection_lane_id = %s
              AND li.status = %s
            ORDER BY o.days_past_due DESC NULLS LAST, o.invoice_number
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
                h.detail_json,
                h.created_at
            FROM {_dedup_cte("collection_lane_history", "h")}
            WHERE h._rn = 1
              AND h.collection_lane_id = %s
            ORDER BY h.created_at DESC NULLS LAST
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
            amount_due=float(row.get("amount_due") or 0),
            amount_due_base=row.get("amount_due_base"),
            currency=row.get("currency"),
            base_currency=row.get("base_currency"),
            document_to_base_rate=row.get("document_to_base_rate"),
            due_date=_date_string(row.get("due_date")),
            days_past_due=int(row.get("days_past_due") or 0),
            state=row.get("state") or "open",
        )
