"""Pure normalization for the AI draft-context contract.

This module deliberately has no reader, SQL, HTTP, or provider dependency.
It transforms already tenant-scoped Silver rows into the stable evidence
shapes consumed by ``CaseContext``.  Keeping this boundary pure lets the
hydrator coordinate I/O without also owning every business representation
rule.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from src.api.models.requests import (
    BehaviorInfo,
    CommunicationInfo,
    ObligationInfo,
    PartyInfo,
)

from .models import DraftCandidate

HELD_COMMITMENT_REASON_TOKENS = frozenset(
    {
        "promised",
        "promise",
        "remittance",
        "remittance_pending",
        "payment_verification",
        "payment_claim",
        "payment_plan",
    }
)
BROKEN_COMMITMENT_REASON_TOKENS = frozenset(
    {
        "broken_promise",
        "promise_broken",
        "broken_remittance",
        "remittance_not_found",
        "payment_verification_not_found",
        "not_found_remittance",
    }
)


def json_value(value: Any, *, fallback: Any) -> Any:
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


def int_or_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return str(value)


def candidate_obligations(
    candidate: DraftCandidate,
    obligations_by_lane: dict[str, list[ObligationInfo]],
    lane_ids: list[str],
) -> list[ObligationInfo]:
    """Select the candidate's exact invoice scope from canonical lane rows."""

    obligations = [
        obligation for lane_id in lane_ids for obligation in obligations_by_lane.get(lane_id, [])
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


def candidate_lane_contexts(
    *,
    candidate: DraftCandidate,
    fallback_context: dict[str, Any],
    obligations: list[ObligationInfo],
) -> list[dict[str, Any]]:
    """Normalize multi-lane candidate scope without inferring ownership."""

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


def row_has_current_open_balance(row: dict[str, Any]) -> bool:
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


def format_lane_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return chronology-first lane events with the payload decoded."""

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
            "detail": json_value(row.get("detail_json"), fallback={}),
            "created_at": row.get("created_at"),
        }
        for row in reversed(rows)
    ]


def format_temporal_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection_case_thread_id": str(row.get("collection_case_thread_id"))
        if row.get("collection_case_thread_id")
        else None,
        "mail_message_id": str(row.get("mail_message_id")) if row.get("mail_message_id") else None,
        "message_time": date_string(row.get("message_time")),
        "invoice_ref_raw": row.get("invoice_ref_raw"),
        "invoice_ref_normalized": row.get("invoice_ref_normalized"),
        "invoice_number": row.get("invoice_number"),
        "obligation_id": str(row.get("obligation_id")) if row.get("obligation_id") else None,
        "current_amount_due": row.get("current_amount_due"),
        "current_amount_due_base": row.get("current_amount_due_base"),
        "current_state": row.get("current_state"),
        "current_state_reason": row.get("current_state_reason"),
        "as_of_amount_due": row.get("as_of_amount_due"),
        "as_of_amount_due_base": row.get("as_of_amount_due_base"),
        "as_of_state": row.get("as_of_state"),
        "as_of_source": row.get("as_of_source"),
        "as_of_confidence": row.get("as_of_confidence"),
        "commitment_event_ids": json_value(row.get("commitment_event_ids_json"), fallback=[]),
        "warnings": json_value(row.get("warnings_json"), fallback=[]),
    }


def format_case_thread_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: dict[str, dict[str, Any]] = {}
    for row in rows:
        message_id = str(row.get("mail_message_id") or "")
        if not message_id:
            continue
        message = messages.setdefault(
            message_id,
            {
                "id": message_id,
                "message_time": row.get("message_time"),
                "invoice_states": [],
            },
        )
        message["invoice_states"].append(
            {
                "mail_message_id": message_id,
                "invoice_ref_raw": row.get("invoice_ref_raw"),
                "invoice_number": row.get("invoice_number"),
                "obligation_id": row.get("obligation_id"),
                "as_of_amount_due": row.get("as_of_amount_due"),
                "as_of_state": row.get("as_of_state"),
                "as_of_source": row.get("as_of_source"),
                "as_of_confidence": row.get("as_of_confidence"),
                "current_amount_due": row.get("current_amount_due"),
                "current_state": row.get("current_state"),
                "warning": "; ".join(str(item) for item in (row.get("warnings") or [])[:3])
                if isinstance(row.get("warnings"), list)
                else None,
            }
        )
    return list(messages.values())[:8]


def format_case_invoice_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invoices: dict[str, dict[str, Any]] = {}
    for row in rows:
        invoice_key = str(row.get("invoice_number") or row.get("invoice_ref_normalized") or "")
        if not invoice_key:
            continue
        invoice = invoices.setdefault(
            invoice_key,
            {
                "invoice_number": row.get("invoice_number"),
                "obligation_id": row.get("obligation_id"),
                "message_states": [],
                "current_amount_due": row.get("current_amount_due"),
                "current_amount_due_base": row.get("current_amount_due_base"),
                "current_state": row.get("current_state"),
                "current_state_reason": row.get("current_state_reason"),
                "will_be_chased_if_adopted": (
                    str(row.get("current_state") or "").lower() == "open"
                    and float(row.get("current_amount_due") or 0) > 0
                ),
            },
        )
        invoice["message_states"].append(
            {
                "mail_message_id": row.get("mail_message_id"),
                "message_time": row.get("message_time"),
                "as_of_amount_due": row.get("as_of_amount_due"),
                "as_of_state": row.get("as_of_state"),
                "as_of_source": row.get("as_of_source"),
                "as_of_confidence": row.get("as_of_confidence"),
            }
        )
    return list(invoices.values())


def commitments_from_lane_contexts(
    lane_contexts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    held: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []
    for lane_context in lane_contexts:
        for obligation_id, raw_reasons in (
            lane_context.get("blocked_reasons_by_obligation_id") or {}
        ).items():
            reasons = [
                str(reason).strip()
                for reason in (raw_reasons if isinstance(raw_reasons, list) else [raw_reasons])
                if str(reason).strip()
            ]
            normalized = {reason.lower() for reason in reasons}
            payload = {
                "obligation_id": str(obligation_id),
                "lane_id": lane_context.get("lane_id") or lane_context.get("collection_lane_id"),
            }
            if normalized & HELD_COMMITMENT_REASON_TOKENS:
                held.append({**payload, "hold_reasons": reasons})
            if normalized & BROKEN_COMMITMENT_REASON_TOKENS:
                broken.append({**payload, "broken_reasons": reasons})
    return held, broken


def format_actual_sent_scope_row(row: dict[str, Any]) -> dict[str, Any]:
    def json_list(value: Any) -> list[str]:
        parsed = json_value(value, fallback=[])
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
        "invoice_refs_generated": json_list(row.get("invoice_refs_generated_json")),
        "invoice_refs_sent": json_list(row.get("invoice_refs_sent_json")),
        "invoice_refs_added": json_list(row.get("invoice_refs_added_json")),
        "invoice_refs_removed": json_list(row.get("invoice_refs_removed_json")),
        "invoice_scope_changed": bool(row.get("invoice_scope_changed")),
        "edit_severity": row.get("edit_severity"),
        "payment_expectation_added": bool(row.get("payment_expectation_added")),
        "payment_expectation_kind": row.get("payment_expectation_kind"),
        "payment_expectation_date": date_string(row.get("payment_expectation_date")),
        "payment_expectation_amount": row.get("payment_expectation_amount"),
        "review_reason_codes": json_list(row.get("review_reason_codes_json")),
    }


def actual_sent_scope_version_ids(rows: list[dict[str, Any]]) -> list[str]:
    version_ids: list[str] = []
    for row in rows:
        event_id = row.get("sent_draft_analysis_event_id")
        content_hash = row.get("application_content_hash")
        if event_id:
            version_ids.append(f"sent_draft_analysis_event:{event_id}")
        if content_hash:
            version_ids.append(f"sent_draft_analysis_hash:{content_hash}")
    return version_ids


def format_party_contact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def party_info(row: dict[str, Any]) -> PartyInfo:
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
        do_not_contact_until=date_string(row.get("do_not_contact_until")),
        monthly_touch_count=int(row.get("monthly_touch_count") or 0),
        is_verified=bool(row.get("is_verified", True)),
        source=provider_type,
        customer_type=row.get("customer_type"),
        size_bucket=row.get("size_bucket"),
    )


def behavior_info(row: dict[str, Any]) -> BehaviorInfo:
    return BehaviorInfo(
        lifetime_value=row.get("lifetime_value"),
        total_collected=row.get("total_collected"),
        avg_days_to_pay=row.get("avg_days_to_pay"),
        on_time_rate=row.get("on_time_rate"),
        partial_payment_rate=row.get("partial_payment_rate"),
        behaviour_profile=json_value(row.get("behaviour_profile"), fallback=None),
        behaviour_segment=row.get("behaviour_segment") or row.get("segment"),
    )


def communication_info(row: dict[str, Any]) -> CommunicationInfo:
    return CommunicationInfo(
        touch_count=int(row.get("touch_count") or 0),
        last_touch_at=row.get("last_touch_at"),
        last_touch_channel=row.get("last_touch_channel"),
        last_sender_level=row.get("last_sender_level"),
        last_tone_used=row.get("last_tone_used"),
        last_response_at=row.get("last_response_at"),
        last_response_type=row.get("last_response_type"),
    )


def obligation_info(row: dict[str, Any]) -> ObligationInfo:
    amount_due = float(row.get("amount_due") or 0)
    days_overdue = int(row.get("days_overdue") or row.get("days_past_due") or 0)
    is_source_disputed = bool(row.get("is_source_disputed")) or bool(row.get("source_query_raw"))
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
        due_date=date_string(row.get("due_date")),
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
