"""Build the V4 AI context from rows already loaded from regional Silver."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.api.models.requests import CaseContext, ObligationInfo

from . import context_evidence as evidence
from .models import DraftCandidate


def assemble_case_context(
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
    """Compose the existing V4 CaseContext without issuing any reads.

    Historical chain evidence is context-only. Current candidate obligations
    remain the only demand scope, and every call receives the same records as
    the prior hydrator-owned assembly path.
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
            + evidence.actual_sent_scope_version_ids(actual_sent_scope_history)
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
        "tone_ladder": evidence.json_value(lane.get("tone_ladder_snapshot_json"), fallback=[]),
    }
    lane_contexts = evidence.candidate_lane_contexts(
        candidate=candidate,
        fallback_context={
            "lane_id": str(lane["id"]),
            "current_level": current_level,
            "entry_level": lane.get("entry_level"),
            "tone_ladder": lane_context["tone_ladder"],
        },
        obligations=obligations,
    )
    held_commitments, broken_commitments = evidence.commitments_from_lane_contexts(lane_contexts)
    temporal_evidence = case_temporal_evidence or []

    return CaseContext(
        schema_version=4,
        party=evidence.party_info(party),
        behavior=evidence.behavior_info(party),
        obligations=obligations,
        communication=evidence.communication_info(party),
        case_state=party.get("case_state"),
        base_currency=base_currency,
        total_outstanding_base=lane.get("outstanding_amount_base")
        or lane.get("outstanding_amount"),
        broken_promises_count=int(party.get("broken_promises_count") or 0),
        active_dispute=bool(party.get("dispute_type")),
        hardship_indicated=bool(party.get("hardship_indicated")),
        brand_tone=party.get("tone_override") or "professional",
        touch_cap=int(party.get("touch_cap_override") or 10),
        grace_days=evidence.int_or_default(party.get("grace_days_override"), 0),
        do_not_contact_until=evidence.date_string(party.get("do_not_contact_until")),
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
        collection_thread_messages=evidence.format_case_thread_messages(temporal_evidence),
        collection_thread_invoice_evidence=evidence.format_case_invoice_evidence(temporal_evidence),
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
        mode=candidate.mode or ("multi_lane" if len(lane_contexts) > 1 else "single_lane"),
        debtor_contact=debtor_contact,
        party_contacts=party_contacts,
        context_version="v4",
        source_sync_run_id=str(candidate.sync_run_id),
        application_run_id=str(lane.get("application_run_id") or f"app:{candidate.sync_run_id}"),
        core_snapshot_watermark=party.get("silver_valid_from") or decision_time,
        application_snapshot_watermark=lane.get("valid_from") or decision_time,
        application_decision_cutoff=decision_time,
        input_silver_version_ids=input_silver_version_ids,
        policy_snapshot_id=str(lane.get("policy_snapshot_id") or ""),
        draft_candidate_id=str(candidate.candidate_id),
        collection_basis=str(lane.get("collection_basis") or lane.get("chase_basis") or "overdue"),
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


__all__ = ["assemble_case_context"]
