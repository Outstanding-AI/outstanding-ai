"""Regression tests for phase-8a duplicate-field deprecations."""

from __future__ import annotations

import logging

import pytest

from src.api.models.requests import (
    ActualSentScopeHistory,
    BehaviorInfo,
    CommunicationInfo,
    GenerateDraftRequest,
    LaneContextInfo,
)
from src.engine.generator_prompts import build_extra_sections


def test_behavior_info_segment_warns_and_backfills():
    """Legacy segment should warn and hydrate behaviour_segment."""
    with pytest.warns(DeprecationWarning, match="BehaviorInfo.segment is deprecated"):
        info = BehaviorInfo(segment="reliable_late_payer")

    assert info.behaviour_segment == "reliable_late_payer"


def test_behavior_info_segment_mismatch_rejected():
    """Duplicate behaviour segment fields must not diverge."""
    with pytest.raises(ValueError, match="must match behaviour_segment"):
        BehaviorInfo(segment="ghost", behaviour_segment="reliable_late_payer")


def test_generate_request_sender_persona_warns_and_backfills(sample_case_context):
    """Legacy sender persona name/title should hydrate top-level sender fields."""
    with pytest.warns(
        DeprecationWarning,
        match="GenerateDraftRequest.sender_persona.name is deprecated",
    ):
        request = GenerateDraftRequest.model_validate(
            {
                "context": sample_case_context.model_dump(mode="python"),
                "tone": "professional",
                "objective": "follow_up",
                "sender_persona": {
                    "name": "Sarah Jones",
                    "title": "Credit Controller",
                    "communication_style": "calm and direct",
                },
            }
        )

    assert request.sender_name == "Sarah Jones"
    assert request.sender_title == "Credit Controller"


def test_generate_request_sender_persona_name_mismatch_rejected(sample_case_context):
    """Top-level sender identity stays canonical during the deprecation window."""
    with pytest.raises(ValueError, match="must match sender_name"):
        GenerateDraftRequest.model_validate(
            {
                "context": sample_case_context.model_dump(mode="python"),
                "tone": "professional",
                "objective": "follow_up",
                "sender_name": "Top Level Sender",
                "sender_persona": {
                    "name": "Nested Sender",
                    "title": "Credit Controller",
                    "communication_style": "calm and direct",
                },
            }
        )


def test_communication_info_last_response_snippet_warns():
    """Legacy response snippets should warn on construction."""
    with pytest.warns(
        DeprecationWarning,
        match="CommunicationInfo.last_response_snippet is deprecated",
    ):
        info = CommunicationInfo(last_response_snippet="Please call me back.")

    assert info.__dict__["last_response_snippet"] == "Please call me back."


def test_case_context_lane_context_duplicate_fields_warn(sample_case_context):
    """Legacy lane-context duplicates should warn and remain backward-compatible."""
    payload = sample_case_context.model_dump(mode="python")
    payload.update(
        {
            "collection_lane_id": "lane-123",
            "lane": {
                "collection_lane_id": "lane-123",
                "current_level": 2,
                "entry_level": 1,
                "scheduled_touch_index": 1,
                "max_touches_for_level": 3,
                "reminder_cadence_days_for_level": 7,
                "max_days_for_level": 21,
                "tone_ladder": ["professional", "firm"],
                "outstanding_amount": 1500.0,
                "invoice_refs": ["INV-12345"],
            },
            "lane_contexts": [
                {
                    "collection_lane_id": "lane-123",
                    "lane_id": "lane-123",
                    "invoice_refs": ["INV-12345"],
                    "outstanding_amount": 1500.0,
                }
            ],
            "mode": "single_lane",
        }
    )

    with pytest.warns(DeprecationWarning, match="LaneContextInfo.invoice_refs is deprecated"):
        request = GenerateDraftRequest.model_validate(
            {
                "context": payload,
                "tone": "professional",
                "objective": "follow_up",
            }
        )

    lane_context = request.context.lane_contexts[0]
    assert lane_context.__dict__["invoice_refs"] == ["INV-12345"]
    assert lane_context.__dict__["outstanding_amount"] == 1500.0


def test_build_extra_sections_renders_grouped_lane_scope(sample_generate_draft_request):
    sample_generate_draft_request.context.mode = "multi_lane"
    sample_generate_draft_request.context.lane = None
    sample_generate_draft_request.context.lane_contexts = [
        LaneContextInfo(
            lane_id="lane-old",
            current_level=1,
            entry_level=1,
            scheduled_touch_index=2,
            max_touches_for_level=3,
            tone_ladder=["professional", "firm"],
            invoice_refs=["INV-OLD"],
            action="reminder",
        ),
        LaneContextInfo(
            lane_id="lane-new",
            current_level=1,
            entry_level=1,
            scheduled_touch_index=1,
            max_touches_for_level=3,
            tone_ladder=["professional"],
            invoice_refs=["INV-NEW"],
            action="initial",
        ),
    ]

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "Coverage Mode: multiple due recovery lanes" in extra_sections
    assert "invoice table is the authoritative scope" in extra_sections
    assert "INV-OLD" in extra_sections
    assert "INV-NEW" in extra_sections


def test_build_extra_sections_renders_single_active_debtor_thread_scope(
    sample_generate_draft_request,
):
    sample_generate_draft_request.context.collection_case_id = "case-1"
    sample_generate_draft_request.context.threading_strategy = "single_active_debtor_thread"
    sample_generate_draft_request.context.threading_mode = "case_continuation"
    sample_generate_draft_request.context.active_thread_subject = "Overdue invoices"
    sample_generate_draft_request.context.lane = {
        "collection_lane_id": "lane-1",
        "current_level": 1,
        "entry_level": 1,
        "invoice_refs": ["INV-1"],
        "outstanding_amount": 100.0,
    }

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "**Collection Case Decision Context:**" in extra_sections
    assert "Threading Strategy: single_active_debtor_thread" in extra_sections
    assert "continue the single active debtor case thread" in extra_sections
    assert "must not add invoices or amounts to the demand" in extra_sections
    assert "Other lanes for the same debtor may exist" not in extra_sections


def test_build_extra_sections_legacy_cohort_strategy_keeps_cohort_scope(
    sample_generate_draft_request,
):
    sample_generate_draft_request.context.threading_strategy = "invoice_cohort_thread"
    sample_generate_draft_request.context.lane = {
        "collection_lane_id": "lane-1",
        "current_level": 1,
        "entry_level": 1,
        "invoice_refs": ["INV-1"],
        "outstanding_amount": 100.0,
    }

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "this email is for this lane/cohort only" in extra_sections
    assert "Other lanes for the same debtor may exist" in extra_sections


def test_conversation_history_without_inbound_trigger_does_not_force_thank_you(
    sample_generate_draft_request,
):
    sample_generate_draft_request.trigger_classification = None
    sample_generate_draft_request.context.collection_case_id = "case-1"
    sample_generate_draft_request.context.collection_thread_messages = [
        {
            "direction": "outbound",
            "sent_at": "2026-06-09T10:00:00Z",
            "subject": "Overdue invoices",
            "body_snippet": "Please see the current overdue invoices.",
            "invoice_states": [
                {
                    "invoice_number": "INV-1",
                    "as_of_state": "open",
                    "current_state": "open",
                    "as_of_confidence": "high",
                }
            ],
        }
    ]

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "Do NOT say" in extra_sections
    assert "thank you for your reply" in extra_sections
    assert "You MUST acknowledge the debtor's most recent response" not in extra_sections


def test_build_extra_sections_ignores_last_response_snippet(sample_generate_draft_request, caplog):
    """Prompt construction should no longer read deprecated last_response_snippet."""
    sample_generate_draft_request.context.recent_messages = None
    sample_generate_draft_request.context.communication.last_response_snippet = (
        "Old snippet that should not be injected."
    )
    sample_generate_draft_request.context.communication.last_response_subject = "Legacy subject"
    sample_generate_draft_request.context.communication.last_response_type = "reply"

    with caplog.at_level(logging.WARNING):
        extra_sections = build_extra_sections(
            sample_generate_draft_request,
            sample_generate_draft_request.context.behavior,
        )

    assert "Debtor's Last Response" not in extra_sections
    assert "last_response_snippet is deprecated and ignored" in caplog.text


def test_build_extra_sections_omits_unverified_procurement_status(sample_generate_draft_request):
    """Without verified PO/POD, procurement_context_status must not leak into the prompt.

    Even neutral/missing/candidate_reference statuses prime the model to mention
    procurement evidence the FactualGroundingGuardrail then has to block. Drop
    the priming entirely; the guardrail remains as the safety net."""
    obligation = sample_generate_draft_request.context.obligations[0]
    obligation.procurement_context_status = "missing"
    obligation.has_verified_purchase_order = False
    obligation.has_verified_pod = False

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
        candidate_obligations=[obligation],
    )

    assert "procurement_context_status" not in extra_sections
    assert "verified_po" not in extra_sections
    assert "verified_pod" not in extra_sections


def test_build_extra_sections_renders_verified_procurement_facts(sample_generate_draft_request):
    """Verified PO/POD facts ARE renderable — those are grounded evidence the
    LLM can reference accurately."""
    obligation = sample_generate_draft_request.context.obligations[0]
    obligation.procurement_context_status = "verified"
    obligation.has_verified_purchase_order = True
    obligation.purchase_order_reference = "PO-12345"
    obligation.has_verified_pod = True
    obligation.pod_reference = "POD-987"

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
        candidate_obligations=[obligation],
    )

    assert "verified_po=PO-12345" in extra_sections
    assert "verified_pod=POD-987" in extra_sections
    assert "procurement_context_status" not in extra_sections


def test_build_extra_sections_renders_stale_draft_lane_history(sample_generate_draft_request):
    sample_generate_draft_request.context.lane_history = [
        {
            "event_type": "stale_draft_replacement_requested",
            "from_level": 0,
            "to_level": 1,
            "created_at": "2026-05-25T08:00:00Z",
            "detail": {
                "replacement_reason": "protocol_drift",
                "stale_changes": [
                    "protocol slot changed from L0:T3:D15 to L1:T1:D22",
                    "sender changed from accounts@eswl-americas.com to charleen.shanks@eswl-ltd.com",
                ],
            },
        }
    ]

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "stale_draft_replacement_requested" in extra_sections
    assert "replacement_reason=protocol_drift" in extra_sections
    assert "protocol slot changed from L0:T3:D15 to L1:T1:D22" in extra_sections


def test_build_extra_sections_renders_actual_sent_scope_history(sample_generate_draft_request):
    sample_generate_draft_request.context.actual_sent_scope_history = [
        ActualSentScopeHistory(
            draft_id="draft-1",
            sent_at="2026-05-26T09:00:00Z",
            invoice_refs_generated=["INV-1001", "INV-1002"],
            invoice_refs_sent=["INV-1002", "INV-1003"],
            invoice_refs_added=["INV-1003"],
            invoice_refs_removed=["INV-1001"],
            invoice_scope_changed=True,
            edit_severity="critical",
            payment_expectation_added=True,
            payment_expectation_kind="promise_to_pay",
            payment_expectation_date="2026-05-30",
        )
    ]

    extra_sections = build_extra_sections(
        sample_generate_draft_request,
        sample_generate_draft_request.context.behavior,
    )

    assert "Actual Sent Scope History" in extra_sections
    assert "actually sent invoices INV-1002, INV-1003" in extra_sections
    assert "AI-generated scope: INV-1001, INV-1002" in extra_sections
    assert "operator removed before send: INV-1001" in extra_sections
    assert "current invoice table remains the authoritative scope" in extra_sections


def test_format_obligation_flags_omits_procurement_status_for_unverified():
    """``_format_obligation_flags`` is the per-obligation tag string emitter for the
    text block of the prompt. Same priming concern: drop unverified procurement."""
    from src.engine.formatters import _format_obligation_flags

    class _Obligation:
        is_source_disputed = False
        source_query_raw = None
        procurement_context_status = "missing"
        has_verified_purchase_order = False
        has_verified_pod = False

    flags = _format_obligation_flags(_Obligation())
    assert "procurement" not in flags


def test_factual_grounding_section_removed_from_system_prompt():
    """System prompt must NOT mention PO/POD/procurement priming tokens.

    The guardrail still validates output, but the model should not be primed
    with the very words it might then hallucinate around."""
    from src.prompts.draft_generation import GENERATE_DRAFT_SYSTEM

    lower_prompt = GENERATE_DRAFT_SYSTEM.lower()
    assert "purchase order" not in lower_prompt
    assert "proof of delivery" not in lower_prompt
    assert "procurement" not in lower_prompt
