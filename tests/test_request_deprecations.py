"""Regression tests for phase-8a duplicate-field deprecations."""

from __future__ import annotations

import logging

import pytest

from src.api.models.requests import BehaviorInfo, CommunicationInfo, GenerateDraftRequest
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
