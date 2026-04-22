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
