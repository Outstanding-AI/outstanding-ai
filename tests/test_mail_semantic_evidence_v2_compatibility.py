"""Compatibility coverage for the deployed V2 semantic-mail endpoint."""

from __future__ import annotations

import json

import pytest
from solvix_contracts.ai import MailSemanticEvidenceRequestV2

from src.api.routes.interpret_collection_email_v2 import router
from src.config.settings import settings
from src.engine.mail_semantic_evidence_v2 import MailSemanticEvidenceInterpreterV2
from src.llm.base import LLMResponse

_HASH = "a" * 64


def _request() -> MailSemanticEvidenceRequestV2:
    return MailSemanticEvidenceRequestV2(
        tenant_ref="synthetic-tenant",
        canonical_message_state_key="synthetic-message-state",
        canonical_message_version_hash=_HASH,
        semantic_revision_key="synthetic-revision-1",
        input_context_hash=_HASH,
        mode="known_collection_inbound",
        current_message={
            "source_id": "message-1",
            "timestamp": "2026-09-17T10:00:00Z",
            "direction": "inbound",
            "subject": "Synthetic only",
            "envelope": {
                "sender_role": "debtor",
                "recipient_roles": ["shared_mailbox"],
                "mailbox_direction": "inbound",
            },
            "body_state": "complete",
            "source_segments": [
                {
                    "source_id": "authored-1",
                    "source_kind": "authored_body",
                    "text": "We will pay INV-SYN-1 tomorrow.",
                    "content_hash": _HASH,
                }
            ],
        },
    )


class _FakeV2Client:
    primary_provider_name = "vertex"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(self, system_prompt: str, user_prompt: str, **kwargs: object) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        return LLMResponse(
            content=json.dumps(
                {
                    "relevance": "collection",
                    "semantic_events": [
                        {
                            "event_id": "synthetic-event-1",
                            "family": "commitment",
                            "transition": "made",
                            "polarity": "affirmed",
                            "temporal_orientation": "future",
                            "invoice_refs": ["INV-SYN-1"],
                            "account_wide": False,
                            "amount": None,
                            "currency": None,
                            "asserted_date": None,
                            "reference": None,
                            "evidence": [
                                {
                                    "source_id": "authored-1",
                                    "evidence_text": "We will pay INV-SYN-1 tomorrow.",
                                    "supports": [
                                        "relevance",
                                        "invoice_scope",
                                        "operational_state",
                                    ],
                                }
                            ],
                            "confidence": 0.9,
                            "reason_codes": [],
                        }
                    ],
                    "response_evidence": [
                        {
                            "source_id": "authored-1",
                            "evidence_text": "We will pay INV-SYN-1 tomorrow.",
                            "supports": ["relevance"],
                        }
                    ],
                    "disposition": "accepted",
                    "adjudication_status": "not_required",
                    "confidence": 0.9,
                    "reason_codes": [],
                    "ambiguity_reason_codes": [],
                }
            ),
            provider="vertex",
            model="synthetic-v2",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def test_v2_route_remains_registered() -> None:
    assert any(route.path == "/interpret-collection-email-v2" for route in router.routes)


@pytest.mark.asyncio
async def test_v2_disabled_is_a_real_deterministic_deferred_response(monkeypatch) -> None:
    interpreter = MailSemanticEvidenceInterpreterV2()
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v2", False)

    result = await interpreter.interpret(_request())

    assert result.disposition == "deferred"
    assert result.reason_codes == ["mail_semantic_evidence_v2_disabled"]
    assert result.operation_summary.operation == "mail_semantic_evidence_v2"
    assert result.operation_summary.total_tokens == 0


@pytest.mark.asyncio
async def test_v2_enabled_uses_v2_contract_and_legacy_provider_path(monkeypatch) -> None:
    interpreter = MailSemanticEvidenceInterpreterV2()
    fake = _FakeV2Client()
    interpreter._client = fake
    monkeypatch.setattr(settings, "enable_mail_semantic_evidence_v2", True)

    result = await interpreter.interpret(_request())

    assert result.disposition == "accepted"
    assert result.semantic_events[0].invoice_refs == ["INV-SYN-1"]
    assert result.operation_summary.operation == "mail_semantic_evidence_v2"
    assert result.operation_summary.response_schema_version == "mail-semantic-evidence.v2"
    assert fake.calls[0]["caller"] == "mail_semantic_evidence_v2"
