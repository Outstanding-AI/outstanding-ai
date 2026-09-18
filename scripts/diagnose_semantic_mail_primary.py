"""Diagnose one restricted semantic-mail packet without writing source or model text.

This local-only runner emits opaque IDs, packet-shape metadata, model enum and
support selections, validation-stage codes, and final scorer comparison. Its
input/output paths are expected to be owner-only evaluation storage outside Git.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from scripts.run_semantic_mail_primary_eval import (
    _apply_sealed_admission,
    _expected_families,
    _request_for,
    _write_restricted,
)
from src.config.settings import settings
from src.engine.collection_email_event_classifier import _parse_response_object
from src.engine.mail_semantic_evidence_v2 import (
    _EVENT_ENUM_GUIDANCE,
    _SYSTEM_PROMPT,
    _LLMResponse,
    _normalize_known_model_aliases,
)
from src.engine.mail_semantic_evidence_v3 import MailSemanticEvidenceInterpreterV3
from src.llm.base import LLMResponse
from src.llm.openrouter_provider import OpenRouterProvider


class _RecordedPrimary:
    """Replay one provider response through the real interpreter without another API call."""

    provider_name = "openrouter"
    model_name = "recorded-primary"

    def __init__(self, response: LLMResponse) -> None:
        self._response = response

    async def complete(self, *_args: object, **_kwargs: object) -> LLMResponse:
        return self._response


def _safe_event_shape(event: object) -> dict[str, object]:
    if not isinstance(event, dict):
        return {"event_shape": type(event).__name__}
    evidence = event.get("evidence")
    spans = []
    if isinstance(evidence, list):
        for span in evidence:
            if isinstance(span, dict):
                spans.append(
                    {
                        "source_id": span.get("source_id"),
                        "supports": span.get("supports"),
                        "evidence_text_length": len(str(span.get("evidence_text") or "")),
                    }
                )
    return {
        "family": event.get("family"),
        "transition": event.get("transition"),
        "polarity": event.get("polarity"),
        "temporal_orientation": event.get("temporal_orientation"),
        "invoice_ref_count": len(event.get("invoice_refs") or []),
        "has_amount": event.get("amount") is not None,
        "has_currency": event.get("currency") is not None,
        "has_asserted_date": event.get("asserted_date") is not None,
        "has_reference": event.get("reference") is not None,
        "evidence": spans,
    }


def _safe_raw_shape(content: str) -> dict[str, object]:
    try:
        parsed = _parse_response_object(content)
    except Exception as exc:
        return {"json_parse_error": type(exc).__name__}
    if not isinstance(parsed, dict):
        return {"decoded_shape": type(parsed).__name__}
    raw_events = parsed.get("semantic_events")
    output: dict[str, object] = {
        "relevance": parsed.get("relevance"),
        "disposition": parsed.get("disposition"),
        "adjudication_status": parsed.get("adjudication_status"),
        "semantic_event_shapes": [_safe_event_shape(event) for event in raw_events]
        if isinstance(raw_events, list)
        else [],
    }
    try:
        normalized = _normalize_known_model_aliases(parsed)
        _LLMResponse.model_validate(normalized)
        output["llm_schema_valid"] = True
    except Exception as exc:
        output["llm_schema_valid"] = False
        output["llm_schema_error_type"] = type(exc).__name__
        if hasattr(exc, "errors"):
            errors = exc.errors()
            if errors:
                output["llm_schema_error_location"] = list(errors[0].get("loc") or ())
                output["llm_schema_error_kind"] = errors[0].get("type")
    return output


async def _diagnose(packet: dict[str, Any], label: dict[str, Any]) -> dict[str, object]:
    request = _request_for(packet)
    settings.enable_mail_semantic_evidence_v3 = True
    settings.openrouter_mail_semantic_max_completion_tokens = 2048
    settings.openrouter_mail_semantic_timeout_seconds = 25.0
    provider = OpenRouterProvider()
    response = await provider.complete(
        system_prompt=f"{_SYSTEM_PROMPT}\n{_EVENT_ENUM_GUIDANCE}",
        user_prompt=json.dumps(
            request.model_dump(mode="json", exclude_none=True),
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ),
        temperature=settings.classification_temperature,
        json_mode=True,
        reasoning_effort=settings.openrouter_mail_semantic_primary_reasoning_effort,
        reasoning_enabled=settings.openrouter_mail_semantic_primary_reasoning_enabled,
        caller="semantic_mail_diagnostic",
    )
    interpreter = MailSemanticEvidenceInterpreterV3(primary_provider=_RecordedPrimary(response))
    final = await interpreter.interpret(request)
    expected = _expected_families(label)
    predicted = {event.family for event in final.semantic_events}
    return {
        "packet_id": packet["packet_id"],
        "split": packet["split"],
        "input_shape": {
            "direction": packet["target"]["direction"],
            "current_segment_count": len(request.current_message.source_segments),
            "current_text_char_count": sum(
                len(segment.text) for segment in request.current_message.source_segments
            ),
            "prior_message_count": len(request.prior_messages),
            "prior_text_char_count": sum(
                len(item.authored_text) for item in request.prior_messages
            ),
            "attachment_evidence_state": request.attachment_evidence_state,
            "candidate_party_count": len(request.candidate_parties),
            "candidate_invoice_count": len(request.candidate_invoices),
            "party_resolution": request.party_resolution,
            "invoice_resolution": request.invoice_resolution,
        },
        "raw_model_shape": _safe_raw_shape(response.content),
        "final_interpreter": {
            "disposition": final.disposition,
            "reason_codes": final.reason_codes,
            "predicted_event_families": sorted(predicted),
            "invented_invoice_reference_count": sum(
                len(event.invoice_refs) for event in final.semantic_events
            ),
            "total_tokens": final.operation_summary.total_tokens,
        },
        "label_comparison": {
            "expected_event_families": sorted(expected),
            "event_set_exact_match": expected == predicted,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packets = _apply_sealed_admission(
        json.loads(args.pack.read_text(encoding="utf-8")),
        json.loads(args.sealed_manifest.read_text(encoding="utf-8")),
    )
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["labels"]
    packet = next(item for item in packets if item["packet_id"] == args.packet_id)
    label = next(item for item in labels if item["packet_id"] == args.packet_id)
    _write_restricted(args.output, asyncio.run(_diagnose(packet, label)))
    print(json.dumps({"output": str(args.output), "packet_id": args.packet_id}))


if __name__ == "__main__":
    main()
