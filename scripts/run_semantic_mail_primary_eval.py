"""Run a bounded DeepSeek semantic-mail evaluation over restricted local packets.

The input pack and labels remain outside the repository. This script writes only
opaque packet IDs, model telemetry, model-derived identifiers and aggregate
scores; it never writes source bodies, subjects, email addresses or evidence
text to its output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solvix_contracts.ai import MailSemanticEvidenceRequestV3

from src.config.settings import settings
from src.engine.mail_semantic_evidence_v3 import MailSemanticEvidenceInterpreterV3

_SOURCE_KIND = {
    "authored_current": "authored_body",
    "quoted_history": "quoted_body",
    "forwarded_inline": "forwarded_body",
}
_REQUEST_INFORMATION = "request_information"


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mailbox_roles(direction: str) -> dict[str, object]:
    if direction == "outbound":
        return {
            "sender_role": "internal",
            "recipient_roles": ["external_unknown"],
            "mailbox_direction": "outbound",
        }
    return {
        "sender_role": "external_unknown",
        "recipient_roles": ["shared_mailbox"],
        "mailbox_direction": "inbound",
    }


def _request_for(packet: dict[str, Any]) -> MailSemanticEvidenceRequestV3:
    target = packet["target"]
    direction = str(target["direction"])
    admission = packet.get("semantic_admission")
    if not isinstance(admission, dict):
        raise ValueError("restricted packet lacks sealed semantic_admission")
    current_segments = []
    for segment in target.get("canonical_segments") or []:
        source_kind = _SOURCE_KIND.get(str(segment.get("kind") or ""))
        text = str(segment.get("text") or "")
        if source_kind and text:
            current_segments.append(
                {
                    "source_id": f"current-{segment['ordinal']}",
                    "source_kind": source_kind,
                    "text": text,
                    "content_hash": segment["content_hash"],
                }
            )

    prior_messages = []
    for ordinal, prior in enumerate(packet.get("earlier_context") or [], start=1):
        prior_messages.append(
            {
                "ordinal": ordinal,
                "source_id": f"prior-{ordinal}",
                "timestamp": prior["occurred_at"],
                "direction": prior["direction"],
                "subject": prior.get("subject") or "",
                "authored_text": prior.get("authored_text") or "",
                "content_hash": prior["body_content_hash"],
            }
        )

    packet_id = str(packet["packet_id"])
    request_input = {
        "packet_id": packet_id,
        "message_version": target["message_version_key"],
        "prior_message_versions": [item.get("source_id") for item in prior_messages],
        "attachment_state": admission.get("attachment_evidence_state"),
        "semantic_admission": admission,
    }
    return MailSemanticEvidenceRequestV3(
        tenant_ref="restricted-local-evaluation",
        canonical_message_state_key=target["message_key"],
        canonical_message_version_hash=target["body_content_hash"],
        semantic_revision_key=f"local-eval-{packet_id[:32]}",
        input_context_hash=_sha256(request_input),
        mode="manual_outbound" if direction == "outbound" else "known_collection_inbound",
        current_message={
            "source_id": "current-message",
            "timestamp": target["occurred_at"],
            "direction": direction,
            "subject": target.get("subject") or "",
            "envelope": _mailbox_roles(direction),
            "body_state": "complete" if current_segments else "missing",
            "source_segments": current_segments,
        },
        prior_messages=prior_messages,
        party_resolution=str(packet.get("party_resolution") or "unresolved"),
        candidate_parties=packet.get("candidate_parties") or [],
        invoice_resolution=str(packet.get("invoice_resolution") or "unresolved"),
        candidate_invoices=packet.get("candidate_invoices") or [],
        attachment_evidence=packet.get("attachment_evidence") or [],
        context_manifest=packet.get("context_manifest") or [],
        admission=admission,
    )


def _expected_families(label: dict[str, Any]) -> set[str]:
    families = set()
    for assertion in label.get("assertions") or []:
        family = str(assertion.get("event_family") or "")
        if family and family != "none":
            families.add(_REQUEST_INFORMATION if family == "document_request" else family)
    return families


def _write_restricted(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _apply_sealed_admission(
    packets: list[dict[str, Any]], sealed_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join restricted packets to the V3 terminal-admission fixture before any provider call."""

    if sealed_manifest.get("schema_version") != "sealed-semantic-admission.local.v2":
        raise ValueError("sealed manifest must use sealed-semantic-admission.local.v2")
    entries = sealed_manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("sealed manifest requires entries")
    by_packet_id = {
        str(entry.get("packet_id")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("packet_id")
    }
    if len(by_packet_id) != len(entries):
        raise ValueError("sealed manifest has missing or duplicate packet ids")
    enriched: list[dict[str, Any]] = []
    for packet in packets:
        packet_id = str(packet.get("packet_id") or "")
        entry = by_packet_id.get(packet_id)
        if entry is None:
            raise ValueError("every evaluation packet requires a sealed admission entry")
        identity = entry.get("identity")
        document = entry.get("document")
        admission = entry.get("semantic_admission")
        if not all(isinstance(value, dict) for value in (identity, document, admission)):
            raise ValueError("sealed entry has incomplete identity/document/admission data")
        merged = dict(packet)
        merged.update(
            {
                "party_resolution": identity["party_resolution"],
                "candidate_parties": identity["candidate_parties"],
                "invoice_resolution": identity["invoice_resolution"],
                "candidate_invoices": identity["candidate_invoices"],
                "attachment_evidence": document["attachment_evidence"],
                "context_manifest": entry["selected_context"],
                "semantic_admission": admission,
            }
        )
        enriched.append(merged)
    return enriched


async def _evaluate(
    packets: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, concurrency: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings.enable_mail_semantic_evidence_v3 = True
    interpreter = MailSemanticEvidenceInterpreterV3()
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_packet(packet: dict[str, Any]) -> dict[str, Any]:
        packet_id = str(packet["packet_id"])
        label = labels[packet_id]
        expected = _expected_families(label)
        try:
            async with semaphore:
                request = _request_for(packet)
                response = await interpreter.interpret(request)
        except Exception as exc:  # Persist a controlled evaluation failure, not customer content.
            return {
                "packet_id": packet_id,
                "split": packet["split"],
                "expected_event_families": sorted(expected),
                "status": "primary_error",
                "error_type": type(exc).__name__,
                "error_code": str(exc)[:120],
            }

        predicted = {event.family for event in response.semantic_events}
        invoice_reference_count = sum(len(event.invoice_refs) for event in response.semantic_events)
        invocation = (
            response.operation_summary.invocations[0]
            if response.operation_summary.invocations
            else None
        )
        return {
            "packet_id": packet_id,
            "split": packet["split"],
            "expected_event_families": sorted(expected),
            "predicted_event_families": sorted(predicted),
            "event_set_exact_match": predicted == expected,
            "disposition": response.disposition,
            "reason_codes": response.reason_codes,
            "semantic_invoice_reference_count": invoice_reference_count,
            "invoice_reference_reconciliation": "complete",
            "provider": invocation.provider if invocation else None,
            "model": invocation.model if invocation else None,
            "total_tokens": response.operation_summary.total_tokens,
        }

    results = await asyncio.gather(*(evaluate_packet(packet) for packet in packets))
    family_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    true_positives = false_positives = false_negatives = 0
    exact_event_set_matches = 0
    admitted_invoice_reference_count = 0
    primary_errors = 0
    completed_results = 0
    accepted_packets = 0
    nonaccepted_packets = 0
    for result in results:
        expected = set(result["expected_event_families"])
        expected_counts.update(expected)
        if result.get("status") == "primary_error":
            primary_errors += 1
            continue
        completed_results += 1
        predicted = set(result["predicted_event_families"])
        family_counts.update(predicted)
        true_positives += len(expected & predicted)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        admitted_invoice_reference_count += int(result["semantic_invoice_reference_count"])
        if result["disposition"] == "accepted":
            accepted_packets += 1
        else:
            nonaccepted_packets += 1
        if result["disposition"] == "accepted" and result["event_set_exact_match"]:
            exact_event_set_matches += 1

    summary = {
        "evaluated_packets": len(packets),
        "completed_packets": completed_results,
        "accepted_packets": accepted_packets,
        "nonaccepted_packets": nonaccepted_packets,
        "primary_errors": primary_errors,
        "exact_event_set_matches": exact_event_set_matches,
        "strict_exact_event_set_accuracy": exact_event_set_matches / len(packets) if packets else 0,
        "accepted_exact_event_set_accuracy": exact_event_set_matches / accepted_packets
        if accepted_packets
        else 0,
        "event_family_true_positives": true_positives,
        "event_family_false_positives": false_positives,
        "event_family_false_negatives": false_negatives,
        "event_family_precision": true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0,
        "event_family_recall": true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0,
        "event_family_f1": (2 * true_positives)
        / (2 * true_positives + false_positives + false_negatives)
        if 2 * true_positives + false_positives + false_negatives
        else 0,
        "expected_event_family_counts": dict(sorted(expected_counts.items())),
        "predicted_event_family_counts": dict(sorted(family_counts.items())),
        "admitted_invoice_reference_count": admitted_invoice_reference_count,
        "identity_resolution_note": (
            "Every scored packet carries a terminal identity/accounting and document "
            "admission envelope. Invoice references that remain after strict local "
            "validation are admitted candidates, not unbound model claims."
        ),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="Restricted local packet JSON")
    parser.add_argument(
        "--sealed-manifest",
        type=Path,
        required=True,
        help="Restricted V3 terminal-admission manifest for the same packets",
    )
    parser.add_argument(
        "--labels", type=Path, required=True, help="Restricted effective-label JSON"
    )
    parser.add_argument("--output", type=Path, required=True, help="New restricted output JSON")
    parser.add_argument(
        "--packet-id",
        help="Optional opaque packet ID for a bounded single-packet diagnostic replay",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Bounded completion cap for local calibration (default: 1024)",
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=25.0,
        help="Per-message provider deadline for local calibration (default: 25)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent provider calls for local calibration (default: 1)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "holdout"),
        help="Optional frozen split to evaluate without reading labels from another split",
    )
    args = parser.parse_args()

    packets = _apply_sealed_admission(
        json.loads(args.pack.read_text(encoding="utf-8")),
        json.loads(args.sealed_manifest.read_text(encoding="utf-8")),
    )
    label_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = {str(label["packet_id"]): label for label in label_payload["labels"]}
    selected_packets = [packet for packet in packets if str(packet["packet_id"]) in labels]
    if len(selected_packets) != len(labels):
        raise ValueError("every effective label must have exactly one packet")
    if args.packet_id:
        selected_packets = [
            packet for packet in selected_packets if str(packet["packet_id"]) == args.packet_id
        ]
        if len(selected_packets) != 1:
            raise ValueError("packet-id must select exactly one labelled packet")
    elif args.split:
        selected_packets = [packet for packet in selected_packets if packet["split"] == args.split]
        if not selected_packets:
            raise ValueError("split selected no labelled packets")

    if (
        args.max_completion_tokens <= 0
        or args.provider_timeout_seconds <= 0
        or args.concurrency <= 0
    ):
        raise ValueError("local calibration limits must be positive")
    settings.openrouter_mail_semantic_max_completion_tokens = args.max_completion_tokens
    settings.openrouter_mail_semantic_timeout_seconds = args.provider_timeout_seconds

    results, summary = asyncio.run(
        _evaluate(selected_packets, labels, concurrency=args.concurrency)
    )
    _write_restricted(
        args.output,
        {
            "schema_version": "semantic-mail-primary-eval.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pack_path": str(args.pack),
            "labels_path": str(args.labels),
            "result_packets": results,
            "summary": summary,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluated_packets": summary["evaluated_packets"],
                "primary_errors": summary["primary_errors"],
            }
        )
    )


if __name__ == "__main__":
    main()
