"""Build a restricted, content-free sealed-admission manifest for semantic evaluation.

The raw message pack remains outside Git. This validator joins it only to
timestamp-pinned accounting and terminal document fixtures, then writes opaque
packet/message/context hashes and candidate provenance for the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from solvix_contracts.ai import (
    CollectionInvoiceCandidateV1,
    MailSemanticContextManifestEntryV3,
    MailSemanticPartyCandidateV2,
    mail_semantic_candidate_set_hash,
    mail_semantic_context_manifest_hash,
)

_PARTY_STATES = {"identified", "ambiguous", "unresolved", "absent"}
_INVOICE_STATES = {"exact", "ambiguous", "unresolved", "absent"}
_TERMINAL_DOCUMENT_STATES = {
    "complete",
    "partial",
    "unsupported",
    "failed",
    "unavailable",
    "unsafe",
    "expired",
    "revoked",
}


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entries_by_packet(payload: dict[str, Any], *, name: str) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{name} fixture requires an entries list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("packet_id"), str):
            raise ValueError(f"{name} fixture entry requires packet_id")
        packet_id = entry["packet_id"]
        if packet_id in indexed:
            raise ValueError(f"{name} fixture has duplicate packet_id")
        indexed[packet_id] = entry
    return indexed


def _validate_accounting(entry: dict[str, Any]) -> dict[str, Any]:
    identity_revision = str(entry.get("identity_terminal_revision") or "")
    accounting_cut = str(entry.get("as_of_accounting_cut") or "")
    party_resolution = str(entry.get("party_resolution") or "")
    invoice_resolution = str(entry.get("invoice_resolution") or "")
    candidates = entry.get("candidate_invoices") or []
    parties = entry.get("candidate_parties") or []
    if not identity_revision or not accounting_cut:
        raise ValueError(
            "accounting entry requires identity_terminal_revision and as_of_accounting_cut"
        )
    if party_resolution not in _PARTY_STATES or invoice_resolution not in _INVOICE_STATES:
        raise ValueError("accounting entry has invalid terminal party/invoice resolution")
    if party_resolution in {"identified", "ambiguous"} and not parties:
        raise ValueError("resolved party outcome requires candidate_parties")
    if invoice_resolution in {"exact", "ambiguous"} and not candidates:
        raise ValueError("resolved invoice outcome requires candidate_invoices")
    for candidate in candidates:
        required = {"obligation_id", "invoice_number", "is_open", "binding_provenance_ref"}
        if not isinstance(candidate, dict) or any(not candidate.get(field) for field in required):
            raise ValueError(
                "invoice candidate requires obligation_id, invoice_number and binding_provenance_ref"
            )
    for candidate in parties:
        if (
            not isinstance(candidate, dict)
            or not candidate.get("party_ref")
            or not candidate.get("binding_provenance_ref")
        ):
            raise ValueError("party candidate requires party_ref and binding_provenance_ref")
    return {
        "identity_terminal_revision": identity_revision,
        "as_of_accounting_cut": accounting_cut,
        "party_resolution": party_resolution,
        "invoice_resolution": invoice_resolution,
        "candidate_parties": parties,
        "candidate_invoices": candidates,
    }


def _validate_document(entry: dict[str, Any], *, has_attachments: bool) -> dict[str, Any]:
    if not has_attachments:
        return {
            "attachment_evidence_state": "not_present",
            "document_evidence_revision": None,
            "attachment_evidence": [],
        }
    state = str(entry.get("attachment_evidence_state") or "")
    revision = str(entry.get("document_evidence_revision") or "")
    attachments = entry.get("attachment_evidence") or []
    if state not in _TERMINAL_DOCUMENT_STATES or not revision or not attachments:
        raise ValueError(
            "material attachment requires a terminal state and document_evidence_revision"
        )
    if not isinstance(attachments, list):
        raise ValueError("attachment_evidence must be a list")
    return {
        "attachment_evidence_state": state,
        "document_evidence_revision": revision,
        "attachment_evidence": attachments,
    }


def _candidate_set_hash(identity: dict[str, Any]) -> str:
    parties = [
        MailSemanticPartyCandidateV2.model_validate(
            {key: value for key, value in candidate.items() if key != "binding_provenance_ref"}
        )
        for candidate in identity["candidate_parties"]
    ]
    invoices = [
        CollectionInvoiceCandidateV1.model_validate(
            {key: value for key, value in candidate.items() if key != "binding_provenance_ref"}
        )
        for candidate in identity["candidate_invoices"]
    ]
    return mail_semantic_candidate_set_hash(
        party_resolution=identity["party_resolution"],
        candidate_parties=parties,
        invoice_resolution=identity["invoice_resolution"],
        candidate_invoices=invoices,
    )


def build_manifest(
    packets: list[dict[str, Any]],
    accounting_fixture: dict[str, Any],
    document_fixture: dict[str, Any],
    *,
    context_selector_version: str,
) -> dict[str, Any]:
    if not context_selector_version:
        raise ValueError("context_selector_version is required")
    accounting = _entries_by_packet(accounting_fixture, name="accounting")
    documents = _entries_by_packet(document_fixture, name="document")
    sealed_entries = []
    for packet in packets:
        packet_id = str(packet.get("packet_id") or "")
        target = packet.get("target") or {}
        if not packet_id or not target.get("message_version_key"):
            raise ValueError("packet requires packet_id and target message_version_key")
        if packet_id not in accounting:
            raise ValueError("packet has no accounting fixture entry")
        if packet_id not in documents:
            raise ValueError("packet has no document fixture entry")
        identity = _validate_accounting(accounting[packet_id])
        document = _validate_document(
            documents[packet_id], has_attachments=bool(target.get("has_attachments"))
        )
        context = packet.get("earlier_context") or []
        selected_context = [
            {
                "ordinal": ordinal,
                "source_id": f"prior-{ordinal}",
                "canonical_message_version": item.get("message_version_key"),
                "content_hash": item.get("body_content_hash"),
                "occurred_at": item.get("occurred_at"),
                "selection_reason": "deterministic_causal_context",
            }
            for ordinal, item in enumerate(context, start=1)
        ]
        context_entries = [
            MailSemanticContextManifestEntryV3.model_validate(item) for item in selected_context
        ]
        sealed_entries.append(
            {
                "packet_id": packet_id,
                "canonical_message_version": target["message_version_key"],
                "message_content_hash": target.get("body_content_hash"),
                "occurred_at": target.get("occurred_at"),
                "context_selector_version": context_selector_version,
                "selected_context": selected_context,
                "identity": identity,
                "document": document,
                "semantic_admission": {
                    "identity_revision": identity["identity_terminal_revision"],
                    "accounting_cut_at": identity["as_of_accounting_cut"],
                    "identity_candidate_set_hash": _candidate_set_hash(identity),
                    "context_manifest_revision": context_selector_version,
                    "context_manifest_hash": mail_semantic_context_manifest_hash(context_entries),
                    "attachment_evidence_state": document["attachment_evidence_state"],
                    "document_evidence_revision": document["document_evidence_revision"],
                },
            }
        )
    return {
        "schema_version": "sealed-semantic-admission.local.v2",
        "source_pack_hash": _sha256(
            [
                {
                    "packet_id": item["packet_id"],
                    "message": item["target"]["message_version_key"],
                }
                for item in packets
            ]
        ),
        "entries": sealed_entries,
    }


def _write_restricted(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--accounting-fixture", type=Path, required=True)
    parser.add_argument("--document-fixture", type=Path, required=True)
    parser.add_argument("--context-selector-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packets = json.loads(args.pack.read_text(encoding="utf-8"))
    accounting = json.loads(args.accounting_fixture.read_text(encoding="utf-8"))
    documents = json.loads(args.document_fixture.read_text(encoding="utf-8"))
    manifest = build_manifest(
        packets,
        accounting,
        documents,
        context_selector_version=args.context_selector_version,
    )
    _write_restricted(args.output, manifest)
    print(json.dumps({"output": str(args.output), "entry_count": len(manifest["entries"])}))


if __name__ == "__main__":
    main()
