"""Synthetic coverage for the local sealed semantic-admission manifest builder."""

from __future__ import annotations

import pytest

from scripts.build_sealed_semantic_manifest import build_manifest


def _packet(*, has_attachments: bool = False) -> dict[str, object]:
    return {
        "packet_id": "packet-1",
        "target": {
            "message_version_key": "message-1:version:hash",
            "body_content_hash": "a" * 64,
            "occurred_at": "2026-09-18T00:00:00Z",
            "has_attachments": has_attachments,
        },
        "earlier_context": [],
    }


def _accounting() -> dict[str, object]:
    return {
        "entries": [
            {
                "packet_id": "packet-1",
                "identity_terminal_revision": "identity-revision-1",
                "as_of_accounting_cut": "2026-09-18T00:00:00Z",
                "party_resolution": "unresolved",
                "invoice_resolution": "unresolved",
                "candidate_parties": [],
                "candidate_invoices": [],
            }
        ]
    }


def _documents(*, state: str = "not_present", revision: str | None = None) -> dict[str, object]:
    return {
        "entries": [
            {
                "packet_id": "packet-1",
                "attachment_evidence_state": state,
                "document_evidence_revision": revision,
            }
        ]
    }


def test_builds_content_free_manifest_for_terminal_unresolved_identity() -> None:
    manifest = build_manifest(
        [_packet()], _accounting(), _documents(), context_selector_version="selector-v1"
    )

    assert manifest["schema_version"] == "sealed-semantic-admission.local.v2"
    assert manifest["entries"][0]["identity"]["invoice_resolution"] == "unresolved"
    assert manifest["entries"][0]["document"]["attachment_evidence_state"] == "not_present"
    assert (
        manifest["entries"][0]["semantic_admission"]["identity_revision"] == "identity-revision-1"
    )
    assert "authored_text" not in str(manifest)


def test_rejects_attachment_without_terminal_document_revision() -> None:
    with pytest.raises(ValueError, match="terminal state and document_evidence_revision"):
        build_manifest(
            [_packet(has_attachments=True)],
            _accounting(),
            _documents(state="pending"),
            context_selector_version="selector-v1",
        )


def test_rejects_missing_accounting_fixture() -> None:
    with pytest.raises(ValueError, match="no accounting fixture entry"):
        build_manifest(
            [_packet()], {"entries": []}, _documents(), context_selector_version="selector-v1"
        )
