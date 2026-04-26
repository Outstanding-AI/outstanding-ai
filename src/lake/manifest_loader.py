"""Load immutable draft-candidate manifests from regional S3 staging."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from .models import DraftCandidate

MANIFEST_SCHEMA_VERSION = 1


class ManifestLoadError(RuntimeError):
    """Raised when a draft-candidate manifest cannot be loaded or validated."""


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket, key)`` for an S3 URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ManifestLoadError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _default_s3_client(region_name: str):
    import boto3

    return boto3.client("s3", region_name=region_name)


def _read_manifest_bytes(manifest_uri: str, *, region_name: str, s3_client: Any | None) -> bytes:
    bucket, key = parse_s3_uri(manifest_uri)
    client = s3_client or _default_s3_client(region_name)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        return body.read()
    except Exception as exc:
        raise ManifestLoadError(
            f"Failed to read draft candidate manifest {manifest_uri}: {exc}"
        ) from exc


def _candidate_payload(raw_payload: Any) -> list[Any]:
    if isinstance(raw_payload, list):
        return raw_payload
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("candidates"), list):
        return raw_payload["candidates"]
    raise ManifestLoadError(
        "Draft candidate manifest must be a JSON list or object with candidates[]"
    )


def _validate_manifest_envelope(
    raw_payload: Any,
    *,
    expected_tenant_id: str | None,
    expected_sync_run_id: str | None,
    expected_data_lake_region: str | None,
) -> None:
    if not isinstance(raw_payload, dict):
        if expected_tenant_id or expected_data_lake_region:
            raise ManifestLoadError(
                "Draft candidate manifest must be an object with tenant_id, sync_run_id, "
                "data_lake_region, and candidates[]"
            )
        return

    schema_version = raw_payload.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestLoadError(
            f"Draft candidate manifest schema_version must be {MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
        )

    expected_fields = {
        "tenant_id": expected_tenant_id,
        "sync_run_id": expected_sync_run_id,
        "data_lake_region": expected_data_lake_region,
    }
    for field_name, expected_value in expected_fields.items():
        if expected_value is None:
            continue
        actual_value = str(raw_payload.get(field_name) or "")
        if actual_value != str(expected_value):
            raise ManifestLoadError(
                f"Draft candidate manifest {field_name} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )


def load_draft_candidate_manifest(
    manifest_uri: str,
    *,
    region_name: str,
    expected_tenant_id: str | None = None,
    expected_sync_run_id: str | None = None,
    expected_data_lake_region: str | None = None,
    s3_client: Any | None = None,
) -> list[DraftCandidate]:
    """Load and validate draft candidates from an S3 staging manifest."""
    raw_bytes = _read_manifest_bytes(manifest_uri, region_name=region_name, s3_client=s3_client)
    try:
        raw_payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestLoadError(f"Draft candidate manifest is not valid UTF-8 JSON: {exc}") from exc

    _validate_manifest_envelope(
        raw_payload,
        expected_tenant_id=expected_tenant_id,
        expected_sync_run_id=expected_sync_run_id,
        expected_data_lake_region=expected_data_lake_region,
    )

    try:
        candidates = [
            DraftCandidate.model_validate(item) for item in _candidate_payload(raw_payload)
        ]
    except ValidationError as exc:
        raise ManifestLoadError(
            f"Draft candidate manifest failed schema validation: {exc}"
        ) from exc

    if expected_sync_run_id is not None:
        expected = str(expected_sync_run_id)
        mismatches = [
            candidate.candidate_id for candidate in candidates if candidate.sync_run_id != expected
        ]
        if mismatches:
            raise ManifestLoadError(
                "Draft candidate manifest contains candidates for a different sync_run_id: "
                + ", ".join(mismatches[:5])
            )

    seen_candidate_ids: set[str] = set()
    duplicates: list[str] = []
    for candidate in candidates:
        if candidate.candidate_id in seen_candidate_ids:
            duplicates.append(candidate.candidate_id)
        seen_candidate_ids.add(candidate.candidate_id)
    if duplicates:
        raise ManifestLoadError(
            "Draft candidate manifest contains duplicate candidate_id values: "
            + ", ".join(duplicates[:5])
        )

    return candidates
