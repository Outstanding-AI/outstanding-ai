from __future__ import annotations

import json

import pytest

from src.lake import ManifestLoadError, load_draft_candidate_manifest, parse_s3_uri


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _S3Client:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def get_object(self, **kwargs):
        self.calls.append({"Bucket": kwargs["Bucket"], "Key": kwargs["Key"]})
        return {"Body": _Body(json.dumps(self.payload).encode("utf-8"))}


def test_parse_s3_uri_requires_bucket_and_key() -> None:
    assert parse_s3_uri("s3://bucket/path/to/manifest.json") == (
        "bucket",
        "path/to/manifest.json",
    )

    with pytest.raises(ManifestLoadError):
        parse_s3_uri("https://bucket/path")
    with pytest.raises(ManifestLoadError):
        parse_s3_uri("s3://bucket")


def test_load_draft_candidate_manifest_accepts_wrapped_candidates() -> None:
    s3 = _S3Client(
        {
            "candidates": [
                {
                    "party_id": "party-1",
                    "lane_id": "lane-1",
                    "sync_run_id": "sync-1",
                    "candidate_id": "candidate-1",
                }
            ]
        }
    )

    candidates = load_draft_candidate_manifest(
        "s3://staging/tenant/sync/draft_candidates_manifest.json",
        region_name="eu-west-2",
        expected_sync_run_id="sync-1",
        s3_client=s3,
    )

    assert candidates[0].party_id == "party-1"
    assert candidates[0].lane_id == "lane-1"
    assert s3.calls == [{"Bucket": "staging", "Key": "tenant/sync/draft_candidates_manifest.json"}]


def test_load_draft_candidate_manifest_rejects_sync_run_mismatch() -> None:
    s3 = _S3Client(
        [
            {
                "party_id": "party-1",
                "lane_id": "lane-1",
                "sync_run_id": "other-sync",
                "candidate_id": "candidate-1",
            }
        ]
    )

    with pytest.raises(ManifestLoadError, match="different sync_run_id"):
        load_draft_candidate_manifest(
            "s3://staging/manifest.json",
            region_name="eu-west-2",
            expected_sync_run_id="sync-1",
            s3_client=s3,
        )


def test_load_draft_candidate_manifest_rejects_extra_candidate_fields() -> None:
    s3 = _S3Client(
        [
            {
                "party_id": "party-1",
                "lane_id": "lane-1",
                "sync_run_id": "sync-1",
                "candidate_id": "candidate-1",
                "case_context": {},
            }
        ]
    )

    with pytest.raises(ManifestLoadError, match="schema validation"):
        load_draft_candidate_manifest(
            "s3://staging/manifest.json",
            region_name="eu-west-2",
            s3_client=s3,
        )
