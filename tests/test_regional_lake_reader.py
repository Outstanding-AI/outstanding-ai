from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

from src.lake import DraftGenerationHandoff, RegionalLakeClients


class _FakeBoto3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def client(self, service_name: str, *, region_name: str):
        self.calls.append((service_name, region_name))
        return {"service": service_name, "region": region_name}


def test_handoff_requires_explicit_data_lake_region() -> None:
    with pytest.raises(ValidationError, match="data_lake_region"):
        DraftGenerationHandoff.model_validate(
            {
                "tenant_id": "tenant-1",
                "sync_run_id": "sync-1",
                "manifest_uri": "s3://bucket/manifest.json",
            }
        )

    with pytest.raises(ValidationError, match="explicit AWS region"):
        DraftGenerationHandoff(
            tenant_id="tenant-1",
            sync_run_id="sync-1",
            manifest_uri="s3://bucket/manifest.json",
            data_lake_region="eu",
        )


def test_regional_clients_are_pinned_to_handoff_region(monkeypatch) -> None:
    fake_boto3 = _FakeBoto3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    handoff = DraftGenerationHandoff(
        tenant_id="tenant-1",
        sync_run_id="sync-1",
        manifest_uri="s3://bucket/manifest.json",
        data_lake_region="eu-west-2",
    )
    clients = RegionalLakeClients.from_handoff(handoff)

    assert clients.athena() == {"service": "athena", "region": "eu-west-2"}
    assert clients.glue() == {"service": "glue", "region": "eu-west-2"}
    assert clients.s3() == {"service": "s3", "region": "eu-west-2"}
    assert fake_boto3.calls == [
        ("athena", "eu-west-2"),
        ("glue", "eu-west-2"),
        ("s3", "eu-west-2"),
    ]
    assert os.environ["AWS_REGION"] == "us-east-1"
