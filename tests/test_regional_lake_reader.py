from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

from src.lake import DraftGenerationHandoff, RegionalLakeClients
from src.lake.regional_reader import RegionalLakeReader


class _FakeBoto3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def client(self, service_name: str, *, region_name: str):
        self.calls.append((service_name, region_name))
        return {"service": service_name, "region": region_name}


class _FakeAthena:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.executions = 0

    def start_query_execution(self, **kwargs):
        self.started.append(kwargs)
        return {"QueryExecutionId": "query-1"}

    def get_query_execution(self, **kwargs):
        self.executions += 1
        assert kwargs["QueryExecutionId"] == "query-1"
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, **kwargs):
        assert kwargs["QueryExecutionId"] == "query-1"
        assert kwargs.get("NextToken") is None
        return {
            "ResultSet": {
                "ResultSetMetadata": {
                    "ColumnInfo": [
                        {"Label": "id", "Type": "varchar"},
                        {"Label": "amount_due", "Type": "double"},
                    ]
                },
                "Rows": [
                    {"Data": [{"VarCharValue": "id"}, {"VarCharValue": "amount_due"}]},
                    {"Data": [{"VarCharValue": "obl-1"}, {"VarCharValue": "12.5"}]},
                ],
            }
        }


class _FakeClients:
    region_name = "eu-west-2"

    def __init__(self) -> None:
        self.athena_client = _FakeAthena()

    def athena(self):
        return self.athena_client


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


def test_regional_lake_reader_executes_against_region_database_and_coerces_rows() -> None:
    clients = _FakeClients()
    reader = RegionalLakeReader(clients=clients, poll_interval_seconds=0)

    rows = reader.execute(
        "SELECT id, amount_due FROM obligations WHERE tenant_id = %s", ["tenant-1"]
    )

    assert rows == [{"id": "obl-1", "amount_due": 12.5}]
    started = clients.athena_client.started[0]
    assert started["QueryExecutionContext"] == {"Database": "outstandingai_eu_west_2"}
    assert started["WorkGroup"] == "primary"
    assert "tenant_id = 'tenant-1'" in started["QueryString"]
