"""Region-pinned AWS clients for regional lake hydration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from solvix_contracts.datalake.athena_dialect import coerce_row, render_params

from .models import DraftGenerationHandoff


class RegionalLakeQueryError(RuntimeError):
    """Raised when a regional Athena query fails or times out."""


@dataclass(frozen=True)
class RegionalLakeClients:
    """Factory for AWS clients pinned to the handoff's data lake region."""

    region_name: str

    @classmethod
    def from_handoff(cls, handoff: DraftGenerationHandoff) -> "RegionalLakeClients":
        return cls(region_name=handoff.data_lake_region)

    def athena(self) -> Any:
        return self._client("athena")

    def glue(self) -> Any:
        return self._client("glue")

    def s3(self) -> Any:
        return self._client("s3")

    def _client(self, service_name: str) -> Any:
        import boto3

        return boto3.client(service_name, region_name=self.region_name)


@dataclass
class RegionalLakeReader:
    """Minimal Athena reader pinned to one data lake region."""

    clients: RegionalLakeClients
    database: str | None = None
    workgroup: str = "primary"
    output_location: str | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 60.0

    @classmethod
    def from_handoff(cls, handoff: DraftGenerationHandoff, **kwargs) -> "RegionalLakeReader":
        return cls(clients=RegionalLakeClients.from_handoff(handoff), **kwargs)

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        schema: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        athena = self.clients.athena()
        rendered_sql = render_params(sql, list(params or []))
        start_kwargs: dict[str, Any] = {
            "QueryString": rendered_sql,
            "QueryExecutionContext": {"Database": self.database or self._database_name()},
            "WorkGroup": self.workgroup,
        }
        if self.output_location:
            start_kwargs["ResultConfiguration"] = {"OutputLocation": self.output_location}

        query_id = athena.start_query_execution(**start_kwargs)["QueryExecutionId"]
        self._wait_for_query(athena, query_id, rendered_sql)
        return self._fetch_rows(athena, query_id, schema=schema)

    def execute_one(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        schema: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        rows = self.execute(sql, params, schema=schema)
        return rows[0] if rows else None

    def _database_name(self) -> str:
        return f"outstandingai_{self.clients.region_name.replace('-', '_')}"

    def _wait_for_query(self, athena: Any, query_id: str, sql: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            execution = athena.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
            status = execution["Status"]
            state = status["State"]
            if state == "SUCCEEDED":
                return
            if state in {"FAILED", "CANCELLED"}:
                reason = status.get("StateChangeReason") or "unknown"
                raise RegionalLakeQueryError(
                    f"Athena query {query_id} {state}: {reason}; SQL={sql[:300]}"
                )
            if time.monotonic() >= deadline:
                raise RegionalLakeQueryError(
                    f"Athena query {query_id} timed out after {self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def _fetch_rows(
        self,
        athena: Any,
        query_id: str,
        *,
        schema: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        result = athena.get_query_results(QueryExecutionId=query_id)
        column_info = result["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        raw_rows = result["ResultSet"].get("Rows", [])
        rows = [coerce_row(row, column_info, schema=schema) for row in raw_rows[1:]]

        next_token = result.get("NextToken")
        while next_token:
            result = athena.get_query_results(QueryExecutionId=query_id, NextToken=next_token)
            raw_rows = result["ResultSet"].get("Rows", [])
            rows.extend(coerce_row(row, column_info, schema=schema) for row in raw_rows)
            next_token = result.get("NextToken")
        return rows
