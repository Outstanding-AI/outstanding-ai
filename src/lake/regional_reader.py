"""Region-pinned AWS clients for regional lake hydration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import DraftGenerationHandoff


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
