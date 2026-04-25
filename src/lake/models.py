"""Pydantic models for the regional draft-generation handoff."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_AWS_REGION_RE = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")


def _non_empty(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


class DraftCandidate(BaseModel):
    """One immutable draft-generation candidate listed in a staging manifest."""

    model_config = ConfigDict(extra="forbid")

    party_id: str = Field(..., max_length=255)
    lane_id: str = Field(..., max_length=255)
    sync_run_id: str = Field(..., max_length=255)
    candidate_id: str = Field(..., max_length=255)

    @field_validator("party_id", "lane_id", "sync_run_id", "candidate_id")
    @classmethod
    def require_non_empty_identity(cls, value: str, info) -> str:
        return _non_empty(value, field_name=info.field_name)


class DraftGenerationHandoff(BaseModel):
    """Backend-to-AI handoff for regional lake-hydrated draft generation."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., max_length=255)
    sync_run_id: str = Field(..., max_length=255)
    manifest_uri: str = Field(..., max_length=2048)
    data_lake_region: str = Field(..., max_length=32)

    @field_validator("tenant_id", "sync_run_id", "manifest_uri", "data_lake_region")
    @classmethod
    def require_non_empty_fields(cls, value: str, info) -> str:
        return _non_empty(value, field_name=info.field_name)

    @field_validator("manifest_uri")
    @classmethod
    def require_s3_manifest_uri(cls, value: str) -> str:
        if not value.startswith("s3://"):
            raise ValueError("manifest_uri must be an s3:// URI")
        return value

    @field_validator("data_lake_region")
    @classmethod
    def require_explicit_aws_region(cls, value: str) -> str:
        if not _AWS_REGION_RE.fullmatch(value):
            raise ValueError("data_lake_region must be an explicit AWS region")
        return value

    @model_validator(mode="after")
    def reject_regionless_handoff(self) -> "DraftGenerationHandoff":
        # This intentionally has no fallback to AWS_REGION. Residency is part
        # of the payload contract, not an AI task runtime default.
        if not self.data_lake_region:
            raise ValueError("data_lake_region is required")
        return self
