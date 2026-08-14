"""Atomic compact-current epoch validation and SQL rendering for AI reads."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from solvix_contracts.datalake import (
        CompactCurrentEpochMemberV1 as _SharedCompactCurrentEpochMemberV1,
    )
    from solvix_contracts.datalake import CompactCurrentEpochV1 as _SharedCompactCurrentEpochV1
    from solvix_contracts.datalake import (
        compact_current_manifest_digest as _shared_manifest_digest,
    )
    from solvix_contracts.datalake import compact_current_relation_for as _shared_relation_for
    from solvix_contracts.datalake import compact_current_relations as _shared_relations
except ImportError:  # Contracts release is deployed backend-first.
    _SharedCompactCurrentEpochMemberV1 = None
    _SharedCompactCurrentEpochV1 = None
    _shared_manifest_digest = None
    _shared_relation_for = None
    _shared_relations = None

from .materialized_current_registry import (
    AI_CONTEXT_CURRENT_VIEWS,
    compact_epoch_table_name,
    logical_relation_for_current_view,
)

COMPACT_CURRENT_EPOCH_SCHEMA_VERSION = "compact-current-epoch.v1"
CurrentReadMode = Literal["off", "enforced", "compact_only"]

_SQL_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,159}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{32,128}$")
_FROM_OR_JOIN_RELATION_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:(?:"?[a-z][a-z0-9_]*"?)\.)?'
    r'(?P<relation>"?[a-z][a-z0-9_]*_current"?)(?![a-z0-9_])',
    re.IGNORECASE,
)

# Names ending in ``_current`` are not assumed to be views merely because of
# their suffix.  Keep physical exceptions explicit and catalog-backed.
PHYSICAL_CURRENT_RELATIONS: frozenset[str] = frozenset({"gold_dashboard_summary_current"})


def _non_empty(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _identifier(value: str, *, field_name: str) -> str:
    normalized = _non_empty(value, field_name=field_name)
    if not _SQL_IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase SQL identifier")
    return normalized


class CompactCurrentEpochMemberV1(BaseModel):
    """One immutable logical relation inside a committed compact epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_relation: str = Field(..., max_length=160)
    materialized_table: str = Field(..., max_length=160)
    compact_epoch_id: str = Field(..., max_length=255)
    canonical_columns: tuple[str, ...]

    @field_validator("logical_relation", "materialized_table")
    @classmethod
    def require_identifier(cls, value: str, info) -> str:
        return _identifier(value, field_name=info.field_name)

    @field_validator("compact_epoch_id")
    @classmethod
    def require_epoch_id(cls, value: str) -> str:
        return _non_empty(value, field_name="compact_epoch_id")

    @field_validator("canonical_columns")
    @classmethod
    def require_canonical_columns(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _identifier(column, field_name="canonical_columns") for column in columns
        )
        if not normalized:
            raise ValueError("canonical_columns must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("canonical_columns must be unique and ordered")
        if "tenant_id" not in normalized:
            raise ValueError("canonical_columns must include tenant_id")
        if "compact_epoch_id" in normalized:
            raise ValueError("compact_epoch_id must not leak into canonical columns")
        return normalized

    @model_validator(mode="after")
    def require_expected_table(self) -> "CompactCurrentEpochMemberV1":
        canonical_view = f"{self.logical_relation}_current"
        expected = compact_epoch_table_name(canonical_view)
        if self.materialized_table != expected:
            raise ValueError(
                f"materialized_table for {self.logical_relation!r} must be {expected!r}"
            )
        if _shared_relation_for is not None:
            relation = _shared_relation_for(self.logical_relation)
            if relation is None:
                raise ValueError(f"unknown compact logical relation: {self.logical_relation!r}")
            if self.canonical_columns != tuple(relation.canonical_columns):
                raise ValueError(
                    f"canonical_columns for {self.logical_relation!r} do not match the shared contract"
                )
        return self


class CompactCurrentEpochV1(BaseModel):
    """Versioned backend-issued manifest for one committed compact epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[COMPACT_CURRENT_EPOCH_SCHEMA_VERSION]
    tenant_id: str = Field(..., max_length=255)
    epoch_id: str = Field(..., max_length=255)
    manifest_digest: str = Field(..., max_length=128)
    committed_at: datetime
    members: tuple[CompactCurrentEpochMemberV1, ...]

    @field_validator("tenant_id", "epoch_id")
    @classmethod
    def require_identity(cls, value: str, info) -> str:
        return _non_empty(value, field_name=info.field_name)

    @field_validator("manifest_digest")
    @classmethod
    def require_manifest_digest(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not _HEX_DIGEST_RE.fullmatch(normalized):
            raise ValueError("manifest_digest must be a lowercase hexadecimal digest")
        return normalized

    @field_validator("committed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("committed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def require_coherent_members(self) -> "CompactCurrentEpochV1":
        if not self.members:
            raise ValueError("members must not be empty")
        logical_relations = [member.logical_relation for member in self.members]
        if len(set(logical_relations)) != len(logical_relations):
            raise ValueError("members must contain unique logical_relation values")
        # An atomic pointer may reuse an unchanged immutable member from a
        # prior epoch.  Its member partition id is therefore intentionally
        # independent from the outer pointer epoch id; non-empty validation
        # is enforced by the member model and every member remains bound to
        # its own relation/table/schema contract below.
        if _shared_manifest_digest is not None:
            expected_digest = str(_shared_manifest_digest())
            if self.manifest_digest != expected_digest:
                raise ValueError(
                    "manifest_digest does not match the shared compact-current contract"
                )
        if _shared_relations is not None:
            expected_relations = {
                str(relation.logical_relation) for relation in _shared_relations()
            }
            actual_relations = set(logical_relations)
            missing = sorted(expected_relations - actual_relations)
            unexpected = sorted(actual_relations - expected_relations)
            if missing or unexpected:
                raise ValueError(
                    "epoch members do not match the complete shared contract: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        return self

    def member_by_logical_relation(self) -> dict[str, CompactCurrentEpochMemberV1]:
        return {member.logical_relation: member for member in self.members}


# Consume the released shared payload models as soon as the backend-first
# contracts tag is installed.  The local definitions keep dark images running
# on the preceding release, but compact-only remains unavailable in that state.
if _SharedCompactCurrentEpochMemberV1 is not None:
    CompactCurrentEpochMemberV1 = _SharedCompactCurrentEpochMemberV1
if _SharedCompactCurrentEpochV1 is not None:
    CompactCurrentEpochV1 = _SharedCompactCurrentEpochV1


class CompactCurrentEpochUnavailable(RuntimeError):  # noqa: N818 - cross-runtime error code
    """The caller requested compact-only reads without a complete valid epoch."""


def validate_ai_context_epoch(
    epoch: CompactCurrentEpochV1 | None,
    *,
    tenant_id: str,
) -> CompactCurrentEpochV1:
    """Require a tenant-matched epoch containing every AI hydration relation."""

    if epoch is None:
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch is required in compact_only mode"
        )
    if epoch.tenant_id != str(tenant_id):
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch tenant_id does not match the handoff tenant"
        )
    if _shared_relations is None:
        raise CompactCurrentEpochUnavailable(
            "compact_only requires the released shared compact-current contract"
        )
    relations = tuple(_shared_relations())
    members = {str(member.logical_relation): member for member in epoch.members}
    expected_digest = str(_shared_manifest_digest())
    if str(epoch.manifest_digest) != expected_digest:
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch manifest_digest does not match the shared contract"
        )
    if epoch.committed_at.tzinfo is None or epoch.committed_at.utcoffset() is None:
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch committed_at must include a timezone"
        )
    expected_relations = {str(relation.logical_relation) for relation in relations}
    missing_epoch_relations = sorted(expected_relations - set(members))
    unexpected_epoch_relations = sorted(set(members) - expected_relations)
    if missing_epoch_relations or unexpected_epoch_relations:
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch does not contain the complete shared contract: "
            f"missing={missing_epoch_relations}, unexpected={unexpected_epoch_relations}"
        )
    for relation in relations:
        member = members[str(relation.logical_relation)]
        if str(member.materialized_table) != str(relation.materialized_table):
            raise CompactCurrentEpochUnavailable(
                f"compact_current_epoch table mismatch for {relation.logical_relation!r}"
            )
        if tuple(member.canonical_columns) != tuple(relation.canonical_columns):
            raise CompactCurrentEpochUnavailable(
                f"compact_current_epoch columns mismatch for {relation.logical_relation!r}"
            )
        if not str(member.compact_epoch_id).strip():
            raise CompactCurrentEpochUnavailable(
                f"compact_current_epoch member partition missing for {relation.logical_relation!r}"
            )
    required = {
        logical_relation_for_current_view(view_name) for view_name in AI_CONTEXT_CURRENT_VIEWS
    }
    missing = sorted(required - set(members))
    if missing:
        raise CompactCurrentEpochUnavailable(
            "compact_current_epoch is missing AI relations: " + ", ".join(missing)
        )
    return epoch


def _quote_identifier(value: str) -> str:
    return f'"{_identifier(value, field_name="SQL identifier")}"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def compact_epoch_source(
    epoch: CompactCurrentEpochV1,
    *,
    tenant_id: str,
    current_view: str,
) -> str:
    """Render one epoch-filtered relation without exposing partition columns."""

    logical_relation = logical_relation_for_current_view(current_view)
    member = next(
        (
            candidate
            for candidate in epoch.members
            if str(candidate.logical_relation) == logical_relation
        ),
        None,
    )
    if member is None:
        raise CompactCurrentEpochUnavailable(
            f"compact_current_epoch has no member for {logical_relation!r}"
        )
    columns = ", ".join(_quote_identifier(column) for column in member.canonical_columns)
    table = _quote_identifier(member.materialized_table)
    alias = _quote_identifier(f"compact_{logical_relation}")
    return (
        f"(SELECT {columns} FROM {table} "
        f'WHERE "tenant_id" = {_sql_literal(tenant_id)} '
        f'AND "compact_epoch_id" = {_sql_literal(member.compact_epoch_id)}) AS {alias}'
    )


def unresolved_virtual_current_relations(sql: str) -> tuple[str, ...]:
    """Return direct ``FROM``/``JOIN`` current-view tokens in executable SQL."""

    matches = {
        match.group("relation").strip('"').lower()
        for match in _FROM_OR_JOIN_RELATION_RE.finditer(str(sql or ""))
    }
    return tuple(sorted(matches - PHYSICAL_CURRENT_RELATIONS))


__all__ = [
    "COMPACT_CURRENT_EPOCH_SCHEMA_VERSION",
    "CompactCurrentEpochMemberV1",
    "CompactCurrentEpochUnavailable",
    "CompactCurrentEpochV1",
    "CurrentReadMode",
    "PHYSICAL_CURRENT_RELATIONS",
    "compact_epoch_source",
    "unresolved_virtual_current_relations",
    "validate_ai_context_epoch",
]
