"""
Party and behavior models for AI operations.

Contains PartyInfo, BehaviorInfo, and EmailContent models used
to describe debtors and their email communications.

April 2026: switched the wrapper bases to the V3 contracts. V3 is a
strict superset of V2 for these two models (adds the remaining override
columns to PartyInfo, no semantic changes to BehaviorInfo) — V2 callers
keep validating cleanly because the V3-only fields are all Optional.
"""

import warnings
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from solvix_contracts.ai.context.v3 import BehaviorInfoV3, PartyInfoV3


class EmailContent(BaseModel):
    """Email content for classification."""

    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=50000)  # 50KB max for email body
    from_address: str = Field(..., min_length=1, max_length=320)  # RFC 5321 max email length
    from_name: Optional[str] = Field(None, max_length=200)
    received_at: Optional[datetime] = None


class PartyInfo(PartyInfoV3):
    """Party (debtor) information."""

    @model_validator(mode="after")
    def validate_party_identity(self) -> "PartyInfo":
        """Keep the AI request contract aligned with canonical provider identity."""
        if self.source != self.provider_type:
            raise ValueError("PartyInfo.source must equal provider_type")
        return self


class BehaviorInfo(BehaviorInfoV3):
    """Historical payment behavior."""

    @model_validator(mode="after")
    def normalize_deprecated_segment(self) -> "BehaviorInfo":
        """Backfill the canonical behaviour_segment during the deprecation window."""
        legacy_segment = self.__dict__.get("segment")
        if legacy_segment is not None:
            warnings.warn(
                "BehaviorInfo.segment is deprecated; use behaviour_segment instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.behaviour_segment and self.behaviour_segment != legacy_segment:
                raise ValueError(
                    "BehaviorInfo.segment must match behaviour_segment when both are provided"
                )
            if not self.behaviour_segment:
                self.behaviour_segment = legacy_segment
        return self
