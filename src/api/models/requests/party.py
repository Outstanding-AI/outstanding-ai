"""
Party and behavior models for AI operations.

Contains PartyInfo, BehaviorInfo, and EmailContent models used
to describe debtors and their email communications.
"""

import warnings
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EmailContent(BaseModel):
    """Email content for classification."""

    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=50000)  # 50KB max for email body
    from_address: str = Field(..., min_length=1, max_length=320)  # RFC 5321 max email length
    from_name: Optional[str] = Field(None, max_length=200)
    received_at: Optional[datetime] = None


class PartyInfo(BaseModel):
    """Party (debtor) information."""

    # Flexible validation for external IDs (come from accounting software like Sage)
    party_id: str = Field(..., min_length=1, max_length=100)
    external_id: Optional[str] = Field(None, min_length=1, max_length=100)
    provider_type: Optional[str] = Field(None, min_length=1, max_length=64)
    customer_code: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=500)
    country_code: Optional[str] = None
    currency: str = Field("GBP", max_length=10)
    credit_limit: Optional[float] = None
    on_hold: bool = False

    # Debtor-level override fields (NEW)
    relationship_tier: str = Field("standard", max_length=50)  # vip, standard, high_risk
    tone_override: Optional[str] = None  # friendly, professional, firm (overrides brand_tone)
    grace_days_override: Optional[int] = None  # Overrides tenant grace_days
    touch_cap_override: Optional[int] = None  # Overrides tenant touch_cap
    do_not_contact_until: Optional[str] = None  # ISO date YYYY-MM-DD
    monthly_touch_count: int = 0  # Touches this month (for monthly cap reset)
    is_verified: bool = True  # False for placeholder parties from unknown emails
    source: str = Field("sage", max_length=50)  # sage, email_inbound, manual

    # Customer segmentation
    customer_type: Optional[str] = Field(None, description="individual / business / unclassified")
    size_bucket: Optional[str] = Field(None, description="large / medium / small")


class BehaviorInfo(BaseModel):
    """Historical payment behavior."""

    lifetime_value: Optional[float] = None
    total_collected: Optional[float] = None
    avg_days_to_pay: Optional[float] = None
    on_time_rate: Optional[float] = None
    partial_payment_rate: Optional[float] = None
    segment: Optional[str] = Field(
        default=None,
        deprecated="Use behaviour_segment instead.",
    )
    # Enhanced behaviour context
    behaviour_profile: Optional[dict] = None
    behaviour_segment: Optional[str] = None

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
