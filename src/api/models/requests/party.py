"""
Party and behavior models for AI operations.

Contains PartyInfo, BehaviorInfo, and EmailContent models used
to describe debtors and their email communications.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
    avg_days_to_pay: Optional[float] = None
    on_time_rate: Optional[float] = None
    partial_payment_rate: Optional[float] = None
    segment: Optional[str] = None
    # Enhanced behaviour context
    behaviour_profile: Optional[dict] = None
    behaviour_segment: Optional[str] = None
