"""Weekly overdue-report summarisation request models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WeeklyReportInvoiceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1, max_length=160)
    invoice_number: str = Field(min_length=1, max_length=160)
    currency: str = Field(min_length=1, max_length=12)
    amount_due: float
    due_date: date | None = None
    days_overdue: int | None = None
    collection_status: str | None = Field(default=None, max_length=80)
    query_reason: str | None = Field(default=None, max_length=1200)
    remittance_state: str | None = Field(default=None, max_length=80)
    remittance_reference: str | None = Field(default=None, max_length=240)
    operator_finance_update: str | None = Field(default=None, max_length=2400)
    comments_to_ai: str | None = Field(default=None, max_length=2400)


class WeeklyReportEvidenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=200)
    occurred_at: datetime
    event_type: Literal["email", "operator_note"]
    direction: Literal["inbound", "outbound", "internal"]
    subject: str | None = Field(default=None, max_length=1000)
    authored_text: str = Field(default="", max_length=12000)
    obligation_ids: list[str] = Field(default_factory=list, max_length=100)


class WeeklyOverdueReportSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    party_id: str = Field(min_length=1, max_length=160)
    account_code: str = Field(min_length=1, max_length=160)
    customer_name: str = Field(min_length=1, max_length=500)
    reporting_window_start: date
    reporting_window_end: date
    generated_at: datetime
    evidence_truncated: bool = False
    invoices: list[WeeklyReportInvoiceFact] = Field(min_length=1, max_length=250)
    evidence_events: list[WeeklyReportEvidenceEvent] = Field(default_factory=list, max_length=250)
