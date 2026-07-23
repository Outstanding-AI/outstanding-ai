"""Weekly overdue-report summarisation request models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WeeklyReportInvoiceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1, max_length=160)
    invoice_number: str = Field(min_length=1, max_length=160)
    invoice_date: date | None = None
    customer_po_number: str | None = Field(default=None, max_length=240)
    customer_po_source: str | None = Field(default=None, max_length=80)
    sales_order_number: str | None = Field(default=None, max_length=240)
    sales_order_status: str | None = Field(default=None, max_length=120)
    sales_order_date: date | None = None
    currency: str = Field(min_length=1, max_length=12)
    amount_due: float
    due_date: date | None = None
    days_overdue: int | None = None
    collection_status: str | None = Field(default=None, max_length=80)
    query_reason: str | None = Field(default=None, max_length=1200)
    commitment_status: str | None = Field(default=None, max_length=80)
    commitment_date: date | None = None
    commitment_amount: float | None = None
    remittance_state: str | None = Field(default=None, max_length=80)
    remittance_reference: str | None = Field(default=None, max_length=240)
    allocated_credit_amount: float | None = None
    allocated_credit_references: list[str] = Field(default_factory=list, max_length=50)
    operator_finance_update: str | None = Field(default=None, max_length=2400)
    comments_to_ai: str | None = Field(default=None, max_length=2400)


class WeeklyReportAccountCreditPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = Field(min_length=1, max_length=12)
    unapplied_credit_amount: float
    unapplied_credit_references: list[str] = Field(default_factory=list, max_length=50)
    credit_review_required: bool = False


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
    invoices: list[WeeklyReportInvoiceFact] = Field(min_length=1, max_length=1)
    account_credit_positions: list[WeeklyReportAccountCreditPosition] = Field(
        default_factory=list,
        max_length=25,
    )
    forbidden_references: list[str] = Field(default_factory=list, max_length=500)
    evidence_events: list[WeeklyReportEvidenceEvent] = Field(default_factory=list, max_length=250)
