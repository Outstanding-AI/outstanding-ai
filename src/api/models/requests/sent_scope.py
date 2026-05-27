"""Request models for sent-draft invoice-scope analysis."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class GeneratedDraftInput(BaseModel):
    """AI-generated draft content and invoice scope before operator edits."""

    subject: Optional[str] = Field(None, max_length=2000)
    body_plain: Optional[str] = Field(None, max_length=50000)
    body_html: Optional[str] = Field(None, max_length=100000)
    invoice_refs: List[str] = Field(default_factory=list)


class SentDraftEmailInput(BaseModel):
    """Captured email that was actually sent to the customer."""

    subject: Optional[str] = Field(None, max_length=2000)
    body_plain: Optional[str] = Field(None, max_length=50000)
    body_html: Optional[str] = Field(None, max_length=100000)
    from_email: Optional[str] = Field(None, max_length=320)
    to_emails: List[str] = Field(default_factory=list)
    cc_emails: List[str] = Field(default_factory=list)
    bcc_emails: List[str] = Field(default_factory=list)
    reply_to: List[str] = Field(default_factory=list)


class SentDraftInvoiceCandidate(BaseModel):
    """One real same-party invoice that the sent email may reference."""

    obligation_id: str = Field(..., max_length=100)
    invoice_number: str = Field(..., max_length=100)
    document_no: Optional[str] = Field(None, max_length=100)
    currency_code: Optional[str] = Field(None, max_length=10)
    amount_due_native: Optional[float] = None
    amount_due_base: Optional[float] = None
    due_date: Optional[str] = Field(None, max_length=20)
    days_overdue: Optional[int] = None
    is_source_disputed: bool = False
    collection_status: Optional[str] = Field(None, max_length=100)
    generated_in_draft: bool = False


class AnalyzeSentDraftScopeRequest(BaseModel):
    """Analyze final sent email scope against generated draft and invoice candidates."""

    tenant_id: str = Field(..., max_length=64)
    party_id: str = Field(..., max_length=100)
    draft_id: str = Field(..., max_length=100)
    touch_id: Optional[str] = Field(None, max_length=100)
    provider_message_id: Optional[str] = Field(None, max_length=255)
    sent_at: Optional[datetime] = None
    generated: GeneratedDraftInput
    sent: SentDraftEmailInput
    invoice_candidates: List[SentDraftInvoiceCandidate] = Field(default_factory=list)
