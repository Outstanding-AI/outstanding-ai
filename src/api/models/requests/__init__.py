"""
Request models for Outstanding AI Engine API.

Re-exports all request models from sub-modules for backward compatibility.
"""

from .context import (
    ActualSentScopeHistory,
    CaseContext,
    CommunicationInfo,
    CommunicationTrackingInfo,
    IndustryInfo,
    LaneContextInfo,
    ObligationInfo,
    PromiseHistory,
    RemittanceHistory,
    TouchHistory,
)
from .party import BehaviorInfo, EmailContent, PartyInfo
from .persona import (
    GeneratePersonaRequest,
    PersonaContact,
    RefinePersonaRequest,
    SenderContext,
    SenderPerformanceStats,
    SenderPersona,
)
from .sent_scope import (
    AnalyzeSentDraftScopeRequest,
    GeneratedDraftInput,
    SentDraftEmailInput,
    SentDraftInvoiceCandidate,
)
from .validation import (
    PROMPT_INJECTION_PATTERNS,
    ClassifyRequest,
    CollectionChainIdentificationRequest,
    CollectionChainRoutingRequest,
    CollectionEmailEventRequest,
    CollectionEmailFactExtractionRequest,
    GenerateDraftRequest,
    HistoricalCollectionThreadRequest,
)
from .weekly_report import (
    WeeklyOverdueReportSummaryRequest,
    WeeklyReportEvidenceEvent,
    WeeklyReportInvoiceFact,
)

__all__ = [
    # Party / Behavior
    "EmailContent",
    "PartyInfo",
    "BehaviorInfo",
    # Context
    "ObligationInfo",
    "CommunicationInfo",
    "CommunicationTrackingInfo",
    "TouchHistory",
    "PromiseHistory",
    "RemittanceHistory",
    "ActualSentScopeHistory",
    "IndustryInfo",
    "LaneContextInfo",
    "CaseContext",
    # Persona
    "SenderPersona",
    "PersonaContact",
    "GeneratePersonaRequest",
    "SenderPerformanceStats",
    "RefinePersonaRequest",
    "SenderContext",
    # Sent draft scope analysis
    "AnalyzeSentDraftScopeRequest",
    "GeneratedDraftInput",
    "SentDraftEmailInput",
    "SentDraftInvoiceCandidate",
    # Validation / Requests
    "ClassifyRequest",
    "CollectionChainIdentificationRequest",
    "CollectionChainRoutingRequest",
    "CollectionEmailEventRequest",
    "CollectionEmailFactExtractionRequest",
    "HistoricalCollectionThreadRequest",
    "GenerateDraftRequest",
    "PROMPT_INJECTION_PATTERNS",
    "WeeklyOverdueReportSummaryRequest",
    "WeeklyReportEvidenceEvent",
    "WeeklyReportInvoiceFact",
]
