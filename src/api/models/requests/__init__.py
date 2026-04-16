"""
Request models for Outstanding AI Engine API.

Re-exports all request models from sub-modules for backward compatibility.
"""

from .context import (
    CaseContext,
    CommunicationInfo,
    CommunicationTrackingInfo,
    IndustryInfo,
    ObligationInfo,
    PromiseHistory,
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
from .validation import (
    PROMPT_INJECTION_PATTERNS,
    ClassifyRequest,
    EvaluateGatesBatchRequest,
    EvaluateGatesRequest,
    GenerateDraftRequest,
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
    "IndustryInfo",
    "CaseContext",
    # Persona
    "SenderPersona",
    "PersonaContact",
    "GeneratePersonaRequest",
    "SenderPerformanceStats",
    "RefinePersonaRequest",
    "SenderContext",
    # Validation / Requests
    "ClassifyRequest",
    "GenerateDraftRequest",
    "EvaluateGatesRequest",
    "EvaluateGatesBatchRequest",
    "PROMPT_INJECTION_PATTERNS",
]
