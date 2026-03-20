"""Shared test fixtures for Outstanding AI Engine tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.models.requests import (
    BehaviorInfo,
    CaseContext,
    ClassifyRequest,
    CommunicationInfo,
    EmailContent,
    EvaluateGatesRequest,
    GenerateDraftRequest,
    ObligationInfo,
    PartyInfo,
    TouchHistory,
)


@pytest.fixture
def sample_email_content() -> EmailContent:
    """Sample inbound email for classification."""
    return EmailContent(
        subject="Re: Invoice #12345",
        body="I cannot pay right now. I lost my job last month. Can we work out a payment plan?",
        from_address="customer@example.com",
        received_at="2024-01-15T10:30:00Z",
    )


@pytest.fixture
def sample_party_info() -> PartyInfo:
    """Sample party/customer info."""
    return PartyInfo(
        party_id="party-123",
        customer_code="CUST001",
        name="Acme Corp",
        country_code="GB",
        currency="GBP",
    )


@pytest.fixture
def sample_behavior_info() -> BehaviorInfo:
    """Sample payment behavior metrics."""
    return BehaviorInfo(
        lifetime_value=50000.0,
        avg_days_to_pay=35.5,
        on_time_rate=0.65,
        segment="medium_risk",
    )


@pytest.fixture
def sample_obligations() -> list[ObligationInfo]:
    """Sample outstanding invoices."""
    return [
        ObligationInfo(
            invoice_number="INV-12345",
            original_amount=1500.0,
            amount_due=1500.0,
            due_date="2024-01-01",
            days_past_due=14,
            state="open",
        ),
        ObligationInfo(
            invoice_number="INV-12346",
            original_amount=2500.0,
            amount_due=2500.0,
            due_date="2024-01-05",
            days_past_due=10,
            state="open",
        ),
    ]


@pytest.fixture
def sample_communication_info() -> CommunicationInfo:
    """Sample communication history summary."""
    return CommunicationInfo(
        touch_count=3,
        last_touch_at="2024-01-10T09:00:00Z",
        last_touch_channel="email",
        last_sender_level=1,
        last_tone_used="friendly_reminder",
    )


@pytest.fixture
def sample_case_context(
    sample_party_info,
    sample_behavior_info,
    sample_obligations,
    sample_communication_info,
) -> CaseContext:
    """Complete case context for AI operations."""
    return CaseContext(
        party=sample_party_info,
        behavior=sample_behavior_info,
        obligations=sample_obligations,
        communication=sample_communication_info,
        recent_touches=[
            TouchHistory(
                sent_at="2024-01-10T09:00:00Z",
                tone="friendly_reminder",
                sender_level=1,
                had_response=False,
            )
        ],
        case_state="ACTIVE",
        days_in_state=30,
        broken_promises_count=1,
        active_dispute=False,
        hardship_indicated=False,
        brand_tone="professional",
        touch_cap=10,
        touch_interval_days=3,
    )


@pytest.fixture
def sample_classify_request(sample_email_content, sample_case_context) -> ClassifyRequest:
    """Complete classification request."""
    return ClassifyRequest(
        email=sample_email_content,
        context=sample_case_context,
    )


@pytest.fixture
def sample_generate_draft_request(sample_case_context) -> GenerateDraftRequest:
    """Complete draft generation request."""
    return GenerateDraftRequest(
        context=sample_case_context,
        tone="concerned_inquiry",
        objective="follow_up",
    )


@pytest.fixture
def sample_evaluate_gates_request(sample_case_context) -> EvaluateGatesRequest:
    """Complete gate evaluation request."""
    return EvaluateGatesRequest(
        context=sample_case_context,
        proposed_action="send_email",
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock = MagicMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = AsyncMock()
    return mock
