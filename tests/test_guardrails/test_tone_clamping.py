from src.api.models.requests import CaseContext, PartyInfo
from src.guardrails.tone_clamping import ToneClampingGuardrail


def _context() -> CaseContext:
    return CaseContext(
        schema_version=4,
        party=PartyInfo(
            party_id="party-1",
            external_id="party-ext-1",
            provider_type="sage_200",
            customer_code="FELT001",
            name="Felton Energy Services Limited",
            source="sage_200",
        ),
    )


def test_final_notice_legal_pressure_requires_authorized_policy():
    results = ToneClampingGuardrail().validate(
        "This is a final notice. The matter may be referred to our legal team.",
        _context(),
        tone="final_notice",
        authorized_policies={"legal_escalation_enabled": False},
    )

    assert not results[0].passed
    assert "without policy authorization" in results[0].message


def test_final_notice_operational_follow_up_passes_without_legal_policy():
    results = ToneClampingGuardrail().validate(
        "Could you please confirm when payment can be expected?",
        _context(),
        tone="final_notice",
        authorized_policies={"legal_escalation_enabled": False},
    )

    assert results[0].passed


def test_authorized_legal_policy_allows_legal_pressure():
    results = ToneClampingGuardrail().validate(
        "If this is not paid, it may be referred to our legal team.",
        _context(),
        tone="final_notice",
        authorized_policies={"legal_escalation_enabled": True},
    )

    assert results[0].passed
