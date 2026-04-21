import json
from types import SimpleNamespace

from src.config.settings import settings
from src.guardrails.entity import EntityVerificationGuardrail


def test_entity_uses_settings_max_tokens(sample_case_context, monkeypatch):
    captured = {}

    async def _fake_complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "customer_code_valid": True,
                    "customer_code_reason": "ok",
                    "party_name_valid": True,
                    "party_name_reason": "ok",
                    "issues_found": [],
                    "passed": True,
                }
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr("src.guardrails.entity.llm_client.complete", _fake_complete)

    guardrail = EntityVerificationGuardrail()
    results = guardrail._validate_entities_with_llm("Dear Acme Corp", sample_case_context)

    assert len(results) == 2
    assert captured["max_tokens"] == settings.openai_max_tokens
    assert captured["caller"] == "entity_verification"
