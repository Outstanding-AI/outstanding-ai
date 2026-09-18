"""Tests for the no-fallback OpenRouter semantic provider."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from src.llm.base import LLMProviderUnavailableError, LLMRateLimitedError, LLMStructuredOutputError
from src.llm.openrouter_provider import OpenRouterProvider


@pytest.mark.asyncio
async def test_openrouter_provider_uses_bounded_json_request_without_leaking_key() -> None:
    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "completion-id",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "provider": "test-provider",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"status":"ok"}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            },
        )

    provider = OpenRouterProvider(
        api_key="test-secret",
        model="deepseek/deepseek-v4-flash",
        base_url="https://router.test/api/v1",
        max_completion_tokens=128,
        transport=httpx.MockTransport(handler),
    )

    response = await provider.complete(
        system_prompt="Return JSON only.",
        user_prompt="Synthetic fixture only.",
        json_mode=True,
        caller="semantic_mail_test",
    )

    payload = observed["payload"]
    assert observed["url"] == "https://router.test/api/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-secret"
    assert payload["model"] == "deepseek/deepseek-v4-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 128
    assert response.content == '{"status":"ok"}'
    assert response.provider == "openrouter"
    assert response.usage["total_tokens"] == 15
    assert response.raw_response == {
        "id": "completion-id",
        "created": 1,
        "finish_reason": "stop",
        "provider": "test-provider",
    }
    assert "test-secret" not in json.dumps(response.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_openrouter_provider_treats_rate_limit_as_retryable_provider_failure() -> None:
    provider = OpenRouterProvider(
        api_key="test-secret",
        model="deepseek/deepseek-v4-flash",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(429, headers={"retry-after": "42"})
        ),
    )

    with pytest.raises(LLMRateLimitedError, match="retry_after_seconds=42"):
        await provider.complete("system", "synthetic", caller="semantic_mail_test")


@pytest.mark.asyncio
async def test_openrouter_provider_emits_only_safe_error_code_for_rejected_request() -> None:
    provider = OpenRouterProvider(
        api_key="test-secret",
        model="synthetic-model",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={"error": {"code": "schema unsupported: Synthetic Fixture"}},
            )
        ),
    )

    with pytest.raises(LLMStructuredOutputError, match="schema_unsupported_synthetic_fixture"):
        await provider.complete("system", "synthetic", caller="semantic_mail_test")


@pytest.mark.asyncio
async def test_openrouter_provider_uses_strict_schema_and_provider_privacy_controls() -> None:
    class _ResponseSchema(BaseModel):
        status: str

    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "synthetic-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"status":"ok"}'}}],
                "usage": {},
            },
        )

    provider = OpenRouterProvider(
        api_key="test-secret",
        model="synthetic-model",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

    await provider.complete(
        "Return JSON only.",
        "Synthetic fixture only.",
        response_schema=_ResponseSchema,
        caller="semantic_mail_test",
    )

    payload = observed["payload"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["provider"] == {
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }


@pytest.mark.asyncio
async def test_openrouter_provider_explicitly_disables_reasoning_over_effort() -> None:
    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "synthetic-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"status":"ok"}'}}],
                "usage": {},
            },
        )

    provider = OpenRouterProvider(
        api_key="test-secret",
        model="synthetic-model",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(handler),
    )

    await provider.complete(
        "Return JSON only.",
        "Synthetic fixture only.",
        json_mode=True,
        reasoning_effort="low",
        reasoning_enabled=False,
        caller="semantic_mail_test",
    )

    payload = observed["payload"]
    assert payload["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_openrouter_provider_does_not_fallback_after_auth_failure() -> None:
    provider = OpenRouterProvider(
        api_key="test-secret",
        model="deepseek/deepseek-v4-flash",
        base_url="https://router.test/api/v1",
        transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
    )

    with pytest.raises(LLMProviderUnavailableError, match="HTTP 401"):
        await provider.complete("system", "synthetic", caller="semantic_mail_test")
