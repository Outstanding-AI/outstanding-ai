"""Tests for Vertex provider and fallback wiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm.base import LLMResponse
from src.llm.factory import LLMProviderWithFallback
from src.llm.vertex_provider import VertexProvider


class _Schema(BaseModel):
    subject: str


def _fake_response(*, parsed=None, text="", prompt_tokens=12, completion_tokens=7, total_tokens=19):
    usage_metadata = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=total_tokens,
    )
    return SimpleNamespace(
        parsed=parsed,
        text=text,
        usage_metadata=usage_metadata,
        response_id="resp-123",
    )


def test_vertex_provider_builds_explicit_wif_credentials(monkeypatch, tmp_path):
    wif_config = tmp_path / "vertex-wif-config.json"
    wif_config.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/pool/providers/provider",
                "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
                "token_url": "https://sts.googleapis.com/v1/token",
                "credential_source": {"ignored": True},
                "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/test@example.com:generateAccessToken",
            }
        ),
        encoding="utf-8",
    )

    fake_creds = object()
    fake_client = MagicMock()

    monkeypatch.setattr("src.llm.vertex_provider.settings.vertex_wif_config_path", str(wif_config))
    monkeypatch.setattr("src.llm.vertex_provider.settings.vertex_project_id", "production-493814")
    monkeypatch.setattr("src.llm.vertex_provider.settings.vertex_location", "europe-west2")
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/abc")
    monkeypatch.setenv("AWS_REGION", "eu-west-2")

    with patch(
        "src.llm.vertex_provider.google_auth_aws.Credentials", return_value=fake_creds
    ) as mock_credentials:
        with patch("src.llm.vertex_provider.Client", return_value=fake_client) as mock_client:
            VertexProvider()

    assert not mock_client.called
    kwargs = mock_credentials.call_args.kwargs
    assert kwargs["audience"].endswith("/providers/provider")
    assert kwargs["service_account_impersonation_url"].startswith(
        "https://iamcredentials.googleapis.com/"
    )
    assert kwargs["aws_security_credentials_supplier"].__class__.__name__ == "EcsTaskRoleSupplier"
    assert "credential_source" not in kwargs


def test_vertex_provider_uses_adc_off_ecs(monkeypatch):
    fake_creds = object()

    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)

    with patch(
        "src.llm.vertex_provider.google_auth_default", return_value=(fake_creds, "local-proj")
    ) as mock_adc:
        with patch("src.llm.vertex_provider.Client"):
            provider = VertexProvider()

    assert provider._credentials is fake_creds
    mock_adc.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_complete_uses_structured_output(monkeypatch):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(
        return_value=_fake_response(parsed={"subject": "Hi"})
    )
    fake_client.aio.aclose = AsyncMock()

    monkeypatch.setattr(VertexProvider, "_build_credentials", lambda self: object())

    with patch("src.llm.vertex_provider.Client", return_value=fake_client):
        provider = VertexProvider()
        response = await provider.complete(
            system_prompt="sys",
            user_prompt="user",
            response_schema=_Schema,
        )

    assert response.provider == "vertex"
    assert response.model == "gemini-2.5-flash"
    assert json.loads(response.content) == {"subject": "Hi"}
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}

    config = fake_client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is _Schema
    assert getattr(config, "max_output_tokens", None) is None
    fake_client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_vertex_complete_json_mode_uses_text(monkeypatch):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(
        return_value=_fake_response(text='{"ok": true}')
    )
    fake_client.aio.aclose = AsyncMock()

    monkeypatch.setattr(VertexProvider, "_build_credentials", lambda self: object())

    with patch("src.llm.vertex_provider.Client", return_value=fake_client):
        provider = VertexProvider()
        response = await provider.complete(system_prompt="sys", user_prompt="user", json_mode=True)

    assert response.content == '{"ok": true}'
    config = fake_client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert getattr(config, "max_output_tokens", None) is None
    fake_client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_vertex_complete_uses_fresh_client_per_call(monkeypatch):
    clients = []

    def _make_client(*args, **kwargs):
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=_fake_response(text="ok"))
        client.aio.aclose = AsyncMock()
        clients.append(client)
        return client

    monkeypatch.setattr(VertexProvider, "_build_credentials", lambda self: object())

    with patch("src.llm.vertex_provider.Client", side_effect=_make_client) as mock_client:
        provider = VertexProvider()
        await provider.complete(system_prompt="sys", user_prompt="user")
        await provider.complete(system_prompt="sys", user_prompt="user")

    assert mock_client.call_count == 2
    assert clients[0] is not clients[1]
    clients[0].aio.aclose.assert_awaited_once()
    clients[1].aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_vertex_complete_logs_caller_on_failure(monkeypatch, caplog):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("boom"))
    fake_client.aio.aclose = AsyncMock()

    monkeypatch.setattr(VertexProvider, "_build_credentials", lambda self: object())

    with patch("src.llm.vertex_provider.Client", return_value=fake_client):
        provider = VertexProvider()
        with pytest.raises(RuntimeError):
            await provider.complete(
                system_prompt="sys", user_prompt="user", caller="draft_generation"
            )

    matching = [record for record in caplog.records if record.message == "Vertex provider error"]
    assert matching
    assert matching[-1].caller == "draft_generation"
    assert matching[-1].error_type == "RuntimeError"
    fake_client.aio.aclose.assert_awaited_once()


def test_vertex_provider_thread_isolation_with_fresh_client(monkeypatch):
    class _LoopBoundGenerate:
        def __init__(self, bound_loop):
            self.bound_loop = bound_loop

        async def __call__(self, **kwargs):
            if asyncio.get_running_loop() is not self.bound_loop:
                raise RuntimeError("bound to a different event loop")
            return _fake_response(text='{"ok": true}')

    class _LoopBoundClient:
        def __init__(self, *args, **kwargs):
            loop = asyncio.get_event_loop()
            self.aio = SimpleNamespace(
                models=SimpleNamespace(generate_content=_LoopBoundGenerate(loop)),
                aclose=AsyncMock(),
            )

    monkeypatch.setattr(VertexProvider, "_build_credentials", lambda self: object())

    with patch("src.llm.vertex_provider.Client", _LoopBoundClient):
        provider = VertexProvider()

        def _run_once():
            response = asyncio.run(provider.complete(system_prompt="sys", user_prompt="user"))
            return response.content

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: _run_once(), range(5)))

    assert results == ['{"ok": true}'] * 5


@pytest.mark.asyncio
async def test_factory_falls_back_to_openai():
    fallback_response = LLMResponse(
        content='{"ok": true}',
        model="gpt-5-mini",
        provider="openai",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    with patch("src.llm.factory.VertexProvider") as mock_vertex:
        with patch("src.llm.factory.OpenAIProvider") as mock_openai:
            mock_vertex.return_value.complete = AsyncMock(side_effect=RuntimeError("vertex down"))
            mock_vertex.return_value.provider_name = "vertex"
            mock_openai.return_value.complete = AsyncMock(return_value=fallback_response)
            mock_openai.return_value.provider_name = "openai"
            client = LLMProviderWithFallback(primary_provider="vertex", fallback_provider="openai")

            response = await client.complete("sys", "user", caller="draft_generation")

    assert response.provider == "openai"
    assert response.is_fallback is True
    assert client.fallback_count == 1
    assert client.get_failure_metrics()["primary_failures_by_caller"] == {"draft_generation": 1}
