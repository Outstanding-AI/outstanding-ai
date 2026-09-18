"""OpenRouter provider for the evidence-only semantic-mail workload.

This adapter uses OpenRouter's OpenAI-compatible chat-completions endpoint but
is deliberately a distinct provider.  In particular, semantic-mail calls do
not inherit the application's Vertex -> OpenAI fallback policy: a provider
failure is returned to the semantic workflow as a controlled failure/defer.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel

from src.config.settings import settings

from .base import (
    BaseLLMProvider,
    LLMProviderUnavailableError,
    LLMRateLimitedError,
    LLMResponse,
    LLMStructuredOutputError,
)

logger = logging.getLogger(__name__)


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    """Normalize OpenRouter's OpenAI-compatible usage payload."""

    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(
        completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": min(max(0, reasoning_tokens), max(0, completion_tokens)),
        "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
    }


def _estimated_input_tokens(*texts: str) -> int:
    """Conservatively bound a request before it reaches a routed provider."""

    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    return max(1, (byte_count + 3) // 4)


def _text_content(choice: dict[str, Any]) -> str:
    """Return a text completion without retaining the provider's full payload."""

    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise LLMStructuredOutputError("OpenRouter completion has no message object")
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    raise LLMStructuredOutputError("OpenRouter completion has no text content")


def _safe_provider_error_code(response: httpx.Response) -> str:
    """Expose a bounded provider error code without retaining request/response bodies."""

    try:
        payload = response.json()
    except ValueError:
        return "unparseable_error"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        candidate = error.get("code") or error.get("type")
    elif isinstance(error, str):
        candidate = error
    else:
        candidate = None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(candidate or "unknown_error").lower())
    return normalized[:80] or "unknown_error"


class OpenRouterProvider(BaseLLMProvider):
    """Direct, no-fallback OpenRouter chat-completions provider."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_input_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        allow_fallbacks: bool | None = None,
        require_parameters: bool | None = None,
        data_collection: str | None = None,
        zdr: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key or settings.openrouter_api_key
        self._model = model or settings.openrouter_mail_semantic_primary_model
        self._base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self._timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.openrouter_mail_semantic_timeout_seconds
        )
        self._max_completion_tokens = int(
            max_completion_tokens
            if max_completion_tokens is not None
            else settings.openrouter_mail_semantic_max_completion_tokens
        )
        self._max_input_tokens = int(
            max_input_tokens
            if max_input_tokens is not None
            else settings.openrouter_mail_semantic_max_input_tokens
        )
        self._allow_fallbacks = (
            settings.openrouter_mail_semantic_allow_fallbacks
            if allow_fallbacks is None
            else allow_fallbacks
        )
        self._require_parameters = (
            settings.openrouter_mail_semantic_require_parameters
            if require_parameters is None
            else require_parameters
        )
        self._data_collection = (
            settings.openrouter_mail_semantic_data_collection
            if data_collection is None
            else data_collection
        )
        self._zdr = settings.openrouter_mail_semantic_zdr if zdr is None else zdr
        self._transport = transport

        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY not provided (set via environment or .env file)")
        if not self._model:
            raise ValueError("OpenRouter semantic model is not configured")
        if not self._base_url.startswith("https://"):
            raise ValueError("OpenRouter base URL must use HTTPS")
        if self._timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        if self._max_completion_tokens <= 0:
            raise ValueError("OpenRouter max completion tokens must be positive")
        if self._max_input_tokens <= 0:
            raise ValueError("OpenRouter max input tokens must be positive")
        if self._data_collection not in {"allow", "deny"}:
            raise ValueError("OpenRouter data collection policy must be allow or deny")

        logger.info(
            "Initialized OpenRouter semantic provider",
            extra={"provider": self.provider_name, "model": self._model},
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        *,
        caller: str = "unknown",
    ) -> LLMResponse:
        """Call OpenRouter with a bounded, content-free audit response."""

        estimated_input_tokens = _estimated_input_tokens(system_prompt, user_prompt)
        if estimated_input_tokens > self._max_input_tokens:
            raise LLMStructuredOutputError(
                "OpenRouter semantic request exceeds its configured input-token budget"
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self._max_completion_tokens,
            "provider": {
                "allow_fallbacks": self._allow_fallbacks,
                "require_parameters": self._require_parameters,
                "data_collection": self._data_collection,
                "zdr": self._zdr,
            },
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": reasoning_enabled}
        elif reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(self._timeout_seconds)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise LLMProviderUnavailableError("OpenRouter request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderUnavailableError("OpenRouter request failed") from exc

        if response.status_code == 429:
            raise LLMRateLimitedError("OpenRouter rate limited the semantic request")
        if response.status_code in {400, 422}:
            raise LLMStructuredOutputError(
                f"OpenRouter rejected the semantic request: {_safe_provider_error_code(response)}"
            )
        if response.status_code >= 400:
            raise LLMProviderUnavailableError(
                f"OpenRouter semantic request failed with HTTP {response.status_code}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise LLMStructuredOutputError("OpenRouter returned non-JSON data") from exc
        if not isinstance(response_payload, dict):
            raise LLMStructuredOutputError("OpenRouter returned an invalid response payload")
        choices = response_payload.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMStructuredOutputError("OpenRouter returned no completion choices")

        content = _text_content(choices[0])
        response_model = str(response_payload.get("model") or self._model)
        response_metadata = response_payload.get("provider")
        usage = _usage(response_payload)
        safe_invocation_config = {
            "base_url": self._base_url,
            "json_mode": json_mode,
            "response_schema": response_schema.__name__ if response_schema else None,
            "max_input_tokens": self._max_input_tokens,
            "max_completion_tokens": self._max_completion_tokens,
            "reasoning_effort": reasoning_effort,
            "reasoning_enabled": reasoning_enabled,
            "allow_fallbacks": self._allow_fallbacks,
            "require_parameters": self._require_parameters,
            "data_collection": self._data_collection,
            "zdr": self._zdr,
        }
        raw_response = {
            "id": str(response_payload.get("id") or ""),
            "created": response_payload.get("created"),
            "finish_reason": choices[0].get("finish_reason"),
            "provider": response_metadata if isinstance(response_metadata, str) else None,
        }

        logger.info(
            "OpenRouter semantic request completed",
            extra={
                "caller": caller,
                "metric_type": "llm_call",
                "provider": self.provider_name,
                "model": response_model,
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "success": True,
            },
        )
        try:
            httpx_version = version("httpx")
        except PackageNotFoundError:
            httpx_version = None
        return LLMResponse(
            content=content,
            model=response_model,
            provider=self.provider_name,
            usage=usage,
            raw_response=raw_response,
            model_invocation_config=safe_invocation_config,
            sdk_library="httpx",
            sdk_version=httpx_version,
        )

    async def health_check(self) -> dict[str, Any]:
        """Run a minimal synthetic health request; never use mail content."""

        try:
            response = await self.complete(
                system_prompt="Return JSON only.",
                user_prompt='{"health":"reply with ok"}',
                json_mode=True,
                caller="openrouter_semantic_health_check",
            )
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self._model,
                "test_response": response.content[:20],
            }
        except Exception as exc:
            logger.warning(
                "OpenRouter semantic health check failed",
                extra={"provider": self.provider_name, "error_type": type(exc).__name__},
            )
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self._model,
                "error": type(exc).__name__,
            }
