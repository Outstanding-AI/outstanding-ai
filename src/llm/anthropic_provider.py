"""Anthropic Claude LLM provider.

Optional third provider alongside Vertex (default primary) and OpenAI (fallback).
Supports structured output via tool_use for guaranteed JSON.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import settings

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Import Anthropic SDK and retryable errors
try:
    import anthropic
    from anthropic import RateLimitError as AnthropicRateLimitError

    ANTHROPIC_RETRYABLE_ERRORS = (AnthropicRateLimitError,)
except ImportError:
    anthropic = None
    ANTHROPIC_RETRYABLE_ERRORS = ()


def _log_retry(retry_state):
    """Log retry attempts with structured metrics."""
    exception = retry_state.outcome.exception()
    logger.warning(
        "Anthropic retry attempt",
        extra={
            "metric_type": "llm_retry_attempt",
            "provider": "anthropic",
            "attempt": retry_state.attempt_number,
            "wait_seconds": (retry_state.next_action.sleep if retry_state.next_action else 0),
            "error": str(exception),
            "error_type": type(exception).__name__,
        },
    )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider. Optional alongside Vertex and OpenAI."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
    ):
        raise ValueError(
            "Anthropic provider is disabled until it supports no application-level max token cap"
        )
        if anthropic is None:
            raise ValueError("anthropic package not installed. Install with: pip install anthropic")

        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        self._model = model or settings.anthropic_model
        self.temperature = (
            temperature if temperature is not None else settings.anthropic_temperature
        )

        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        logger.info("Anthropic provider initialized with model=%s", self._model)

    @retry(
        retry=retry_if_exception_type(ANTHROPIC_RETRYABLE_ERRORS),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=_log_retry,
    )
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        *,
        caller: str = "unknown",
    ) -> LLMResponse:
        """Generate completion using Anthropic Claude."""
        raise ValueError(
            "Anthropic provider is disabled until it supports no application-level max token cap"
        )
        temp = temperature if temperature is not None else self.temperature
        start_time = time.perf_counter()

        messages = [{"role": "user", "content": user_prompt}]

        if response_schema:
            # Use tool_use for structured output
            schema = response_schema.model_json_schema()
            response = await self._client.messages.create(
                model=self._model,
                temperature=temp,
                system=system_prompt,
                messages=messages,
                tools=[
                    {
                        "name": "structured_output",
                        "description": "Return the result in the required format",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": "structured_output"},
            )
            # Extract tool use content
            content_text = ""
            for block in response.content:
                if block.type == "tool_use":
                    content_text = json.dumps(block.input)
                    break
        else:
            response = await self._client.messages.create(
                model=self._model,
                temperature=temp,
                system=system_prompt,
                messages=messages,
            )
            content_text = response.content[0].text if response.content else ""

        latency_ms = (time.perf_counter() - start_time) * 1000

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        logger.info(
            "Anthropic completion",
            extra={
                "metric_type": "llm_completion",
                "provider": "anthropic",
                "model": self._model,
                "latency_ms": round(latency_ms, 2),
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "caller": caller,
            },
        )

        return LLMResponse(
            content=content_text,
            model=self._model,
            provider="anthropic",
            usage=usage,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API availability."""
        try:
            await self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
            )
            return {
                "status": "healthy",
                "provider": "anthropic",
                "model": self._model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic",
                "error": str(e),
            }

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model
