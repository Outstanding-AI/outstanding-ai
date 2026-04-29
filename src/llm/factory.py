"""LLM provider factory with automatic fallback."""

import logging
import time

from src.config.settings import settings

from .base import (
    LLMFallbackExhaustedError,
    LLMProviderUnavailableError,
    LLMRateLimitedError,
    LLMResponse,
)
from .openai_provider import OpenAIProvider
from .vertex_provider import VertexProvider

logger = logging.getLogger(__name__)


class LLMProviderWithFallback:
    """
    LLM provider with automatic fallback from Vertex → OpenAI.
    """

    def __init__(self, primary_provider: str = None, fallback_provider: str = "openai"):
        self.primary_provider_name = primary_provider or settings.llm_provider
        self.fallback_provider_name = (
            None if fallback_provider == self.primary_provider_name else fallback_provider
        )

        # Lazy initialization - providers created on first use
        self._primary = None
        self._fallback = None
        self.fallback_count = 0
        self._primary_failures_by_caller: dict[str, int] = {}
        self._cooldowns: dict[tuple[str, str], float] = {}

        logger.info(
            "LLM factory created with primary=%s, fallback=%s",
            self.primary_provider_name,
            self.fallback_provider_name,
        )

    @property
    def primary(self):
        """Lazy-initialize primary provider."""
        if self._primary is None:
            self._primary = self._create_provider(self.primary_provider_name)
        return self._primary

    @property
    def fallback(self):
        """Lazy-initialize fallback provider."""
        if self._fallback is None and self.fallback_provider_name:
            try:
                self._fallback = self._create_provider(self.fallback_provider_name)
            except ValueError as e:
                # API key not configured - disable fallback gracefully
                logger.warning("Fallback provider unavailable: %s", e)
                self.fallback_provider_name = None  # Disable fallback
                return None
        return self._fallback

    @property
    def fallback_enabled(self):
        return self.fallback_provider_name is not None

    def _create_provider(self, name: str):
        """Create a provider instance by name."""
        if name == "vertex":
            return VertexProvider(
                model=settings.vertex_model,
                temperature=settings.vertex_temperature,
            )
        if name == "openai":
            return OpenAIProvider(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
            )
        if name == "anthropic":
            raise ValueError(
                "Anthropic provider is disabled until it supports no application-level max token cap"
            )
        raise ValueError(f"Unknown provider: {name}")

    async def complete(
        self, system_prompt: str, user_prompt: str, *, caller: str = "unknown", **kwargs
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Tries primary provider first, falls back to secondary on failure.
        """
        start_time = time.perf_counter()

        try:
            self._raise_if_cooling_down(self.primary.provider_name, caller)
            response = await self.primary.complete(
                system_prompt,
                user_prompt,
                caller=caller,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM request completed",
                extra={
                    "metric_type": "llm_factory_call",
                    "provider": response.provider,
                    "model": response.model,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": response.usage.get("prompt_tokens", 0),
                    "output_tokens": response.usage.get("completion_tokens", 0),
                    "total_tokens": response.usage.get("total_tokens", 0),
                    "success": True,
                    "used_fallback": False,
                    "caller": caller,
                },
            )
            return response
        except Exception as e:
            primary_latency_ms = (time.perf_counter() - start_time) * 1000
            if isinstance(e, (LLMRateLimitedError, LLMProviderUnavailableError)):
                self._record_cooldown(self.primary.provider_name, caller)
            self._primary_failures_by_caller[caller] = (
                self._primary_failures_by_caller.get(caller, 0) + 1
            )
            logger.error(
                "Primary provider failed",
                extra={
                    "caller": caller,
                    "metric_type": "llm_primary_failed",
                    "provider": self.primary.provider_name,
                    "latency_ms": round(primary_latency_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            if not self.fallback_enabled:
                logger.error("No fallback provider configured, raising error")
                raise

            fallback = self.fallback
            if fallback is None:
                raise
            try:
                self._raise_if_cooling_down(fallback.provider_name, caller)
            except LLMProviderUnavailableError as cooldown_error:
                raise LLMFallbackExhaustedError(str(cooldown_error)) from e

            logger.warning("Falling back to %s", fallback.provider_name)
            self.fallback_count += 1
            fallback_start = time.perf_counter()

            try:
                response = await fallback.complete(
                    system_prompt,
                    user_prompt,
                    caller=caller,
                    **kwargs,
                )
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                fallback_latency_ms = (time.perf_counter() - fallback_start) * 1000
                logger.info(
                    "LLM fallback succeeded",
                    extra={
                        "metric_type": "llm_factory_call",
                        "provider": response.provider,
                        "model": response.model,
                        "latency_ms": round(total_latency_ms, 2),
                        "fallback_latency_ms": round(fallback_latency_ms, 2),
                        "input_tokens": response.usage.get("prompt_tokens", 0),
                        "output_tokens": response.usage.get("completion_tokens", 0),
                        "total_tokens": response.usage.get("total_tokens", 0),
                        "success": True,
                        "used_fallback": True,
                        "fallback_count": self.fallback_count,
                        "caller": caller,
                    },
                )
                return response.model_copy(update={"is_fallback": True})
            except Exception as fallback_error:
                if isinstance(
                    fallback_error,
                    (LLMRateLimitedError, LLMProviderUnavailableError),
                ):
                    self._record_cooldown(fallback.provider_name, caller)
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "Fallback provider also failed",
                    extra={
                        "caller": caller,
                        "metric_type": "llm_factory_call",
                        "latency_ms": round(total_latency_ms, 2),
                        "success": False,
                        "used_fallback": True,
                        "error": str(fallback_error),
                        "error_type": type(fallback_error).__name__,
                    },
                    exc_info=True,
                )
                raise LLMFallbackExhaustedError(str(fallback_error)) from fallback_error

    async def health_check(self) -> dict:
        """Check health of both providers."""
        primary_health = await self.primary.health_check()
        fallback_health = (
            await self.fallback.health_check() if self.fallback else {"status": "disabled"}
        )

        return {
            "primary": primary_health,
            "fallback": fallback_health,
            "fallback_count": self.fallback_count,
        }

    def get_failure_metrics(self) -> dict[str, dict[str, int]]:
        return {"primary_failures_by_caller": dict(self._primary_failures_by_caller)}

    def _cooldown_key(self, provider: str, caller: str) -> tuple[str, str]:
        return (provider, caller or "unknown")

    def _record_cooldown(self, provider: str, caller: str) -> None:
        until = time.monotonic() + max(0, settings.llm_fallback_cooldown_seconds)
        self._cooldowns[self._cooldown_key(provider, caller)] = until

    def _raise_if_cooling_down(self, provider: str, caller: str) -> None:
        until = self._cooldowns.get(self._cooldown_key(provider, caller))
        if not until:
            return
        remaining = until - time.monotonic()
        if remaining <= 0:
            self._cooldowns.pop(self._cooldown_key(provider, caller), None)
            return
        raise LLMProviderUnavailableError(
            f"{provider} is cooling down for caller={caller} ({remaining:.0f}s remaining)"
        )

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def model_name(self) -> str:
        return self.primary.model_name


# Singleton instance (backwards compatibility with existing code)
llm_client = LLMProviderWithFallback()
