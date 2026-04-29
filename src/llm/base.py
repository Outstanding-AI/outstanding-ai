"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized LLM response across all providers."""

    content: str
    model: str
    provider: str  # "vertex", "openai", etc.
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Dict[str, Any]] = None
    is_fallback: bool = False


class LLMProviderError(RuntimeError):
    """Base class for provider failures that callers may classify."""


class LLMRateLimitedError(LLMProviderError):
    """Provider rejected the request due to rate/resource limits."""


class LLMProviderUnavailableError(LLMProviderError):
    """Provider is temporarily unavailable."""


class LLMStructuredOutputError(LLMProviderError):
    """Provider returned no usable structured/text output."""


class LLMFallbackExhaustedError(LLMProviderError):
    """Primary and fallback providers both failed."""


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        *,
        caller: str = "unknown",
    ) -> LLMResponse:
        """
        Generate completion from prompts.

        Args:
            system_prompt: System message for the model
            user_prompt: User message/query
            temperature: Sampling temperature (0-1)
            json_mode: If True, request JSON output format
            response_schema: Optional Pydantic model to enforce structured output.
                When provided, the model is constrained to output valid JSON
                matching this schema. More reliable than json_mode alone.
            caller: Logical caller identifier used for diagnostics.
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider availability and return model info."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (vertex, openai, etc.)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return current model name."""
        pass
