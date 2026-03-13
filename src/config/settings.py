from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    debug: bool = False

    # CORS - Allowed origins for cross-origin requests
    # In production, set this to your actual frontend domains
    # Example: "https://app.solvix.com,https://admin.solvix.com"
    cors_allowed_origins: str = ""  # Comma-separated list, empty = allow all in debug mode

    @field_validator("cors_allowed_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        # Just validate it's a string, parsing happens in get_cors_origins()
        return v

    def get_cors_origins(self) -> List[str]:
        """
        Get list of allowed CORS origins.

        Returns:
            List of allowed origins. If empty and debug=True, allows all origins.
            If empty and debug=False, returns empty list (no CORS allowed).
        """
        if not self.cors_allowed_origins:
            if self.debug:
                return ["*"]  # Allow all in debug mode only
            return []  # No CORS in production without explicit config
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    # LLM Provider Selection
    llm_provider: str = "gemini"  # "openai" or "gemini"

    # Gemini Configuration (PRIMARY)
    # Using gemini-2.5-pro for reliability and best performance
    # Options: gemini-2.5-flash (fast), gemini-2.5-pro (most capable/reliable)
    # Upgrade path: gemini-3.1-pro ($2/$12 per 1M tokens) when JSON-mode stability is confirmed
    gemini_api_key: Optional[str] = Field(None, repr=False)
    gemini_model: str = "gemini-2.5-pro"
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 8192  # High for longer drafts/complex structured output

    # OpenAI Configuration (FALLBACK)
    # Note: gpt-5-nano is a reasoning model. Reasoning tokens consume from max_tokens budget.
    # With max_tokens=2000 and 2000 reasoning tokens, there's 0 left for output.
    # 32768 provides headroom for reasoning models that consume tokens for "thinking".
    # Upgrade path: gpt-5.3-instant ($0.50/$2 per 1M) for better quality at ~3x cost
    openai_api_key: Optional[str] = Field(None, repr=False)
    openai_model: str = "gpt-5-nano"
    openai_temperature: float = 0.3
    openai_max_tokens: int = (
        32768  # Very high for reasoning models (reasoning tokens eat this budget)
    )

    # Anthropic Configuration (OPTIONAL third provider)
    anthropic_api_key: Optional[str] = Field(None, repr=False)
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_temperature: float = 0.3
    anthropic_classification_model: str = "claude-haiku-4-5-20251001"

    # Task-specific LLM temperatures (override provider defaults per use case)
    draft_temperature: float = 0.7  # Higher for creative draft generation
    classification_temperature: float = 0.2  # Lower for deterministic classification
    persona_gen_temperature: float = 0.7  # Creative persona generation
    persona_refine_temperature: float = 0.5  # Moderate for consistent refinement

    # Guardrail retry
    max_guardrail_retries: int = 2  # Max retries when guardrails fail during draft generation

    # Timeouts and Retries
    llm_timeout_seconds: int = 60  # Per-LLM-call timeout (increased for concurrent calls)
    llm_max_retries: int = 3  # Used by tenacity retry decorator

    # Logging
    log_level: str = "INFO"

    # Service-to-service authentication
    # When set, all requests (except /health) must include Authorization: Bearer <token>
    service_auth_token: Optional[str] = Field(None, repr=False)

    # Rate Limiting (per-IP, per-minute)
    # Higher limits for internal service-to-service calls
    rate_limit_classify: str = "100/minute"
    rate_limit_generate: str = "100/minute"
    rate_limit_gates: str = "100/minute"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
