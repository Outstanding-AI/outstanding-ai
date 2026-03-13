"""Solvix AI Engine application settings.

Load configuration from environment variables (and ``.env`` file) using
Pydantic Settings.  All settings have sensible defaults for local
development; production values are injected via Docker environment or
a secrets manager.

Provider hierarchy:
    1. **Gemini** (primary) -- ``gemini-2.5-pro``, best reliability and
       structured output support.  Set via ``GEMINI_API_KEY``.
    2. **OpenAI** (fallback) -- ``gpt-5-nano``, activated automatically
       when Gemini is unavailable.  Set via ``OPENAI_API_KEY``.
       Note: reasoning models consume ``max_tokens`` for "thinking",
       so the budget is set very high (32768).
    3. **Anthropic** (optional) -- ``claude-sonnet-4-20250514`` for
       generation, ``claude-haiku-4-5-20251001`` for classification.
       Only used when explicitly configured via ``ANTHROPIC_API_KEY``.

Task-specific temperatures override provider defaults:
    - Draft generation: 0.7 (creative)
    - Classification: 0.2 (deterministic)
    - Persona generation: 0.7 (creative)
    - Persona refinement: 0.5 (moderate consistency)

Dev/prod switching:
    - ``DEBUG=true``: enables wildcard CORS, verbose logging.
    - ``SERVICE_AUTH_TOKEN``: when set, all endpoints (except /health,
      /ping) require ``Authorization: Bearer <token>``.
    - ``CORS_ALLOWED_ORIGINS``: comma-separated list of allowed
      origins; empty + debug=false = no CORS allowed.

Rate limiting:
    - Per-IP, per-minute via slowapi.  Defaults to 100/minute for all
      endpoints (appropriate for internal service-to-service calls).
    - Configurable per-endpoint via ``RATE_LIMIT_CLASSIFY``,
      ``RATE_LIMIT_GENERATE``, ``RATE_LIMIT_GATES`` env vars.
"""

from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All fields map 1:1 to environment variables (case-insensitive).
    For example, ``gemini_api_key`` reads from ``GEMINI_API_KEY``.
    A ``.env`` file in the project root is also loaded automatically.
    """

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    debug: bool = False

    # --- CORS ---
    # Comma-separated list of allowed origins.
    # Empty + debug=True  -> allow all ("*")
    # Empty + debug=False -> no CORS allowed (production safety)
    # Example: "https://app.solvix.com,https://admin.solvix.com"
    cors_allowed_origins: str = ""

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

    # --- LLM Provider Selection ---
    # Controls which provider is used as primary.
    # The fallback provider is the other one (if its API key is set).
    llm_provider: str = "gemini"  # "openai" or "gemini"

    # --- Gemini Configuration (PRIMARY) ---
    # gemini-2.5-pro: best reliability and structured output support.
    # Options: gemini-2.5-flash (fast/cheap), gemini-2.5-pro (capable).
    # Upgrade path: gemini-3.1-pro when JSON-mode stability confirmed.
    gemini_api_key: Optional[str] = Field(None, repr=False)
    gemini_model: str = "gemini-2.5-pro"
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 8192  # High for structured output

    # --- OpenAI Configuration (FALLBACK) ---
    # gpt-5-nano is a reasoning model: reasoning tokens consume from
    # max_tokens budget, so the budget must be very high (32768).
    # Upgrade path: gpt-5.3-instant for better quality at ~3x cost.
    openai_api_key: Optional[str] = Field(None, repr=False)
    openai_model: str = "gpt-5-nano"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 32768  # Headroom for reasoning tokens

    # --- Anthropic Configuration (OPTIONAL third provider) ---
    # Only activated when ANTHROPIC_API_KEY is set.
    # Sonnet for generation, Haiku for classification (cheaper).
    anthropic_api_key: Optional[str] = Field(None, repr=False)
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_temperature: float = 0.3
    anthropic_classification_model: str = "claude-haiku-4-5-20251001"

    # --- Task-specific temperatures ---
    # Override provider defaults per use case for optimal output.
    draft_temperature: float = 0.7  # Creative draft generation
    classification_temperature: float = 0.2  # Deterministic classification
    persona_gen_temperature: float = 0.7  # Creative persona generation
    persona_refine_temperature: float = 0.5  # Balanced refinement

    # --- Guardrail retry ---
    # Max retries when guardrails fail during draft generation.
    # Each retry passes failure details back to the LLM as feedback.
    max_guardrail_retries: int = 2

    # --- Timeouts and Retries ---
    llm_timeout_seconds: int = 60  # Per-call timeout (seconds)
    llm_max_retries: int = 3  # Tenacity retry decorator max attempts

    # --- Logging ---
    log_level: str = "INFO"

    # --- Service-to-service authentication ---
    # When set, all requests (except /health, /ping) must include
    # Authorization: Bearer <token>.  Leave unset for local dev.
    service_auth_token: Optional[str] = Field(None, repr=False)

    # --- Rate Limiting (per-IP, per-minute) ---
    # Defaults are high (100/min) since callers are internal services,
    # not end users.  Adjust downward for public-facing deployments.
    rate_limit_classify: str = "100/minute"
    rate_limit_generate: str = "100/minute"
    rate_limit_gates: str = "100/minute"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
