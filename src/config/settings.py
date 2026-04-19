"""Outstanding AI Engine application settings.

Load configuration from environment variables (and ``.env`` file) using
Pydantic Settings. Production uses Vertex AI via AWS workload identity
federation. Local development can use ADC when ECS task-role credentials
are not available.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All fields map 1:1 to environment variables (case-insensitive).
    A ``.env`` file in the project root is also loaded automatically.
    """

    # --- Deployment environment ---
    # "local" | "development" | "staging" | "production"
    # Production enforces fail-closed checks (service auth required, debug=false).
    environment: str = "local"

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    debug: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, v):
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
        return v

    # --- CORS ---
    # Comma-separated list of allowed origins.
    # Empty + debug=True  -> allow all ("*")
    # Empty + debug=False -> no CORS allowed (production safety)
    # Example: "https://app.outstandingai.com,https://admin.outstandingai.com"
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
    llm_provider: str = "vertex"  # "vertex", "openai", "anthropic"

    # --- Vertex AI Configuration (PRIMARY) ---
    vertex_project_id: str = "production-493814"
    vertex_location: str = "europe-west2"
    vertex_model: str = "gemini-2.5-flash"
    vertex_temperature: float = 0.3
    vertex_max_tokens: int = 8192
    vertex_wif_config_path: str = "/app/infra/vertex-wif-config.json"

    # --- OpenAI Configuration (FALLBACK) ---
    openai_api_key: Optional[str] = Field(None, repr=False)
    openai_model: str = "gpt-5-nano"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 32768

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

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def running_on_ecs(self) -> bool:
        return bool(
            os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        )

    def vertex_wif_path(self) -> Path:
        return Path(self.vertex_wif_config_path)

    def model_for_provider(self, provider: Optional[str] = None) -> str:
        provider_name = provider or self.llm_provider
        if provider_name == "vertex":
            return self.vertex_model
        if provider_name == "openai":
            return self.openai_model
        if provider_name == "anthropic":
            return self.anthropic_model
        return "unknown"

    def provider_status(self) -> dict[str, bool]:
        return {
            "vertex_config": self.vertex_wif_path().is_file(),
            "ecs_task_role": self.running_on_ecs(),
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
        }

    @model_validator(mode="after")
    def _enforce_production_invariants(self) -> "Settings":
        if self.environment == "production":
            if not self.service_auth_token:
                raise ValueError("SERVICE_AUTH_TOKEN is required when ENVIRONMENT=production")
            if self.debug:
                raise ValueError("DEBUG must be false when ENVIRONMENT=production")
            if self.llm_provider == "vertex":
                if not self.vertex_project_id or self.vertex_project_id.startswith("REPLACE_"):
                    raise ValueError(
                        "VERTEX_PROJECT_ID must be set to a non-placeholder value in production"
                    )
                if not self.vertex_wif_path().is_file():
                    raise ValueError(
                        "VERTEX_WIF_CONFIG_PATH must point to a readable file in production"
                    )
                if not self.running_on_ecs():
                    raise ValueError(
                        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI or "
                        "AWS_CONTAINER_CREDENTIALS_FULL_URI is required in production"
                    )
                if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
                    raise ValueError("AWS_REGION or AWS_DEFAULT_REGION is required in production")
        return self


settings = Settings()
