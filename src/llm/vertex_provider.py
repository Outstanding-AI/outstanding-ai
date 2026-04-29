"""Vertex AI provider using google-genai and AWS WIF."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Type

from google.api_core.exceptions import InternalServerError, ResourceExhausted, ServiceUnavailable
from google.auth import aws as google_auth_aws
from google.auth import default as google_auth_default
from google.genai import Client, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import settings

from .aws_ecs_supplier import EcsTaskRoleSupplier
from .base import (
    BaseLLMProvider,
    LLMProviderUnavailableError,
    LLMRateLimitedError,
    LLMResponse,
    LLMStructuredOutputError,
)

logger = logging.getLogger(__name__)

VERTEX_RETRYABLE_ERRORS = (InternalServerError, ResourceExhausted, ServiceUnavailable)


def _log_retry(retry_state):
    """Log retry attempts with structured metrics."""
    exception = retry_state.outcome.exception()
    logger.warning(
        "Vertex retry attempt",
        extra={
            "metric_type": "llm_retry_attempt",
            "provider": "vertex",
            "attempt": retry_state.attempt_number,
            "wait_seconds": retry_state.next_action.sleep if retry_state.next_action else 0,
            "error": str(exception),
            "error_type": type(exception).__name__,
        },
    )


class VertexProvider(BaseLLMProvider):
    """Vertex AI provider using google-genai with ECS WIF."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
    ):
        self._model = model or settings.vertex_model
        self._temperature = temperature if temperature is not None else settings.vertex_temperature
        self._project = settings.vertex_project_id
        self._location = settings.vertex_location
        self._credentials = self._build_credentials()
        logger.info(
            "Initialized Vertex provider with model=%s project=%s location=%s",
            self._model,
            self._project,
            self._location,
        )

    @property
    def provider_name(self) -> str:
        return "vertex"

    @property
    def model_name(self) -> str:
        return self._model

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
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature if temperature is not None else self._temperature,
        )

        if json_mode or response_schema:
            config.response_mime_type = "application/json"
        if response_schema:
            config.response_schema = response_schema

        client = Client(
            vertexai=True,
            project=self._project,
            location=self._location,
            credentials=self._credentials,
        )

        @retry(
            retry=retry_if_exception_type(VERTEX_RETRYABLE_ERRORS),
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=_log_retry,
            reraise=True,
        )
        async def _generate() -> types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=config,
            )

        start_time = time.perf_counter()
        try:
            response = await _generate()
            usage = self._usage_from_response(response)
            content = self._serialize_response_content(response)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM call completed",
                extra={
                    "metric_type": "llm_call",
                    "provider": "vertex",
                    "model": self._model,
                    "latency_ms": round(latency_ms, 2),
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "success": True,
                    "structured": bool(response_schema),
                    "caller": caller,
                },
            )
            return LLMResponse(
                content=content,
                model=self._model,
                provider="vertex",
                usage=usage,
                raw_response={"response_id": response.response_id},
            )
        except ResourceExhausted as exc:
            logger.error(
                "Vertex provider rate limited",
                extra={
                    "caller": caller,
                    "provider": "vertex",
                    "model": self._model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "structured": bool(response_schema),
                },
                exc_info=True,
            )
            raise LLMRateLimitedError(str(exc)) from exc
        except (InternalServerError, ServiceUnavailable) as exc:
            logger.error(
                "Vertex provider unavailable",
                extra={
                    "caller": caller,
                    "provider": "vertex",
                    "model": self._model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "structured": bool(response_schema),
                },
                exc_info=True,
            )
            raise LLMProviderUnavailableError(str(exc)) from exc
        except ValueError as exc:
            logger.error(
                "Vertex provider returned unusable output",
                extra={
                    "caller": caller,
                    "provider": "vertex",
                    "model": self._model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "structured": bool(response_schema),
                },
                exc_info=True,
            )
            raise LLMStructuredOutputError(str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Vertex provider error",
                extra={
                    "caller": caller,
                    "provider": "vertex",
                    "model": self._model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "structured": bool(response_schema),
                },
                exc_info=True,
            )
            raise
        finally:
            try:
                await client.aio.aclose()
            except Exception as close_exc:
                logger.warning(
                    "Vertex async client close failed",
                    extra={
                        "caller": caller,
                        "provider": "vertex",
                        "model": self._model,
                        "error": str(close_exc),
                        "error_type": type(close_exc).__name__,
                    },
                    exc_info=True,
                )

    async def health_check(self) -> Dict[str, Any]:
        """Check Vertex availability with a small live request."""
        try:
            response = await self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with OK",
                caller="health_check",
            )
            return {
                "status": "healthy",
                "provider": "vertex",
                "model": self._model,
                "test_response": response.content[:20],
            }
        except Exception as exc:
            logger.error("Vertex health check failed: %s", exc)
            return {
                "status": "unhealthy",
                "provider": "vertex",
                "model": self._model,
                "error": str(exc),
            }

    def _build_credentials(self):
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if settings.running_on_ecs():
            info = self._load_wif_config()
            credentials = google_auth_aws.Credentials(
                audience=info["audience"],
                subject_token_type=info["subject_token_type"],
                token_url=info["token_url"],
                aws_security_credentials_supplier=EcsTaskRoleSupplier(),
                service_account_impersonation_url=info.get("service_account_impersonation_url"),
                service_account_impersonation_options=info.get(
                    "service_account_impersonation_options"
                ),
                token_info_url=info.get("token_info_url"),
                client_id=info.get("client_id"),
                client_secret=info.get("client_secret"),
                quota_project_id=info.get("quota_project_id"),
                workforce_pool_user_project=info.get("workforce_pool_user_project"),
                universe_domain=info.get("universe_domain", "googleapis.com"),
                trust_boundary=info.get("trust_boundary"),
                scopes=scopes,
            )
            return credentials

        credentials, discovered_project = google_auth_default(scopes=scopes)
        logger.info(
            "Using local ADC for Vertex auth (discovered_project=%s)",
            discovered_project or "unknown",
        )
        return credentials

    def _load_wif_config(self) -> dict[str, Any]:
        path = settings.vertex_wif_path()
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Vertex WIF config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Vertex WIF config file is not valid JSON: {path}") from exc

        required = {"audience", "subject_token_type", "token_url"}
        missing = sorted(required - info.keys())
        if missing:
            raise ValueError(f"Vertex WIF config missing required fields: {', '.join(missing)}")
        if info.get("type") != "external_account":
            raise ValueError("Vertex WIF config must be of type 'external_account'")

        # The ECS-specific supplier replaces the static credential_source block.
        info.pop("credential_source", None)
        return info

    def _serialize_response_content(self, response: types.GenerateContentResponse) -> str:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, BaseModel):
                return parsed.model_dump_json()
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed)
            return json.dumps(parsed)

        text = getattr(response, "text", None)
        if text:
            return text

        raise ValueError("Vertex returned no parsed payload or text content")

    def _usage_from_response(self, response: types.GenerateContentResponse) -> Dict[str, int]:
        usage_metadata = getattr(response, "usage_metadata", None)
        if not usage_metadata:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
