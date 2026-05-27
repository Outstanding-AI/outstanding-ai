"""Audit metadata helpers for prompt/model lineage."""

import hashlib
import json
from typing import Any, Literal

from src.api.models.responses import AIAuditMetadata
from src.config.settings import settings

# Fixed enum for ``inference_profile`` — keep in lockstep with the values
# documented in ``AIAuditMetadata.inference_profile`` and CLAUDE.md note 51.
# Free-form values are rejected at ``build_ai_audit`` so dashboards and
# cost-by-purpose queries have a stable join key.
InferenceProfile = Literal[
    "draft_generation",
    "classification",
    "persona_gen",
    "persona_refine",
    "sent_scope_analysis",
]
_VALID_INFERENCE_PROFILES: frozenset[str] = frozenset(
    {"draft_generation", "classification", "persona_gen", "persona_refine", "sent_scope_analysis"}
)


def hash_text(value: str) -> str:
    """Return a stable SHA-256 hash for prompt text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_payload(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-serializable prompt inputs."""
    canonical = json.dumps(
        payload,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hash_text(canonical)


def _json_list_or_none(values: set[str]) -> str | None:
    cleaned = sorted(value for value in values if value)
    return json.dumps(cleaned, sort_keys=True) if cleaned else None


def _actual_sent_scope_audit_inputs(context: Any) -> tuple[str | None, str | None]:
    """Return lineage for sent-draft analysis rows included in context."""
    event_ids: set[str] = set()
    hashes: set[str] = set()
    for row in getattr(context, "actual_sent_scope_history", None) or []:
        event_id = getattr(row, "sent_draft_analysis_event_id", None)
        content_hash = getattr(row, "application_content_hash", None)
        if event_id:
            event_ids.add(str(event_id))
        if content_hash:
            hashes.add(str(content_hash))
    return _json_list_or_none(event_ids), _json_list_or_none(hashes)


def build_ai_audit(
    *,
    response: Any,
    context: Any = None,
    prompt_template_id: str,
    prompt_template_version: str,
    system_prompt: str,
    user_prompt: str,
    prompt_input: Any,
    guardrail_pipeline_version: str | None = None,
    token_count: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    latency_ms: float | None = None,
    inference_profile: InferenceProfile | None = None,
) -> AIAuditMetadata:
    """Build the shared audit object returned by AI endpoints.

    Args:
        inference_profile: One of ``draft_generation``, ``classification``,
            ``persona_gen``, ``persona_refine``. Validated against
            ``_VALID_INFERENCE_PROFILES``; unknown values raise ValueError so
            a typo can't pollute audit dashboards. Pass ``None`` only for
            test fixtures or experimental call sites — production callers
            should always set this.
    """
    if inference_profile is not None and inference_profile not in _VALID_INFERENCE_PROFILES:
        raise ValueError(
            f"Unknown inference_profile {inference_profile!r}; "
            f"must be one of {sorted(_VALID_INFERENCE_PROFILES)}"
        )

    # Defense-in-depth: the per-provider helpers in src/llm/_invocation_audit.py
    # already sanitize before constructing LLMResponse. This re-sanitization
    # at the response-building boundary catches any provider regression
    # that leaks unsafe nested keys into LLMResponse.model_invocation_config
    # before the HTTP response leaves the AI Engine. Any consumer that
    # reads the response (logs, traces, debug tooling) sees only the
    # sanitized dict; backend has its own filter as the final layer.
    # When sanitization changes the dict, recompute the hash so the
    # persisted hash always matches the persisted config.
    from src.llm._invocation_audit import (
        hash_invocation_config,
        sanitize_persisted_invocation_config,
    )

    raw_invocation_config = getattr(response, "model_invocation_config", None)
    sanitized_invocation_config: dict[str, Any] | None = None
    sanitized_invocation_hash: str | None = None
    if raw_invocation_config:
        sanitized_invocation_config = sanitize_persisted_invocation_config(raw_invocation_config)
        if sanitized_invocation_config:
            # Invariant: the hash on the response always describes the exact
            # config dict on the response. Recompute unconditionally so stale
            # or forged provider hashes cannot cross the AI HTTP boundary.
            sanitized_invocation_hash = hash_invocation_config(sanitized_invocation_config)

    provider = getattr(response, "provider", None)
    input_versions_json = None
    sent_analysis_event_ids_json = None
    sent_analysis_hashes_json = None
    if context is not None:
        input_versions_json = getattr(context, "input_silver_version_ids_json", None)
        if not input_versions_json and getattr(context, "input_silver_version_ids", None):
            input_versions_json = json.dumps(context.input_silver_version_ids, sort_keys=True)
        sent_analysis_event_ids_json, sent_analysis_hashes_json = _actual_sent_scope_audit_inputs(
            context
        )

    return AIAuditMetadata(
        ai_provider=provider,
        ai_model=getattr(response, "model", None),
        ai_region=settings.vertex_location if provider == "vertex" else None,
        prompt_template_id=prompt_template_id,
        prompt_template_version=prompt_template_version,
        system_prompt_hash=hash_text(system_prompt),
        user_prompt_hash=hash_text(user_prompt),
        prompt_input_hash=hash_payload(prompt_input),
        guardrail_pipeline_version=guardrail_pipeline_version,
        input_silver_version_ids_json=input_versions_json,
        input_sent_draft_analysis_event_ids_json=sent_analysis_event_ids_json,
        input_sent_draft_analysis_hashes_json=sent_analysis_hashes_json,
        policy_snapshot_id=getattr(context, "policy_snapshot_id", None),
        draft_candidate_id=getattr(context, "draft_candidate_id", None),
        draft_generation_run_id=getattr(context, "draft_generation_run_id", None),
        source_sync_run_id=getattr(context, "source_sync_run_id", None),
        application_run_id=getattr(context, "application_run_id", None),
        token_count=token_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        # Model invocation audit (May 2026): use the re-sanitized values
        # computed above so the persisted dict and persisted hash agree
        # even if a provider helper regressed and let unsafe keys through.
        # None-safe: older test fixtures with no model_invocation_config
        # attribute land None on both fields.
        model_invocation_config=sanitized_invocation_config,
        model_invocation_config_hash=sanitized_invocation_hash,
        model_version_fingerprint=getattr(response, "model_version_fingerprint", None),
        sdk_library=getattr(response, "sdk_library", None),
        sdk_version=getattr(response, "sdk_version", None),
        inference_profile=inference_profile,
    )
