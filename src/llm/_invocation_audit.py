"""Per-provider sanitized invocation-audit helpers.

EU AI Act Article 13 transparency obligation: backend audit logs need a
record of which non-content knobs we passed to each LLM call. These
helpers build that record from a FIXED, per-provider allow list of
explicit keys — never from the raw SDK config object.

Hard rules (enforced by tests in tests/test_invocation_audit_pii_safety.py):
  - NEVER serialize ``GenerateContentConfig`` (Vertex), ``ChatOpenAI`` raw
    kwargs, ``Messages.create`` raw kwargs (Anthropic), or any object that
    transitively holds prompt text, system instructions, message lists, or
    tool/function bodies.
  - The sanitized dict contains ONLY the keys in the per-provider
    ``_*_AUDIT_KEYS`` set below. Add a key to that set if and only if it is
    a non-content knob (sampling param, output cap, format flag, etc.).
  - The hash is computed over the sanitized dict alone. SDK library and
    SDK version have their own dedicated fields on ``AIAuditMetadata``;
    they are NOT folded into the hash. Hash stability is a property of
    our explicit allow list, not of the SDK's config object.

Public surface:
  - ``vertex_invocation_audit(config, response)``
  - ``openai_invocation_audit(client_kwargs, response)``
  - ``anthropic_invocation_audit(call_kwargs, response)``
  - ``InvocationAudit`` dataclass — five-field bundle returned by each
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import metadata as _pkg_metadata
from typing import Any, Dict, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Per-provider allow-key sets
#
# Every key listed here MUST be a non-content knob. See module docstring.
# ---------------------------------------------------------------------------

_VERTEX_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "candidate_count",
        "seed",
        "stop_sequences",
        "response_mime_type",
        # Schema is captured by IDENTITY only (class name + canonical-JSON
        # hash of the schema fields) so we can detect schema drift without
        # storing the schema body. Computed in ``vertex_invocation_audit``.
        "response_schema_name",
        "response_schema_hash",
        "structured",
    }
)

_OPENAI_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        # ``model`` is intentionally NOT here — it's redundant with the
        # top-level ``ai_model`` on AIAuditMetadata. Keeping providers
        # symmetric (Vertex / Anthropic also don't carry ``model`` inside
        # the sanitized config) so the backend nested allowlist stays a
        # single source of truth.
        "temperature",
        "top_p",
        "max_completion_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
        # ``response_format_type`` only — never the schema body.
        "response_format_type",
        # structured-output identity, mirrors Vertex.
        # ``response_schema_name`` is the Pydantic class __name__;
        # ``response_schema_hash`` is sha256 of canonical-JSON of the
        # JSON schema fields. Schema body is NEVER stored.
        "response_schema_name",
        "response_schema_hash",
        "structured",
    }
)

_ANTHROPIC_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop_sequences",
    }
)

# Keys that MUST NEVER appear in any sanitized config — backstop for
# defense-in-depth. The per-provider builders only copy from the allow list
# above so these can't slip in via key-name; this set guards against future
# refactors that might widen the allow list.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "system_instruction",
        "system",
        "messages",
        "contents",
        "user_prompt",
        "user_message",
        "tools",
        "tool_choice",
        "tool_config",
        "safety_settings",  # contents include category mappings; if ever needed, capture as fingerprint
        "response_format",  # full body; capture only the type via response_format_type
        "response_schema",  # full body; capture only schema_name + schema_hash
        "openai_api_key",
        "api_key",
        "credentials",
    }
)


@dataclass(frozen=True)
class InvocationAudit:
    """Five-field bundle attached to every LLMResponse.

    Mirrors the new ``AIAuditMetadata`` fields one-to-one so the
    per-endpoint caller can copy them straight onto the response.
    """

    model_invocation_config: Dict[str, Any]
    model_invocation_config_hash: str
    model_version_fingerprint: Optional[str]
    sdk_library: str
    sdk_version: Optional[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Stable canonical-JSON serialization for hash input.

    sort_keys=True + compact separators + ensure_ascii=True so the same
    sanitized dict always hashes to the same string regardless of insertion
    order or platform.
    """
    return json.dumps(
        dict(payload),
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_config(sanitized: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(sanitized).encode("utf-8")).hexdigest()


def _filter(source: Mapping[str, Any], allowed: Iterable[str]) -> Dict[str, Any]:
    """Pick only keys in ``allowed`` whose values are non-None.

    Defense-in-depth: also drop anything in ``_FORBIDDEN_KEYS`` even if
    somehow listed in ``allowed``. The forbidden check is a backstop;
    the allow lists above are the primary contract.
    """
    out: Dict[str, Any] = {}
    for key in allowed:
        if key in _FORBIDDEN_KEYS:
            continue
        if key not in source:
            continue
        value = source[key]
        if value is None:
            continue
        out[key] = value
    return out


def _try_pkg_version(name: str) -> Optional[str]:
    try:
        return _pkg_metadata.version(name)
    except _pkg_metadata.PackageNotFoundError:
        return None


def _schema_identity(schema: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (class name, canonical-JSON hash of schema fields).

    Captures schema IDENTITY without storing the schema body. ``schema``
    is expected to be a Pydantic class (or None). Returns (None, None) if
    we can't introspect it — never raise; audit metadata is best-effort.
    """
    if schema is None:
        return None, None
    name = getattr(schema, "__name__", None)
    try:
        # Pydantic v2 model_json_schema returns a stable dict
        body = schema.model_json_schema()
        return name, hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    except Exception:
        return name, None


# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------


def vertex_invocation_audit(
    *,
    explicit_config: Mapping[str, Any],
    response_schema: Any = None,
    response: Any = None,
) -> InvocationAudit:
    """Build the audit bundle for a Vertex / google-genai call.

    Args:
        explicit_config: dict of the kwargs WE explicitly set on
            ``GenerateContentConfig``. Caller must build this — DO NOT pass
            the raw ``GenerateContentConfig`` object (it carries
            ``system_instruction``).
        response_schema: optional Pydantic class — captured by name + hash
            of its JSON schema, never as a body.
        response: the SDK response object. ``response.model_version`` is
            read defensively for ``model_version_fingerprint``.
    """
    sanitized = _filter(explicit_config, _VERTEX_AUDIT_KEYS)
    schema_name, schema_hash = _schema_identity(response_schema)
    if schema_name:
        sanitized["response_schema_name"] = schema_name
    if schema_hash:
        sanitized["response_schema_hash"] = schema_hash
    sanitized.setdefault("structured", response_schema is not None)

    fingerprint = getattr(response, "model_version", None) if response is not None else None

    return InvocationAudit(
        model_invocation_config=sanitized,
        model_invocation_config_hash=_hash_config(sanitized),
        model_version_fingerprint=fingerprint,
        sdk_library="google-genai",
        sdk_version=_try_pkg_version("google-genai"),
    )


# ---------------------------------------------------------------------------
# OpenAI / LangChain
# ---------------------------------------------------------------------------


def openai_invocation_audit(
    *,
    client_kwargs: Mapping[str, Any],
    structured: bool,
    response_format_type: Optional[str] = None,
    response_schema: Any = None,
    raw_response: Any = None,
) -> InvocationAudit:
    """Build the audit bundle for an OpenAI call routed via LangChain.

    ``sdk_library`` is ``"langchain-openai"`` because LangChain (not the raw
    ``openai`` SDK) owns default-shape decisions for our call sites.

    ``response_schema`` (Pydantic class) is captured by IDENTITY only —
    class ``__name__`` plus sha256 of canonical-JSON of the JSON schema.
    Schema body is NEVER serialized. Mirrors the Vertex contract.

    ``raw_response`` may be either the LangChain ``AIMessage`` (text path) or
    the dict returned by ``with_structured_output(..., include_raw=True)``
    (structured path). For the structured path, the raw langchain response
    is at ``raw_response["raw"]``. We probe ``response_metadata`` defensively
    for ``system_fingerprint``.
    """
    sanitized = _filter(client_kwargs, _OPENAI_AUDIT_KEYS)
    sanitized["structured"] = bool(structured)
    if response_format_type:
        sanitized["response_format_type"] = response_format_type
    schema_name, schema_hash = _schema_identity(response_schema)
    if schema_name:
        sanitized["response_schema_name"] = schema_name
    if schema_hash:
        sanitized["response_schema_hash"] = schema_hash

    # Defensive system_fingerprint extraction. Two shapes:
    #   1. raw_response is an AIMessage with .response_metadata
    #   2. raw_response is the structured dict with raw_response["raw"].response_metadata
    fingerprint: Optional[str] = None
    candidate = raw_response
    if isinstance(raw_response, dict) and "raw" in raw_response:
        candidate = raw_response.get("raw")
    metadata = getattr(candidate, "response_metadata", None) if candidate is not None else None
    if isinstance(metadata, Mapping):
        fp = metadata.get("system_fingerprint")
        if isinstance(fp, str) and fp:
            fingerprint = fp

    return InvocationAudit(
        model_invocation_config=sanitized,
        model_invocation_config_hash=_hash_config(sanitized),
        model_version_fingerprint=fingerprint,
        sdk_library="langchain-openai",
        sdk_version=_try_pkg_version("langchain-openai"),
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def anthropic_invocation_audit(
    *,
    explicit_config: Mapping[str, Any],
    response: Any = None,
) -> InvocationAudit:
    """Build the audit bundle for an Anthropic call.

    Provider is currently disabled (no live calls in production), but the
    helper is kept symmetric with the others so reactivation needs no new
    plumbing.
    """
    sanitized = _filter(explicit_config, _ANTHROPIC_AUDIT_KEYS)
    # Anthropic has no system_fingerprint analogue. response.id is per-call
    # rather than per-model-build, so it is not useful for drift detection.
    fingerprint: Optional[str] = None

    return InvocationAudit(
        model_invocation_config=sanitized,
        model_invocation_config_hash=_hash_config(sanitized),
        model_version_fingerprint=fingerprint,
        sdk_library="anthropic",
        sdk_version=_try_pkg_version("anthropic"),
    )
