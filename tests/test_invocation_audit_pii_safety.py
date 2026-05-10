"""Hard PII-safety + correctness tests for the per-provider invocation-audit helpers.

These are CI-enforced contract tests for ``src/llm/_invocation_audit.py``:

  - The sanitized ``model_invocation_config`` MUST NEVER contain prompt
    text, system instructions, message content, customer data, or
    schema bodies. (Codex Point 2 turned into regression tests.)
  - The hash MUST be stable across two builds with identical sanitized
    inputs and MUST NOT include SDK / library version (those have
    dedicated fields per the user's constraint).
  - ``sdk_library`` MUST report the wrapper-of-record (``langchain-openai``
    for OpenAI, ``google-genai`` for Vertex, ``anthropic`` for Anthropic).
  - ``inference_profile`` MUST reject unknown values at ``build_ai_audit``.
  - The structured OpenAI path MUST defensively pull ``system_fingerprint``
    from ``raw_output["raw"].response_metadata`` (Codex Point 3 / final
    constraint #3).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.engine.audit import build_ai_audit
from src.llm._invocation_audit import (
    anthropic_invocation_audit,
    openai_invocation_audit,
    vertex_invocation_audit,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SECRET_PROMPT = "This is the secret system_instruction containing customer name JOHN_DOE_ACME"
SECRET_USER_MSG = "User says: please update invoice INV-12345 amount £8421.00"


def _vertex_response(model_version: str | None = "models/gemini-2.5-flash@002") -> SimpleNamespace:
    return SimpleNamespace(model_version=model_version, response_id="vrtx-fake-1")


def _openai_text_response(system_fingerprint: str | None = "fp_openai_xyz") -> SimpleNamespace:
    metadata: dict[str, Any] = {}
    if system_fingerprint is not None:
        metadata["system_fingerprint"] = system_fingerprint
    return SimpleNamespace(content="reply", response_metadata=metadata)


def _openai_structured_raw_output(system_fingerprint: str | None) -> dict[str, Any]:
    """Shape of ``with_structured_output(..., include_raw=True)`` return."""
    raw_msg = _openai_text_response(system_fingerprint=system_fingerprint)
    return {"raw": raw_msg, "parsed": SimpleNamespace(model_dump_json=lambda: "{}")}


# ---------------------------------------------------------------------------
# PII non-leak — Codex Point 2 hardened as regression test
# ---------------------------------------------------------------------------


class TestPiiNonLeak:
    """Sanitized config must NEVER contain prompt or customer content,
    regardless of how the caller builds the explicit-config dict."""

    def test_vertex_drops_system_instruction_even_if_passed(self) -> None:
        # Defense-in-depth: caller accidentally includes system_instruction.
        # The allow-list filter in _filter() must drop it.
        audit = vertex_invocation_audit(
            explicit_config={
                "temperature": 0.7,
                "system_instruction": SECRET_PROMPT,  # forbidden — must be dropped
                "contents": [{"role": "user", "parts": [{"text": SECRET_USER_MSG}]}],
            },
            response=_vertex_response(),
        )
        serialized = json.dumps(audit.model_invocation_config)
        assert SECRET_PROMPT not in serialized
        assert SECRET_USER_MSG not in serialized
        assert "system_instruction" not in audit.model_invocation_config
        assert "contents" not in audit.model_invocation_config

    def test_vertex_response_schema_captured_by_identity_only(self) -> None:
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            field_with_potentially_sensitive_description: str = "x"

        audit = vertex_invocation_audit(
            explicit_config={"temperature": 0.3, "response_mime_type": "application/json"},
            response_schema=TestSchema,
            response=_vertex_response(),
        )
        cfg = audit.model_invocation_config
        # Identity captured...
        assert cfg["response_schema_name"] == "TestSchema"
        assert isinstance(cfg["response_schema_hash"], str)
        assert len(cfg["response_schema_hash"]) == 64  # sha256 hex
        # ...but body not stored. The class itself isn't serialized into the
        # config dict.
        assert "response_schema" not in cfg
        # And the structured flag is set.
        assert cfg["structured"] is True

    def test_openai_drops_messages_and_keys(self) -> None:
        audit = openai_invocation_audit(
            client_kwargs={
                "model": "gpt-5-mini",
                "temperature": 0.3,
                "openai_api_key": "sk-FAKE-LEAK-DETECTOR",  # forbidden
                "messages": [{"role": "system", "content": SECRET_PROMPT}],  # forbidden
                "tools": [{"name": "leak"}],  # forbidden
            },
            structured=False,
            response_format_type=None,
            raw_response=_openai_text_response(),
        )
        serialized = json.dumps(audit.model_invocation_config)
        assert "sk-FAKE-LEAK-DETECTOR" not in serialized
        assert SECRET_PROMPT not in serialized
        assert "leak" not in serialized
        assert "openai_api_key" not in audit.model_invocation_config
        assert "messages" not in audit.model_invocation_config
        assert "tools" not in audit.model_invocation_config

    def test_anthropic_drops_system_and_messages(self) -> None:
        audit = anthropic_invocation_audit(
            explicit_config={
                "temperature": 0.4,
                "system": SECRET_PROMPT,  # forbidden
                "messages": [{"role": "user", "content": SECRET_USER_MSG}],  # forbidden
                "max_tokens": 4096,
            },
            response=SimpleNamespace(),
        )
        serialized = json.dumps(audit.model_invocation_config)
        assert SECRET_PROMPT not in serialized
        assert SECRET_USER_MSG not in serialized
        assert "system" not in audit.model_invocation_config
        assert "messages" not in audit.model_invocation_config
        assert audit.model_invocation_config["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Hash stability — user constraint #2 enforced
# ---------------------------------------------------------------------------


class TestHashStability:
    def test_vertex_hash_stable_across_calls(self) -> None:
        a = vertex_invocation_audit(
            explicit_config={"temperature": 0.3, "response_mime_type": "application/json"},
            response=_vertex_response(),
        )
        b = vertex_invocation_audit(
            explicit_config={"response_mime_type": "application/json", "temperature": 0.3},
            response=_vertex_response(),
        )
        # Same sanitized inputs → identical hash, even with different key order.
        assert a.model_invocation_config_hash == b.model_invocation_config_hash

    def test_hash_changes_when_temperature_changes(self) -> None:
        a = vertex_invocation_audit(
            explicit_config={"temperature": 0.3},
            response=_vertex_response(),
        )
        b = vertex_invocation_audit(
            explicit_config={"temperature": 0.7},
            response=_vertex_response(),
        )
        assert a.model_invocation_config_hash != b.model_invocation_config_hash

    def test_hash_does_not_include_sdk_version(self) -> None:
        """User constraint #2: SDK version has its own field; mixing it
        into the hash would make hash equality dependent on a library
        upgrade rather than on our explicit knobs.
        """
        audit = vertex_invocation_audit(
            explicit_config={"temperature": 0.3},
            response=_vertex_response(),
        )
        # The SDK version, if present, is on the audit object but NOT in the
        # config dict. Inverse check: the config dict has no key that looks
        # like an SDK version field.
        cfg = audit.model_invocation_config
        for forbidden in ("sdk_version", "sdk_library", "_underlying_sdk_version"):
            assert forbidden not in cfg, f"{forbidden} leaked into hashed config"
        # And recomputing the hash from a stripped sanitized config matches
        # — i.e. the hash is over the published config dict, not augmented.
        import hashlib

        canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert (
            hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            == audit.model_invocation_config_hash
        )


# ---------------------------------------------------------------------------
# sdk_library values
# ---------------------------------------------------------------------------


class TestSdkLibraryValues:
    def test_vertex_reports_google_genai(self) -> None:
        audit = vertex_invocation_audit(
            explicit_config={"temperature": 0.3}, response=_vertex_response()
        )
        assert audit.sdk_library == "google-genai"

    def test_openai_reports_langchain_openai(self) -> None:
        audit = openai_invocation_audit(
            client_kwargs={"model": "gpt-5-mini", "temperature": 0.3},
            structured=False,
            response_format_type=None,
            raw_response=_openai_text_response(),
        )
        # User-confirmed: LangChain wraps the OpenAI call and owns
        # default-shape decisions, so sdk_library is langchain-openai.
        assert audit.sdk_library == "langchain-openai"

    def test_anthropic_reports_anthropic(self) -> None:
        audit = anthropic_invocation_audit(
            explicit_config={"temperature": 0.4, "max_tokens": 1024},
            response=SimpleNamespace(),
        )
        assert audit.sdk_library == "anthropic"


# ---------------------------------------------------------------------------
# system_fingerprint extraction — Codex constraint #3
# ---------------------------------------------------------------------------


class TestOpenAIFingerprint:
    def test_text_path_reads_from_response_metadata(self) -> None:
        audit = openai_invocation_audit(
            client_kwargs={"model": "gpt-5-mini", "temperature": 0.3},
            structured=False,
            response_format_type=None,
            raw_response=_openai_text_response(system_fingerprint="fp_text_path"),
        )
        assert audit.model_version_fingerprint == "fp_text_path"

    def test_structured_path_unwraps_raw_dict(self) -> None:
        """User constraint #3: ``with_structured_output(..., include_raw=True)``
        returns ``{"raw": <AIMessage>, "parsed": <BaseModel>}``. The helper
        must drill through ``raw_output["raw"].response_metadata``.
        """
        raw = _openai_structured_raw_output(system_fingerprint="fp_structured_path")
        audit = openai_invocation_audit(
            client_kwargs={"model": "gpt-5-mini", "temperature": 0.3},
            structured=True,
            response_format_type=None,
            raw_response=raw,
        )
        assert audit.model_version_fingerprint == "fp_structured_path"

    def test_missing_fingerprint_returns_none_not_raise(self) -> None:
        # gpt-5 sometimes omits system_fingerprint; helper must tolerate.
        audit = openai_invocation_audit(
            client_kwargs={"model": "gpt-5-mini", "temperature": 0.3},
            structured=False,
            response_format_type=None,
            raw_response=_openai_text_response(system_fingerprint=None),
        )
        assert audit.model_version_fingerprint is None

    def test_empty_response_metadata_returns_none(self) -> None:
        # Edge: response_metadata is missing or not a Mapping.
        bad_response = SimpleNamespace()  # no response_metadata at all
        audit = openai_invocation_audit(
            client_kwargs={"model": "gpt-5-mini", "temperature": 0.3},
            structured=False,
            response_format_type=None,
            raw_response=bad_response,
        )
        assert audit.model_version_fingerprint is None


# ---------------------------------------------------------------------------
# inference_profile validation
# ---------------------------------------------------------------------------


class TestInferenceProfile:
    def _audit_kwargs(self) -> dict:
        return dict(
            response=SimpleNamespace(provider="vertex", model="gemini-2.5-flash"),
            context=None,
            prompt_template_id="t",
            prompt_template_version="v",
            system_prompt="sys",
            user_prompt="usr",
            prompt_input={"a": 1},
        )

    def test_accepts_known_profile(self) -> None:
        for profile in ("draft_generation", "classification", "persona_gen", "persona_refine"):
            built = build_ai_audit(**self._audit_kwargs(), inference_profile=profile)
            assert built.inference_profile == profile

    def test_rejects_unknown_profile(self) -> None:
        with pytest.raises(ValueError, match="Unknown inference_profile"):
            build_ai_audit(**self._audit_kwargs(), inference_profile="typo_profile")

    def test_none_is_allowed_for_test_fixtures(self) -> None:
        built = build_ai_audit(**self._audit_kwargs(), inference_profile=None)
        assert built.inference_profile is None


# ---------------------------------------------------------------------------
# build_ai_audit copies provider fields off LLMResponse
# ---------------------------------------------------------------------------


class TestBuildAiAuditCopiesProviderFields:
    def test_copies_all_five_provider_fields(self) -> None:
        # LLMResponse-like duck type carrying the audit fields the providers attach.
        response = SimpleNamespace(
            provider="vertex",
            model="gemini-2.5-flash",
            model_invocation_config={"temperature": 0.7, "structured": False},
            model_invocation_config_hash="abc123",
            model_version_fingerprint="models/gemini-2.5-flash@002",
            sdk_library="google-genai",
            sdk_version="1.2.3",
        )
        built = build_ai_audit(
            response=response,
            context=None,
            prompt_template_id="draft_generation",
            prompt_template_version="silver_application_v1",
            system_prompt="sys",
            user_prompt="usr",
            prompt_input={"a": 1},
            inference_profile="draft_generation",
        )
        assert built.model_invocation_config == {"temperature": 0.7, "structured": False}
        assert built.model_invocation_config_hash == "abc123"
        assert built.model_version_fingerprint == "models/gemini-2.5-flash@002"
        assert built.sdk_library == "google-genai"
        assert built.sdk_version == "1.2.3"
        assert built.inference_profile == "draft_generation"

    def test_tolerates_response_missing_audit_fields(self) -> None:
        # Older test fixtures and the tests that build LLMResponse without
        # the new fields still need to work.
        response = SimpleNamespace(provider="vertex", model="gemini-2.5-flash")
        built = build_ai_audit(
            response=response,
            context=None,
            prompt_template_id="draft_generation",
            prompt_template_version="silver_application_v1",
            system_prompt="sys",
            user_prompt="usr",
            prompt_input={"a": 1},
        )
        assert built.model_invocation_config is None
        assert built.model_invocation_config_hash is None
        assert built.sdk_library is None
        assert built.sdk_version is None
