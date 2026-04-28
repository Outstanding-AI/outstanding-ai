"""Schema-bump regression tests for ExtractedData (Sprint A item #2).

Pins the two new fields:

- ``promise_strength``: captured for future use only as of 2026-04-28
  (Codex P2). The lane reply router currently treats every
  PROMISE_TO_PAY identically; strength-aware suppression intensity
  (``firm`` vs ``soft`` vs ``aspirational``) is Sprint B+ work. These
  tests pin the schema so the downstream consumer has a stable shape
  to switch on once it ships.
- ``account_wide``: ACTIVE — gates the scope-resolver fallback. ``True``
  means the debtor used explicit account-wide language ("all invoices",
  "everything outstanding"); ``False`` (default) means scope must come
  from ``invoice_refs`` or the message's tracked-thread lane. Without
  this field, an empty ``invoice_refs`` defaulted to ALL open
  obligations for the party (the bug at ETL
  ``classification_service.py:139``).
"""

from __future__ import annotations

import pytest

from src.api.models.responses import ExtractedData as ResponseExtractedData
from src.engine.classifier import _build_extracted_data
from src.llm.schemas import LLMExtractedData


class TestPromiseStrength:
    @pytest.mark.parametrize("value", ["firm", "soft", "aspirational"])
    def test_accepts_valid_values(self, value):
        raw = LLMExtractedData(promise_amount=500, promise_strength=value)
        assert raw.promise_strength == value

    def test_default_is_none(self):
        """Default ``None`` lets ETL apply backwards-compat (treat as firm)
        without forcing a Pydantic-side default. Keeps the Vertex JSON
        compact when the LLM doesn't emit the field."""
        raw = LLMExtractedData(promise_amount=500)
        assert raw.promise_strength is None

    def test_rejects_invalid_value(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMExtractedData(promise_amount=500, promise_strength="maybe_pay")  # type: ignore[arg-type]

    def test_propagates_through_build_extracted_data(self):
        raw = LLMExtractedData(promise_amount=500, promise_strength="aspirational")
        built = _build_extracted_data(raw)
        assert built is not None
        assert built.promise_strength == "aspirational"


class TestAccountWide:
    def test_default_is_none(self):
        """ETL treats ``None`` as ``False`` — that's the safe scope
        default (don't fall back to all-open). True must be explicit."""
        raw = LLMExtractedData(invoice_refs=[])
        assert raw.account_wide is None

    def test_explicit_true(self):
        raw = LLMExtractedData(invoice_refs=[], account_wide=True)
        assert raw.account_wide is True

    def test_explicit_false(self):
        raw = LLMExtractedData(invoice_refs=[], account_wide=False)
        assert raw.account_wide is False

    def test_propagates_through_build_extracted_data(self):
        raw = LLMExtractedData(invoice_refs=["INV-001"], account_wide=True)
        built = _build_extracted_data(raw)
        assert built is not None
        assert built.account_wide is True
        assert built.invoice_refs == ["INV-001"]


class TestResponseSchemaParity:
    """``ExtractedData`` (response) must mirror ``LLMExtractedData`` (LLM)
    so consumers see what the LLM produced. Catches drift between the
    two parallel schemas."""

    def test_response_has_promise_strength(self):
        # Pydantic v2: model_fields gives us declared field names.
        assert "promise_strength" in ResponseExtractedData.model_fields

    def test_response_has_account_wide(self):
        assert "account_wide" in ResponseExtractedData.model_fields

    def test_response_field_names_superset_of_llm_schema(self):
        """If we add a field to the LLM schema, it must also exist on
        the response schema — otherwise ``_build_extracted_data`` will
        silently drop it."""
        llm_fields = set(LLMExtractedData.model_fields.keys())
        response_fields = set(ResponseExtractedData.model_fields.keys())
        missing = llm_fields - response_fields
        assert not missing, f"LLM-side fields not exposed in response: {missing}"


class TestPromptDocumentsNewFields:
    """Source-check: classifier prompt explicitly asks the LLM for the
    new fields. Without prompt instructions the LLM won't emit them
    even though the schema accepts them."""

    def _read_prompt(self) -> str:
        with open(
            "/Users/bijitdeka23/Downloads/Solvix_repo/solvix-ai/src/prompts/classification.py",
            "r",
        ) as fh:
            return fh.read()

    def test_prompt_documents_promise_strength(self):
        prompt = self._read_prompt()
        assert "promise_strength" in prompt
        # The three valid values must be enumerated for the LLM to choose.
        for level in ("firm", "soft", "aspirational"):
            assert level in prompt, f"prompt missing promise_strength value {level!r}"

    def test_prompt_documents_account_wide(self):
        prompt = self._read_prompt()
        assert "account_wide" in prompt
        # At least one of the canonical phrases must appear so the LLM
        # has concrete examples to anchor on.
        anchors = ("all invoices", "full balance", "everything outstanding", "whole account")
        assert any(a in prompt.lower() for a in anchors), (
            f"prompt must enumerate account-wide language anchors; none of {anchors} found"
        )
