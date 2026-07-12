"""Vertex-first bounded fact extraction for one collection-mail event."""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailFactExtractionRequest
from src.api.models.responses import CollectionEmailFactExtractionResponse
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionEmailFactExtractionLLMResponse

from .audit import build_ai_audit
from .collection_email_event_classifier import _invalid_response_telemetry, _parse_response_object

PROMPT_TEMPLATE_ID = "collection_email_fact_extraction"
PROMPT_TEMPLATE_VERSION = "v4"
_SYSTEM_PROMPT = """Extract only invoice references, monetary assertions, and date assertions from one email event.
Use the current message and bounded prior context only to resolve references. Ignore quoted/forwarded text as authored intent.
Do not decide whether the chain is collection-related, do not classify a debtor response, and do not infer Sage truth, policy,
recipients, a route, or a draft. Return only strict JSON with keys invoice_assertions, amount_assertions, date_assertions,
confidence, and reason_codes. Bind amount and due-date assertions to an
invoice_ref only when the authored text supports that relationship. Never
invent a pairing. Preserve unbound amount-only and due-date-only assertions so
the deterministic reconciler can abstain or use its unique combined fallback.

``prior_chain_invoice_context`` is a body-free ledger of invoice references
explicitly established by *earlier messages in this same chain*. It is not
Sage data. For a deictic current-message statement such as "we will pay it on
Friday" or "we dispute that invoice", you may emit one ledger invoice_ref
only when candidate_count is exactly 1, is_truncated is false, and the current
authored text clearly adopts that one invoice. Add
``contextual_single_invoice_link`` to reason_codes. If there are zero,
multiple, or truncated candidates, do not choose one: leave invoice_assertions
empty and add ``ambiguous_contextual_invoice_scope`` or
``missing_contextual_invoice_scope``. Never copy several prior invoice refs,
never fabricate an identifier, and never use quoted text as the current
message's assertion.

Always include all five keys. When the authored message contains no asserted
invoice, amount, or date, return exactly this shape:
{"invoice_assertions":[],"amount_assertions":[],"date_assertions":[],"confidence":0.0,"reason_codes":["no_explicit_fact"]}
invoice_assertions is a list of invoice-reference strings only. Each
amount_assertions item has only invoice_ref (string or null), amount
(number or null), currency (string or null), and assertion_type
(claimed_paid, claimed_due, promised_payment, disputed_amount,
remittance_amount, or unknown). Each date_assertions item has only invoice_ref
(string or null), date_value (string or null), and assertion_type
(promise_date, payment_date, due_date, remittance_date, or other).
Do not add prose or any other key."""

_FACT_KEYS = {
    "invoice_assertions",
    "invoice_refs",
    "invoices",
    "amount_assertions",
    "amounts",
    "date_assertions",
    "dates",
    "confidence",
    "reason_codes",
    "reasons",
}


def _canonical_fact_response_object(content: object) -> dict:
    """Accept only documented compatibility aliases from generic JSON mode.

    Vertex rejects the nested provider-native schema for this response.  This
    narrow normalizer preserves a closed semantic contract without accepting
    arbitrary model fields or inventing any fact.
    """

    raw = _parse_response_object(content)
    unknown = set(raw) - _FACT_KEYS
    if unknown:
        raise ValueError("collection_email_fact_response_unknown_fields")
    if not set(raw).intersection(_FACT_KEYS - {"confidence", "reason_codes", "reasons"}):
        raise ValueError("collection_email_fact_response_missing_fact_fields")

    def _value(*keys: str, default):
        for key in keys:
            if key in raw:
                return raw[key]
        return default

    invoice_assertions = _value("invoice_assertions", "invoice_refs", "invoices", default=[])
    amount_assertions = _value("amount_assertions", "amounts", default=[])
    date_assertions = _value("date_assertions", "dates", default=[])
    reason_codes = _value("reason_codes", "reasons", default=[])
    if invoice_assertions is None:
        invoice_assertions = []
    if amount_assertions is None:
        amount_assertions = []
    if date_assertions is None:
        date_assertions = []
    if reason_codes is None:
        reason_codes = []
    if not all(
        isinstance(value, list)
        for value in (invoice_assertions, amount_assertions, date_assertions)
    ):
        raise ValueError("collection_email_fact_response_assertions_must_be_lists")
    if isinstance(reason_codes, str):
        reason_codes = [reason_codes]
    if not isinstance(reason_codes, list):
        raise ValueError("collection_email_fact_response_reason_codes_must_be_list")
    return {
        "invoice_assertions": invoice_assertions,
        "amount_assertions": amount_assertions,
        "date_assertions": date_assertions,
        # Missing confidence is conservative: it cannot make an inferred
        # assertion more trustworthy than the model explicitly stated.
        "confidence": raw.get("confidence", 0.0),
        "reason_codes": reason_codes,
    }


class CollectionEmailFactExtractor:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def extract(
        self, request: CollectionEmailFactExtractionRequest
    ) -> CollectionEmailFactExtractionResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            # Vertex rejects this nested response schema as too large for its
            # serving-state budget. JSON mode plus the closed normalizer below
            # is the supported, fail-closed transport for this endpoint.
            json_mode=True,
            caller="collection_email_fact_extraction",
        )
        try:
            parsed = CollectionEmailFactExtractionLLMResponse(
                **_canonical_fact_response_object(response.content)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid collection-email fact response",
                details={
                    "operation": "collection_email_fact_extraction",
                    "telemetry": _invalid_response_telemetry(response),
                },
            ) from exc
        return CollectionEmailFactExtractionResponse(
            invoice_assertions=parsed.invoice_assertions,
            amount_assertions=[
                item.model_dump(exclude_none=True) for item in parsed.amount_assertions
            ],
            date_assertions=[item.model_dump(exclude_none=True) for item in parsed.date_assertions],
            confidence=parsed.confidence,
            reason_codes=parsed.reason_codes,
            tokens_used=response.usage.get("total_tokens", 0),
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            provider=response.provider,
            model=response.model,
            is_fallback=response.provider != self._client.primary_provider_name,
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_input=prompt_input,
                token_count=response.usage.get("total_tokens", 0),
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                inference_profile="classification",
            ),
        )


collection_email_fact_extractor = CollectionEmailFactExtractor()
