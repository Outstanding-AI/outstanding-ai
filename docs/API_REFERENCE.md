# Outstanding AI Engine — API Reference

Stateless FastAPI microservice (`title="Outstanding AI Engine"`, `version=0.1.0`, default port **8001**, Python >= 3.12). Called exclusively by the Django backend over HTTP (with circuit breaker). Verified against source.

- Contracts pin: `solvix-contracts==0.12.40` (`pyproject.toml`). Request models extend the contracts case-context bases (`CaseContextV2`, `PartyInfoV3`, `BehaviorInfoV3`, `ObligationInfoV3` + V3 history blocks).
- Repo-local Pydantic response schemas in the **backend** mirror (`Solvix/services/ai_engine/schemas.py`) are warn-only — they never block.

_Last updated: 2026-06-14._

---

## Authentication

| Setting | Local | Production |
|---------|-------|-----------|
| `SERVICE_AUTH_TOKEN` | Empty → auth disabled (local dev only) | Required (AWS Secrets Manager); startup fails without it |

When `SERVICE_AUTH_TOKEN` is set, `ServiceAuthMiddleware` (outermost middleware) requires `Authorization: Bearer <token>` on every non-public path and compares it with constant-time `hmac.compare_digest` (401 otherwise).

**Public (no-auth) paths are dynamic** via `settings.public_paths()`:
- Always public: `/health`, `/ping`.
- `/docs`, `/openapi.json`, `/redoc` are public **only when docs are enabled** (`ENABLE_API_DOCS=true` or non-production environment).
- `/health/llm` is intentionally **not** public (it burns provider quota and requires bearer auth).

Other app-level middleware:
- `RequestIDMiddleware`: accepts/generates `X-Request-ID`, stores it in a contextvar, echoes it back in the response header.
- CORS: explicit origins from `CORS_ALLOWED_ORIGINS` (comma-separated). Empty + debug → `["*"]`; empty + non-debug → CORS disabled. Allowed methods `GET, POST, OPTIONS`; allowed headers `Authorization, Content-Type, X-Tenant-ID, X-Request-ID`.

---

## Endpoint Summary

| Method | Path | Request model | Response model | Rate limit setting | Tag |
|---|---|---|---|---|---|
| GET | `/ping` | — | `PingResponse` | none | Health |
| GET | `/health` | — | `ShallowHealthResponse` | none | Health |
| GET | `/health/llm` | — | `HealthResponse` | none (bearer auth required) | Health |
| POST | `/classify` | `ClassifyRequest` | `ClassifyResponse` | `rate_limit_classify` (100/minute) | Classification |
| POST | `/classify-collection-email-event` | `CollectionEmailEventRequest` | `CollectionEmailEventResponse` | `rate_limit_classify` | Collection email shadow |
| POST | `/extract-collection-email-facts` | `CollectionEmailFactExtractionRequest` | `CollectionEmailFactExtractionResponse` | `rate_limit_classify` | Collection email shadow |
| POST | `/identify-collection-chain` | `CollectionChainIdentificationRequest` | `CollectionChainIdentificationResponse` | `rate_limit_classify` | Collection email shadow |
| POST | `/generate-draft` | `GenerateDraftRequest` | `GenerateDraftResponse` | `rate_limit_generate` (100/minute) | Generation |
| POST | `/generate-draft-from-manifest` | `DraftGenerationHandoff` | `GenerateDraftFromManifestResponse` | `rate_limit_generate` | Generation |
| POST | `/analyze-sent-draft-scope` | `AnalyzeSentDraftScopeRequest` | `AnalyzeSentDraftScopeResponse` | `rate_limit_classify` | Sent Scope |
| POST | `/generate-persona` | `GeneratePersonaRequest` | `GeneratePersonaResponse` | `rate_limit_generate` | Persona |
| POST | `/refine-persona` | `RefinePersonaRequest` | `RefinePersonaResponse` | `rate_limit_generate` | Persona |

Every response also carries the safe `X-AI-Engine-Class` header (`medium`,
`large`, or `xlarge`). The backend stores requested and served class in the
sanitized LLM audit metadata alongside provider, tokens, cost, and latency.
Class-specific backend routing is valid only when each class has a distinct
service-discovery endpoint; otherwise the backend intentionally uses the
single medium service.

> Gate evaluation is **not** an AI Engine endpoint. The historical `/evaluate-gates` route + `GateEvaluator` were deleted on 2026-04-26. All 10 compliance gates (G1–G9 + `workflow_hold`) are evaluated in the Django backend (`Solvix/services/gate_checker.py`). A generic `GateResult` model survives in `responses.py` but no gate route exists.

---

## Endpoints

### POST /classify

Classify an inbound debtor email into one of **25** categories (multi-intent capable). Called by the backend during `ai.process_email_classification`; drives draft discard/regeneration, verification-task creation, and OCS updates.

**Request** (`ClassifyRequest`):
- `email: EmailContent`
- `context: CaseContext` — classification accepts `schema_version` 2, 3, or 4.

`EmailContent`:
- `subject: str` (1..500)
- `body: str` (1..50000)
- `from_address: str` (1..320)
- `from_name: Optional[str]` (<=200)
- `received_at: Optional[datetime]` — anchor for relative-date resolution
- `forwarded_context: Optional[dict]` — forwarded/validated reply-source context

`X-Tenant-ID` header is used for per-tenant rate limiting.

**Response** (`ClassifyResponse`):
- `classification: str` — one of the 25 `CLASSIFICATION_CATEGORIES`
- `confidence: float` (0.0–1.0)
- `reasoning: Optional[str]`
- `secondary_intents: Optional[List[str]]`
- `extracted_data: Optional[ExtractedData]` — flat, primary-intent extraction (backward compat)
- `intent_details: Optional[List[IntentDetail]]` — per-intent extraction; `intent_details[0]` is primary and matches `classification`, the rest align with `secondary_intents`
- `tokens_used` / `prompt_tokens` / `completion_tokens: Optional[int]`
- `forbidden_content_detected: List[dict]` (default `[]`)
- `guardrail_validation: Optional[GuardrailValidation]` — guardrails run over the LLM **reasoning** text
- `provider`, `model: Optional[str]`
- `is_fallback: bool = False`
- `classification_evidence_only: bool = True`
- `ai_audit: Optional[AIAuditMetadata]`

---

### POST /generate-draft

Generate a collection email draft (persona-aware, lane-scoped, with the 12-guardrail pipeline). AI is a wording service only — it never selects sender, recipient, escalation level, scope, thread, or tone.

**Hard cut to `context.schema_version == 4`.** Any other version → **HTTP 422** (`"Draft generation requires current datalake context schema_version=4"`). The body returns a `{INVOICE_TABLE}` placeholder that the backend replaces post-generation (unless `skip_invoice_table`/`closure_mode`). On the manifest path, the tone may be overridden from `context.lane.tone_ladder[0]`.

**Request** (`GenerateDraftRequest`):
- `context: CaseContext` (V4 required; see lineage/sendability rules below)
- `sender_persona: Optional[SenderPersona]` — `.name`/`.title` are **deprecated** (backfill `sender_name`/`sender_title` with a `DeprecationWarning`; a mismatch raises)
- `sender_name`, `sender_title`, `sender_company`, `sender_email: Optional[str]` (canonical sender fields)
- `cc_emails: List[str]` (default `[]`)
- `sender_context: Optional[SenderContext]`
- `tone: str = "professional"` — must match `CANONICAL_TONES` (the exact backend-selected tone for this touch)
- `objective: Optional[str]` — `follow_up | promise_reminder | escalation | initial_contact`
- `closure_mode: bool = False`
- `skip_invoice_table: bool = False`
- `trigger_classification: Optional[str]` (<=50) — follow-up context
- `escalation_level: Optional[int]` (0–4)
- `tone_preference: Optional[str]` — `diplomatic | professional | direct`
- `custom_instructions: Optional[str]` (<=1000) — prompt-injection screened (rejected if it contains phrases such as "ignore previous", "system prompt", "you are now", "act as", "override", "bypass", etc.)
- `follow_up_context: Optional[FollowUpContext]` — debtor's original payment claim + Sage verification outcome for follow-up drafts (`trigger_classification`, `verification_id`, `claimed_amount/date/reference`, `matched_amount`, `residual_amount`, `obligation_ids`)

**V4 fail-closed validation** (`validation.py::validate_current_datalake_context`). When `schema_version == 4`, the request requires:
- Lineage fields: `source_sync_run_id`, `application_run_id`, `core_snapshot_watermark`, `application_snapshot_watermark`, `application_decision_cutoff`, `policy_snapshot_id`, `draft_candidate_id`.
- A resolvable recipient email (in `debtor_contact` or `party_contacts`).
- Unless `closure_mode`/follow-up: at least one **sendable** obligation. Sendability (`_is_sendable_candidate`): in `sendable_obligation_ids` if that list is set, not in `blocked_obligation_ids`, not source-disputed/queried, `is_sendable is not False`, `is_chase_eligible is not False`, and overdue under `chase_basis`/`collection_basis` (default `"overdue"`).
- Canonical identity for all payloads: `party.external_id`, `party.provider_type`, and per-obligation `id`/`external_id`/`provider_type`; `PartyInfo.source` must equal `provider_type`.

**Response** (`GenerateDraftResponse`):
- `subject: str`
- `body: str` — single field containing the `{INVOICE_TABLE}` placeholder for standard drafts (no `body_html`/`body_plain`)
- `tone_used: str`
- `invoices_referenced: List[str]` (default `[]`)
- `tokens_used` / `prompt_tokens` / `completion_tokens: Optional[int]`
- `guardrail_validation: Optional[GuardrailValidation]`
- `provider`, `model: Optional[str]`
- `is_fallback: bool = False`
- `reasoning: Optional[dict]` — `tone_rationale`, `strategy`, `key_factors`
- `primary_cta: Optional[str]`
- `follow_up_days: Optional[int]`
- `ai_audit: Optional[AIAuditMetadata]`
- `usage_breakdown: Optional[UsageBreakdown]` — `main_generation` (guardrail tokens excluded) + per-guardrail rollup keyed by guardrail name

---

### POST /generate-draft-from-manifest

Phase 4.4 regional-lake handoff. Loads a draft-candidate manifest from regional S3, batch-hydrates each candidate's `CaseContext` (`CaseContextHydrator.hydrate_batch` — bulk Athena reads in the tenant's `data_lake_region`), then generates per candidate. Partial-failure tolerant. An empty manifest → HTTP 500.

**Request** (`DraftGenerationHandoff`, `src/lake/models.py`, `extra="forbid"`):
- `tenant_id: str`
- `sync_run_id: str`
- `manifest_uri: str` — must be an `s3://` URI
- `data_lake_region: str` — must match an explicit AWS region; regionless handoffs are rejected by the validator

**Response** (`GenerateDraftFromManifestResponse`):
- `tenant_id`, `sync_run_id`, `data_lake_region: str`
- `total`, `generated_count`, `failed_count: int`
- `status: Literal["completed", "partial_failed", "failed"]`
- `results: List[GenerateDraftFromManifestCandidateResult]` — each: `candidate_id`, `party_id`, `lane_id`, `status: Literal["generated", "failed"]`, `draft: Optional[GenerateDraftResponse]`, `error: Optional[str]`

---

### POST /analyze-sent-draft-scope

(New 2026-05-28.) Compares the actually-sent email against the AI-generated draft and a candidate invoice list to determine final invoice scope. Powers AI-attributed sent-scope analysis on the backend. Prompt template `sent_draft_scope_analysis`, version `2026-05-28.v1`. LLM extraction runs at temperature 0.0, caller `sent_scope_analysis`.

**Request** (`AnalyzeSentDraftScopeRequest`):
- `tenant_id`, `party_id`, `draft_id: str`
- `touch_id`, `provider_message_id`, `sent_at: Optional`
- `generated: GeneratedDraftInput` — `subject`, `body_plain` (<=50k), `body_html` (<=100k), `invoice_refs`
- `sent: SentDraftEmailInput` — `subject`, `body_plain`, `body_html`, `from_email`, `to_emails`, `cc_emails`, `bcc_emails`, `reply_to`
- `invoice_candidates: List[SentDraftInvoiceCandidate]` — each: `obligation_id`, `invoice_number`, `document_no`, `currency_code`, `amount_due_native/base`, `due_date`, `days_overdue`, `is_source_disputed`, `collection_status`, `generated_in_draft`

**Response** (`AnalyzeSentDraftScopeResponse`):
- `invoice_refs_sent / retained / operator_added / removed / ambiguous: List[str]`
- `invoice_scope_changed: bool`
- `scope_extraction_confidence: float`
- `scope_extraction_status: Literal["succeeded", "review_required", "failed"]`
- `review_recommended: bool`, `review_reason_codes: List[str]`
- `decisions: List[SentDraftInvoiceScopeDecision]` — each: `invoice_number`, `obligation_id`, `status: Literal["retained_generated_invoice", "operator_added_invoice", "removed_generated_invoice", "not_present", "ambiguous"]`, `confidence`, `evidence`
- `reasoning: Optional[str]`, token fields, `provider`, `model`, `scope_extraction_llm_request_id`, `is_fallback`, `ai_audit`

---

### POST /generate-persona

Cold-start persona generation for escalation contacts (called when an admin saves the escalation hierarchy). One LLM call per contact; per-contact failures are non-fatal. Emits one aggregate `ai_audit` per HTTP response (`inference_profile=persona_gen`).

**Request** (`GeneratePersonaRequest`):
- `contacts: List[PersonaContact]` (max 10) — each: `name`, `title`, `level` (1–4), `style_description` (<=2000), `style_examples: List[str]`
- `total_levels: int = 4` (1–4)

**Response** (`GeneratePersonaResponse`):
- `personas: List[PersonaResult]` — each `PersonaResult` = `name`, `level`, `communication_style`, `formality_level`, `emphasis`
- token fields, `provider`, `model`, `is_fallback`, `ai_audit`

---

### POST /refine-persona

Performance-driven persona refinement, called during sync for senders with >= 10 touches (`inference_profile=persona_refine`).

**Request** (`RefinePersonaRequest`):
- `name`, `title`, `level`
- `current_persona: SenderPersona`
- `performance: SenderPerformanceStats`
- `sender_performance_current: Optional[dict]` — takes precedence over `performance.model_dump()` when provided
- `persona_version: int = 0`, `style_description`, `style_examples`

**Response** (`RefinePersonaResponse`):
- `communication_style: str`, `formality_level: str`, `emphasis: str`, `reasoning: str` (all required, flat — not nested under `persona`)
- token fields, `provider`, `model`, `is_fallback`, `ai_audit`

---

### GET /ping

Liveness check, no LLM calls. Returns `PingResponse` (`status`, `uptime_seconds`).

### GET /health

Shallow health check, no LLM calls. Returns `ShallowHealthResponse` (`status`, `version`, `uptime_seconds`). Safe for load balancers and ECS probes.

### GET /health/llm

Deep provider-aware health check — actually calls the configured LLM provider(s), so it burns quota and requires bearer auth. Use for operator diagnostics, **not** ECS health checks.

Returns `HealthResponse`: `status` (`healthy|degraded|unhealthy`), `version`, `provider`, `model`, `fallback_provider`, `fallback_model`, `fallback_count`, `primary_failures_by_caller: dict[str,int]`, `model_available`, `fallback_available`, `uptime_seconds`.

---

## Classification Categories (25)

`CLASSIFICATION_CATEGORIES` (`src/config/constants.py`), enforced by `ClassificationLLMResponse.validate_classification`:

- **Legal/Compliance (priority):** `INSOLVENCY`, `UNSUBSCRIBE`, `HOSTILE`
- **Payment claims:** `ALREADY_PAID`, `PAYMENT_CONFIRMATION`, `REMITTANCE_ADVICE`, `PARTIAL_PAYMENT_NOTIFICATION`
- **Disputes:** `DISPUTE`, `AMOUNT_DISAGREEMENT`, `PAYMENT_TIMING_DISPUTE`, `DEBTOR_INTERNAL_PROCESSING_BLOCKER`, `RETENTION_CLAIM`
- **Commitments & requests:** `PROMISE_TO_PAY`, `HARDSHIP`, `PLAN_REQUEST`, `REQUEST_INFO`, `REDIRECT`, `ESCALATION_REQUEST`, `QUERY_QUESTION`
- **Engagement:** `COOPERATIVE`, `LEGAL_RESPONSE`
- **Non-actionable:** `OUT_OF_OFFICE`, `EMAIL_BOUNCE`, `GENERIC_ACKNOWLEDGEMENT`
- **Fallback:** `UNCLEAR`

Per-category extraction fields live on `ExtractedData`. Notable shapes:
- `PROMISE_TO_PAY`: `promise_date`, `promise_amount`, `promise_strength` (`firm | soft | aspirational` — firm = full grace-days suppression downstream, soft = half, aspirational = defers next-touch only).
- `PAYMENT_TIMING_DISPUTE`: `claimed_due_date`, `claimed_payment_date`, `payment_timing_reason`.
- `DEBTOR_INTERNAL_PROCESSING_BLOCKER`: `internal_blocker_type` (`goods_receipt_missing | po_issue | approval_pending | payment_run_pending | portal_processing | internal_review | other`), `internal_blocker_reason`, `internal_blocker_owner_hint`.
- `DISPUTE`: `dispute_type`, `dispute_reason`, `invoice_refs`, `disputed_amount`.
- `ALREADY_PAID`: `claimed_amount`, `claimed_date`, `claimed_reference`, `claimed_details`.
- `INSOLVENCY`: `insolvency_type`, `insolvency_details`, `administrator_name`, `administrator_email`, `reference_number`.
- `OUT_OF_OFFICE`: `return_date`; `REDIRECT`: `redirect_name/contact/email`; `EMAIL_BOUNCE`: `bounced_email`.
- `account_wide: Optional[bool]` — explicit account-wide language; drives the ETL scope-resolver fallback.

Multi-intent: `secondary_intents` + per-intent `intent_details`; invoice refs are deduped per intent and a single invoice ref cannot drive two material intents (first interpretation wins).

---

## Guardrails Pipeline

`GuardrailPipeline._get_default_guardrails` registers **12 guardrails** (singleton `guardrail_pipeline`, shared by the generator and the classifier). The pipeline sorts by severity (CRITICAL first). Default execution is **parallel** via a module-level `ThreadPoolExecutor(max_workers=6, thread_name_prefix="guardrail")` (sequential mode supports fail-fast on first CRITICAL).

Severity → blocking: CRITICAL and HIGH **block** (`should_block=True`); MEDIUM = warn/allow; LOW = log-only; REVIEW = non-blocking operator-review finding. Any exception inside a guardrail is converted to a HIGH (blocking) failure by the executor.

| # | Name (registered) | Severity | Blocking? | Checks |
|---|---|---|---|---|
| 1 | `placeholder_validation` | CRITICAL | yes | Hallucinated `[ALL_CAPS]`/`{ALL_CAPS}` placeholders. Only `{INVOICE_TABLE}` is allowed (disallowed under `skip_invoice_table`/`closure_mode`). Pure regex, zero LLM cost, runs first. |
| 2 | `factual_grounding` | CRITICAL | yes | Invoice numbers and monetary amounts in the draft must exist in context obligations. Primary anti-hallucination defense. |
| 3 | `numerical_consistency` | CRITICAL | yes | Stated totals match the authoritative sum of `amount_due` (±0.01); days-overdue match `days_past_due` (±1 day). Skipped in closure mode. |
| 4 | `lane_scope` | CRITICAL | yes | Draft must not reference invoices outside the lane cohort or blocked obligations; validates lane totals. |
| 5 | `identity_scope` | HIGH | yes | Deterministic greeting/sign-off/embedded-email identity checks (recipient first name, sender name, reply-to scope). Replaced the old LLM "entity verification" guardrail. |
| 6 | `overdue_terminology` | HIGH | yes | For `schema_version>=4` with basis `"overdue"`: blocks "outstanding invoice/balance/amount" wording (must say "overdue"). Skipped for closure mode / non-overdue scope. |
| 7 | `policy_grounding` | HIGH | yes | LLM-judge: does the draft make a hard commitment/offer under an unauthorised policy category (`legal_escalation_enabled`, `statutory_interest_enabled`, `discount_allowed`, `settlement_allowed`)? Falls back to a strict narrow regex on LLM unavailability. |
| 8 | `forbidden_content` | REVIEW | no (review finding) | Regex detection of bank/payment details (IBAN, sort code, account number, SWIFT/BIC), legal statute references, and external URLs — surfaced to operator review, never blocks. |
| 9 | `tone_clamping` | HIGH | yes | Confirms a non-empty runtime-selected tone was supplied and the draft honors it (drift safety net). |
| 10 | `semantic_coherence` | MEDIUM | no | LLM-backed: is the draft a coherent, tone-aligned response to the last inbound lane message? Skipped for `mail_mode == "initial"` / no prior inbound. |
| 11 | `temporal_consistency` | MEDIUM | no | Promise dates in the future (today OK; >90 days flagged); prose due dates match obligation `due_date` ±1 day. |
| 12 | `contextual_coherence` | LOW | no | Situational-awareness heuristics + structural checks (refs in prose vs structured obligations, no chasing of paid invoices). Log-only by design. |

`GuardrailValidation` (on both classify and generate responses): `all_passed`, `guardrails_run`, `guardrails_passed`, `blocking_failures`, `warnings`, `review_findings`, `factual_accuracy` (0..1, default 1.0), `results` (per-check pass/fail, severity, expected/found).

Generator retry: when guardrails `should_block` and `len(blocking_guardrails) <= 2`, the generator retries main generation with targeted feedback up to `MAX_GUARDRAIL_RETRIES` (default 2 → 3 attempts total). Pipeline result version constant: `silver_application_v1`.

---

## LLM Providers, Fallback, Retry

`LLMProviderWithFallback` singleton (`src/llm/factory.py`). Every call carries a `caller=` kwarg (`draft_generation`, `classification`, `persona_*`, `sent_scope_analysis`, guardrail callers).

Historical thread relevance uses the same provider policy: Vertex AI is primary
(`gemini-2.5-flash` in `europe-west2`) and OpenAI `gpt-5-mini` is the fallback.
The `/classify-historical-collection-thread` endpoint accepts
`mode=thread_collection_relevance` for the Stage 3 CTR-only shadow. That mode
returns only `collection_related`, `non_collection`, or `uncertain` relevance
evidence; it does not classify promises, disputes, remittances, or create
draft/routing decisions.
The bidirectional Stage 1B–3B shadow uses prompt template `v2` on the same
mode. It supplies a single chronological thread containing debtor inbound,
manual internal outbound, system-generated outbound, and unknown roles. It
still returns only the three relevance labels and remains downstream-disabled.

The collection-email status foundation adds two bounded event endpoints.
`/extract-collection-email-facts` extracts asserted invoice references,
amounts, and due dates from one message plus bounded prior context.  Its output
is reconciled by the backend/ETL against Sage before it can influence chain
state. `/identify-collection-chain` receives that event, bounded context, and
reconciliation outcome codes to decide only `collection`, `non_collection`, or
`uncertain` and the event effect. Neither endpoint receives debtor policy or
can choose a recipient, route, or draft. Both are Vertex-primary, use OpenAI
only for transient provider failures, and return provider/model/fallback and
token audit metadata. The backend records request latency, calculates cost
from those tokens, and persists the request-log identifier.

`/classify-collection-email-event` runs only for an inbound event after its
chain is confirmed as collection. It reuses the operational debtor-response
taxonomy and per-intent extraction contract without executing operational
side effects. Multi-intent replies return one `intent_details` entry per
intent so invoice A payment evidence cannot be conflated with invoice B
promise evidence. Its email-native lifecycle value is persisted as revised
chain-identifier evidence before deterministic status reduction.

The same endpoint accepts `mode=chain_selection_tiebreak` only for a bounded
Stage 4 tie-break after deterministic candidate eligibility. The response may
select one supplied candidate key or abstain; it cannot invent invoice scope,
policy, recipients, provider identifiers, or draft content. The call is
Vertex-primary with OpenAI fallback and its token, cost, latency, provider, and
prompt hashes are returned through the normal `ai_audit` contract.

- **Primary: Vertex AI** (`gemini-2.5-flash` @ `europe-west2`, temperature 0.3). `google-genai` builds a **new `Client` per call**; credentials via AWS→GCP Workload Identity Federation (ECS task-role supplier) in production, ADC locally. Structured output via `response_schema` + `response_mime_type="application/json"`. Retry: tenacity `stop_after_attempt(LLM_MAX_RETRIES=3)`, `wait_exponential(min=2, max=30)`, on `(InternalServerError, ResourceExhausted, ServiceUnavailable)`.
- **Fallback: OpenAI** (`gpt-5-mini`, temperature 0.3, LangChain `ChatOpenAI`). Disabled if it equals the primary or `OPENAI_API_KEY` is unset. Same tenacity retry shape. **No application-level max-token cap** is set (`OPENAI_MAX_TOKENS`/`VERTEX_MAX_TOKENS` env vars were removed 2026-04-29).
- **Anthropic: disabled.** `_create_provider("anthropic")` raises `ValueError` ("disabled until it supports no application-level max token cap"), and the production settings validator rejects `LLM_PROVIDER=anthropic`. `anthropic_provider.py` still exists (defaults `claude-sonnet-4-20250514` / classification `claude-haiku-4-5-20251001`) but is unreachable via the factory.

Fallback behaviour: `complete()` uses OpenAI only for transient quota, timeout, or provider-infrastructure failures. Authentication/configuration failures and invalid structured output fail closed without fallback. Both transient providers failing raises `LLMFallbackExhaustedError`. A per-`(provider, caller)` **cooldown** of `LLM_FALLBACK_COOLDOWN_SECONDS` (300s) is recorded on `LLMRateLimitedError`/`LLMProviderUnavailableError`: a cooling-down primary skips straight to fallback; a cooling-down fallback → `LLMFallbackExhaustedError`. `fallback_count` and `primary_failures_by_caller` are exposed on `/health/llm`.

For Gemini thinking models, `completion_tokens` is the billable output count:
visible candidate tokens plus `thoughts_token_count`. This keeps backend cost
telemetry aligned with Google's output-plus-thinking pricing; `tokens_used`
continues to use the provider's `total_token_count`.

Collection-email and historical collection callers enforce
`LLM_TIMEOUT_SECONDS` independently for Vertex and OpenAI. With the default
60-second provider bound, a timed-out Vertex request can still fall back and
complete within the backend client's 180-second HTTP deadline. Sent-scope
analysis retains its dedicated 40-second provider timeout; other legacy callers
continue to rely on the backend deadline.

Error taxonomy (`src/llm/base.py`): `LLMProviderError → LLMRateLimitedError / LLMProviderUnavailableError / LLMStructuredOutputError / LLMFallbackExhaustedError`.

---

## `ai_audit` / `AIAuditMetadata`

Built by `src/engine/audit.py::build_ai_audit`; returned by classify, generate-draft, both persona endpoints, and sent-scope.

```
ai_provider, ai_model, ai_region: str|None
prompt_template_id, prompt_template_version: str|None
system_prompt_hash, user_prompt_hash, prompt_input_hash: str|None       # SHA-256
guardrail_pipeline_version: str|None                                    # "silver_application_v1"
guardrail_result_ids: list[str]|None
input_silver_version_ids_json: str|None
input_sent_draft_analysis_event_ids_json, input_sent_draft_analysis_hashes_json: str|None
policy_snapshot_id, draft_candidate_id, draft_generation_run_id: str|None
source_sync_run_id, application_run_id: str|None
token_count, prompt_tokens, completion_tokens: int|None
latency_ms: float|None
model_invocation_config: dict|None         # SANITIZED explicit SDK knobs only
model_invocation_config_hash: str|None      # hash over the sanitized dict only
model_version_fingerprint: str|None         # vertex response.model_version / openai system_fingerprint
sdk_library: str|None                       # google-genai | langchain-openai | anthropic
sdk_version: str|None
inference_profile: str|None                 # draft_generation|classification|persona_gen|persona_refine|sent_scope_analysis
```

- `inference_profile` is validated against the fixed 5-value enum in `build_ai_audit`.
- `model_invocation_config` is built from per-provider allow-key lists in `src/llm/_invocation_audit.py` (EU AI Act Article 13 readiness). It never contains prompt text, messages, system instructions, tools, customer data, or PII — a `_FORBIDDEN_KEYS` backstop blocks those, and the response schema is captured by identity only (class name + canonical-JSON hash). PII safety is enforced by `tests/test_invocation_audit_pii_safety.py`.
- Backend telemetry filters this object via its `_AI_AUDIT_ALLOWED_KEYS` allowlist — **new audit fields must be added there or they silently drop** (backend-side contract). Nested `model_invocation_config` has its own allowlist.

---

## Rate Limiting

Per-tenant via the `X-Tenant-ID` header. Key function: `tenant_rate_limit_key` in `src/api/middleware.py`, used by the classify, generate, generate-from-manifest, sent-scope, and persona routes. It keys on `X-Tenant-ID`, falls back to a fixed `"no-tenant"` bucket (never IP), and uses an `"unauthenticated"` penalty bucket when the auth flag is missing. The app-level limiter is keyed by remote IP.

Defaults: `RATE_LIMIT_CLASSIFY="100/minute"` (classify, analyze-sent-draft-scope), `RATE_LIMIT_GENERATE="100/minute"` (generate-draft, generate-draft-from-manifest, generate-persona, refine-persona). `RATE_LIMIT_GATES` still exists in settings but is vestigial (the gates route is deleted).

---

## ECS Idle Shutdown

When `IDLE_SHUTDOWN_SECONDS` > 0 (default `0` = disabled, read in `src/main.py`), a watchdog checks every 30s and sends SIGTERM after sustained idle. Reduces Fargate cost when the AI Engine is unused between syncs.

---

## Configuration / Environment Variables

`Settings(BaseSettings)` in `src/config/settings.py`; each field maps 1:1 to a case-insensitive env var; `.env` auto-loaded; `extra="ignore"`.

| Env var | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `"local"` | local/development/staging/production. Production validator enforces: SERVICE_AUTH_TOKEN set, DEBUG false, non-placeholder VERTEX_PROJECT_ID, readable WIF config, ECS container creds, AWS_REGION/AWS_DEFAULT_REGION set, and **rejects `LLM_PROVIDER=anthropic`**. |
| `API_HOST` | `"0.0.0.0"` | |
| `API_PORT` | `8001` | |
| `DEBUG` | `False` | |
| `ENABLE_API_DOCS` | `False` | Docs auto-enabled outside production regardless; gates `/docs`,`/redoc`,`/openapi.json` exposure + public paths. |
| `CORS_ALLOWED_ORIGINS` | `""` | comma-separated; empty+debug → `*`, empty+prod → none |
| `LLM_PROVIDER` | `"vertex"` | `vertex \| openai \| anthropic` (anthropic disabled) |
| `VERTEX_PROJECT_ID` | `"production-493814"` | GCP project id |
| `VERTEX_LOCATION` | `"europe-west2"` | |
| `VERTEX_MODEL` | `"gemini-2.5-flash"` | |
| `VERTEX_TEMPERATURE` | `0.3` | |
| `VERTEX_WIF_CONFIG_PATH` | `"/app/infra/vertex-wif-config.json"` | AWS→GCP workload identity federation config |
| `OPENAI_API_KEY` | `None` | fallback provider enabler |
| `OPENAI_MODEL` | `"gpt-5-mini"` | |
| `OPENAI_TEMPERATURE` | `0.3` | |
| `ANTHROPIC_API_KEY` | `None` | optional third provider, disabled in factory |
| `ANTHROPIC_MODEL` | `"claude-sonnet-4-20250514"` | |
| `ANTHROPIC_TEMPERATURE` | `0.3` | |
| `ANTHROPIC_CLASSIFICATION_MODEL` | `"claude-haiku-4-5-20251001"` | |
| `DRAFT_TEMPERATURE` | `0.7` | generator |
| `CLASSIFICATION_TEMPERATURE` | `0.2` | classifier |
| `PERSONA_GEN_TEMPERATURE` | `0.7` | |
| `PERSONA_REFINE_TEMPERATURE` | `0.5` | |
| `MAX_GUARDRAIL_RETRIES` | `2` | generator retry loop (`retries + 1` attempts) |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-provider timeout for collection-email and historical collection calls |
| `LLM_MAX_RETRIES` | `3` | tenacity `stop_after_attempt` |
| `LLM_FALLBACK_COOLDOWN_SECONDS` | `300` | per-(provider, caller) cooldown |
| `LOG_LEVEL` | `"INFO"` | |
| `SERVICE_AUTH_TOKEN` | `None` | bearer token for service-to-service auth |
| `RATE_LIMIT_CLASSIFY` | `"100/minute"` | /classify, /analyze-sent-draft-scope |
| `RATE_LIMIT_GENERATE` | `"100/minute"` | /generate-draft, /generate-draft-from-manifest, /generate-persona, /refine-persona |
| `RATE_LIMIT_GATES` | `"100/minute"` | vestigial (gates route deleted) |
| `ATHENA_WORKGROUP` | `"primary"` | regional lake reads (manifest path) |
| `ATHENA_OUTPUT_LOCATION` | `None` | |
| `REGIONAL_LAKE_POLL_INTERVAL_SECONDS` | `1.0` | |
| `REGIONAL_LAKE_QUERY_TIMEOUT_SECONDS` | `60.0` | |

Read outside `Settings`: `IDLE_SHUTDOWN_SECONDS` (`src/main.py`); `AWS_REGION`/`AWS_DEFAULT_REGION`, `AWS_CONTAINER_CREDENTIALS_*` (ECS task-role credential supplier for Vertex WIF).

---

## Error Responses

`OutstandingAIBaseError` hierarchy → structured `ErrorResponse {error, error_code, details, request_id}`. `ErrorCode` enum: `VALIDATION_ERROR, INVALID_REQUEST, INVALID_CLASSIFICATION, MISSING_REQUIRED_FIELD, LLM_PROVIDER_ERROR, LLM_RESPONSE_INVALID, LLM_TIMEOUT, LLM_RATE_LIMITED, INTERNAL_ERROR, SERVICE_UNAVAILABLE`.

| HTTP Code | Cause |
|-----------|-------|
| 422 | Request validation error (Pydantic); also draft generation with `schema_version != 4` |
| 401 | Missing or invalid `SERVICE_AUTH_TOKEN` |
| 429 | Rate limit exceeded for tenant |
| 500 | Internal error (e.g. empty manifest on generate-draft-from-manifest) |
| 503 | LLM provider unavailable / fallback exhausted (`LLMFallbackExhaustedError`) |

---

## Source Files

| What | File |
|------|------|
| Route handlers | `src/api/routes/{classify,generate,sent_scope,persona,health}.py` |
| Request models | `src/api/models/requests/` (`context.py`, `party.py`, `persona.py`, `sent_scope.py`, `validation.py`) |
| Response models | `src/api/models/responses.py` |
| Middleware | `src/api/middleware.py` (RequestID + ServiceAuth + `tenant_rate_limit_key`) |
| Error types | `src/api/errors.py` |
| Constants (categories, tones, placeholders) | `src/config/constants.py` |
| Settings / env | `src/config/settings.py` |
| Engine | `src/engine/` (`classifier.py`, `generator.py`, `generator_prompts.py`, `persona.py`, `sent_scope.py`, `audit.py`, `escalation_validator.py`, `formatters.py`) |
| Guardrails | `src/guardrails/` (`pipeline.py`, `executor.py`, `base.py`, per-guardrail modules) |
| LLM providers | `src/llm/` (`factory.py`, `vertex_provider.py`, `openai_provider.py`, `anthropic_provider.py`, `base.py`, `schemas.py`, `aws_ecs_supplier.py`, `_invocation_audit.py`) |
| Prompts | `src/prompts/` (`classification.py`, `draft_generation.py`, `sent_scope.py`, `_sanitize.py`) |
| Regional lake / manifest handoff | `src/lake/` (`models.py`, `manifest_loader.py`, `regional_reader.py`, `context_hydrator.py`) |
| CaseContext builder (caller) | Backend `services/context_builder.py` |
| Response mirror (caller) | Backend `services/ai_engine/schemas.py` (warn-only Pydantic) |

Cross-repo contract: see `Solvix/docs/CONTRACTS.md`.
