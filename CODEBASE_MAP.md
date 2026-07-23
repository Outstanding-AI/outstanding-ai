# Outstanding AI Engine — Codebase Map

Concept → file navigation index.

## Engine (Core AI Logic)

| Concept | File |
|---------|------|
| Email classification | `src/engine/classifier.py` |
| Draft generation (orchestration) | `src/engine/generator.py` | `DraftGenerator.generate()` orchestrates `_assemble_prompt`, `_run_llm_with_guardrails`, `_build_response`; honors upstream collection-policy blocks before model work, blocks non-current/held obligations, and treats temporal thread evidence as continuity context only |
| Draft prompt section composer | `src/engine/generator_prompts.py` | preserves prompt-section ordering and composes request-level drafting guidance; no direct model or data-lake work |
| Shared formatters | `src/engine/formatters.py` |
| Persona management | `src/engine/persona.py` |
| Weekly overdue-report summarisation | `src/engine/weekly_overdue_report_summarizer.py` | one debtor-scoped evidence request; returns one validated earlier/this-week/current/next update per supplied obligation and uses the shared Vertex → OpenAI fallback |

> Gate evaluation is **backend-only** — see `Solvix/services/gate_checker.py`. The AI Engine no longer hosts a gate evaluator.

## Prompts

| Concept | File |
|---------|------|
| Classification prompt | `src/prompts/classification.py` |
| Draft generation prompt | `src/prompts/draft_generation.py` | strategy-aware wording: `single_active_debtor_thread` continues one debtor thread; `invoice_cohort_thread` keeps legacy cohort behavior; collection policy is upstream authority |
| Draft prompt context formatters | `src/prompts/draft_context.py` | pure, bounded rendering of supplied obligation, protocol, thread-temporal, promise, and scheduled-preparation facts; no SQL, HTTP, or LLM calls |
| Prompt sanitization helpers | `src/prompts/_sanitize.py` |

## Guardrails

| Concept | File |
|---------|------|
| Pipeline orchestrator | `src/guardrails/pipeline.py` (12 registered guardrails, 6-worker ThreadPoolExecutor), `executor.py` (execution), `feedback.py` (feedback) |
| Base validator | `src/guardrails/base.py` |
| Placeholder detection | `src/guardrails/placeholder.py` |
| Factual grounding | `src/guardrails/factual_grounding.py` | current demand amounts must come from candidate obligations/current credit context; temporal evidence amounts are continuity-only |
| Numerical consistency | `src/guardrails/numerical.py` |
| Candidate scope | `src/guardrails/lane_scope.py` | validates generated invoice refs against current candidate scope and blocked ids; invoices only present in temporal evidence cannot be chased |
| Identity scope (renamed from entity) | `src/guardrails/identity_scope.py` |
| Overdue terminology | `src/guardrails/overdue_terminology.py` |
| Policy grounding | `src/guardrails/policy_grounding.py` |
| Forbidden content | `src/guardrails/forbidden_content.py` |
| Tone clamping | `src/guardrails/tone_clamping.py` |
| Semantic coherence | `src/guardrails/semantic_coherence.py` |
| Temporal consistency | `src/guardrails/temporal.py` |
| Contextual coherence | `src/guardrails/contextual.py` |

## LLM Providers

| Concept | File |
|---------|------|
| Provider factory | `src/llm/factory.py` |
| Vertex provider | `src/llm/vertex_provider.py` |
| AWS ECS WIF credential supplier | `src/llm/aws_ecs_supplier.py` |
| OpenAI provider | `src/llm/openai_provider.py` |
| Anthropic provider | `src/llm/anthropic_provider.py` |

## API Layer

| Concept | File |
|---------|------|
| Classification endpoint | `src/api/routes/classify.py` — reused by historical collection-thread backfill audit; inbound historical rows use debtor-reply semantics, outbound rows classify operator/Outstanding AI collection actions; backend controls no-cache/no-persist audit mode and token/cost caps; AI returns semantic evidence only, never computes protocol level/touch/escalation state, and never writes App DB/data-lake/mailbox state |
| Generation endpoint | `src/api/routes/generate.py` |
| Persona endpoints | `src/api/routes/persona.py` |
| Weekly overdue-report summary endpoint | `src/api/routes/summarize_weekly_overdue_report.py` |
| Health checks | `src/api/routes/health.py` |
| Request models | `src/api/models/requests/` (package: context.py, party.py, persona.py, validation.py) |
| Response models | `src/api/models/responses.py` |
| Middleware | `src/api/middleware.py` |
| Error types | `src/api/errors.py` |

## Data Lake Hydration

| Concept | File |
|---------|------|
| Context hydration coordinator | `src/lake/context_hydrator.py` | preserves the public hydration API and coordinates batch results into the V4 request context |
| Context projection reads | `src/lake/context_reader.py` | identifier-safe serving-source selection plus tenant-scoped/bounded current-view reads and row grouping; it never constructs `CaseContext` |
| Context evidence normalization | `src/lake/context_evidence.py` | pure Silver-row to `CaseContext` evidence/model transformation; no SQL, HTTP, or provider dependency |
| Context contract assembly | `src/lake/context_projection.py` | composes already-read, shaped evidence into the V4 `CaseContext`; no Athena or provider dependency |
| Case/thread evidence reads | `src/lake/context_reader.py` (`COLLECTION_CASE_THREADS_CURRENT`, `COLLECTION_THREAD_MESSAGE_INVOICE_EVIDENCE_CURRENT`) |
| Case context model fields | `src/api/models/requests/context.py` |

Hydration rules:
- join collection case evidence by `collection_case_id`, `collection_case_thread_id`, and `mail_message_id`;
- never use generic `thread_id` as a cross-table join key;
- do not widen `obligations` from historical message evidence; current collectible obligations remain the only chase scope;
- `collection_policy_context.ai_email_chase_allowed = false` means no draft should be generated.
- Silver Core delta manifests, Silver Application delta signals, Gold skip proofs, and Athena/MSCK partition controls are upstream ETL/backend operational mechanics. AI consumes the current request context it receives from backend/Gold/Silver readers and does not make skip or manifest decisions.

## Config & Utils

| Concept | File |
|---------|------|
| Settings (Pydantic) | `src/config/settings.py` |
| Constants | `src/config/constants.py` |
| JSON extractor | `src/utils/json_extractor.py` |
| App entry point | `src/main.py` |
