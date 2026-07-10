"""API integration tests for Outstanding AI Engine."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


def _mark_current_datalake_context(context):
    watermark = datetime(2026, 5, 6, tzinfo=timezone.utc)
    context.schema_version = 4
    context.source_sync_run_id = "sync-1"
    context.application_run_id = "app-run-1"
    context.core_snapshot_watermark = watermark
    context.application_snapshot_watermark = watermark
    context.application_decision_cutoff = watermark
    context.policy_snapshot_id = "policy-1"
    context.draft_candidate_id = "candidate-1"
    context.collection_basis = "overdue"
    context.chase_basis = "overdue"
    context.debtor_contact = {"email": "ap@example.com"}
    context.sendable_obligation_ids = [str(obligation.id) for obligation in context.obligations]
    for obligation in context.obligations:
        obligation.is_sendable = True
        obligation.is_chase_eligible = True
        obligation.is_overdue = True
        obligation.days_overdue = obligation.days_overdue or obligation.days_past_due or 1
    return context


@pytest.fixture
def client():
    """Unauthenticated test client. Auth is ON (token set in conftest.py)."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test shallow health endpoint returns probe-safe response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "provider" not in data
        assert "model" not in data

    @patch("src.api.routes.health.llm_client")
    def test_llm_health_check(self, mock_llm_client, authed_client):
        """Test deep LLM health endpoint returns provider-aware status."""
        mock_llm_client.health_check = AsyncMock(
            return_value={
                "primary": {"status": "healthy"},
                "fallback": {"status": "healthy"},
            }
        )
        mock_llm_client.provider_name = "vertex"
        mock_llm_client.model_name = "gemini-2.5-flash"
        mock_fallback = type(
            "Fallback", (), {"provider_name": "openai", "model_name": "gpt-5-mini"}
        )()
        mock_llm_client.fallback = mock_fallback
        mock_llm_client.fallback_count = 0
        mock_llm_client.get_failure_metrics.return_value = {"primary_failures_by_caller": {}}

        response = authed_client.get("/health/llm")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["provider"] == "vertex"
        assert data["model"] == "gemini-2.5-flash"
        assert data["fallback_provider"] == "openai"
        assert data["fallback_model"] == "gpt-5-mini"
        assert data["primary_failures_by_caller"] == {}
        assert "uptime_seconds" in data


class TestClassifyEndpoint:
    """Tests for /classify endpoint."""

    def test_classify_requires_auth(self, client):
        """Test classify endpoint rejects unauthenticated requests."""
        response = client.post("/classify", json={})
        assert response.status_code == 401

    def test_classify_requires_email(self, authed_client):
        """Test classify endpoint requires email field."""
        response = authed_client.post("/classify", json={})

        assert response.status_code == 422

    def test_classify_requires_context(self, authed_client):
        """Test classify endpoint requires context field."""
        response = authed_client.post(
            "/classify",
            json={
                "email": {
                    "subject": "Test",
                    "body": "Test body",
                    "from_address": "test@example.com",
                }
            },
        )

        assert response.status_code == 422

    @patch("src.api.routes.classify.classifier")
    def test_classify_defaults_context_schema_version_to_v4(self, mock_classifier, authed_client):
        """Test classify defaults sparse contexts to the current datalake schema."""
        from src.api.models.responses import ClassifyResponse

        mock_classifier.classify = AsyncMock(
            return_value=ClassifyResponse(
                classification="COOPERATIVE",
                confidence=0.88,
                reasoning="Canonical context parsed as v4",
            )
        )

        response = authed_client.post(
            "/classify",
            json={
                "email": {
                    "subject": "Test",
                    "body": "Test body",
                    "from_address": "test@example.com",
                },
                "context": {
                    "party": {
                        "party_id": "party-1",
                        "external_id": "party-ext-1",
                        "provider_type": "sage_200",
                        "customer_code": "CUST001",
                        "name": "Acme Corp",
                        "source": "sage_200",
                    },
                    "obligations": [
                        {
                            "id": "obl-1",
                            "external_id": "INV-1",
                            "provider_type": "sage_200",
                            "invoice_number": "INV-1",
                            "original_amount": 100.0,
                            "amount_due": 100.0,
                        }
                    ],
                },
            },
        )

        assert response.status_code == 200
        assert mock_classifier.classify.await_args.args[0].context.schema_version == 4

    @patch("src.api.routes.classify.classifier")
    def test_classify_success(self, mock_classifier, authed_client, sample_classify_request):
        """Test successful classification."""
        from src.api.models.responses import ClassifyResponse

        mock_response = ClassifyResponse(
            classification="HARDSHIP", confidence=0.92, reasoning="Job loss mentioned"
        )
        mock_classifier.classify = AsyncMock(return_value=mock_response)

        response = authed_client.post(
            "/classify", json=sample_classify_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "HARDSHIP"


class TestHistoricalCollectionThreadEndpoint:
    """Tests for /classify-historical-collection-thread endpoint."""

    def test_historical_collection_thread_requires_auth(self, client):
        response = client.post("/classify-historical-collection-thread", json={})

        assert response.status_code == 401

    @patch(
        "src.api.routes.classify_historical_collection_thread.historical_collection_thread_classifier"
    )
    def test_historical_collection_thread_message_protocol_success(
        self, mock_classifier, authed_client
    ):
        from src.api.models.responses import HistoricalCollectionThreadResponse

        mock_classifier.classify = AsyncMock(
            return_value=HistoricalCollectionThreadResponse(
                classification="same_level_follow_up",
                protocol_touch_type="same_level_follow_up",
                is_escalation=False,
                escalation_kind="none",
                debtor_reply_response=False,
                confidence=0.93,
                reason="Outbound follow-up stayed in the same conversation and same contact level.",
                evidence_message_ids=["msg-1", "msg-2"],
                thread_actions={},
                tokens_used=32,
                prompt_tokens=20,
                completion_tokens=12,
                provider="openai",
                model="gpt-5-mini",
                is_fallback=True,
            )
        )

        response = authed_client.post(
            "/classify-historical-collection-thread",
            json={
                "mode": "message_protocol",
                "message": {
                    "mail_message_id": "msg-2",
                    "conversation_id": "conv-1",
                    "message_role": "outbound_operator",
                    "subject": "Re: Invoice 0000007926",
                    "body": "Following up on the overdue invoice.",
                },
                "prior_messages_summary": [
                    {
                        "mail_message_id": "msg-1",
                        "message_role": "outbound_operator",
                        "subject": "Invoice 0000007926",
                    }
                ],
                "rolling_invoice_state_before": ["0000007926"],
                "rolling_invoice_state_after": ["0000007926"],
                "deterministic_facts": {
                    "contact_transition_fact": "none",
                    "days_since_prior_touch": 7,
                },
                "current_sage_validation": [
                    {
                        "invoice_number": "0000007926",
                        "current_state": "open_overdue",
                        "will_be_chased_if_adopted": True,
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "same_level_follow_up"
        assert data["provider"] == "openai"
        assert data["is_fallback"] is True
        assert data["tokens_used"] == 32
        mock_classifier.classify.assert_awaited_once()

    @patch(
        "src.api.routes.classify_historical_collection_thread.historical_collection_thread_classifier"
    )
    def test_historical_collection_thread_adjudication_success(
        self, mock_classifier, authed_client
    ):
        from src.api.models.responses import HistoricalCollectionThreadResponse

        mock_classifier.classify = AsyncMock(
            return_value=HistoricalCollectionThreadResponse(
                classification="needs_review",
                confidence=0.51,
                reason="Two candidate chains compete for the same currently open exposure.",
                recommended_active_thread_id=None,
                thread_actions={"conv-a": "needs_review", "conv-b": "needs_review"},
                guardrail_warnings=["multiple_competing_threads"],
                provider="openai",
                model="gpt-5-mini",
                is_fallback=True,
            )
        )

        response = authed_client.post(
            "/classify-historical-collection-thread",
            json={
                "mode": "debtor_thread_adjudication",
                "party_id": "party-1",
                "candidate_threads": [
                    {
                        "conversation_id": "conv-a",
                        "current_open_overdue_invoice_numbers": ["0000007926"],
                    },
                    {
                        "conversation_id": "conv-b",
                        "current_open_overdue_invoice_numbers": ["0000007926"],
                    },
                ],
                "current_sage_validation": [
                    {
                        "invoice_number": "0000007926",
                        "current_state": "open_overdue",
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "needs_review"
        assert data["thread_actions"] == {
            "conv-a": "needs_review",
            "conv-b": "needs_review",
        }
        assert data["guardrail_warnings"] == ["multiple_competing_threads"]
        mock_classifier.classify.assert_awaited_once()

    @patch(
        "src.api.routes.classify_historical_collection_thread.historical_collection_thread_classifier"
    )
    def test_historical_collection_thread_relevance_success(self, mock_classifier, authed_client):
        from src.api.models.responses import HistoricalCollectionThreadResponse

        mock_classifier.classify = AsyncMock(
            return_value=HistoricalCollectionThreadResponse(
                relevance_label="collection_related",
                confidence=0.88,
                signal_codes=["explicit_collection_request"],
                evidence_message_ordinals=[1],
                reason="The authored thread contains a collection request.",
                provider="vertex",
                model="gemini-2.5-flash",
            )
        )

        response = authed_client.post(
            "/classify-historical-collection-thread",
            json={
                "mode": "thread_collection_relevance",
                "prior_messages_summary": [
                    {"ordinal": 1, "direction": "outbound", "body": "Please confirm payment."}
                ],
                "deterministic_facts": {"visible_party_match": True},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["relevance_label"] == "collection_related"
        assert data["provider"] == "vertex"
        mock_classifier.classify.assert_awaited_once()


class TestGenerateEndpoint:
    """Tests for /generate-draft endpoint."""

    def test_generate_requires_auth(self, client):
        """Test generate endpoint rejects unauthenticated requests."""
        response = client.post("/generate-draft", json={})
        assert response.status_code == 401

    def test_generate_requires_context(self, authed_client):
        """Test generate endpoint requires context field."""
        response = authed_client.post("/generate-draft", json={"tone": "firm"})

        assert response.status_code == 422

    @patch("src.api.routes.generate.generator")
    def test_generate_success(self, mock_generator, authed_client, sample_generate_draft_request):
        """Test successful draft generation."""
        from src.api.models.responses import GenerateDraftResponse

        _mark_current_datalake_context(sample_generate_draft_request.context)
        mock_response = GenerateDraftResponse(
            subject="Re: Your Account",
            body="Dear Customer,\n\nThank you for reaching out.",
            tone_used="concerned_inquiry",
            invoices_referenced=["INV-123"],
        )
        mock_generator.generate = AsyncMock(return_value=mock_response)

        response = authed_client.post(
            "/generate-draft", json=sample_generate_draft_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["subject"] == "Re: Your Account"
        assert data["body"] == "Dear Customer,\n\nThank you for reaching out."

    @patch("src.api.routes.generate.generator")
    def test_generate_accepts_candidate_credit_context(
        self, mock_generator, authed_client, sample_generate_draft_request
    ):
        from src.api.models.responses import GenerateDraftResponse

        _mark_current_datalake_context(sample_generate_draft_request.context)
        sample_generate_draft_request.context.candidate_credit_context = {
            "currency": "USD",
            "candidate_overdue_amount": 413.78,
            "unapplied_credit_amount": 861.65,
            "net_candidate_amount": 0.0,
            "full_cover": True,
            "invoice_refs": ["0000008064"],
        }
        mock_generator.generate = AsyncMock(
            return_value=GenerateDraftResponse(
                subject="Credit allocation update",
                body="Can you please confirm how the unapplied credit should be allocated?",
                tone_used="professional",
                invoices_referenced=["0000008064"],
            )
        )

        response = authed_client.post(
            "/generate-draft", json=sample_generate_draft_request.model_dump(mode="json")
        )

        assert response.status_code == 200
        request_arg = mock_generator.generate.await_args.args[0]
        assert request_arg.context.candidate_credit_context["full_cover"] is True

    @patch("src.api.routes.generate.generator")
    def test_generate_rejects_legacy_context(
        self, mock_generator, authed_client, sample_generate_draft_request
    ):
        """Production draft generation requires current schema-version 4 context."""
        response = authed_client.post(
            "/generate-draft", json=sample_generate_draft_request.model_dump(mode="json")
        )

        assert response.status_code == 422
        assert "schema_version=4" in response.json()["detail"]
        mock_generator.generate.assert_not_called()

    def test_generate_from_manifest_requires_auth(self, client):
        """Test regional manifest generation rejects unauthenticated requests."""
        response = client.post("/generate-draft-from-manifest", json={})
        assert response.status_code == 401

    def test_generate_from_manifest_success(self, authed_client, sample_case_context):
        """Test regional manifest generation hydrates context and calls generator."""
        from src.api.models.responses import GenerateDraftResponse
        from src.lake import BatchHydrationResult, DraftCandidate

        _mark_current_datalake_context(sample_case_context)
        fake_clients = MagicMock()
        fake_clients.s3.return_value = object()
        fake_reader = object()
        candidate = DraftCandidate(
            party_id="party-1",
            lane_id="lane-1",
            sync_run_id="sync-1",
            candidate_id="candidate-1",
        )
        fake_hydrator = MagicMock()
        fake_hydrator.hydrate_batch.return_value = [
            BatchHydrationResult(candidate=candidate, context=sample_case_context),
        ]
        mock_response = GenerateDraftResponse(
            subject="Re: Your Account",
            body="Dear Customer,\n\nPlease see the attached summary.",
            tone_used="professional",
            invoices_referenced=["INV-123"],
        )

        with (
            patch("src.api.routes.generate.RegionalLakeClients") as mock_clients_cls,
            patch("src.api.routes.generate.RegionalLakeReader") as mock_reader_cls,
            patch("src.api.routes.generate.CaseContextHydrator") as mock_hydrator_cls,
            patch("src.api.routes.generate.load_draft_candidate_manifest") as mock_loader,
            patch("src.api.routes.generate.generator") as mock_generator,
        ):
            mock_clients_cls.from_handoff.return_value = fake_clients
            mock_reader_cls.from_handoff.return_value = fake_reader
            mock_hydrator_cls.return_value = fake_hydrator
            mock_loader.return_value = [candidate]
            mock_generator.generate = AsyncMock(return_value=mock_response)

            response = authed_client.post(
                "/generate-draft-from-manifest",
                json={
                    "tenant_id": "tenant-1",
                    "sync_run_id": "sync-1",
                    "manifest_uri": "s3://bucket/manifest.json",
                    "data_lake_region": "eu-west-2",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total"] == 1
        assert data["generated_count"] == 1
        assert data["failed_count"] == 0
        assert data["results"][0]["candidate_id"] == "candidate-1"
        assert data["results"][0]["status"] == "generated"
        assert data["results"][0]["draft"]["subject"] == "Re: Your Account"
        mock_loader.assert_called_once_with(
            "s3://bucket/manifest.json",
            region_name="eu-west-2",
            expected_tenant_id="tenant-1",
            expected_sync_run_id="sync-1",
            expected_data_lake_region="eu-west-2",
            s3_client=fake_clients.s3.return_value,
        )
        from src.config.settings import settings

        mock_reader_cls.from_handoff.assert_called_once()
        _, reader_kwargs = mock_reader_cls.from_handoff.call_args
        assert reader_kwargs == {
            "workgroup": settings.athena_workgroup,
            "output_location": settings.athena_output_location,
            "poll_interval_seconds": settings.regional_lake_poll_interval_seconds,
            "timeout_seconds": settings.regional_lake_query_timeout_seconds,
        }
        mock_hydrator_cls.assert_called_once_with("tenant-1", fake_reader)
        mock_generator.generate.assert_awaited_once()
        assert mock_generator.generate.await_args.args[0].context == sample_case_context

    def test_generate_from_manifest_rejects_backend_context_payload(self, authed_client):
        """Test the regional handoff accepts only manifest coordinates, never backend context."""
        response = authed_client.post(
            "/generate-draft-from-manifest",
            json={
                "tenant_id": "tenant-1",
                "sync_run_id": "sync-1",
                "manifest_uri": "s3://bucket/manifest.json",
                "data_lake_region": "eu-west-2",
                "context": {"party": {"party_id": "party-1"}},
            },
        )

        assert response.status_code == 422

    def test_generate_from_manifest_returns_candidate_failure(self, authed_client):
        """Test candidate hydration failures are returned explicitly."""
        from src.lake import BatchHydrationResult, ContextHydrationError, DraftCandidate

        fake_clients = MagicMock()
        fake_clients.s3.return_value = object()
        candidate = DraftCandidate(
            party_id="party-1",
            lane_id="lane-1",
            sync_run_id="sync-1",
            candidate_id="candidate-1",
        )
        fake_hydrator = MagicMock()
        fake_hydrator.hydrate_batch.return_value = [
            BatchHydrationResult(
                candidate=candidate,
                error=ContextHydrationError("lane exploded"),
            ),
        ]

        with (
            patch("src.api.routes.generate.RegionalLakeClients") as mock_clients_cls,
            patch("src.api.routes.generate.RegionalLakeReader") as mock_reader_cls,
            patch("src.api.routes.generate.CaseContextHydrator") as mock_hydrator_cls,
            patch("src.api.routes.generate.load_draft_candidate_manifest") as mock_loader,
            patch("src.api.routes.generate.generator") as mock_generator,
        ):
            mock_clients_cls.from_handoff.return_value = fake_clients
            mock_reader_cls.from_handoff.return_value = object()
            mock_hydrator_cls.return_value = fake_hydrator
            mock_loader.return_value = [candidate]

            response = authed_client.post(
                "/generate-draft-from-manifest",
                json={
                    "tenant_id": "tenant-1",
                    "sync_run_id": "sync-1",
                    "manifest_uri": "s3://bucket/manifest.json",
                    "data_lake_region": "eu-west-2",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["generated_count"] == 0
        assert data["failed_count"] == 1
        assert data["results"][0]["status"] == "failed"
        assert "ContextHydrationError: lane exploded" in data["results"][0]["error"]
        mock_generator.generate.assert_not_called()

    def test_generate_from_manifest_rejects_empty_manifest(self, authed_client):
        """Test empty regional manifests fail closed instead of returning zero work."""
        fake_clients = MagicMock()
        fake_clients.s3.return_value = object()

        with (
            patch("src.api.routes.generate.RegionalLakeClients") as mock_clients_cls,
            patch("src.api.routes.generate.RegionalLakeReader") as mock_reader_cls,
            patch("src.api.routes.generate.CaseContextHydrator") as mock_hydrator_cls,
            patch("src.api.routes.generate.load_draft_candidate_manifest") as mock_loader,
            patch("src.api.routes.generate.generator") as mock_generator,
        ):
            mock_clients_cls.from_handoff.return_value = fake_clients
            mock_loader.return_value = []

            response = authed_client.post(
                "/generate-draft-from-manifest",
                json={
                    "tenant_id": "tenant-1",
                    "sync_run_id": "sync-1",
                    "manifest_uri": "s3://bucket/manifest.json",
                    "data_lake_region": "eu-west-2",
                },
            )

        assert response.status_code == 500
        assert "contained no candidates" in response.json()["detail"]
        mock_reader_cls.from_handoff.assert_not_called()
        mock_hydrator_cls.assert_not_called()
        mock_generator.generate.assert_not_called()


# TestGatesEndpoint removed 2026-04-26 — /evaluate-gates route + GateEvaluator
# deleted. Gate evaluation lives in backend services/gate_checker.py
# (CLAUDE.md note #40); the AI-side endpoint had no production callers.
