"""Tests for ECS task-role credential supplier used by Vertex WIF."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.auth import aws, exceptions

from src.llm.aws_ecs_supplier import EcsTaskRoleSupplier


def _response(payload: str, status: int = 200):
    return SimpleNamespace(status=status, data=payload.encode("utf-8"))


def test_get_region_from_aws_region(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    supplier = EcsTaskRoleSupplier()
    assert supplier.get_aws_region(None, None) == "eu-west-2"


def test_get_credentials_from_relative_uri(monkeypatch):
    calls = []

    def request(*, url, method, headers=None):
        calls.append((url, method, headers))
        return _response(
            """
            {
              "AccessKeyId": "AKIA123",
              "SecretAccessKey": "secret",
              "Token": "session",
              "Expiration": "2099-01-01T00:00:00Z"
            }
            """
        )

    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/abc")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", "Bearer test")

    supplier = EcsTaskRoleSupplier()
    credentials = supplier.get_aws_security_credentials(None, request)

    assert isinstance(credentials, aws.AwsSecurityCredentials)
    assert credentials.access_key_id == "AKIA123"
    assert calls == [
        (
            "http://169.254.170.2/v2/credentials/abc",
            "GET",
            {"Authorization": "Bearer test"},
        )
    ]


def test_get_credentials_from_full_uri(monkeypatch):
    def request(*, url, method, headers=None):
        assert url == "http://127.0.0.1/creds"
        assert method == "GET"
        assert headers is None
        return _response(
            """
            {
              "AccessKeyId": "AKIA456",
              "SecretAccessKey": "secret",
              "Token": "session",
              "Expiration": "2099-01-01T00:00:00Z"
            }
            """
        )

    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1/creds")
    supplier = EcsTaskRoleSupplier()

    credentials = supplier.get_aws_security_credentials(None, request)
    assert credentials.access_key_id == "AKIA456"


def test_reads_authorization_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("Bearer file-token\n", encoding="utf-8")

    def request(*, url, method, headers=None):
        assert headers == {"Authorization": "Bearer file-token"}
        return _response(
            """
            {
              "AccessKeyId": "AKIA789",
              "SecretAccessKey": "secret",
              "Token": "session",
              "Expiration": "2099-01-01T00:00:00Z"
            }
            """
        )

    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/creds")
    monkeypatch.delenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", str(token_file))

    supplier = EcsTaskRoleSupplier()
    credentials = supplier.get_aws_security_credentials(None, request)
    assert credentials.access_key_id == "AKIA789"


def test_credential_responses_are_cached(monkeypatch):
    call_count = 0

    def request(*, url, method, headers=None):
        nonlocal call_count
        del url, method, headers
        call_count += 1
        return _response(
            """
            {
              "AccessKeyId": "AKIA999",
              "SecretAccessKey": "secret",
              "Token": "session",
              "Expiration": "2099-01-01T00:00:00Z"
            }
            """
        )

    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/cached")
    supplier = EcsTaskRoleSupplier()

    first = supplier.get_aws_security_credentials(None, request)
    second = supplier.get_aws_security_credentials(None, request)

    assert first.access_key_id == second.access_key_id
    assert call_count == 1


def test_invalid_payload_raises_refresh_error(monkeypatch):
    def request(*, url, method, headers=None):
        del url, method, headers
        return _response('{"AccessKeyId":"AKIA","Expiration":"2099-01-01T00:00:00Z"}')

    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/broken")
    supplier = EcsTaskRoleSupplier()

    with pytest.raises(exceptions.RefreshError, match="missing required fields"):
        supplier.get_aws_security_credentials(None, request)
