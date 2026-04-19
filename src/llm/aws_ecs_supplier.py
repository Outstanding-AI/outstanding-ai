"""AWS ECS task-role credential supplier for Google WIF."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from google.auth import aws, exceptions

ECS_METADATA_HOST = "http://169.254.170.2"
EXPIRY_SKEW_SECONDS = 60


class EcsTaskRoleSupplier(aws.AwsSecurityCredentialsSupplier):
    """Load AWS task-role credentials from the ECS metadata endpoint."""

    def __init__(self) -> None:
        self._cached_credentials: aws.AwsSecurityCredentials | None = None
        self._cached_expiry: datetime | None = None

    def get_aws_region(self, context, request) -> str:
        del context, request
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            raise exceptions.RefreshError(
                "AWS_REGION or AWS_DEFAULT_REGION must be set for Vertex WIF on ECS"
            )
        return region

    def get_aws_security_credentials(self, context, request) -> aws.AwsSecurityCredentials:
        del context
        if self._cached_credentials and self._cached_expiry:
            if (
                datetime.now(timezone.utc) + timedelta(seconds=EXPIRY_SKEW_SECONDS)
                < self._cached_expiry
            ):
                return self._cached_credentials

        endpoint = self._credentials_endpoint()
        headers = self._authorization_headers()
        response = request(url=endpoint, method="GET", headers=headers or None)
        payload = self._decode_response(response)
        required = {"AccessKeyId", "SecretAccessKey", "Token", "Expiration"}
        missing = sorted(required - payload.keys())
        if missing:
            raise exceptions.RefreshError(
                f"ECS credential response missing required fields: {', '.join(missing)}"
            )

        expiry = self._parse_expiry(payload["Expiration"])
        credentials = aws.AwsSecurityCredentials(
            access_key_id=payload["AccessKeyId"],
            secret_access_key=payload["SecretAccessKey"],
            session_token=payload["Token"],
        )
        self._cached_credentials = credentials
        self._cached_expiry = expiry
        return credentials

    def _credentials_endpoint(self) -> str:
        full_uri = os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        if full_uri:
            return full_uri

        relative_uri = os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        if relative_uri:
            return f"{ECS_METADATA_HOST}{relative_uri}"

        raise exceptions.RefreshError(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI or AWS_CONTAINER_CREDENTIALS_FULL_URI "
            "must be set for ECS task-role auth"
        )

    def _authorization_headers(self) -> dict[str, str]:
        token = os.environ.get("AWS_CONTAINER_AUTHORIZATION_TOKEN")
        token_file = os.environ.get("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
        if not token and token_file:
            try:
                with open(token_file, encoding="utf-8") as fh:
                    token = fh.read().strip()
            except OSError as exc:
                raise exceptions.RefreshError(
                    f"Unable to read AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE: {exc}"
                ) from exc
        return {"Authorization": token} if token else {}

    def _decode_response(self, response: Any) -> dict[str, Any]:
        body = response.data.decode("utf-8") if hasattr(response.data, "decode") else response.data
        if response.status != 200:
            raise exceptions.RefreshError(
                f"ECS credential endpoint returned {response.status}: {body}"
            )
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise exceptions.RefreshError(
                f"ECS credential endpoint returned invalid JSON: {body}"
            ) from exc
        if not isinstance(payload, dict):
            raise exceptions.RefreshError(
                f"ECS credential endpoint returned unexpected payload type: {type(payload).__name__}"
            )
        return payload

    def _parse_expiry(self, value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        try:
            expiry = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise exceptions.RefreshError(
                f"ECS credential expiration is not a valid ISO timestamp: {value}"
            ) from exc
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry
