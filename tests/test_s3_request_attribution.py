from __future__ import annotations

from pathlib import Path

from src.common.s3_request_attribution import (
    S3RequestAccumulator,
    create_instrumented_s3_client,
    s3_request_attribution_context,
)


def test_ai_s3_request_accumulator_snapshot():
    with s3_request_attribution_context() as accumulator:
        assert isinstance(accumulator, S3RequestAccumulator)

    assert accumulator.snapshot()["total_requests"] == 0


def test_ai_instrumented_s3_client_preserves_real_client_interface(monkeypatch):
    class _Events:
        def __init__(self) -> None:
            self.registered: list[str] = []

        def register(self, name, _handler):
            self.registered.append(name)

    class _Client:
        def __init__(self) -> None:
            self.meta = type("Meta", (), {"events": _Events()})()

    class _Boto3:
        def __init__(self) -> None:
            self.client_obj = _Client()

        def client(self, service_name, **kwargs):
            assert service_name == "s3"
            assert kwargs == {"region_name": "eu-west-2"}
            return self.client_obj

    fake = _Boto3()
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)

    client = create_instrumented_s3_client(region_name="eu-west-2", source="ai.lake_reader.test")

    assert client is fake.client_obj
    assert "before-parameter-build.s3.*" in client.meta.events.registered
    assert "after-call.s3.*" in client.meta.events.registered


def test_ai_raw_s3_client_construction_is_limited_to_instrumented_factory():
    repo = Path(__file__).resolve().parents[1]
    hits: list[str] = []
    for file in (repo / "src").rglob("*.py"):
        text = file.read_text(encoding="utf-8")
        if 'boto3.client("s3"' in text or "boto3.client('s3'" in text:
            hits.append(str(file.relative_to(repo)))

    assert sorted(hits) == ["src/common/s3_request_attribution.py"]
