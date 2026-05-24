from src.config.settings import Settings


def test_api_docs_are_disabled_by_default_in_production():
    settings = Settings(environment="production", service_auth_token="test-token")

    assert settings.api_docs_enabled() is False
    assert settings.public_paths() == {"/health", "/ping"}


def test_api_docs_remain_available_outside_production():
    settings = Settings(environment="local")

    assert settings.api_docs_enabled() is True
    assert {"/docs", "/openapi.json", "/redoc"}.issubset(settings.public_paths())
