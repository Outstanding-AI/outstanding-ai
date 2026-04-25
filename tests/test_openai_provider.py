from types import SimpleNamespace

import pytest

from src.llm.openai_provider import OpenAIProvider


class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        return SimpleNamespace(
            content="OK",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
            response_metadata={"id": "resp-test"},
        )


@pytest.mark.asyncio
async def test_openai_complete_non_structured_invokes_plain_client(monkeypatch):
    monkeypatch.setattr("src.llm.openai_provider.ChatOpenAI", _FakeChatOpenAI)

    provider = OpenAIProvider(api_key="test-key", model="gpt-5-mini", max_tokens=10)

    response = await provider.complete(
        system_prompt="sys",
        user_prompt="Reply with OK",
        max_tokens=10,
        caller="health_check",
    )

    assert response.content == "OK"
    assert response.provider == "openai"
    assert response.model == "gpt-5-mini"
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert response.raw_response == {"response_metadata": {"id": "resp-test"}}
