from types import SimpleNamespace

import pytest

from src.llm.openai_provider import OpenAIProvider


class _FakeChatOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

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
    _FakeChatOpenAI.instances = []
    monkeypatch.setattr("src.llm.openai_provider.ChatOpenAI", _FakeChatOpenAI)

    provider = OpenAIProvider(api_key="test-key", model="gpt-5-mini")

    response = await provider.complete(
        system_prompt="sys",
        user_prompt="Reply with OK",
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
    assert all("max_tokens" not in instance.kwargs for instance in _FakeChatOpenAI.instances)
