from __future__ import annotations

import json

import pytest

from adaptlab.evaluation.errors import TransientProviderError
from adaptlab.evaluation.providers import ModelRequest, OllamaModelProvider
from adaptlab.evaluation.providers.ollama import _HttpResult


def _request(*, seed: int | None = 1729) -> ModelRequest:
    return ModelRequest(
        system_prompt="Follow the requested format precisely.",
        user_prompt="Question?",
        temperature=0.0,
        max_tokens=64,
        seed=seed,
    )


def _body(*, content: str = "answer", model: str = "qwen3:8b") -> bytes:
    return json.dumps(
        {
            "model": model,
            "message": {"role": "assistant", "content": content, "thinking": "hidden"},
            "done_reason": "stop",
            "prompt_eval_count": 11,
            "eval_count": 3,
        }
    ).encode("utf-8")


def test_ollama_provider_builds_request_with_explicit_controls() -> None:
    captured = {}

    def transport(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["headers"] = dict(http_request.headers)
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _HttpResult(status=200, headers={}, body=_body())

    provider = OllamaModelProvider(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
        context_length=40960,
        think=False,
        stream=False,
        transport=transport,
        max_retries=0,
    )
    response = provider.generate(_request())

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 60.0
    assert captured["payload"] == {
        "messages": [
            {"content": "Follow the requested format precisely.", "role": "system"},
            {"content": "Question?", "role": "user"},
        ],
        "model": "qwen3:8b",
        "options": {"num_ctx": 40960, "num_predict": 64, "seed": 1729, "temperature": 0.0},
        "stream": False,
        "think": False,
    }
    assert response.text == "answer"
    assert response.input_tokens == 11
    assert response.output_tokens == 3
    assert response.model_revision is None
    assert response.provider_metadata == {
        "done_reason": "stop",
        "model": "qwen3:8b",
        "thinking_present": True,
    }


def test_ollama_provider_translates_transient_http_errors() -> None:
    def transport(http_request, timeout):
        return _HttpResult(status=503, headers={}, body=b'{"error":{"message":"unavailable"}}')

    provider = OllamaModelProvider(
        model_id="qwen3:8b",
        transport=transport,
        max_retries=0,
    )
    with pytest.raises(TransientProviderError, match="unavailable"):
        provider.generate(_request())
