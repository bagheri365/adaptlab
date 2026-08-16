from __future__ import annotations

import pytest

from adaptlab.evaluation.errors import PermanentProviderError, TransientProviderError
from adaptlab.evaluation.providers import (
    FakeModelProvider,
    FakePermanentFailure,
    FakeSuccess,
    FakeTransientFailure,
    ModelRequest,
    ModelResponse,
)


def request() -> ModelRequest:
    return ModelRequest(
        system_prompt="Follow the format.",
        user_prompt="Question?",
        temperature=0.0,
        max_tokens=32,
        seed=7,
    )


def test_model_request_serializes_deterministically() -> None:
    first = request()
    second = request()
    assert first.to_json_bytes() == second.to_json_bytes()
    assert first.to_json_bytes().endswith(b"\n")


def test_model_request_validates_generation_parameters() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ModelRequest("", "", -0.1, 10)
    with pytest.raises(ValueError, match="max_tokens"):
        ModelRequest("", "", 0.0, 0)
    with pytest.raises(ValueError, match="seed"):
        ModelRequest("", "", 0.0, 10, True)


def test_model_response_serializes_and_validates_usage() -> None:
    response = ModelResponse(
        text="answer",
        input_tokens=10,
        output_tokens=2,
        latency_ms=12.5,
        model_revision="snapshot-1",
        provider_metadata={"request_id": "abc"},
    )
    assert b'"text": "answer"' in response.to_json_bytes()
    with pytest.raises(ValueError, match="input_tokens"):
        ModelResponse(text="x", input_tokens=-1)


def test_fake_provider_returns_scripted_successes_in_order() -> None:
    first = ModelResponse(text="one", input_tokens=2, output_tokens=1)
    second = ModelResponse(text="two")
    provider = FakeModelProvider([FakeSuccess(first), second])

    assert provider.generate(request()) == first
    assert provider.generate(request()) == second
    assert provider.remaining_steps == 0
    assert provider.requests == [request(), request()]


def test_fake_provider_supports_transient_then_success() -> None:
    expected = ModelResponse(text="recovered")
    provider = FakeModelProvider([FakeTransientFailure("rate limit"), expected])

    with pytest.raises(TransientProviderError, match="rate limit") as exc_info:
        provider.generate(request())
    assert exc_info.value.retryable is True
    assert exc_info.value.provider == "fake"
    assert provider.generate(request()) == expected


def test_fake_provider_supports_permanent_failure() -> None:
    provider = FakeModelProvider([FakePermanentFailure("invalid request")])

    with pytest.raises(PermanentProviderError, match="invalid request") as exc_info:
        provider.generate(request())
    assert exc_info.value.retryable is False
    assert exc_info.value.provider == "fake"


def test_fake_provider_script_exhaustion_is_typed_permanent_error() -> None:
    provider = FakeModelProvider([])
    with pytest.raises(PermanentProviderError) as exc_info:
        provider.generate(request())
    assert exc_info.value.metadata == {"reason": "script_exhausted"}


def test_provider_error_metadata_is_copied() -> None:
    metadata = {"status": 503}
    error = TransientProviderError("unavailable", provider="x", metadata=metadata)
    metadata["status"] = 200
    assert error.metadata == {"status": 503}


def test_fake_provider_does_not_read_or_require_credentials() -> None:
    provider = FakeModelProvider([ModelResponse(text="ok")])
    assert provider.generate(request()).text == "ok"
