"""Deterministic fake inference provider for evaluation tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from adaptlab.evaluation.errors import PermanentProviderError, TransientProviderError
from adaptlab.evaluation.providers.base import ModelProvider, ModelRequest, ModelResponse


@dataclass(frozen=True)
class FakeSuccess:
    response: ModelResponse


@dataclass(frozen=True)
class FakeTransientFailure:
    message: str = "scripted transient provider failure"


@dataclass(frozen=True)
class FakePermanentFailure:
    message: str = "scripted permanent provider failure"


FakeProviderStep = FakeSuccess | FakeTransientFailure | FakePermanentFailure | ModelResponse


class FakeModelProvider(ModelProvider):
    """Consume a fixed script, one deterministic step per generate call."""

    def __init__(
        self,
        script: Iterable[FakeProviderStep],
        *,
        provider_name: str = "fake",
    ) -> None:
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        self._provider_name = provider_name
        self._script = deque(script)
        self.requests: list[ModelRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def remaining_steps(self) -> int:
        return len(self._script)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        self.requests.append(request)
        if not self._script:
            raise PermanentProviderError(
                "fake provider script exhausted",
                provider=self.provider_name,
                metadata={"reason": "script_exhausted"},
            )

        step = self._script.popleft()
        if isinstance(step, ModelResponse):
            return step
        if isinstance(step, FakeSuccess):
            return step.response
        if isinstance(step, FakeTransientFailure):
            raise TransientProviderError(step.message, provider=self.provider_name)
        if isinstance(step, FakePermanentFailure):
            raise PermanentProviderError(step.message, provider=self.provider_name)
        raise PermanentProviderError(
            f"unsupported fake provider step: {type(step).__name__}",
            provider=self.provider_name,
        )
