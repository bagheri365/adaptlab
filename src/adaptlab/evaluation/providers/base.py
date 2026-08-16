"""Vendor-neutral model-provider contracts for evaluation inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptlab.benchmark.io import canonical_json_bytes


@dataclass(frozen=True)
class ModelRequest:
    """Canonical request passed from evaluation code to a model provider."""

    system_prompt: str
    user_prompt: str
    temperature: float
    max_tokens: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if not isinstance(self.user_prompt, str):
            raise ValueError("user_prompt must be a string")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a non-negative number")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True)
class ModelResponse:
    """Vendor-neutral response returned by a model provider."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    model_revision: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be non-negative or None")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be a non-empty string or None")
        if not isinstance(self.provider_metadata, dict):
            raise ValueError("provider_metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


class ModelProvider(ABC):
    """Minimal inference interface implemented by isolated provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier recorded in run provenance."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response or raise a typed ProviderError."""
