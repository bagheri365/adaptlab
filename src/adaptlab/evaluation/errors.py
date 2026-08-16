"""Typed errors emitted by the AdaptLab evaluation provider boundary."""

from __future__ import annotations

from typing import Any


class EvaluationError(RuntimeError):
    """Base class for failures normalized by the evaluation harness."""


class ProviderError(EvaluationError):
    """Normalized inference-provider failure.

    Provider implementations should translate vendor-specific exceptions into one
    of the typed subclasses below before an error reaches the evaluation runner.
    """

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("provider error message must be a non-empty string")
        self.provider = provider
        self.metadata = dict(metadata or {})
        super().__init__(message)


class TransientProviderError(ProviderError):
    """A provider failure that may be retried by runner policy."""

    retryable = True


class PermanentProviderError(ProviderError):
    """A provider failure that must not be retried as transient."""

    retryable = False
