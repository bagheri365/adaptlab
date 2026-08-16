"""Ollama chat-completions provider adapter for local evaluation smoke tests."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from adaptlab.evaluation.errors import PermanentProviderError, TransientProviderError
from adaptlab.evaluation.providers.base import ModelProvider, ModelRequest, ModelResponse

_TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class _HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes


Transport = Callable[[urlrequest.Request, float], _HttpResult]
SleepFn = Callable[[float], None]


def _default_transport(request: urlrequest.Request, timeout: float) -> _HttpResult:
    with urlrequest.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local daemon endpoint
        return _HttpResult(
            status=int(response.status),
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.read(),
        )


def _error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(payload.get("message"), str):
            return payload["message"]
    return fallback


class OllamaModelProvider(ModelProvider):
    """Minimal Ollama HTTP provider with explicit request controls."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.5,
        context_length: int = 4096,
        think: bool = False,
        stream: bool = False,
        transport: Transport | None = None,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.initial_backoff_seconds = float(initial_backoff_seconds)
        self.context_length = context_length
        self.think = bool(think)
        self.stream = bool(stream)
        self._transport = transport or _default_transport
        self._sleep = sleep_fn or time.sleep

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": self.stream,
            "think": self.think,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": self.context_length,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        return payload

    def _request(self, request: ModelRequest) -> urlrequest.Request:
        body = json.dumps(self._payload(request), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return urlrequest.Request(
            f"{self.base_url}/api/chat",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    def _send_once(self, request: ModelRequest) -> tuple[_HttpResult, float]:
        http_request = self._request(request)
        started = time.perf_counter()
        try:
            result = self._transport(http_request, self.timeout_seconds)
        except urlerror.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            metadata = {"http_status": exc.code}
            message = _error_message(payload, f"Ollama HTTP {exc.code}")
            error_cls = TransientProviderError if exc.code in _TRANSIENT_HTTP_STATUS else PermanentProviderError
            raise error_cls(message, provider=self.provider_name, metadata=metadata) from exc
        except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
            raise TransientProviderError(
                f"Ollama transport failure: {exc}",
                provider=self.provider_name,
                metadata={"reason": "transport_error"},
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result, elapsed_ms

    def _parse_response(self, result: _HttpResult, elapsed_ms: float) -> ModelResponse:
        if result.status < 200 or result.status >= 300:
            try:
                payload = json.loads(result.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            error_cls = TransientProviderError if result.status in _TRANSIENT_HTTP_STATUS else PermanentProviderError
            raise error_cls(
                _error_message(payload, f"Ollama HTTP {result.status}"),
                provider=self.provider_name,
                metadata={"http_status": result.status},
            )

        try:
            payload = json.loads(result.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PermanentProviderError(
                "malformed Ollama response payload",
                provider=self.provider_name,
                metadata={"reason": "malformed_response"},
            ) from exc
        message = payload.get("message")
        if not isinstance(message, dict):
            raise PermanentProviderError(
                "malformed Ollama response payload",
                provider=self.provider_name,
                metadata={"reason": "missing_message"},
            )
        text = message.get("content")
        if not isinstance(text, str):
            raise PermanentProviderError(
                "Ollama response content is not text",
                provider=self.provider_name,
                metadata={"reason": "non_text_response"},
            )
        metadata: dict[str, Any] = {}
        if isinstance(payload.get("model"), str):
            metadata["model"] = payload["model"]
        if isinstance(payload.get("done_reason"), str):
            metadata["done_reason"] = payload["done_reason"]
        if "thinking" in message:
            metadata["thinking_present"] = bool(message.get("thinking") not in (None, ""))
        if "reasoning" in message:
            metadata["reasoning_present"] = bool(message.get("reasoning") not in (None, ""))
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        return ModelResponse(
            text=text,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            latency_ms=elapsed_ms,
            model_revision=None,
            provider_metadata=metadata,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        attempt = 0
        while True:
            try:
                result, elapsed_ms = self._send_once(request)
                return self._parse_response(result, elapsed_ms)
            except TransientProviderError:
                if attempt >= self.max_retries:
                    raise
                delay = self.initial_backoff_seconds * (2**attempt)
                attempt += 1
                if delay > 0:
                    self._sleep(delay)
