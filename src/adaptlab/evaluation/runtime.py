"""Runtime provenance capture for local evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from adaptlab.evaluation.providers.ollama import OllamaModelProvider

_OLLAMA_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?)")


@dataclass(frozen=True)
class OllamaRuntimeProvenance:
    """Recorded Ollama runtime identity for one canonical run."""

    ollama_version: str | None
    ollama_base_url_policy: str | None
    model_tag: str | None
    model_digest: str | None
    context_length: int | None
    stream: bool | None
    think: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ollama_version": self.ollama_version,
            "ollama_base_url_policy": self.ollama_base_url_policy,
            "model_tag": self.model_tag,
            "model_digest": self.model_digest,
            "context_length": self.context_length,
            "stream": self.stream,
            "think": self.think,
        }


def capture_ollama_version() -> str | None:
    """Return the local Ollama CLI version when available."""

    try:
        completed = subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    match = _OLLAMA_VERSION_RE.search(output)
    if match:
        return match.group(1)
    return output or None


def capture_ollama_model_digest(base_url: str, model_tag: str) -> str | None:
    """Return the digest for *model_tag* from the local Ollama registry if exposed."""

    try:
        with urlrequest.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5.0) as response:  # noqa: S310 - local daemon endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urlerror.URLError, TimeoutError, json.JSONDecodeError):
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("name") == model_tag or model.get("model") == model_tag:
            digest = model.get("digest")
            if isinstance(digest, str) and digest.strip():
                return digest
    return None


def capture_ollama_runtime(provider: OllamaModelProvider) -> OllamaRuntimeProvenance:
    """Capture the runtime provenance needed to audit a local Ollama run."""

    if not isinstance(provider, OllamaModelProvider):
        raise TypeError("provider must be an OllamaModelProvider")
    model_tag = provider.model_id
    return OllamaRuntimeProvenance(
        ollama_version=capture_ollama_version(),
        ollama_base_url_policy=provider.base_url,
        model_tag=model_tag,
        model_digest=capture_ollama_model_digest(provider.base_url, model_tag),
        context_length=provider.context_length,
        stream=provider.stream,
        think=provider.think,
    )
