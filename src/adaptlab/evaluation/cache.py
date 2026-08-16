"""Strict exact-request cache primitives for evaluation inference.

The cache stores only successful raw provider responses.  Scoring identity is kept
separate so preserved raw responses can be rescored when scorer/normalizer
versions change without issuing another model request.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.schemas import AdaptationMethod

CACHE_SCHEMA_VERSION = "3"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class InferenceCacheKey:
    """Exact identity of a provider request whose raw response can be reused."""

    benchmark_manifest_hash: str
    example_id: str
    provider: str
    ollama_base_url_policy: str | None
    ollama_version: str | None
    model_id: str
    model_tag: str | None
    model_digest: str | None
    model_revision: str | None
    prompt_hash: str
    method: AdaptationMethod
    temperature: float
    context_length: int | None
    max_tokens: int
    seed: int | None
    stream: bool | None
    think: bool | None
    input_hash: str
    retrieval_run_id: str | None = None
    retrieval_artifact_hash: str | None = None
    retriever_config_hash: str | None = None
    retrieved_context_hash: str | None = None

    def __post_init__(self) -> None:
        _require_hash("benchmark_manifest_hash", self.benchmark_manifest_hash)
        _require_hash("prompt_hash", self.prompt_hash)
        _require_hash("input_hash", self.input_hash)
        retrieval_fields = (
            self.retrieval_run_id,
            self.retrieval_artifact_hash,
            self.retriever_config_hash,
            self.retrieved_context_hash,
        )
        if self.method is AdaptationMethod.RAG:
            if any(value is None for value in retrieval_fields):
                raise ValueError("RAG cache identity requires retrieval run/artifact/config/context provenance")
            if not isinstance(self.retrieval_run_id, str) or not self.retrieval_run_id.strip():
                raise ValueError("retrieval_run_id must be a non-empty string for RAG")
            _require_hash("retrieval_artifact_hash", self.retrieval_artifact_hash)
            _require_hash("retriever_config_hash", self.retriever_config_hash)
            _require_hash("retrieved_context_hash", self.retrieved_context_hash)
        elif any(value is not None for value in retrieval_fields):
            raise ValueError("retrieval provenance fields are only valid for RAG cache identities")
        for name in ("example_id", "model_id", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.ollama_base_url_policy is not None and (
            not isinstance(self.ollama_base_url_policy, str) or not self.ollama_base_url_policy.strip()
        ):
            raise ValueError("ollama_base_url_policy must be a non-empty string or None")
        if self.ollama_version is not None and (
            not isinstance(self.ollama_version, str) or not self.ollama_version.strip()
        ):
            raise ValueError("ollama_version must be a non-empty string or None")
        if self.model_tag is not None and (not isinstance(self.model_tag, str) or not self.model_tag.strip()):
            raise ValueError("model_tag must be a non-empty string or None")
        if self.model_digest is not None:
            _require_hash("model_digest", self.model_digest)
        if self.model_revision is not None and (not isinstance(self.model_revision, str) or not self.model_revision.strip()):
            raise ValueError("model_revision must be a non-empty string or None")
        if self.method not in {AdaptationMethod.PROMPT, AdaptationMethod.ORACLE_CONTEXT, AdaptationMethod.RAG}:
            raise ValueError("cache only supports PROMPT, ORACLE_CONTEXT, and RAG")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.context_length is not None and (
            not isinstance(self.context_length, int)
            or isinstance(self.context_length, bool)
            or self.context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer or None")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or None")
        if self.stream is not None and not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean or None")
        if self.think is not None and not isinstance(self.think, bool):
            raise ValueError("think must be a boolean or None")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["method"] = self.method.value
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def request_hash(self) -> str:
        return _sha256_bytes(self.to_json_bytes())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceCacheKey":
        return cls(
            benchmark_manifest_hash=data["benchmark_manifest_hash"],
            example_id=data["example_id"],
            provider=data["provider"],
            ollama_base_url_policy=data.get("ollama_base_url_policy"),
            ollama_version=data.get("ollama_version"),
            model_id=data["model_id"],
            model_tag=data.get("model_tag"),
            model_digest=data.get("model_digest"),
            model_revision=data.get("model_revision"),
            prompt_hash=data["prompt_hash"],
            method=AdaptationMethod(data["method"]),
            temperature=data["temperature"],
            context_length=data.get("context_length"),
            max_tokens=data["max_tokens"],
            seed=data.get("seed"),
            stream=data.get("stream"),
            think=data.get("think"),
            input_hash=data["input_hash"],
            retrieval_run_id=data.get("retrieval_run_id"),
            retrieval_artifact_hash=data.get("retrieval_artifact_hash"),
            retriever_config_hash=data.get("retriever_config_hash"),
            retrieved_context_hash=data.get("retrieved_context_hash"),
        )


@dataclass(frozen=True)
class ResultArtifactIdentity:
    """Identity of a scored result derived from an exact inference request."""

    inference: InferenceCacheKey
    scorer_version: str
    normalizer_version: str

    def __post_init__(self) -> None:
        for name in ("scorer_version", "normalizer_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference": self.inference.to_dict(),
            "scorer_version": self.scorer_version,
            "normalizer_version": self.normalizer_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultArtifactIdentity":
        return cls(
            inference=InferenceCacheKey.from_dict(data["inference"]),
            scorer_version=data["scorer_version"],
            normalizer_version=data["normalizer_version"],
        )


class ExactRequestCache:
    """Filesystem cache with canonical content hashing and fail-closed reads."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: InferenceCacheKey) -> Path:
        return self.directory / f"{key.request_hash}.json"

    def put(self, key: InferenceCacheKey, response: ModelResponse) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key.to_dict(),
            "response": response.to_dict(),
        }
        payload_bytes = canonical_json_bytes(payload)
        envelope = {
            "payload": payload,
            "payload_sha256": _sha256_bytes(payload_bytes),
        }
        self._path(key).write_bytes(canonical_json_bytes(envelope))

    def get(self, key: InferenceCacheKey) -> ModelResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
                raise ValueError("invalid cache envelope")
            payload = envelope["payload"]
            expected_hash = envelope["payload_sha256"]
            if _sha256_bytes(canonical_json_bytes(payload)) != expected_hash:
                raise ValueError("cache payload hash mismatch")
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("unsupported cache schema version")
            stored_key = InferenceCacheKey.from_dict(payload["key"])
            if stored_key != key or path.stem != key.request_hash:
                raise ValueError("cache identity mismatch")
            response_data = payload["response"]
            return ModelResponse(
                text=response_data["text"],
                input_tokens=response_data.get("input_tokens"),
                output_tokens=response_data.get("output_tokens"),
                latency_ms=response_data.get("latency_ms"),
                model_revision=response_data.get("model_revision"),
                provider_metadata=response_data.get("provider_metadata", {}),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupted cache entry {path.name}: {exc}") from exc
