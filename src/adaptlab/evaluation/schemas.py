"""Typed, deterministic schemas for AdaptLab evaluation artifacts.

Evaluation artifacts are deliberately separate from benchmark-generation artifacts.
This module contains no benchmark write path: a run records the immutable benchmark
manifest identity it consumed, but never mutates that benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    ScoringRule,
    Split,
    SplitType,
    TaskFamily,
)

EVALUATION_RUN_SCHEMA_VERSION = "1"
EVALUATION_RESULT_SCHEMA_VERSION = "1"
FROZEN_BENCHMARK_TAG = "v0.0-benchmark"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AdaptationMethod(str, Enum):
    """Stable adaptation-method vocabulary.

    Milestone 3 execution supports PROMPT and ORACLE_CONTEXT only. The remaining
    values reserve the experiment vocabulary without implementing those methods.
    """

    PROMPT = "PROMPT"
    ORACLE_CONTEXT = "ORACLE_CONTEXT"
    RAG = "RAG"
    LORA = "LORA"
    LORA_RAG = "LORA_RAG"


MILESTONE_3_METHODS = frozenset(
    {AdaptationMethod.PROMPT, AdaptationMethod.ORACLE_CONTEXT}
)


class EvaluationRunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


def _require_nonempty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(field_name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class ModelInput:
    """Canonical system/user input captured in each evaluation result."""

    system: str
    user: str

    def __post_init__(self) -> None:
        if not isinstance(self.system, str):
            raise ValueError("system must be a string")
        if not isinstance(self.user, str):
            raise ValueError("user must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"system": self.system, "user": self.user}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInput":
        return cls(system=data["system"], user=data["user"])


@dataclass(frozen=True)
class EvaluationRun:
    """Run-level provenance for one evaluation condition."""

    run_id: str
    benchmark_version: str
    benchmark_manifest_hash: str
    model_id: str
    model_tag: str | None
    model_digest: str | None
    model_revision: str | None
    provider: str
    ollama_version: str | None
    ollama_base_url_policy: str | None
    adaptation_method: AdaptationMethod
    prompt_version: str
    prompt_hash: str
    scorer_version: str
    normalizer_version: str
    temperature: float
    max_tokens: int
    seed: int | None
    context_length: int | None
    stream: bool | None
    think: bool | None
    started_at: str
    completed_at: str | None
    status: EvaluationRunStatus
    expected_count: int = 0
    completed_successful_responses: int = 0
    completeness_valid: bool = False
    benchmark_tag: str = FROZEN_BENCHMARK_TAG
    schema_version: str = EVALUATION_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "benchmark_version",
            "model_id",
            "provider",
            "prompt_version",
            "scorer_version",
            "normalizer_version",
            "started_at",
        ):
            _require_nonempty(field_name, getattr(self, field_name))
        if self.model_revision is not None:
            _require_nonempty("model_revision", self.model_revision)
        if self.completed_at is not None:
            _require_nonempty("completed_at", self.completed_at)
        _require_sha256("benchmark_manifest_hash", self.benchmark_manifest_hash)
        _require_sha256("prompt_hash", self.prompt_hash)
        if self.benchmark_tag != FROZEN_BENCHMARK_TAG:
            raise ValueError(
                f"benchmark_tag must reference frozen {FROZEN_BENCHMARK_TAG!r}"
            )
        if self.model_tag is not None:
            _require_nonempty("model_tag", self.model_tag)
        if self.model_digest is not None:
            _require_sha256("model_digest", self.model_digest)
        if self.ollama_version is not None:
            _require_nonempty("ollama_version", self.ollama_version)
        if self.ollama_base_url_policy is not None:
            _require_nonempty("ollama_base_url_policy", self.ollama_base_url_policy)
        if self.adaptation_method not in MILESTONE_3_METHODS:
            raise ValueError(
                "Milestone 3 only implements PROMPT and ORACLE_CONTEXT"
            )
        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative number")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or None")
        if self.context_length is not None and (
            not isinstance(self.context_length, int)
            or isinstance(self.context_length, bool)
            or self.context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer or None")
        if self.stream is not None and not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean or None")
        if self.think is not None and not isinstance(self.think, bool):
            raise ValueError("think must be a boolean or None")
        if not isinstance(self.expected_count, int) or isinstance(self.expected_count, bool) or self.expected_count < 0:
            raise ValueError("expected_count must be a non-negative integer")
        if (
            not isinstance(self.completed_successful_responses, int)
            or isinstance(self.completed_successful_responses, bool)
            or self.completed_successful_responses < 0
        ):
            raise ValueError("completed_successful_responses must be a non-negative integer")
        if not isinstance(self.completeness_valid, bool):
            raise ValueError("completeness_valid must be a boolean")
        if self.schema_version != EVALUATION_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EVALUATION_RUN_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["adaptation_method"] = self.adaptation_method.value
        data["status"] = self.status.value
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRun":
        return cls(
            run_id=data["run_id"],
            benchmark_version=data["benchmark_version"],
            benchmark_manifest_hash=data["benchmark_manifest_hash"],
            model_id=data["model_id"],
            model_tag=data.get("model_tag"),
            model_digest=data.get("model_digest"),
            model_revision=data.get("model_revision"),
            provider=data["provider"],
            ollama_version=data.get("ollama_version"),
            ollama_base_url_policy=data.get("ollama_base_url_policy"),
            adaptation_method=AdaptationMethod(data["adaptation_method"]),
            prompt_version=data["prompt_version"],
            prompt_hash=data["prompt_hash"],
            scorer_version=data["scorer_version"],
            normalizer_version=data["normalizer_version"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            seed=data.get("seed"),
            context_length=data.get("context_length"),
            stream=data.get("stream"),
            think=data.get("think"),
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=EvaluationRunStatus(data["status"]),
            expected_count=data.get("expected_count", 0),
            completed_successful_responses=data.get("completed_successful_responses", 0),
            completeness_valid=data.get("completeness_valid", False),
            benchmark_tag=data.get("benchmark_tag", FROZEN_BENCHMARK_TAG),
            schema_version=data.get("schema_version", EVALUATION_RUN_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Per-example inference, normalization, scoring, and provider metadata."""

    example_id: str
    split: Split
    task_family: TaskFamily
    difficulty: Difficulty
    behavior_type: BehaviorType | None
    knowledge_state: KnowledgeState
    evidence_status: EvidenceStatus
    split_type: SplitType
    input_hash: str
    model_input: ModelInput
    raw_output: str | None
    normalized_output: Any
    expected_output: Any
    score: float | None
    scoring_rule: ScoringRule
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    provider_error: str | None
    retry_count: int
    schema_version: str = EVALUATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty("example_id", self.example_id)
        _require_sha256("input_hash", self.input_hash)
        if self.raw_output is not None and not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string or None")
        if self.score is not None and (
            not isinstance(self.score, (int, float)) or isinstance(self.score, bool)
        ):
            raise ValueError("score must be numeric or None")
        if self.latency_ms is not None and (
            not isinstance(self.latency_ms, (int, float)) or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be non-negative or None")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if self.provider_error is not None:
            _require_nonempty("provider_error", self.provider_error)
        if (
            not isinstance(self.retry_count, int)
            or isinstance(self.retry_count, bool)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a non-negative integer")
        if self.schema_version != EVALUATION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EVALUATION_RESULT_SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in (
            "split",
            "task_family",
            "difficulty",
            "behavior_type",
            "knowledge_state",
            "evidence_status",
            "split_type",
            "scoring_rule",
        ):
            value = getattr(self, field_name)
            data[field_name] = value.value if value is not None else None
        data["model_input"] = self.model_input.to_dict()
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        return cls(
            example_id=data["example_id"],
            split=Split(data["split"]),
            task_family=TaskFamily(data["task_family"]),
            difficulty=Difficulty(data["difficulty"]),
            behavior_type=(
                BehaviorType(data["behavior_type"])
                if data.get("behavior_type") is not None
                else None
            ),
            knowledge_state=KnowledgeState(data["knowledge_state"]),
            evidence_status=EvidenceStatus(data["evidence_status"]),
            split_type=SplitType(data["split_type"]),
            input_hash=data["input_hash"],
            model_input=ModelInput.from_dict(data["model_input"]),
            raw_output=data.get("raw_output"),
            normalized_output=data.get("normalized_output"),
            expected_output=data.get("expected_output"),
            score=data.get("score"),
            scoring_rule=ScoringRule(data["scoring_rule"]),
            latency_ms=data.get("latency_ms"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            provider_error=data.get("provider_error"),
            retry_count=data["retry_count"],
            schema_version=data.get("schema_version", EVALUATION_RESULT_SCHEMA_VERSION),
        )
