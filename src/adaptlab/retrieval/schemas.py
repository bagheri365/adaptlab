"""Typed, deterministic schemas for AdaptLab retrieval artifacts.

Retrieval artifacts are intentionally separate from the frozen benchmark/corpus.
They record immutable provenance needed to trace every result back to the corpus
and policies that produced it, but expose no benchmark write path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import Difficulty, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily

RETRIEVAL_RESULT_SCHEMA_VERSION = "1"
RETRIEVAL_RUN_SCHEMA_VERSION = "1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_nonempty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(field_name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_probability(field_name: str, value: float | None) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be between 0 and 1 or None")


def _require_optional_bool(field_name: str, value: bool | None) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean or None")


@dataclass(frozen=True)
class RetrievalResult:
    """Per-example retrieval output and retrieval-only diagnostics."""

    retrieval_run_id: str
    corpus_hash: str
    example_id: str
    split: Split
    task_family: TaskFamily
    difficulty: Difficulty
    knowledge_state: KnowledgeState
    evidence_status: EvidenceStatus
    split_type: SplitType
    retrieval_eligible: bool
    query_text: str
    query_hash: str
    retriever_name: str
    retriever_version: str
    retriever_config_hash: str
    indexing_policy_version: str
    tokenization_policy_version: str
    top_k: int
    candidate_chunk_ids: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    candidate_ranks: tuple[int, ...]
    gold_chunk_ids: tuple[str, ...]
    required_gold_chunk_ids: tuple[str, ...]
    any_gold_at_1: bool | None
    any_gold_at_3: bool | None
    any_gold_at_5: bool | None
    any_gold_at_k: bool | None
    all_required_gold_at_1: bool | None
    all_required_gold_at_3: bool | None
    all_required_gold_at_5: bool | None
    all_required_gold_at_k: bool | None
    gold_recall_at_1: float | None
    gold_recall_at_3: float | None
    gold_recall_at_5: float | None
    gold_recall_at_k: float | None
    first_gold_reciprocal_rank: float | None
    wrong_version_top1: bool | None
    current_gold_retrieved: bool | None
    obsolete_only_retrieved: bool | None
    current_and_obsolete_retrieved: bool | None
    top1_chunk_id: str | None = None
    top1_score: float | None = None
    top_k_chunk_ids: tuple[str, ...] = ()
    score_margin_top1_top2: float | None = None
    retrieval_returned_any_context: bool | None = None
    wrongly_high_confidence: bool | None = None
    schema_version: str = RETRIEVAL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "retrieval_run_id",
            "example_id",
            "retriever_name",
            "retriever_version",
            "indexing_policy_version",
            "tokenization_policy_version",
        ):
            _require_nonempty(field_name, getattr(self, field_name))
        if not isinstance(self.retrieval_eligible, bool):
            raise ValueError("retrieval_eligible must be a boolean")
        if not isinstance(self.query_text, str):
            raise ValueError("query_text must be a string")
        if not self.retrieval_eligible and self.query_text:
            raise ValueError("retrieval-ineligible results must have an empty query_text")
        for field_name in ("corpus_hash", "query_hash", "retriever_config_hash"):
            _require_sha256(field_name, getattr(self, field_name))
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not (
            len(self.candidate_chunk_ids)
            == len(self.candidate_scores)
            == len(self.candidate_ranks)
        ):
            raise ValueError("candidate chunk IDs, scores, and ranks must have equal lengths")
        if len(self.candidate_chunk_ids) > self.top_k:
            raise ValueError("candidate count cannot exceed top_k")
        if len(set(self.candidate_chunk_ids)) != len(self.candidate_chunk_ids):
            raise ValueError("candidate_chunk_ids must be unique")
        if tuple(self.candidate_ranks) != tuple(range(1, len(self.candidate_ranks) + 1)):
            raise ValueError("candidate_ranks must be contiguous and start at 1")
        for field_name, values in (
            ("candidate_chunk_ids", self.candidate_chunk_ids),
            ("gold_chunk_ids", self.gold_chunk_ids),
            ("required_gold_chunk_ids", self.required_gold_chunk_ids),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if not set(self.required_gold_chunk_ids).issubset(self.gold_chunk_ids):
            raise ValueError("required_gold_chunk_ids must be a subset of gold_chunk_ids")
        for field_name in (
            "any_gold_at_1", "any_gold_at_3", "any_gold_at_5", "any_gold_at_k",
            "all_required_gold_at_1", "all_required_gold_at_3", "all_required_gold_at_5", "all_required_gold_at_k",
            "wrong_version_top1", "current_gold_retrieved", "obsolete_only_retrieved", "current_and_obsolete_retrieved",
            "retrieval_returned_any_context", "wrongly_high_confidence",
        ):
            _require_optional_bool(field_name, getattr(self, field_name))
        for field_name in (
            "gold_recall_at_1", "gold_recall_at_3", "gold_recall_at_5", "gold_recall_at_k",
            "first_gold_reciprocal_rank",
        ):
            _require_probability(field_name, getattr(self, field_name))
        if self.top1_chunk_id is not None:
            _require_nonempty("top1_chunk_id", self.top1_chunk_id)
        for field_name in ("top1_score", "score_margin_top1_top2"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError(f"{field_name} must be numeric or None")
        if any(not isinstance(value, str) or not value.strip() for value in self.top_k_chunk_ids):
            raise ValueError("top_k_chunk_ids must contain non-empty strings")
        if self.schema_version != RETRIEVAL_RESULT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RETRIEVAL_RESULT_SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in (
            "split", "task_family", "difficulty", "knowledge_state", "evidence_status", "split_type"
        ):
            data[field_name] = getattr(self, field_name).value
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        values = dict(data)
        values.update(
            split=Split(data["split"]),
            task_family=TaskFamily(data["task_family"]),
            difficulty=Difficulty(data["difficulty"]),
            knowledge_state=KnowledgeState(data["knowledge_state"]),
            evidence_status=EvidenceStatus(data["evidence_status"]),
            split_type=SplitType(data["split_type"]),
        )
        for field_name in (
            "candidate_chunk_ids", "candidate_scores", "candidate_ranks",
            "gold_chunk_ids", "required_gold_chunk_ids", "top_k_chunk_ids",
        ):
            values[field_name] = tuple(values[field_name])
        return cls(**values)


@dataclass(frozen=True)
class RetrievalRunManifest:
    """Run-level provenance for one immutable retrieval execution."""

    run_id: str
    benchmark_version: str
    benchmark_manifest_hash: str
    git_commit_sha: str
    git_dirty: bool
    corpus_hash: str
    query_policy_version: str
    query_policy_hash: str
    indexing_policy_version: str
    indexing_policy_hash: str
    tokenization_policy_version: str
    tokenization_policy_hash: str
    retriever_name: str
    retriever_version: str
    retriever_config_hash: str
    top_k_values: tuple[int, ...]
    example_count: int
    completed_count: int
    result_hashes: dict[str, str]
    metric_hashes: dict[str, str]
    schema_version: str = RETRIEVAL_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "run_id", "benchmark_version", "git_commit_sha", "query_policy_version",
            "indexing_policy_version", "tokenization_policy_version", "retriever_name",
            "retriever_version",
        ):
            _require_nonempty(field_name, getattr(self, field_name))
        for field_name in (
            "benchmark_manifest_hash", "corpus_hash", "query_policy_hash",
            "indexing_policy_hash", "tokenization_policy_hash", "retriever_config_hash",
        ):
            _require_sha256(field_name, getattr(self, field_name))
        if not isinstance(self.git_dirty, bool):
            raise ValueError("git_dirty must be a boolean")
        if not self.top_k_values:
            raise ValueError("top_k_values must not be empty")
        if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in self.top_k_values):
            raise ValueError("top_k_values must contain positive integers")
        if tuple(sorted(set(self.top_k_values))) != self.top_k_values:
            raise ValueError("top_k_values must be unique and ascending")
        for field_name in ("example_count", "completed_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.completed_count > self.example_count:
            raise ValueError("completed_count cannot exceed example_count")
        for field_name in ("result_hashes", "metric_hashes"):
            mapping = getattr(self, field_name)
            if not isinstance(mapping, dict):
                raise ValueError(f"{field_name} must be a mapping")
            for key, value in mapping.items():
                _require_nonempty(f"{field_name} key", key)
                _require_sha256(f"{field_name}[{key!r}]", value)
        if self.schema_version != RETRIEVAL_RUN_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {RETRIEVAL_RUN_SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalRunManifest":
        values = dict(data)
        values["top_k_values"] = tuple(values["top_k_values"])
        return cls(**values)
