"""Canonical Milestone 4 RAG run completeness and resume contracts.

These contracts deliberately distinguish retrieval quality from infrastructure
success. A retrieval miss is still a valid model-inference example; only an
unresolved provider failure makes a canonical RAG run incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes

RAG_COMPLETENESS_SCHEMA_VERSION = "rag-completeness-v1"
CANONICAL_RAG_EXPECTED_COUNT = 400


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RAGExampleCompletion:
    """Per-example completion state for a canonical RAG run.

    ``retrieval_succeeded`` means the frozen retrieval record was available and
    consumed. It does *not* mean relevant/gold evidence was retrieved.
    """

    example_id: str
    retrieval_succeeded: bool
    model_response_succeeded: bool
    provider_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if self.model_response_succeeded and self.provider_error is not None:
            raise ValueError("successful model response cannot also have provider_error")
        if not self.model_response_succeeded and not self.provider_error:
            raise ValueError("failed model response must preserve provider_error")
        if not self.retrieval_succeeded:
            raise ValueError("canonical RAG example must consume a frozen retrieval/bypass record")


@dataclass(frozen=True, slots=True)
class RAGRunIdentity:
    """Identity fields that must change when the frozen retrieval artifact changes."""

    canonical_rag_config_hash: str
    retrieval_artifact_hash: str
    benchmark_manifest_hash: str

    def __post_init__(self) -> None:
        _require_sha256("canonical_rag_config_hash", self.canonical_rag_config_hash)
        _require_sha256("retrieval_artifact_hash", self.retrieval_artifact_hash)
        _require_sha256("benchmark_manifest_hash", self.benchmark_manifest_hash)

    @property
    def run_identity_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "schema_version": RAG_COMPLETENESS_SCHEMA_VERSION,
            "canonical_rag_config_hash": self.canonical_rag_config_hash,
            "retrieval_artifact_hash": self.retrieval_artifact_hash,
            "benchmark_manifest_hash": self.benchmark_manifest_hash,
        }))


@dataclass(frozen=True, slots=True)
class RAGCompletenessRecord:
    expected_count: int
    represented_count: int
    completed_successful_model_responses: int
    unresolved_provider_failures: int
    valid: bool

    def __post_init__(self) -> None:
        for name in (
            "expected_count",
            "represented_count",
            "completed_successful_model_responses",
            "unresolved_provider_failures",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.completed_successful_model_responses + self.unresolved_provider_failures != self.represented_count:
            raise ValueError("successful responses + provider failures must equal represented_count")


def canonical_rag_completeness(
    completions: Iterable[RAGExampleCompletion],
    *,
    expected_count: int = CANONICAL_RAG_EXPECTED_COUNT,
) -> RAGCompletenessRecord:
    """Compute the strict canonical RAG completion gate.

    Retrieval misses/partial retrieval/wrong-version retrieval do not enter the
    infrastructure completeness calculation. Every represented example simply
    needs a valid frozen retrieval or bypass record plus a successful model
    response.
    """

    rows = tuple(completions)
    ids = [row.example_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical RAG completion records contain duplicate example IDs")
    successful = sum(row.model_response_succeeded for row in rows)
    provider_failures = sum(not row.model_response_succeeded for row in rows)
    valid = (
        len(rows) == expected_count
        and successful == expected_count
        and provider_failures == 0
    )
    return RAGCompletenessRecord(
        expected_count=expected_count,
        represented_count=len(rows),
        completed_successful_model_responses=successful,
        unresolved_provider_failures=provider_failures,
        valid=valid,
    )


def require_canonical_rag_complete(record: RAGCompletenessRecord) -> None:
    if not record.valid:
        raise ValueError(
            "canonical RAG run incomplete: "
            f"expected_count={record.expected_count}, "
            f"represented_count={record.represented_count}, "
            f"completed_successful_model_responses={record.completed_successful_model_responses}, "
            f"unresolved_provider_failures={record.unresolved_provider_failures}"
        )


def resume_identity_matches(prior: RAGRunIdentity, current: RAGRunIdentity) -> bool:
    """Return whether preserved RAG results may be resumed under current identity."""

    return prior == current and prior.run_identity_hash == current.run_identity_hash
