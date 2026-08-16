"""Frozen canonical BM25 configuration for AdaptLab Milestone 4.

The canonical configuration is derived only from the precommitted top-k policy
and its validation-only selection decision.  It is intended to be frozen before
any primary-test retrieval is executed or inspected.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.retrieval.bm25 import (
    BM25_RETRIEVER_NAME,
    BM25_RETRIEVER_VERSION,
    DEFAULT_B,
    DEFAULT_K1,
    bm25_config_hash,
)
from adaptlab.retrieval.policies import (
    INDEXING_POLICY_VERSION,
    TOKENIZATION_POLICY_VERSION,
    TIE_BREAK_POLICY,
    indexing_policy_hash,
    tokenization_policy_hash,
)
from adaptlab.retrieval.query_policy import QUERY_POLICY_VERSION, query_policy_hash
from adaptlab.retrieval.top_k_selection import (
    TOP_K_SELECTION_VERSION,
    top_k_selection_policy_hash,
)

CANONICAL_BM25_CONFIG_VERSION = "canonical-bm25-v1"
CANONICAL_BM25_CONFIG_FILENAME = "canonical_bm25_v1.json"


@dataclass(frozen=True, slots=True)
class CanonicalBM25Config:
    config_version: str
    retriever_name: str
    retriever_version: str
    retriever_config_hash: str
    query_policy_version: str
    query_policy_hash: str
    indexing_policy_version: str
    indexing_policy_hash: str
    tokenization_policy_version: str
    tokenization_policy_hash: str
    k1: float
    b: float
    top_k: int
    tie_break_policy: str
    corpus_hash: str
    benchmark_manifest_hash: str
    selection_policy_version: str
    selection_policy_hash: str
    validation_retrieval_run_id: str
    validation_selection_decision_hash: str

    def __post_init__(self) -> None:
        if self.config_version != CANONICAL_BM25_CONFIG_VERSION:
            raise ValueError("unexpected canonical BM25 config version")
        if self.retriever_name != BM25_RETRIEVER_NAME:
            raise ValueError("canonical retriever_name must be BM25")
        if self.retriever_version != BM25_RETRIEVER_VERSION:
            raise ValueError("canonical retriever_version mismatch")
        if self.retriever_config_hash != bm25_config_hash(k1=self.k1, b=self.b):
            raise ValueError("canonical retriever_config_hash mismatch")
        if self.query_policy_version != QUERY_POLICY_VERSION or self.query_policy_hash != query_policy_hash():
            raise ValueError("canonical query policy mismatch")
        if self.indexing_policy_version != INDEXING_POLICY_VERSION or self.indexing_policy_hash != indexing_policy_hash():
            raise ValueError("canonical indexing policy mismatch")
        if self.tokenization_policy_version != TOKENIZATION_POLICY_VERSION or self.tokenization_policy_hash != tokenization_policy_hash():
            raise ValueError("canonical tokenization policy mismatch")
        if self.tie_break_policy != TIE_BREAK_POLICY:
            raise ValueError("canonical tie-break policy mismatch")
        if self.selection_policy_version != TOP_K_SELECTION_VERSION:
            raise ValueError("canonical top-k selection policy version mismatch")
        if self.selection_policy_hash != top_k_selection_policy_hash():
            raise ValueError("canonical top-k selection policy hash mismatch")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        for field_name in (
            "retriever_config_hash", "query_policy_hash", "indexing_policy_hash",
            "tokenization_policy_hash", "corpus_hash", "benchmark_manifest_hash",
            "selection_policy_hash", "validation_selection_decision_hash",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if not self.validation_retrieval_run_id:
            raise ValueError("validation_retrieval_run_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "b": self.b,
            "benchmark_manifest_hash": self.benchmark_manifest_hash,
            "config_version": self.config_version,
            "corpus_hash": self.corpus_hash,
            "indexing_policy_hash": self.indexing_policy_hash,
            "indexing_policy_version": self.indexing_policy_version,
            "k1": self.k1,
            "query_policy_hash": self.query_policy_hash,
            "query_policy_version": self.query_policy_version,
            "retriever_config_hash": self.retriever_config_hash,
            "retriever_name": self.retriever_name,
            "retriever_version": self.retriever_version,
            "selection_policy_hash": self.selection_policy_hash,
            "selection_policy_version": self.selection_policy_version,
            "tie_break_policy": self.tie_break_policy,
            "tokenization_policy_hash": self.tokenization_policy_hash,
            "tokenization_policy_version": self.tokenization_policy_version,
            "top_k": self.top_k,
            "validation_retrieval_run_id": self.validation_retrieval_run_id,
            "validation_selection_decision_hash": self.validation_selection_decision_hash,
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def canonical_config_hash(self) -> str:
        return sha256_bytes(self.to_json_bytes())

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "CanonicalBM25Config":
        return cls(**raw)  # type: ignore[arg-type]


def build_canonical_bm25_config(
    *,
    selection_decision_path: Path,
    validation_manifest_path: Path,
) -> CanonicalBM25Config:
    """Build the canonical config solely from frozen validation retrieval artifacts."""
    decision_bytes = selection_decision_path.read_bytes()
    decision = json.loads(decision_bytes)
    manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))

    if decision.get("selection_input") != "validation_retrieval_metrics_only":
        raise ValueError("top_k decision was not produced from validation retrieval metrics only")
    if decision.get("policy_version") != TOP_K_SELECTION_VERSION:
        raise ValueError("selection decision policy version mismatch")
    if decision.get("policy_hash") != top_k_selection_policy_hash():
        raise ValueError("selection decision policy hash mismatch")
    if decision.get("run_id") != manifest.get("run_id"):
        raise ValueError("selection decision and validation manifest run IDs differ")
    if manifest.get("retriever_config_hash") != bm25_config_hash(k1=DEFAULT_K1, b=DEFAULT_B):
        raise ValueError("validation retriever config differs from executable frozen BM25 config")

    return CanonicalBM25Config(
        config_version=CANONICAL_BM25_CONFIG_VERSION,
        retriever_name=BM25_RETRIEVER_NAME,
        retriever_version=BM25_RETRIEVER_VERSION,
        retriever_config_hash=str(manifest["retriever_config_hash"]),
        query_policy_version=QUERY_POLICY_VERSION,
        query_policy_hash=query_policy_hash(),
        indexing_policy_version=INDEXING_POLICY_VERSION,
        indexing_policy_hash=indexing_policy_hash(),
        tokenization_policy_version=TOKENIZATION_POLICY_VERSION,
        tokenization_policy_hash=tokenization_policy_hash(),
        k1=DEFAULT_K1,
        b=DEFAULT_B,
        top_k=int(decision["selected_top_k"]),
        tie_break_policy=TIE_BREAK_POLICY,
        corpus_hash=str(manifest["corpus_hash"]),
        benchmark_manifest_hash=str(manifest["benchmark_manifest_hash"]),
        selection_policy_version=TOP_K_SELECTION_VERSION,
        selection_policy_hash=top_k_selection_policy_hash(),
        validation_retrieval_run_id=str(manifest["run_id"]),
        validation_selection_decision_hash=sha256_bytes(decision_bytes),
    )


def verify_frozen_canonical_bm25_config(
    path: Path,
    *,
    selection_decision_path: Path,
    validation_manifest_path: Path,
) -> CanonicalBM25Config:
    """Verify the checked-in canonical config exactly matches its frozen inputs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    frozen = CanonicalBM25Config.from_dict(raw)
    expected = build_canonical_bm25_config(
        selection_decision_path=selection_decision_path,
        validation_manifest_path=validation_manifest_path,
    )
    if frozen.to_dict() != expected.to_dict():
        raise ValueError("frozen canonical BM25 config does not match validation-only selection inputs")
    return frozen
