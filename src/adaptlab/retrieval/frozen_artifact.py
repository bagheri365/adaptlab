"""Freeze canonical retrieval results for deterministic downstream RAG consumption."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.retrieval.schemas import RetrievalResult, RetrievalRunManifest

FROZEN_RETRIEVAL_ARTIFACT_VERSION = "canonical-retrieval-artifact-v1"


@dataclass(frozen=True)
class FrozenRetrievalEntry:
    example_id: str
    retrieval_eligible: bool
    chunk_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not isinstance(self.retrieval_eligible, bool):
            raise ValueError("retrieval_eligible must be a boolean")
        if not (len(self.chunk_ids) == len(self.ranks) == len(self.scores)):
            raise ValueError("chunk_ids, ranks, and scores must have equal lengths")
        if tuple(self.ranks) != tuple(range(1, len(self.ranks) + 1)):
            raise ValueError("ranks must be contiguous and start at 1")
        if not self.retrieval_eligible and (self.chunk_ids or self.ranks or self.scores):
            raise ValueError("retrieval bypass entries must not contain retrieved chunks")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenRetrievalEntry":
        return cls(
            example_id=data["example_id"],
            retrieval_eligible=data["retrieval_eligible"],
            chunk_ids=tuple(data["chunk_ids"]),
            ranks=tuple(data["ranks"]),
            scores=tuple(data["scores"]),
        )


@dataclass(frozen=True)
class FrozenRetrievalArtifact:
    retrieval_run_id: str
    retriever_config_hash: str
    corpus_hash: str
    benchmark_manifest_hash: str
    source_results_hash: str
    entries: tuple[FrozenRetrievalEntry, ...]
    retrieval_artifact_hash: str
    artifact_version: str = FROZEN_RETRIEVAL_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not self.retrieval_run_id:
            raise ValueError("retrieval_run_id must be non-empty")
        if self.artifact_version != FROZEN_RETRIEVAL_ARTIFACT_VERSION:
            raise ValueError("unexpected frozen retrieval artifact version")
        ids = tuple(entry.example_id for entry in self.entries)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("entries must have unique example_id values in ascending order")
        for name in (
            "retriever_config_hash", "corpus_hash", "benchmark_manifest_hash",
            "source_results_hash", "retrieval_artifact_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if expected != self.retrieval_artifact_hash:
            raise ValueError("retrieval_artifact_hash does not match artifact content")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "retrieval_run_id": self.retrieval_run_id,
            "retriever_config_hash": self.retriever_config_hash,
            "corpus_hash": self.corpus_hash,
            "benchmark_manifest_hash": self.benchmark_manifest_hash,
            "source_results_hash": self.source_results_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.hash_payload()
        data["retrieval_artifact_hash"] = self.retrieval_artifact_hash
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenRetrievalArtifact":
        return cls(
            retrieval_run_id=data["retrieval_run_id"],
            retriever_config_hash=data["retriever_config_hash"],
            corpus_hash=data["corpus_hash"],
            benchmark_manifest_hash=data["benchmark_manifest_hash"],
            source_results_hash=data["source_results_hash"],
            entries=tuple(FrozenRetrievalEntry.from_dict(x) for x in data["entries"]),
            retrieval_artifact_hash=data["retrieval_artifact_hash"],
            artifact_version=data.get("artifact_version", FROZEN_RETRIEVAL_ARTIFACT_VERSION),
        )


def build_frozen_retrieval_artifact(
    *, results: Iterable[RetrievalResult], manifest: RetrievalRunManifest
) -> FrozenRetrievalArtifact:
    ordered = tuple(sorted(results, key=lambda r: r.example_id))
    if len(ordered) != manifest.completed_count:
        raise ValueError("result count must match completed_count in retrieval manifest")
    if any(r.retrieval_run_id != manifest.run_id for r in ordered):
        raise ValueError("all retrieval results must belong to the manifest run_id")
    if any(r.corpus_hash != manifest.corpus_hash for r in ordered):
        raise ValueError("all retrieval results must match the manifest corpus_hash")
    if any(r.retriever_config_hash != manifest.retriever_config_hash for r in ordered):
        raise ValueError("all retrieval results must match the manifest retriever_config_hash")
    source_results_hash = sha256_bytes(canonical_json_bytes([r.to_dict() for r in ordered]))
    if manifest.result_hashes.get("canonical") != source_results_hash:
        raise ValueError("retrieval results do not match frozen manifest canonical result hash")
    entries = tuple(
        FrozenRetrievalEntry(
            example_id=r.example_id,
            retrieval_eligible=r.retrieval_eligible,
            chunk_ids=tuple(r.candidate_chunk_ids) if r.retrieval_eligible else (),
            ranks=tuple(r.candidate_ranks) if r.retrieval_eligible else (),
            scores=tuple(r.candidate_scores) if r.retrieval_eligible else (),
        )
        for r in ordered
    )
    payload = {
        "artifact_version": FROZEN_RETRIEVAL_ARTIFACT_VERSION,
        "retrieval_run_id": manifest.run_id,
        "retriever_config_hash": manifest.retriever_config_hash,
        "corpus_hash": manifest.corpus_hash,
        "benchmark_manifest_hash": manifest.benchmark_manifest_hash,
        "source_results_hash": source_results_hash,
        "entries": [entry.to_dict() for entry in entries],
    }
    artifact_hash = sha256_bytes(canonical_json_bytes(payload))
    return FrozenRetrievalArtifact(
        retrieval_run_id=manifest.run_id,
        retriever_config_hash=manifest.retriever_config_hash,
        corpus_hash=manifest.corpus_hash,
        benchmark_manifest_hash=manifest.benchmark_manifest_hash,
        source_results_hash=source_results_hash,
        entries=entries,
        retrieval_artifact_hash=artifact_hash,
    )


def freeze_canonical_retrieval_results(*, results_path: Path, manifest_path: Path, output_path: Path) -> FrozenRetrievalArtifact:
    results = tuple(RetrievalResult.from_dict(x) for x in json.loads(results_path.read_text()))
    manifest = RetrievalRunManifest.from_dict(json.loads(manifest_path.read_text()))
    artifact = build_frozen_retrieval_artifact(results=results, manifest=manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(artifact.to_json_bytes())
    return artifact


def load_and_verify_frozen_retrieval_artifact(path: Path) -> FrozenRetrievalArtifact:
    """Load a canonical retrieval artifact and verify its embedded content hash."""
    return FrozenRetrievalArtifact.from_dict(json.loads(path.read_text()))
