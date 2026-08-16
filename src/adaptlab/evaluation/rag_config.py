"""Frozen canonical Milestone 4 RAG condition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.evaluation.inputs import EVIDENCE_RENDERER_VERSION, evidence_renderer_hash
from adaptlab.retrieval.canonical_config import CanonicalBM25Config
from adaptlab.retrieval.frozen_artifact import FrozenRetrievalArtifact

CANONICAL_RAG_CONFIG_VERSION = "canonical-rag-v1"


def _sha(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CanonicalRAGConfig:
    config_version: str
    condition_id: str
    adaptation_method: str
    model: str
    provider: str
    prompt_version: str
    prompt_hash: str
    temperature: float
    seed: int
    context_length: int
    max_tokens: int
    stream: bool
    think: bool
    scorer_version: str
    normalizer_version: str
    benchmark_manifest_hash: str
    retrieval_run_id: str
    retrieval_artifact_hash: str
    retriever_config_hash: str
    query_policy_hash: str
    indexing_policy_hash: str
    tokenization_policy_hash: str
    top_k: int
    corpus_hash: str
    evidence_renderer_version: str
    evidence_renderer_hash: str
    retrieval_execution: str

    def __post_init__(self) -> None:
        if self.config_version != CANONICAL_RAG_CONFIG_VERSION:
            raise ValueError("unexpected canonical RAG config version")
        if self.adaptation_method != "RAG":
            raise ValueError("canonical adaptation_method must be RAG")
        if self.retrieval_execution != "consume_frozen_artifact_only":
            raise ValueError("canonical RAG must consume the frozen retrieval artifact")
        if self.evidence_renderer_version != EVIDENCE_RENDERER_VERSION:
            raise ValueError("evidence renderer version mismatch")
        if self.evidence_renderer_hash != evidence_renderer_hash():
            raise ValueError("evidence renderer hash mismatch")
        for n in ("prompt_hash","benchmark_manifest_hash","retrieval_artifact_hash","retriever_config_hash","query_policy_hash","indexing_policy_hash","tokenization_policy_hash","corpus_hash","evidence_renderer_hash"):
            _sha(n, getattr(self,n))
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def canonical_config_hash(self) -> str:
        return sha256_bytes(self.to_json_bytes())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CanonicalRAGConfig":
        return cls(**raw)


def build_canonical_rag_config(*, prompt_condition: dict[str, Any], bm25: CanonicalBM25Config, retrieval: FrozenRetrievalArtifact) -> CanonicalRAGConfig:
    if prompt_condition["adaptation_method"] != "PROMPT":
        raise ValueError("RAG freeze must derive non-retrieval controls from canonical PROMPT")
    if retrieval.retriever_config_hash != bm25.retriever_config_hash:
        raise ValueError("frozen retrieval artifact retriever config mismatch")
    if retrieval.corpus_hash != bm25.corpus_hash:
        raise ValueError("frozen retrieval artifact corpus mismatch")
    if retrieval.benchmark_manifest_hash != bm25.benchmark_manifest_hash:
        raise ValueError("frozen retrieval artifact benchmark mismatch")
    if prompt_condition["benchmark"]["benchmark_manifest_hash"] != retrieval.benchmark_manifest_hash:
        raise ValueError("Prompt and retrieval benchmark manifests differ")
    return CanonicalRAGConfig(
        config_version=CANONICAL_RAG_CONFIG_VERSION,
        condition_id="milestone4_ollama_rag_v1",
        adaptation_method="RAG",
        model=prompt_condition["provider"]["model_tag"],
        provider=prompt_condition["provider"]["name"],
        prompt_version=prompt_condition["prompt"]["prompt_version"],
        prompt_hash=prompt_condition["prompt"]["prompt_hash"],
        temperature=prompt_condition["request"]["temperature"],
        seed=prompt_condition["request"]["seed"],
        context_length=prompt_condition["request"]["context_length"],
        max_tokens=prompt_condition["request"]["max_tokens"],
        stream=prompt_condition["request"]["stream"],
        think=prompt_condition["request"]["think"],
        scorer_version=prompt_condition["scoring"]["scorer_version"],
        normalizer_version=prompt_condition["scoring"]["normalizer_version"],
        benchmark_manifest_hash=retrieval.benchmark_manifest_hash,
        retrieval_run_id=retrieval.retrieval_run_id,
        retrieval_artifact_hash=retrieval.retrieval_artifact_hash,
        retriever_config_hash=bm25.retriever_config_hash,
        query_policy_hash=bm25.query_policy_hash,
        indexing_policy_hash=bm25.indexing_policy_hash,
        tokenization_policy_hash=bm25.tokenization_policy_hash,
        top_k=bm25.top_k,
        corpus_hash=bm25.corpus_hash,
        evidence_renderer_version=EVIDENCE_RENDERER_VERSION,
        evidence_renderer_hash=evidence_renderer_hash(),
        retrieval_execution="consume_frozen_artifact_only",
    )


def load_canonical_rag_config(path: Path) -> CanonicalRAGConfig:
    raw=yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("canonical RAG config must be a mapping")
    return CanonicalRAGConfig.from_dict(raw)
