"""Validation-only Milestone 4 RAG smoke-test utilities.

This path consumes persisted validation BM25 results; it never reruns retrieval and
must not be used to tune retrieval from generation accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.domain.enums import Split
from adaptlab.evaluation.cache import ExactRequestCache, InferenceCacheKey
from adaptlab.evaluation.inputs import construct_rag_model_input
from adaptlab.evaluation.providers import ModelProvider, ModelRequest, ModelResponse
from adaptlab.evaluation.runner import load_benchmark_split, load_chunks, verify_frozen_benchmark
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.evaluation.scoring import score_output
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.retrieval.frozen_artifact import FrozenRetrievalArtifact, FrozenRetrievalEntry
from adaptlab.retrieval.schemas import RetrievalResult, RetrievalRunManifest

SMOKE_SCHEMA_VERSION = "1"
SMOKE_SAMPLE_SIZE = 24


def freeze_validation_candidate(*, results_path: Path, manifest_path: Path) -> FrozenRetrievalArtifact:
    """Freeze an already-persisted validation candidate result file without rerunning retrieval."""
    results = tuple(RetrievalResult.from_dict(x) for x in json.loads(Path(results_path).read_text()))
    manifest = RetrievalRunManifest.from_dict(json.loads(Path(manifest_path).read_text()))
    ordered = tuple(sorted(results, key=lambda r: r.example_id))
    source_hash = sha256_bytes(canonical_json_bytes([r.to_dict() for r in ordered]))
    expected_hash = manifest.result_hashes.get("k10")
    if source_hash != expected_hash:
        raise ValueError("validation k10 results do not match the persisted retrieval manifest")
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
        "artifact_version": "canonical-retrieval-artifact-v1",
        "retrieval_run_id": manifest.run_id,
        "retriever_config_hash": manifest.retriever_config_hash,
        "corpus_hash": manifest.corpus_hash,
        "benchmark_manifest_hash": manifest.benchmark_manifest_hash,
        "source_results_hash": source_hash,
        "entries": [entry.to_dict() for entry in entries],
    }
    return FrozenRetrievalArtifact(
        retrieval_run_id=manifest.run_id,
        retriever_config_hash=manifest.retriever_config_hash,
        corpus_hash=manifest.corpus_hash,
        benchmark_manifest_hash=manifest.benchmark_manifest_hash,
        source_results_hash=source_hash,
        entries=entries,
        retrieval_artifact_hash=sha256_bytes(canonical_json_bytes(payload)),
    )


def select_validation_smoke_examples(examples: Iterable, *, count: int = SMOKE_SAMPLE_SIZE):
    """Deterministically choose a compact validation smoke sample.

    Round-robin across task families after sorting by example_id so the sample
    exercises retrieval bypass and knowledge-bearing paths without using scores.
    """
    if count < 20 or count > 30:
        raise ValueError("validation RAG smoke sample must contain 20-30 examples")
    buckets: dict[str, list] = {}
    for ex in sorted(examples, key=lambda e: e.example_id):
        buckets.setdefault(ex.task_family.value, []).append(ex)
    selected = []
    names = sorted(buckets)
    i = 0
    while len(selected) < count:
        made_progress = False
        for name in names:
            if i < len(buckets[name]) and len(selected) < count:
                selected.append(buckets[name][i])
                made_progress = True
        if not made_progress:
            break
        i += 1
    if len(selected) != count:
        raise ValueError(f"validation split contains only {len(selected)} selectable examples")
    return tuple(sorted(selected, key=lambda e: e.example_id))


@dataclass(frozen=True)
class RAGSmokeSummary:
    selected_count: int
    successful_count: int
    provider_failure_count: int
    cache_hit_count: int
    retrieval_run_id: str
    retrieval_artifact_hash: str
    retriever_config_hash: str
    sample_hash: str

    def to_dict(self):
        return self.__dict__.copy()


def _cache_key(*, example_id: str, input_data, manifest_hash: str, provider_name: str,
               model_id: str, prompt_hash: str, temperature: float, context_length: int,
               max_tokens: int, seed: int | None, stream: bool, think: bool,
               ollama_base_url_policy: str | None = None, ollama_version: str | None = None,
               model_tag: str | None = None, model_digest: str | None = None) -> InferenceCacheKey:
    return InferenceCacheKey(
        benchmark_manifest_hash=manifest_hash,
        example_id=example_id,
        provider=provider_name,
        ollama_base_url_policy=ollama_base_url_policy,
        ollama_version=ollama_version,
        model_id=model_id,
        model_tag=model_tag,
        model_digest=model_digest,
        model_revision=None,
        prompt_hash=prompt_hash,
        method=AdaptationMethod.RAG,
        temperature=temperature,
        context_length=context_length,
        max_tokens=max_tokens,
        seed=seed,
        stream=stream,
        think=think,
        input_hash=input_data.input_hash,
        retrieval_run_id=input_data.retrieval_run_id,
        retrieval_artifact_hash=input_data.retrieval_artifact_hash,
        retriever_config_hash=input_data.retriever_config_hash,
        retrieved_context_hash=input_data.retrieved_context_hash,
    )


def run_validation_rag_smoke(*, benchmark_dir: Path, prompt_config: Path,
                             retrieval_results_path: Path, retrieval_manifest_path: Path,
                             provider: ModelProvider, model_id: str, output_dir: Path,
                             sample_size: int = SMOKE_SAMPLE_SIZE, temperature: float = 0.0,
                             context_length: int = 40960, max_tokens: int = 256,
                             seed: int | None = 1729, stream: bool = False, think: bool = False,
                             ollama_base_url_policy: str | None = None,
                             ollama_version: str | None = None, model_tag: str | None = None,
                             model_digest: str | None = None) -> RAGSmokeSummary:
    """Execute a validation-only RAG smoke test from a persisted BM25 candidate artifact."""
    _, manifest_hash = verify_frozen_benchmark(Path(benchmark_dir))
    prompt = load_prompt_contract(Path(prompt_config))
    artifact = freeze_validation_candidate(results_path=retrieval_results_path, manifest_path=retrieval_manifest_path)
    if artifact.benchmark_manifest_hash != manifest_hash:
        raise ValueError("validation retrieval artifact benchmark hash mismatch")
    examples = select_validation_smoke_examples(load_benchmark_split(Path(benchmark_dir), Split.validation), count=sample_size)
    chunks = load_chunks(Path(benchmark_dir))
    cache = ExactRequestCache(Path(output_dir) / "cache")
    rows = []
    cache_hits = successes = failures = 0
    for example in examples:
        constructed = construct_rag_model_input(
            example=example, prompt_contract=prompt, chunks=chunks, retrieval_artifact=artifact
        )
        key = _cache_key(
            example_id=example.example_id, input_data=constructed, manifest_hash=manifest_hash,
            provider_name=provider.provider_name, model_id=model_id, prompt_hash=prompt.prompt_hash,
            temperature=temperature, context_length=context_length, max_tokens=max_tokens,
            seed=seed, stream=stream, think=think, ollama_base_url_policy=ollama_base_url_policy,
            ollama_version=ollama_version, model_tag=model_tag, model_digest=model_digest,
        )
        response = cache.get(key)
        cache_hit = response is not None
        error = None
        if response is None:
            try:
                response = provider.generate(ModelRequest(
                    system_prompt=constructed.model_input.system,
                    user_prompt=constructed.model_input.user,
                    temperature=temperature, max_tokens=max_tokens, seed=seed,
                ))
                cache.put(key, response)
            except Exception as exc:  # provider adapters normalize failures; smoke artifact must retain them
                error = f"{type(exc).__name__}: {exc}"
        else:
            cache_hits += 1
        if response is not None:
            scored = score_output(example, response.text)
            successes += 1
            raw_output = response.text
            normalized_output = scored.normalized_output
            score = scored.score
        else:
            failures += 1
            raw_output = normalized_output = score = None
        rows.append({
            "example_id": example.example_id,
            "retrieval_run_id": constructed.retrieval_run_id,
            "retrieval_artifact_hash": constructed.retrieval_artifact_hash,
            "retriever_config_hash": constructed.retriever_config_hash,
            "retrieved_context_hash": constructed.retrieved_context_hash,
            "input_hash": constructed.input_hash,
            "retrieved_chunk_ids": list(constructed.evidence_chunk_ids),
            "cache_request_hash": key.request_hash,
            "cache_hit": cache_hit,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "score": score,
            "provider_error": error,
        })
    sample_hash = sha256_bytes(canonical_json_bytes([e.example_id for e in examples]))
    summary = RAGSmokeSummary(
        selected_count=len(examples), successful_count=successes, provider_failure_count=failures,
        cache_hit_count=cache_hits, retrieval_run_id=artifact.retrieval_run_id,
        retrieval_artifact_hash=artifact.retrieval_artifact_hash,
        retriever_config_hash=artifact.retriever_config_hash, sample_hash=sample_hash,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "results.json", rows)
    write_json(output_dir / "summary.json", {"schema_version": SMOKE_SCHEMA_VERSION, **summary.to_dict()})
    write_json(output_dir / "frozen_validation_retrieval_artifact.json", artifact.to_dict())
    return summary
