"""Canonical Milestone 4 primary-test RAG execution.

The runner consumes the frozen primary-test retrieval artifact and frozen RAG
configuration. It never invokes retrieval dynamically and persists one row per
benchmark example even when the model provider fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from types import SimpleNamespace

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.domain.enums import Split
from adaptlab.evaluation.cache import ExactRequestCache, InferenceCacheKey
from adaptlab.evaluation.metrics import aggregate_metrics, AccuracyMetric
from adaptlab.evaluation.provenance import capture_git_state
from adaptlab.evaluation.inputs import construct_rag_model_input
from adaptlab.evaluation.runtime import capture_ollama_runtime
from adaptlab.evaluation.providers import ModelProvider, ModelRequest
from adaptlab.evaluation.providers.ollama import OllamaModelProvider
from adaptlab.evaluation.rag_completeness import (
    CANONICAL_RAG_EXPECTED_COUNT,
    RAGExampleCompletion,
    RAGRunIdentity,
    canonical_rag_completeness,
)
from adaptlab.evaluation.rag_config import CanonicalRAGConfig
from adaptlab.evaluation.runner import load_benchmark_split, load_chunks, verify_frozen_benchmark
from adaptlab.evaluation.schemas import AdaptationMethod, EvaluationRunStatus
from adaptlab.evaluation.scoring import score_output
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.retrieval.frozen_artifact import FrozenRetrievalArtifact

PRIMARY_RAG_RUN_SCHEMA_VERSION = "canonical-primary-rag-v1"
RAG_RUN_MANIFEST_FILENAME = "run_manifest.json"
RAG_METRICS_FILENAME = "metrics.json"
RAG_SUMMARY_FILENAME = "summary.txt"
RAG_COMPLETION_FILENAME = "completion.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CanonicalRAGRunSummary:
    run_id: str
    expected_count: int
    represented_count: int
    completed_successful_model_responses: int
    unresolved_provider_failures: int
    valid: bool
    cache_hit_count: int
    retrieval_run_id: str
    retrieval_artifact_hash: str
    canonical_rag_config_hash: str
    benchmark_manifest_hash: str
    results_hash: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def canonical_rag_run_id(*, config: CanonicalRAGConfig) -> str:
    ident = RAGRunIdentity(
        canonical_rag_config_hash=config.canonical_config_hash,
        retrieval_artifact_hash=config.retrieval_artifact_hash,
        benchmark_manifest_hash=config.benchmark_manifest_hash,
    )
    return f"m4-primary-test-rag-{ident.run_identity_hash[:16]}"


def _cache_key(*, example_id: str, constructed, config: CanonicalRAGConfig,
               provider_name: str, ollama_base_url_policy: str | None,
               ollama_version: str | None, model_tag: str | None,
               model_digest: str | None) -> InferenceCacheKey:
    return InferenceCacheKey(
        benchmark_manifest_hash=config.benchmark_manifest_hash,
        example_id=example_id,
        provider=provider_name,
        ollama_base_url_policy=ollama_base_url_policy,
        ollama_version=ollama_version,
        model_id=config.model,
        model_tag=model_tag,
        model_digest=model_digest,
        model_revision=None,
        prompt_hash=config.prompt_hash,
        method=AdaptationMethod.RAG,
        temperature=config.temperature,
        context_length=config.context_length,
        max_tokens=config.max_tokens,
        seed=config.seed,
        stream=config.stream,
        think=config.think,
        input_hash=constructed.input_hash,
        retrieval_run_id=constructed.retrieval_run_id,
        retrieval_artifact_hash=constructed.retrieval_artifact_hash,
        retriever_config_hash=constructed.retriever_config_hash,
        retrieved_context_hash=constructed.retrieved_context_hash,
    )


def run_canonical_primary_rag(*, benchmark_dir: Path, prompt_config: Path,
                              retrieval_artifact: FrozenRetrievalArtifact,
                              config: CanonicalRAGConfig, provider: ModelProvider,
                              output_dir: Path, ollama_base_url_policy: str | None = None,
                              ollama_version: str | None = None,
                              model_tag: str | None = None,
                              model_digest: str | None = None) -> CanonicalRAGRunSummary:
    """Run canonical RAG over all 400 primary-test examples.

    Provider failures are persisted as represented examples and make the summary
    incomplete. Successful raw responses alone are cached; reruns can resume via
    exact cache identity.
    """
    if config.retrieval_execution != "consume_frozen_artifact_only":
        raise ValueError("canonical RAG must consume frozen retrieval artifact only")
    if provider.provider_name != config.provider:
        raise ValueError("provider does not match frozen canonical RAG config")
    if retrieval_artifact.retrieval_artifact_hash != config.retrieval_artifact_hash:
        raise ValueError("retrieval artifact hash does not match frozen RAG config")
    if retrieval_artifact.retrieval_run_id != config.retrieval_run_id:
        raise ValueError("retrieval run ID does not match frozen RAG config")
    if retrieval_artifact.retriever_config_hash != config.retriever_config_hash:
        raise ValueError("retriever config hash does not match frozen RAG config")

    started_at = _now()
    git_state = capture_git_state(Path.cwd())
    benchmark_manifest, manifest_hash = verify_frozen_benchmark(Path(benchmark_dir))
    if manifest_hash != config.benchmark_manifest_hash:
        raise ValueError("benchmark manifest hash does not match frozen RAG config")
    prompt = load_prompt_contract(Path(prompt_config))
    if prompt.prompt_hash != config.prompt_hash:
        raise ValueError("prompt hash does not match frozen RAG config")
    ollama_runtime = capture_ollama_runtime(provider) if isinstance(provider, OllamaModelProvider) else None

    examples = tuple(sorted(load_benchmark_split(Path(benchmark_dir), Split.test), key=lambda e: e.example_id))
    if len(examples) != CANONICAL_RAG_EXPECTED_COUNT:
        raise ValueError(f"canonical primary test must contain 400 examples, found {len(examples)}")
    chunks = load_chunks(Path(benchmark_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ExactRequestCache(output_dir / "cache")
    rows: list[dict[str, Any]] = []
    completions: list[RAGExampleCompletion] = []
    cache_hits = 0

    for example in examples:
        constructed = construct_rag_model_input(
            example=example,
            prompt_contract=prompt,
            chunks=chunks,
            retrieval_artifact=retrieval_artifact,
        )
        key = _cache_key(
            example_id=example.example_id,
            constructed=constructed,
            config=config,
            provider_name=provider.provider_name,
            ollama_base_url_policy=ollama_base_url_policy,
            ollama_version=ollama_version,
            model_tag=model_tag,
            model_digest=model_digest,
        )
        response = cache.get(key)
        cache_hit = response is not None
        if cache_hit:
            cache_hits += 1
        provider_error = None
        if response is None:
            try:
                response = provider.generate(ModelRequest(
                    system_prompt=constructed.model_input.system,
                    user_prompt=constructed.model_input.user,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                ))
                cache.put(key, response)
            except Exception as exc:
                provider_error = f"{type(exc).__name__}: {exc}"

        if response is None:
            raw_output = normalized_output = score = None
            runtime = None
            succeeded = False
        else:
            scored = score_output(example, response.text)
            raw_output = response.text
            normalized_output = scored.normalized_output
            score = scored.score
            runtime = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "model_revision": response.model_revision,
                "provider_metadata": response.provider_metadata,
            }
            succeeded = True

        rows.append({
            "schema_version": PRIMARY_RAG_RUN_SCHEMA_VERSION,
            "example_id": example.example_id,
            "retrieval_result_reference": {
                "retrieval_run_id": constructed.retrieval_run_id,
                "retrieval_artifact_hash": constructed.retrieval_artifact_hash,
            },
            "retrieved_chunk_ids": list(constructed.evidence_chunk_ids),
            "retrieved_context_hash": constructed.retrieved_context_hash,
            "input_hash": constructed.input_hash,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "score": score,
            "runtime_provenance": runtime,
            "cache_metadata": {"request_hash": key.request_hash, "cache_hit": cache_hit},
            "provider_error": provider_error,
        })
        completions.append(RAGExampleCompletion(
            example_id=example.example_id,
            retrieval_succeeded=True,
            model_response_succeeded=succeeded,
            provider_error=provider_error,
        ))
        # Durable progress without dropping failures.
        write_json(output_dir / "results.json", rows)

    completeness = canonical_rag_completeness(completions)
    results_hash = sha256_bytes(canonical_json_bytes(rows))
    pseudo_results = [
        SimpleNamespace(
            example_id=example.example_id,
            task_family=example.task_family,
            difficulty=example.difficulty,
            behavior_type=example.behavior_type,
            knowledge_state=example.knowledge_state,
            evidence_status=example.evidence_status,
            split_type=example.split_type,
            score=row["score"],
        )
        for example, row in zip(examples, rows, strict=True)
    ]
    metrics = aggregate_metrics(pseudo_results)
    if not completeness.valid:
        metrics = metrics.__class__(
            primary={
                **metrics.primary,
                "overall_accuracy": AccuracyMetric(
                    n=completeness.completed_successful_model_responses,
                    accuracy=None,
                ),
            },
            confirmatory=metrics.confirmatory,
            exploratory=metrics.exploratory,
            schema_version=metrics.schema_version,
        )
    run_id = canonical_rag_run_id(config=config)
    summary = CanonicalRAGRunSummary(
        run_id=run_id,
        expected_count=completeness.expected_count,
        represented_count=completeness.represented_count,
        completed_successful_model_responses=completeness.completed_successful_model_responses,
        unresolved_provider_failures=completeness.unresolved_provider_failures,
        valid=completeness.valid,
        cache_hit_count=cache_hits,
        retrieval_run_id=config.retrieval_run_id,
        retrieval_artifact_hash=config.retrieval_artifact_hash,
        canonical_rag_config_hash=config.canonical_config_hash,
        benchmark_manifest_hash=config.benchmark_manifest_hash,
        results_hash=results_hash,
    )
    status = EvaluationRunStatus.COMPLETED if completeness.valid else EvaluationRunStatus.INCOMPLETE
    summary_text = metrics.human_summary()
    completion = {
        "schema_version": PRIMARY_RAG_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status.value,
        "expected_count": completeness.expected_count,
        "represented_count": completeness.represented_count,
        "completed_successful_model_responses": completeness.completed_successful_model_responses,
        "unresolved_provider_failures": completeness.unresolved_provider_failures,
        "valid": completeness.valid,
        "timestamp": _now(),
    }
    run_manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "benchmark_version": benchmark_manifest["benchmark_version"],
        "benchmark_manifest_hash": config.benchmark_manifest_hash,
        "benchmark_tag": "v0.0-benchmark",
        "canonical": True,
        "canonical_accuracy_emitted": completeness.valid,
        "completed_at": _now(),
        "completed_count": completeness.completed_successful_model_responses,
        "completed_successful_responses": completeness.completed_successful_model_responses,
        "completeness_valid": completeness.valid,
        "context_length": (
            ollama_runtime.context_length if ollama_runtime is not None else config.context_length
        ),
        "dirty_git_override": True,
        "example_count": len(examples),
        "expected_count": completeness.expected_count,
        "git_commit_sha": git_state.commit_sha,
        "git_dirty": git_state.dirty,
        "git_provenance_limitation": git_state.limitation,
        "git_state_available": git_state.available,
        "inference_determinism_claimed": False,
        "max_tokens": config.max_tokens,
        "metric_hashes": {},
        "model_digest": ollama_runtime.model_digest if ollama_runtime is not None else None,
        "model_id": config.model,
        "model_revision": None,
        "model_revision_limitation": "Provider/model exposed only an unresolved or mutable model alias; immutable revision unavailable.",
        "model_tag": ollama_runtime.model_tag if ollama_runtime is not None else config.model,
        "normalizer_version": config.normalizer_version,
        "ollama_base_url_policy": (
            ollama_runtime.ollama_base_url_policy if ollama_runtime is not None else ollama_base_url_policy
        ),
        "ollama_model_digest_limitation": (
            None if (ollama_runtime is None or ollama_runtime.model_digest is not None)
            else "Local Ollama registry did not expose a digest for the requested model tag."
        ),
        "ollama_version": ollama_runtime.ollama_version if ollama_runtime is not None else ollama_version,
        "prompt_hash": config.prompt_hash,
        "prompt_version": config.prompt_version,
        "provider": provider.provider_name,
        "provider_error_count": sum(row.get("provider_error") is not None for row in rows),
        "result_hashes": {"results.json": sha256_bytes(canonical_json_bytes(rows))},
        "scorer_version": config.scorer_version,
        "seed": config.seed,
        "seed_policy": f"FIXED_REQUEST_SEED:{config.seed}",
        "started_at": started_at,
        "status": status.value,
        "stream": ollama_runtime.stream if ollama_runtime is not None else config.stream,
        "temperature": config.temperature,
        "think": ollama_runtime.think if ollama_runtime is not None else config.think,
        "timestamp": _now(),
    }
    metric_bytes = canonical_json_bytes(metrics.to_dict())
    run_manifest["metric_hashes"] = {
        RAG_METRICS_FILENAME: sha256_bytes(metric_bytes),
        RAG_SUMMARY_FILENAME: sha256_bytes(summary_text.encode("utf-8")),
        RAG_COMPLETION_FILENAME: sha256_bytes(canonical_json_bytes(completion)),
    }
    write_json(output_dir / RAG_METRICS_FILENAME, metrics.to_dict())
    (output_dir / RAG_SUMMARY_FILENAME).write_text(summary_text, encoding="utf-8")
    write_json(output_dir / RAG_RUN_MANIFEST_FILENAME, run_manifest)
    write_json(output_dir / RAG_COMPLETION_FILENAME, completion)
    write_json(output_dir / "summary.json", {"schema_version": PRIMARY_RAG_RUN_SCHEMA_VERSION, **summary.to_dict()})
    return summary
