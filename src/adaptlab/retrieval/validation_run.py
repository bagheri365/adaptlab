"""Canonical-candidate BM25 validation retrieval for Milestone 4 Prompt 10.

This module intentionally operates on the validation split only. It consumes the
already-frozen query/index/tokenization/BM25/top-k contracts, persists retrieval
artifacts outside the benchmark directory, and applies the precommitted top-k
selection rule using retrieval metrics only.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.retrieval.bm25 import (
    BM25_RETRIEVER_NAME,
    BM25_RETRIEVER_VERSION,
    BM25Retriever,
)
from adaptlab.retrieval.metrics import summarize_retrieval_metrics, with_retrieval_metrics
from adaptlab.retrieval.policies import (
    INDEXING_POLICY_VERSION,
    TOKENIZATION_POLICY_VERSION,
    indexing_policy_hash,
    tokenization_policy_hash,
)
from adaptlab.retrieval.query_policy import (
    QUERY_POLICY_VERSION,
    construct_retrieval_query,
    query_policy_hash,
)
from adaptlab.retrieval.schemas import RetrievalResult, RetrievalRunManifest
from adaptlab.retrieval.top_k_selection import (
    TOP_K_CANDIDATES,
    select_top_k,
    top_k_selection_policy_hash,
)
from adaptlab.retrieval.version_metrics import summarize_version_diagnostics, with_version_diagnostics

VALIDATION_RUNNER_VERSION = "validation-bm25-candidates-v1"


def _load_examples(path: Path) -> tuple[BenchmarkExample, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = tuple(BenchmarkExample.from_dict(item) for item in raw)
    if any(example.split.value != "validation" for example in examples):
        raise ValueError("validation runner accepts validation examples only")
    return tuple(sorted(examples, key=lambda x: x.example_id))


def _load_chunks(path: Path) -> tuple[DocumentChunk, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(sorted((DocumentChunk.from_dict(item) for item in raw), key=lambda x: x.chunk_id))


def _benchmark_manifest_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _result_for_example(
    example: BenchmarkExample,
    retriever: BM25Retriever,
    chunks: Sequence[DocumentChunk],
    *,
    run_id: str,
    top_k: int,
) -> RetrievalResult:
    query = construct_retrieval_query(example)
    hits = retriever.retrieve(query.query_text, top_k=top_k) if query.retrieval_eligible else ()
    result = RetrievalResult(
        retrieval_run_id=run_id,
        corpus_hash=retriever.corpus_hash,
        example_id=example.example_id,
        split=example.split,
        task_family=example.task_family,
        difficulty=example.difficulty,
        knowledge_state=example.knowledge_state,
        evidence_status=example.evidence_status,
        split_type=example.split_type,
        retrieval_eligible=query.retrieval_eligible,
        query_text=query.query_text,
        query_hash=query.query_hash,
        retriever_name=retriever.retriever_name,
        retriever_version=retriever.retriever_version,
        retriever_config_hash=retriever.retriever_config_hash,
        indexing_policy_version=INDEXING_POLICY_VERSION,
        tokenization_policy_version=TOKENIZATION_POLICY_VERSION,
        top_k=top_k,
        candidate_chunk_ids=tuple(hit.chunk_id for hit in hits),
        candidate_scores=tuple(hit.score for hit in hits),
        candidate_ranks=tuple(hit.rank for hit in hits),
        gold_chunk_ids=tuple(example.gold_chunk_ids),
        required_gold_chunk_ids=tuple(example.gold_chunk_ids),
        any_gold_at_1=None,
        any_gold_at_3=None,
        any_gold_at_5=None,
        any_gold_at_k=None,
        all_required_gold_at_1=None,
        all_required_gold_at_3=None,
        all_required_gold_at_5=None,
        all_required_gold_at_k=None,
        gold_recall_at_1=None,
        gold_recall_at_3=None,
        gold_recall_at_5=None,
        gold_recall_at_k=None,
        first_gold_reciprocal_rank=None,
        wrong_version_top1=None,
        current_gold_retrieved=None,
        obsolete_only_retrieved=None,
        current_and_obsolete_retrieved=None,
    )
    result = with_retrieval_metrics(result)
    result = with_version_diagnostics(result, chunks)
    return result


def run_validation_candidates(
    *,
    validation_path: Path,
    chunks_path: Path,
    benchmark_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Execute all precommitted candidate k values and persist deterministic artifacts."""
    examples = _load_examples(validation_path)
    chunks = _load_chunks(chunks_path)
    retriever = BM25Retriever(chunks)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stable run identity is bound to all frozen inputs/policies, not wall-clock time.
    identity_payload = {
        "benchmark_manifest_hash": _benchmark_manifest_hash(benchmark_manifest_path),
        "corpus_hash": retriever.corpus_hash,
        "query_policy_hash": query_policy_hash(),
        "indexing_policy_hash": indexing_policy_hash(),
        "tokenization_policy_hash": tokenization_policy_hash(),
        "retriever_config_hash": retriever.retriever_config_hash,
        "top_k_values": list(TOP_K_CANDIDATES),
        "validation_hash": sha256_bytes(validation_path.read_bytes()),
        "runner_version": VALIDATION_RUNNER_VERSION,
    }
    run_id = "m4-validation-bm25-" + sha256_bytes(canonical_json_bytes(identity_payload))[:16]

    result_hashes: dict[str, str] = {}
    metric_hashes: dict[str, str] = {}
    candidate_primary: dict[int, float] = {}
    reports: dict[str, object] = {}

    for top_k in TOP_K_CANDIDATES:
        results = tuple(
            _result_for_example(example, retriever, chunks, run_id=run_id, top_k=top_k)
            for example in examples
        )
        result_bytes = canonical_json_bytes([result.to_dict() for result in results])
        result_name = f"results_k{top_k}.json"
        (output_dir / result_name).write_bytes(result_bytes)
        result_hashes[f"k{top_k}"] = sha256_bytes(result_bytes)

        metric_report = summarize_retrieval_metrics(results)
        version_report = summarize_version_diagnostics(results, chunks)
        metric_payload = {
            "top_k": top_k,
            "retrieval_metrics": metric_report.to_dict(),
            "version_diagnostics": version_report.to_dict(),
        }
        metric_bytes = canonical_json_bytes(metric_payload)
        metric_name = f"metrics_k{top_k}.json"
        (output_dir / metric_name).write_bytes(metric_bytes)
        metric_hashes[f"k{top_k}"] = sha256_bytes(metric_bytes)
        (output_dir / f"metrics_k{top_k}.txt").write_text(
            metric_report.to_text() + "\n" + version_report.to_text(), encoding="utf-8"
        )

        eligible_present = tuple(
            r for r in results
            if r.retrieval_eligible
            and r.task_family is not TaskFamily.behavior_only
            and r.evidence_status is EvidenceStatus.PRESENT
        )
        if not eligible_present:
            raise ValueError("validation split has no eligible evidence-present examples")
        candidate_primary[top_k] = sum(r.all_required_gold_at_k is True for r in eligible_present) / len(eligible_present)
        reports[str(top_k)] = metric_payload

    decision = select_top_k(candidate_primary)
    decision_payload = {
        **decision.to_dict(),
        "selection_input": "validation_retrieval_metrics_only",
        "top_k_selection_policy_hash": top_k_selection_policy_hash(),
        "run_id": run_id,
    }
    decision_bytes = canonical_json_bytes(decision_payload)
    (output_dir / "top_k_selection_decision.json").write_bytes(decision_bytes)

    manifest = RetrievalRunManifest(
        run_id=run_id,
        benchmark_version=examples[0].benchmark_version,
        benchmark_manifest_hash=identity_payload["benchmark_manifest_hash"],
        git_commit_sha="UNAVAILABLE_NO_GIT_METADATA",
        git_dirty=True,
        corpus_hash=retriever.corpus_hash,
        query_policy_version=QUERY_POLICY_VERSION,
        query_policy_hash=query_policy_hash(),
        indexing_policy_version=INDEXING_POLICY_VERSION,
        indexing_policy_hash=indexing_policy_hash(),
        tokenization_policy_version=TOKENIZATION_POLICY_VERSION,
        tokenization_policy_hash=tokenization_policy_hash(),
        retriever_name=BM25_RETRIEVER_NAME,
        retriever_version=BM25_RETRIEVER_VERSION,
        retriever_config_hash=retriever.retriever_config_hash,
        top_k_values=TOP_K_CANDIDATES,
        example_count=len(examples),
        completed_count=len(examples),
        result_hashes=result_hashes,
        metric_hashes=metric_hashes,
    )
    (output_dir / "run_manifest.json").write_bytes(manifest.to_json_bytes())

    summary = {
        "run_id": run_id,
        "split": "validation",
        "example_count": len(examples),
        "eligible_count": sum(construct_retrieval_query(e).retrieval_eligible for e in examples),
        "candidate_all_required_gold_at_k": {str(k): candidate_primary[k] for k in TOP_K_CANDIDATES},
        "selected_top_k": decision.selected_top_k,
        "selection_decision_hash": sha256_bytes(decision_bytes),
        "retriever_config_hash": retriever.retriever_config_hash,
        "corpus_hash": retriever.corpus_hash,
        "benchmark_manifest_hash": identity_payload["benchmark_manifest_hash"],
    }
    (output_dir / "summary.json").write_bytes(canonical_json_bytes(summary))
    return summary
