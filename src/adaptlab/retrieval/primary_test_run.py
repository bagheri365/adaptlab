"""Frozen canonical BM25 retrieval execution on the 400-example primary test."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.retrieval.absent_diagnostics import with_absent_diagnostics
from adaptlab.retrieval.bm25 import BM25Retriever
from adaptlab.retrieval.canonical_config import CanonicalBM25Config
from adaptlab.retrieval.metrics import with_retrieval_metrics
from adaptlab.retrieval.policies import INDEXING_POLICY_VERSION, TOKENIZATION_POLICY_VERSION
from adaptlab.retrieval.query_policy import construct_retrieval_query
from adaptlab.retrieval.schemas import RetrievalResult, RetrievalRunManifest
from adaptlab.retrieval.version_metrics import with_version_diagnostics
from adaptlab.retrieval.query_policy import QUERY_POLICY_VERSION, query_policy_hash
from adaptlab.retrieval.policies import indexing_policy_hash, tokenization_policy_hash

PRIMARY_TEST_RUNNER_VERSION = "primary-test-bm25-v1"


def _load_examples(path: Path) -> tuple[BenchmarkExample, ...]:
    items = tuple(BenchmarkExample.from_dict(x) for x in json.loads(path.read_text()))
    if len(items) != 400 or any(x.split.value != "test" for x in items):
        raise ValueError("canonical primary-test retrieval requires exactly 400 test examples")
    return tuple(sorted(items, key=lambda x: x.example_id))


def _load_chunks(path: Path) -> tuple[DocumentChunk, ...]:
    return tuple(sorted((DocumentChunk.from_dict(x) for x in json.loads(path.read_text())), key=lambda x: x.chunk_id))


def _result(example: BenchmarkExample, retriever: BM25Retriever, chunks: Sequence[DocumentChunk], run_id: str, top_k: int) -> RetrievalResult:
    q = construct_retrieval_query(example)
    hits = retriever.retrieve(q.query_text, top_k=top_k) if q.retrieval_eligible else ()
    r = RetrievalResult(
        retrieval_run_id=run_id, corpus_hash=retriever.corpus_hash,
        example_id=example.example_id, split=example.split, task_family=example.task_family,
        difficulty=example.difficulty, knowledge_state=example.knowledge_state,
        evidence_status=example.evidence_status, split_type=example.split_type,
        retrieval_eligible=q.retrieval_eligible, query_text=q.query_text, query_hash=q.query_hash,
        retriever_name=retriever.retriever_name, retriever_version=retriever.retriever_version,
        retriever_config_hash=retriever.retriever_config_hash,
        indexing_policy_version=INDEXING_POLICY_VERSION, tokenization_policy_version=TOKENIZATION_POLICY_VERSION,
        top_k=top_k, candidate_chunk_ids=tuple(h.chunk_id for h in hits),
        candidate_scores=tuple(h.score for h in hits), candidate_ranks=tuple(h.rank for h in hits),
        gold_chunk_ids=tuple(example.gold_chunk_ids), required_gold_chunk_ids=tuple(example.gold_chunk_ids),
        any_gold_at_1=None, any_gold_at_3=None, any_gold_at_5=None, any_gold_at_k=None,
        all_required_gold_at_1=None, all_required_gold_at_3=None, all_required_gold_at_5=None, all_required_gold_at_k=None,
        gold_recall_at_1=None, gold_recall_at_3=None, gold_recall_at_5=None, gold_recall_at_k=None,
        first_gold_reciprocal_rank=None, wrong_version_top1=None, current_gold_retrieved=None,
        obsolete_only_retrieved=None, current_and_obsolete_retrieved=None,
    )
    return with_absent_diagnostics(with_version_diagnostics(with_retrieval_metrics(r), chunks))


def run_primary_test_retrieval(*, test_path: Path, chunks_path: Path, benchmark_manifest_path: Path,
                               canonical_config_path: Path, output_dir: Path) -> dict[str, object]:
    cfg = CanonicalBM25Config.from_dict(json.loads(canonical_config_path.read_text()))
    examples, chunks = _load_examples(test_path), _load_chunks(chunks_path)
    retriever = BM25Retriever(chunks, k1=cfg.k1, b=cfg.b)
    manifest_hash = sha256_bytes(benchmark_manifest_path.read_bytes())
    if retriever.corpus_hash != cfg.corpus_hash or retriever.retriever_config_hash != cfg.retriever_config_hash or manifest_hash != cfg.benchmark_manifest_hash:
        raise ValueError("canonical retrieval inputs do not match frozen BM25 configuration")
    identity = {"runner_version": PRIMARY_TEST_RUNNER_VERSION, "test_hash": sha256_bytes(test_path.read_bytes()),
                "canonical_config_hash": cfg.canonical_config_hash, "corpus_hash": retriever.corpus_hash,
                "benchmark_manifest_hash": manifest_hash}
    run_id = "m4-primary-test-bm25-" + sha256_bytes(canonical_json_bytes(identity))[:16]
    results = tuple(_result(e, retriever, chunks, run_id, cfg.top_k) for e in examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_bytes = canonical_json_bytes([r.to_dict() for r in results])
    (output_dir / "results.json").write_bytes(result_bytes)
    result_hash = sha256_bytes(result_bytes)
    retrieval_errors = 0
    eligible_count = sum(r.retrieval_eligible for r in results)
    summary = {"run_id": run_id, "split": "test", "example_count": 400, "represented_count": len(results),
               "eligible_retrieval_count": eligible_count, "retrieval_errors": retrieval_errors,
               "corpus_hash": retriever.corpus_hash, "retriever_config_hash": retriever.retriever_config_hash,
               "canonical_config_hash": cfg.canonical_config_hash, "results_hash": result_hash}
    (output_dir / "summary.json").write_bytes(canonical_json_bytes(summary))
    manifest = RetrievalRunManifest(
        run_id=run_id, benchmark_version=examples[0].benchmark_version, benchmark_manifest_hash=manifest_hash,
        git_commit_sha="UNAVAILABLE_NO_GIT_METADATA", git_dirty=True, corpus_hash=retriever.corpus_hash,
        query_policy_version=QUERY_POLICY_VERSION, query_policy_hash=query_policy_hash(),
        indexing_policy_version=INDEXING_POLICY_VERSION, indexing_policy_hash=indexing_policy_hash(),
        tokenization_policy_version=TOKENIZATION_POLICY_VERSION, tokenization_policy_hash=tokenization_policy_hash(),
        retriever_name=retriever.retriever_name, retriever_version=retriever.retriever_version,
        retriever_config_hash=retriever.retriever_config_hash, top_k_values=(cfg.top_k,), example_count=400,
        completed_count=400, result_hashes={"canonical": result_hash}, metric_hashes={})
    (output_dir / "run_manifest.json").write_bytes(manifest.to_json_bytes())
    return summary
