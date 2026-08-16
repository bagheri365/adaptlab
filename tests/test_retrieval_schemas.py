from __future__ import annotations

import json

import pytest

from adaptlab.domain.enums import Difficulty, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily
from adaptlab.retrieval.schemas import (
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    RETRIEVAL_RUN_SCHEMA_VERSION,
    RetrievalResult,
    RetrievalRunManifest,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def make_result(**overrides) -> RetrievalResult:
    values = dict(
        retrieval_run_id="retrieval-run-001",
        corpus_hash=HASH_A,
        example_id="test-0001",
        split=Split.test,
        task_family=TaskFamily.knowledge_only,
        difficulty=Difficulty.MEDIUM,
        knowledge_state=KnowledgeState.UPDATED,
        evidence_status=EvidenceStatus.PRESENT,
        split_type=SplitType.iid,
        retrieval_eligible=True,
        query_text="What is the current Nimbus value?",
        query_hash=HASH_B,
        retriever_name="BM25",
        retriever_version="bm25-v1",
        retriever_config_hash=HASH_C,
        indexing_policy_version="index-v1",
        tokenization_policy_version="tokens-v1",
        top_k=5,
        candidate_chunk_ids=("chunk-001", "chunk-004"),
        candidate_scores=(4.2, 2.1),
        candidate_ranks=(1, 2),
        gold_chunk_ids=("chunk-001", "chunk-002"),
        required_gold_chunk_ids=("chunk-001", "chunk-002"),
        any_gold_at_1=True,
        any_gold_at_3=True,
        any_gold_at_5=True,
        any_gold_at_k=True,
        all_required_gold_at_1=False,
        all_required_gold_at_3=False,
        all_required_gold_at_5=False,
        all_required_gold_at_k=False,
        gold_recall_at_1=0.5,
        gold_recall_at_3=0.5,
        gold_recall_at_5=0.5,
        gold_recall_at_k=0.5,
        first_gold_reciprocal_rank=1.0,
        wrong_version_top1=False,
        current_gold_retrieved=True,
        obsolete_only_retrieved=False,
        current_and_obsolete_retrieved=False,
    )
    values.update(overrides)
    return RetrievalResult(**values)


def make_manifest(**overrides) -> RetrievalRunManifest:
    values = dict(
        run_id="retrieval-run-001",
        benchmark_version="0.0.0",
        benchmark_manifest_hash=HASH_A,
        git_commit_sha="deadbeef",
        git_dirty=False,
        corpus_hash=HASH_B,
        query_policy_version="query-v1",
        query_policy_hash=HASH_C,
        indexing_policy_version="index-v1",
        indexing_policy_hash=HASH_A,
        tokenization_policy_version="tokens-v1",
        tokenization_policy_hash=HASH_B,
        retriever_name="BM25",
        retriever_version="bm25-v1",
        retriever_config_hash=HASH_C,
        top_k_values=(1, 3, 5),
        example_count=40,
        completed_count=40,
        result_hashes={"results.json": HASH_A},
        metric_hashes={"metrics.json": HASH_B},
    )
    values.update(overrides)
    return RetrievalRunManifest(**values)


def test_retrieval_result_contains_prompt_1_contract_and_corpus_traceability() -> None:
    payload = make_result().to_dict()
    required = {
        "example_id", "split", "task_family", "difficulty", "knowledge_state",
        "evidence_status", "split_type", "retrieval_eligible", "query_text", "query_hash", "retriever_name",
        "retriever_version", "retriever_config_hash", "indexing_policy_version",
        "tokenization_policy_version", "top_k", "candidate_chunk_ids", "candidate_scores",
        "candidate_ranks", "gold_chunk_ids", "required_gold_chunk_ids", "any_gold_at_1",
        "any_gold_at_3", "any_gold_at_5", "any_gold_at_k", "all_required_gold_at_1",
        "all_required_gold_at_3", "all_required_gold_at_5", "all_required_gold_at_k",
        "gold_recall_at_1", "gold_recall_at_3", "gold_recall_at_5", "gold_recall_at_k",
        "first_gold_reciprocal_rank", "wrong_version_top1", "current_gold_retrieved",
        "obsolete_only_retrieved", "current_and_obsolete_retrieved",
    }
    assert required <= set(payload)
    assert payload["retrieval_run_id"] == "retrieval-run-001"
    assert payload["corpus_hash"] == HASH_A
    assert payload["schema_version"] == RETRIEVAL_RESULT_SCHEMA_VERSION


def test_retrieval_result_is_deterministic_and_round_trips() -> None:
    result = make_result()
    assert result.to_json_bytes() == result.to_json_bytes()
    assert result.to_json_bytes().endswith(b"\n")
    assert RetrievalResult.from_dict(json.loads(result.to_json_bytes())) == result


def test_retrieval_result_supports_not_applicable_metrics_without_fake_values() -> None:
    result = make_result(
        evidence_status=EvidenceStatus.ABSENT,
        gold_chunk_ids=(),
        required_gold_chunk_ids=(),
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
    )
    assert result.gold_recall_at_k is None


def test_retrieval_result_rejects_misaligned_candidates_and_bad_provenance() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        make_result(candidate_scores=(1.0,))
    with pytest.raises(ValueError, match="corpus_hash"):
        make_result(corpus_hash="not-a-hash")
    with pytest.raises(ValueError, match="subset"):
        make_result(required_gold_chunk_ids=("chunk-999",))


def test_retrieval_manifest_contains_prompt_1_contract() -> None:
    payload = make_manifest().to_dict()
    required = {
        "run_id", "benchmark_version", "benchmark_manifest_hash", "git_commit_sha",
        "git_dirty", "corpus_hash", "query_policy_version", "query_policy_hash",
        "indexing_policy_version", "indexing_policy_hash", "tokenization_policy_version",
        "tokenization_policy_hash", "retriever_name", "retriever_version",
        "retriever_config_hash", "top_k_values", "example_count", "completed_count",
        "result_hashes", "metric_hashes",
    }
    assert required <= set(payload)
    assert payload["schema_version"] == RETRIEVAL_RUN_SCHEMA_VERSION


def test_retrieval_manifest_is_deterministic_and_round_trips() -> None:
    manifest = make_manifest(result_hashes={"z.json": HASH_C, "a.json": HASH_A})
    encoded = manifest.to_json_bytes()
    assert encoded.index(b'"a.json"') < encoded.index(b'"z.json"')
    assert RetrievalRunManifest.from_dict(json.loads(encoded)) == manifest


def test_retrieval_manifest_rejects_invalid_counts_top_k_and_hashes() -> None:
    with pytest.raises(ValueError, match="completed_count"):
        make_manifest(example_count=1, completed_count=2)
    with pytest.raises(ValueError, match="unique and ascending"):
        make_manifest(top_k_values=(5, 1, 5))
    with pytest.raises(ValueError, match="retriever_config_hash"):
        make_manifest(retriever_config_hash="bad")
