from __future__ import annotations

from dataclasses import replace
import json

import pytest

from adaptlab.domain.enums import Difficulty, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily
from adaptlab.retrieval.metrics import (
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
    with_retrieval_metrics,
)
from adaptlab.retrieval.schemas import RetrievalResult

H1 = "a" * 64
H2 = "b" * 64
H3 = "c" * 64


def result(**overrides) -> RetrievalResult:
    values = dict(
        retrieval_run_id="run-1", corpus_hash=H1, example_id="ex-1", split=Split.test,
        task_family=TaskFamily.knowledge_only, difficulty=Difficulty.EASY,
        knowledge_state=KnowledgeState.UNCHANGED, evidence_status=EvidenceStatus.PRESENT,
        split_type=SplitType.iid, retrieval_eligible=True, query_text="alpha", query_hash=H2,
        retriever_name="BM25", retriever_version="bm25-v1", retriever_config_hash=H3,
        indexing_policy_version="index-v1", tokenization_policy_version="token-v1", top_k=5,
        candidate_chunk_ids=("g1",), candidate_scores=(2.0,), candidate_ranks=(1,),
        gold_chunk_ids=("g1",), required_gold_chunk_ids=("g1",),
        any_gold_at_1=None, any_gold_at_3=None, any_gold_at_5=None, any_gold_at_k=None,
        all_required_gold_at_1=None, all_required_gold_at_3=None, all_required_gold_at_5=None,
        all_required_gold_at_k=None, gold_recall_at_1=None, gold_recall_at_3=None,
        gold_recall_at_5=None, gold_recall_at_k=None, first_gold_reciprocal_rank=None,
        wrong_version_top1=None, current_gold_retrieved=None, obsolete_only_retrieved=None,
        current_and_obsolete_retrieved=None,
    )
    values.update(overrides)
    return RetrievalResult(**values)


def test_single_gold_any_and_all_required_are_equal() -> None:
    metrics = compute_retrieval_metrics(("x", "g1"), ("g1",), ("g1",), top_k=5)
    assert metrics.any_gold_at_1 is False
    assert metrics.all_required_gold_at_1 is False
    assert metrics.any_gold_at_3 is True
    assert metrics.all_required_gold_at_3 is True
    assert metrics.gold_recall_at_1 == 0.0
    assert metrics.gold_recall_at_3 == 1.0
    assert metrics.first_gold_reciprocal_rank == 0.5


def test_multi_chunk_distinguishes_any_all_and_partial_recall() -> None:
    metrics = compute_retrieval_metrics(
        ("g1", "distractor", "g2", "other", "g3"),
        ("g1", "g2", "g3"),
        ("g1", "g2", "g3"),
        top_k=5,
    )
    assert metrics.any_gold_at_1 is True
    assert metrics.all_required_gold_at_1 is False
    assert metrics.gold_recall_at_1 == pytest.approx(1 / 3)
    assert metrics.all_required_gold_at_3 is False
    assert metrics.gold_recall_at_3 == pytest.approx(2 / 3)
    assert metrics.all_required_gold_at_5 is True
    assert metrics.gold_recall_at_5 == 1.0
    assert metrics.first_gold_reciprocal_rank == 1.0


def test_any_gold_can_use_permitted_nonrequired_gold_without_inflating_required_recall() -> None:
    metrics = compute_retrieval_metrics(
        ("alternate-gold",),
        ("required", "alternate-gold"),
        ("required",),
        top_k=1,
    )
    assert metrics.any_gold_at_1 is True
    assert metrics.all_required_gold_at_1 is False
    assert metrics.gold_recall_at_1 == 0.0


def test_no_gold_in_ranked_candidates_has_zero_metrics_and_mrr() -> None:
    metrics = compute_retrieval_metrics(("x", "y"), ("g1", "g2"), ("g1", "g2"), top_k=5)
    assert metrics.any_gold_at_5 is False
    assert metrics.all_required_gold_at_5 is False
    assert metrics.gold_recall_at_5 == 0.0
    assert metrics.first_gold_reciprocal_rank == 0.0


def test_metric_inputs_require_evidence_present_gold_contract() -> None:
    with pytest.raises(ValueError, match="at least one gold"):
        compute_retrieval_metrics((), (), (), top_k=5)
    with pytest.raises(ValueError, match="required_gold"):
        compute_retrieval_metrics((), ("g",), (), top_k=5)


def test_with_metrics_leaves_absent_and_behavior_only_not_applicable() -> None:
    absent = result(
        evidence_status=EvidenceStatus.ABSENT,
        gold_chunk_ids=(), required_gold_chunk_ids=(), candidate_chunk_ids=("x",),
        candidate_scores=(1.0,), candidate_ranks=(1,),
    )
    scored_absent = with_retrieval_metrics(absent)
    assert scored_absent.any_gold_at_1 is None
    assert scored_absent.gold_recall_at_5 is None

    behavior = result(
        task_family=TaskFamily.behavior_only,
        knowledge_state=KnowledgeState.NOT_APPLICABLE,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        retrieval_eligible=False,
        query_text="",
        candidate_chunk_ids=(), candidate_scores=(), candidate_ranks=(),
        gold_chunk_ids=(), required_gold_chunk_ids=(),
    )
    scored_behavior = with_retrieval_metrics(behavior)
    assert scored_behavior.any_gold_at_1 is None
    assert scored_behavior.first_gold_reciprocal_rank is None


def test_report_excludes_behavior_only_and_absent_denominators_and_always_has_n() -> None:
    present_a = result(example_id="a", candidate_chunk_ids=("g1",), candidate_scores=(1.0,), candidate_ranks=(1,))
    present_b = result(
        example_id="b", task_family=TaskFamily.changed_knowledge, difficulty=Difficulty.HARD,
        knowledge_state=KnowledgeState.UPDATED, split_type=SplitType.structural_holdout,
        candidate_chunk_ids=("x",), candidate_scores=(1.0,), candidate_ranks=(1,),
    )
    absent = result(
        example_id="c", evidence_status=EvidenceStatus.ABSENT,
        gold_chunk_ids=(), required_gold_chunk_ids=(), candidate_chunk_ids=("x",),
        candidate_scores=(1.0,), candidate_ranks=(1,),
    )
    behavior = result(
        example_id="d", task_family=TaskFamily.behavior_only,
        knowledge_state=KnowledgeState.NOT_APPLICABLE, evidence_status=EvidenceStatus.NOT_APPLICABLE,
        retrieval_eligible=False, query_text="", candidate_chunk_ids=(), candidate_scores=(),
        candidate_ranks=(), gold_chunk_ids=(), required_gold_chunk_ids=(),
    )

    report = summarize_retrieval_metrics((present_a, present_b, absent, behavior))
    overall = report.rows[0]
    assert overall.dimension == "overall"
    assert overall.n == 2
    assert overall.any_gold_at_1 == 0.5
    assert overall.all_required_gold_at_1 == 0.5
    assert all(row.n > 0 for row in report.rows)
    assert {row.dimension for row in report.rows} == {
        "overall", "task_family", "difficulty", "knowledge_state", "split_type"
    }


def test_report_has_deterministic_machine_and_human_readable_outputs() -> None:
    report = summarize_retrieval_metrics((result(),))
    machine = report.to_json_bytes()
    assert machine == report.to_json_bytes()
    parsed = json.loads(machine)
    assert parsed["rows"][0]["n"] == 1
    assert parsed["rows"][0]["ANY_GOLD@1"] == 1.0

    human = report.to_text()
    assert "Retrieval quality" in human
    assert "task_family" in human
    assert "MRR" in human
    assert "1.000000" in human
