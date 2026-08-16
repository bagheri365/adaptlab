from __future__ import annotations

import json

from adaptlab.domain.enums import Difficulty, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily
from adaptlab.retrieval.absent_diagnostics import summarize_absent_diagnostics, with_absent_diagnostics
from adaptlab.retrieval.metrics import with_retrieval_metrics
from adaptlab.retrieval.schemas import RetrievalResult

H1 = "a" * 64
H2 = "b" * 64
H3 = "c" * 64


def result(**overrides) -> RetrievalResult:
    values = dict(
        retrieval_run_id="run-1", corpus_hash=H1, example_id="absent-1", split=Split.test,
        task_family=TaskFamily.knowledge_only, difficulty=Difficulty.HARD,
        knowledge_state=KnowledgeState.UPDATED, evidence_status=EvidenceStatus.ABSENT,
        split_type=SplitType.structural_holdout, retrieval_eligible=True, query_text="missing fact",
        query_hash=H2, retriever_name="BM25", retriever_version="bm25-v1",
        retriever_config_hash=H3, indexing_policy_version="index-v1",
        tokenization_policy_version="token-v1", top_k=5,
        candidate_chunk_ids=("c1", "c2", "c3"), candidate_scores=(3.5, 2.0, 1.0),
        candidate_ranks=(1, 2, 3), gold_chunk_ids=(), required_gold_chunk_ids=(),
        any_gold_at_1=None, any_gold_at_3=None, any_gold_at_5=None, any_gold_at_k=None,
        all_required_gold_at_1=None, all_required_gold_at_3=None, all_required_gold_at_5=None,
        all_required_gold_at_k=None, gold_recall_at_1=None, gold_recall_at_3=None,
        gold_recall_at_5=None, gold_recall_at_k=None, first_gold_reciprocal_rank=None,
        wrong_version_top1=None, current_gold_retrieved=None, obsolete_only_retrieved=None,
        current_and_obsolete_retrieved=None,
    )
    values.update(overrides)
    return RetrievalResult(**values)


def test_absent_diagnostics_record_ranked_context_without_fake_gold_metrics() -> None:
    diagnosed = with_absent_diagnostics(result())
    assert diagnosed.top1_chunk_id == "c1"
    assert diagnosed.top1_score == 3.5
    assert diagnosed.top_k_chunk_ids == ("c1", "c2", "c3")
    assert diagnosed.score_margin_top1_top2 == 1.5
    assert diagnosed.retrieval_returned_any_context is True
    assert diagnosed.wrongly_high_confidence is None

    scored = with_retrieval_metrics(diagnosed)
    assert scored.any_gold_at_1 is None
    assert scored.gold_recall_at_5 is None


def test_absent_diagnostics_handle_no_results_and_single_result_margin() -> None:
    empty = with_absent_diagnostics(result(candidate_chunk_ids=(), candidate_scores=(), candidate_ranks=()))
    assert empty.top1_chunk_id is None
    assert empty.top1_score is None
    assert empty.top_k_chunk_ids == ()
    assert empty.score_margin_top1_top2 is None
    assert empty.retrieval_returned_any_context is False

    single = with_absent_diagnostics(result(candidate_chunk_ids=("c1",), candidate_scores=(1.25,), candidate_ranks=(1,)))
    assert single.top1_chunk_id == "c1"
    assert single.score_margin_top1_top2 is None


def test_non_absent_examples_do_not_receive_absent_diagnostics() -> None:
    present = result(
        evidence_status=EvidenceStatus.PRESENT,
        gold_chunk_ids=("c1",), required_gold_chunk_ids=("c1",),
    )
    diagnosed = with_absent_diagnostics(present)
    assert diagnosed.top1_chunk_id is None
    assert diagnosed.retrieval_returned_any_context is None


def test_absent_report_is_separate_deterministic_and_sorted() -> None:
    report = summarize_absent_diagnostics((
        result(example_id="z"),
        result(example_id="a", candidate_chunk_ids=(), candidate_scores=(), candidate_ranks=()),
        result(example_id="present", evidence_status=EvidenceStatus.PRESENT,
               gold_chunk_ids=("c1",), required_gold_chunk_ids=("c1",)),
    ))
    assert [row.example_id for row in report.rows] == ["a", "z"]
    payload = json.loads(report.to_json_bytes())
    assert payload["scope"] == "retrieval_eligible AND evidence_status=ABSENT"
    assert payload["confidence_policy"] == "not_defined"
    assert payload["rows"][1]["retrieval_returned_any_context"] is True
    assert "unverified" in report.to_text()


def test_schema_round_trip_preserves_absent_diagnostic_fields() -> None:
    diagnosed = with_absent_diagnostics(result())
    restored = RetrievalResult.from_dict(json.loads(diagnosed.to_json_bytes()))
    assert restored == diagnosed
