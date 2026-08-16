from __future__ import annotations

from dataclasses import replace

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.domain.enums import Difficulty, DocumentStyle, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily
from adaptlab.retrieval.schemas import RetrievalResult
from adaptlab.retrieval.version_metrics import summarize_version_diagnostics, with_version_diagnostics

H1, H2, H3 = "a"*64, "b"*64, "c"*64


def chunk(cid: str, logical: str, *, obsolete: bool, authoritative: bool = True) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, document_id=f"doc-{cid}", version="v1" if obsolete else "v2",
        component_family="authentication", document_style=DocumentStyle.reference_documentation,
        content=f"content {cid}", record_ids=(f"rec-{cid}",), logical_fact_ids=(logical,),
        is_authoritative=authoritative, is_obsolete=obsolete,
    )


def result(state: KnowledgeState, candidates: tuple[str, ...], *, gold: str = "cur") -> RetrievalResult:
    return RetrievalResult(
        retrieval_run_id="run-1", corpus_hash=H1, example_id=f"ex-{state.value}", split=Split.test,
        task_family=TaskFamily.changed_knowledge, difficulty=Difficulty.EASY, knowledge_state=state,
        evidence_status=EvidenceStatus.PRESENT, split_type=SplitType.iid, retrieval_eligible=True,
        query_text="question", query_hash=H2, retriever_name="BM25", retriever_version="bm25-v1",
        retriever_config_hash=H3, indexing_policy_version="index-v1", tokenization_policy_version="token-v1",
        top_k=5, candidate_chunk_ids=candidates, candidate_scores=tuple(float(5-i) for i in range(len(candidates))),
        candidate_ranks=tuple(range(1, len(candidates)+1)), gold_chunk_ids=(gold,), required_gold_chunk_ids=(gold,),
        any_gold_at_1=None, any_gold_at_3=None, any_gold_at_5=None, any_gold_at_k=None,
        all_required_gold_at_1=None, all_required_gold_at_3=None, all_required_gold_at_5=None, all_required_gold_at_k=None,
        gold_recall_at_1=None, gold_recall_at_3=None, gold_recall_at_5=None, gold_recall_at_k=None,
        first_gold_reciprocal_rank=None, wrong_version_top1=None, current_gold_retrieved=None,
        obsolete_only_retrieved=None, current_and_obsolete_retrieved=None,
    )


CORPUS = (
    chunk("cur", "FACT", obsolete=False),
    chunk("obs", "FACT", obsolete=True, authoritative=False),
    chunk("other", "OTHER", obsolete=True, authoritative=False),
)


def test_unchanged_current_only_is_not_wrong_version() -> None:
    out = with_version_diagnostics(result(KnowledgeState.UNCHANGED, ("cur",)), CORPUS)
    assert out.current_gold_retrieved is True
    assert out.obsolete_only_retrieved is False
    assert out.current_and_obsolete_retrieved is False
    assert out.wrong_version_top1 is False


def test_updated_obsolete_instead_is_detected_mechanically() -> None:
    out = with_version_diagnostics(result(KnowledgeState.UPDATED, ("obs", "other")), CORPUS)
    assert out.current_gold_retrieved is False
    assert out.obsolete_only_retrieved is True
    assert out.current_and_obsolete_retrieved is False
    assert out.wrong_version_top1 is True


def test_updated_both_current_and_obsolete_is_distinct() -> None:
    out = with_version_diagnostics(result(KnowledgeState.UPDATED, ("cur", "obs")), CORPUS)
    assert out.current_gold_retrieved is True
    assert out.obsolete_only_retrieved is False
    assert out.current_and_obsolete_retrieved is True
    assert out.wrong_version_top1 is False


def test_updated_neither_is_distinct() -> None:
    out = with_version_diagnostics(result(KnowledgeState.UPDATED, ("other",)), CORPUS)
    assert out.current_gold_retrieved is False
    assert out.obsolete_only_retrieved is False
    assert out.current_and_obsolete_retrieved is False
    assert out.wrong_version_top1 is False


def test_removed_uses_retirement_gold_and_obsolete_history_without_replacement_assumption() -> None:
    retirement = chunk("retirement", "REMOVED_FACT", obsolete=False)
    old = chunk("old-value", "REMOVED_FACT", obsolete=True, authoritative=False)
    out = with_version_diagnostics(
        result(KnowledgeState.REMOVED, ("old-value",), gold="retirement"),
        (retirement, old),
    )
    assert out.current_gold_retrieved is False
    assert out.obsolete_only_retrieved is True
    assert out.wrong_version_top1 is True


def test_non_applicable_examples_get_no_fake_version_labels() -> None:
    base = result(KnowledgeState.UNCHANGED, ("cur",))
    out = with_version_diagnostics(replace(base, task_family=TaskFamily.knowledge_only), CORPUS)
    assert out.current_gold_retrieved is None
    assert out.wrong_version_top1 is None


def test_report_has_counts_and_rates_by_knowledge_state_and_deterministic_outputs() -> None:
    inputs = (
        result(KnowledgeState.UNCHANGED, ("cur",)),
        result(KnowledgeState.UPDATED, ("obs",)),
        result(KnowledgeState.UPDATED, ("cur", "obs")),
        result(KnowledgeState.REMOVED, ("other",)),
    )
    report = summarize_version_diagnostics(inputs, CORPUS)
    by_state = {row.knowledge_state: row for row in report.rows}
    assert by_state["UPDATED"].n == 2
    assert by_state["UPDATED"].current_gold_retrieved_count == 1
    assert by_state["UPDATED"].current_gold_retrieved_rate == 0.5
    assert by_state["UPDATED"].obsolete_only_retrieved_count == 1
    assert report.to_json_bytes() == report.to_json_bytes()
    assert "UPDATED" in report.to_text()
