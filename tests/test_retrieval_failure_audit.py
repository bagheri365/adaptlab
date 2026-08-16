from __future__ import annotations

from dataclasses import replace

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.domain.enums import Difficulty, DocumentStyle, EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily
from adaptlab.retrieval.failure_audit import (
    RetrievalFailureCategory,
    audit_retrieval_failure,
    summarize_retrieval_failures,
)
from adaptlab.retrieval.schemas import RetrievalResult

H1, H2, H3 = "a" * 64, "b" * 64, "c" * 64


def chunk(
    cid: str,
    logical: tuple[str, ...],
    *,
    component: str = "authentication",
    content: str | None = None,
    obsolete: bool = False,
    authoritative: bool = True,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id=f"doc-{cid}",
        version="v1" if obsolete else "v2",
        component_family=component,
        document_style=DocumentStyle.reference_documentation,
        content=content or f"content {cid}",
        record_ids=(f"rec-{cid}",) if logical else (),
        logical_fact_ids=logical,
        is_authoritative=authoritative,
        is_obsolete=obsolete,
    )


def result(
    candidates: tuple[str, ...],
    *,
    gold: tuple[str, ...] = ("gold-a", "gold-b"),
    required: tuple[str, ...] | None = None,
    query: str = "How does Nimbus authentication work?",
    evidence_status: EvidenceStatus = EvidenceStatus.PRESENT,
    knowledge_state: KnowledgeState = KnowledgeState.UPDATED,
) -> RetrievalResult:
    if required is None:
        required = gold
    return RetrievalResult(
        retrieval_run_id="run-1",
        corpus_hash=H1,
        example_id="ex-1",
        split=Split.test,
        task_family=TaskFamily.changed_knowledge,
        difficulty=Difficulty.HARD,
        knowledge_state=knowledge_state,
        evidence_status=evidence_status,
        split_type=SplitType.structural_holdout,
        retrieval_eligible=True,
        query_text=query,
        query_hash=H2,
        retriever_name="BM25",
        retriever_version="bm25-v1",
        retriever_config_hash=H3,
        indexing_policy_version="index-v1",
        tokenization_policy_version="token-v1",
        top_k=max(1, len(candidates)),
        candidate_chunk_ids=candidates,
        candidate_scores=tuple(float(len(candidates) - i) for i in range(len(candidates))),
        candidate_ranks=tuple(range(1, len(candidates) + 1)),
        gold_chunk_ids=gold,
        required_gold_chunk_ids=required,
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


CORPUS = (
    chunk("gold-a", ("FACT_A",)),
    chunk("gold-b", ("FACT_B",)),
    chunk("obs-a", ("FACT_A",), obsolete=True, authoritative=False),
    chunk("near-a", ("FACT_A",), authoritative=False),
    chunk("same-component", (), authoritative=False),
    chunk("other-component", (), component="storage", authoritative=False),
)


def categories(audit) -> set[str]:
    return set(audit.categories)


def test_no_gold_in_top_k_is_gold_outside_top_k() -> None:
    audit = audit_retrieval_failure(result(("other-component",)), CORPUS)
    assert RetrievalFailureCategory.GOLD_OUTSIDE_TOP_K.value in categories(audit)
    assert audit.missing_required_gold_chunk_ids == ("gold-a", "gold-b")


def test_partial_gold_is_distinguished_from_total_miss() -> None:
    audit = audit_retrieval_failure(result(("gold-a", "other-component")), CORPUS)
    assert RetrievalFailureCategory.PARTIAL_GOLD.value in categories(audit)
    assert RetrievalFailureCategory.GOLD_OUTSIDE_TOP_K.value not in categories(audit)
    assert audit.retrieved_gold_chunk_ids == ("gold-a",)
    assert audit.missing_required_gold_chunk_ids == ("gold-b",)


def test_obsolete_only_and_wrong_version_top1_reuse_frozen_provenance_logic() -> None:
    audit = audit_retrieval_failure(result(("obs-a",)), CORPUS)
    assert RetrievalFailureCategory.OBSOLETE_ONLY.value in categories(audit)
    assert RetrievalFailureCategory.WRONG_VERSION_TOP1.value in categories(audit)


def test_near_duplicate_and_same_component_are_mechanical() -> None:
    audit = audit_retrieval_failure(result(("near-a",)), CORPUS)
    assert RetrievalFailureCategory.NEAR_DUPLICATE_DISTRACTOR.value in categories(audit)
    assert RetrievalFailureCategory.SAME_COMPONENT_DISTRACTOR.value in categories(audit)


def test_same_component_distractor_does_not_require_near_duplicate_provenance() -> None:
    audit = audit_retrieval_failure(result(("same-component",)), CORPUS)
    assert RetrievalFailureCategory.SAME_COMPONENT_DISTRACTOR.value in categories(audit)
    assert RetrievalFailureCategory.NEAR_DUPLICATE_DISTRACTOR.value not in categories(audit)


def test_identifier_shortcut_requires_unique_query_identifier_on_a_retrieved_distractor() -> None:
    special = chunk(
        "special",
        (),
        component="storage",
        content="Operator note for UNIQUE_123 only.",
        authoritative=False,
    )
    corpus = CORPUS + (special,)
    audit = audit_retrieval_failure(
        result(("special",), query="What is UNIQUE_123 configured to do?"), corpus
    )
    assert RetrievalFailureCategory.IDENTIFIER_SHORTCUT.value in categories(audit)


def test_absent_examples_are_no_gold_exists_without_fake_gold_failure() -> None:
    absent = result(
        ("other-component",),
        gold=(),
        required=(),
        evidence_status=EvidenceStatus.ABSENT,
        knowledge_state=KnowledgeState.UNCHANGED,
    )
    audit = audit_retrieval_failure(absent, CORPUS)
    assert audit.categories == (RetrievalFailureCategory.NO_GOLD_EXISTS.value,)


def test_successful_all_required_retrieval_does_not_label_harmless_distractors_as_failures() -> None:
    audit = audit_retrieval_failure(result(("gold-a", "gold-b", "same-component")), CORPUS)
    assert audit.categories == ()


def test_report_groups_counts_by_all_required_dimensions_deterministically() -> None:
    inputs = (
        result(("gold-a", "other-component")),
        replace(result(("other-component",)), example_id="ex-2", difficulty=Difficulty.EASY),
    )
    report = summarize_retrieval_failures(inputs, CORPUS)
    dimensions = {row.dimension for row in report.groups}
    assert dimensions == {"task_family", "difficulty", "knowledge_state", "split_type"}
    hard = next(row for row in report.groups if row.dimension == "difficulty" and row.value == "HARD")
    assert hard.n == 1
    assert hard.category_counts[RetrievalFailureCategory.PARTIAL_GOLD.value] == 1
    assert report.to_json_bytes() == report.to_json_bytes()
    assert "difficulty=HARD" in report.to_text()


def test_unknown_retrieved_chunk_is_rejected() -> None:
    try:
        audit_retrieval_failure(result(("missing",)), CORPUS)
    except ValueError as exc:
        assert "missing from frozen corpus" in str(exc)
    else:
        raise AssertionError("expected frozen-corpus traceability validation")
