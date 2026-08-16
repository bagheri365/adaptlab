from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    BehaviorType, Difficulty, EvidenceStatus, KnowledgeState, ScoringRule,
    Split, SplitType, TaskFamily,
)
from adaptlab.retrieval.query_policy import (
    QUERY_POLICY_VERSION,
    construct_retrieval_query,
    query_policy_hash,
    retrieve_if_eligible,
    verify_frozen_query_policy,
)


def _example(family: TaskFamily) -> BenchmarkExample:
    behavior = BehaviorType.SCHEMA_ADHERENCE if family in {TaskFamily.behavior_only, TaskFamily.behavior_knowledge} else None
    if family is TaskFamily.behavior_only:
        evidence = EvidenceStatus.NOT_APPLICABLE
        state = KnowledgeState.NOT_APPLICABLE
        records = facts = docs = chunks = ()
    else:
        evidence = EvidenceStatus.PRESENT
        state = KnowledgeState.UPDATED if family is TaskFamily.changed_knowledge else KnowledgeState.UNCHANGED
        records = ("record-secret",)
        facts = ("fact-secret",)
        docs = ("doc-secret",)
        chunks = ("chunk-secret",)
    return BenchmarkExample(
        example_id=f"example-{family.value}", benchmark_version="0.0.0", task_family=family,
        behavior_type=behavior, difficulty=Difficulty.EASY, split=Split.validation,
        split_type=SplitType.iid, holdout_dimension=None, holdout_group=None,
        knowledge_version="v2" if family is not TaskFamily.behavior_only else None,
        knowledge_state=state, evidence_status=evidence,
        question="What is Nimbus mode alpha?", expected_output="SECRET_EXPECTED_OUTPUT",
        required_record_ids=records, required_logical_fact_ids=facts,
        gold_document_ids=docs, gold_chunk_ids=chunks, generation_seed=1729,
        scoring_rule=ScoringRule.FACT_VALUE,
        scoring_parameters={"hidden": "SECRET_SCORING_METADATA"},
        lifecycle_logical_fact_id="lifecycle-secret" if family is TaskFamily.changed_knowledge else None,
    )


def test_frozen_policy_file_matches_executable_contract() -> None:
    path = Path("config/retrieval/query_policy_v1.json")
    assert verify_frozen_query_policy(path) == query_policy_hash()
    assert QUERY_POLICY_VERSION == "retrieval-query-v1"
    assert len(query_policy_hash()) == 64


def test_all_eligible_families_use_exact_question_only() -> None:
    for family in (TaskFamily.knowledge_only, TaskFamily.behavior_knowledge, TaskFamily.changed_knowledge):
        example = _example(family)
        query = construct_retrieval_query(example)
        assert query.retrieval_eligible is True
        assert query.query_text == example.question


def test_hidden_benchmark_metadata_is_excluded_from_query() -> None:
    example = _example(TaskFamily.changed_knowledge)
    query = construct_retrieval_query(example)
    forbidden = (
        "SECRET_EXPECTED_OUTPUT", "changed_knowledge", "EASY", "UPDATED", "PRESENT",
        "iid", "chunk-secret", "fact-secret", "record-secret", "lifecycle-secret",
        "SECRET_SCORING_METADATA",
    )
    assert query.query_text == "What is Nimbus mode alpha?"
    assert all(value not in query.query_text for value in forbidden)

    mutated = replace(
        example,
        expected_output="OTHER_OUTPUT",
        required_record_ids=("other-record",),
        required_logical_fact_ids=("other-fact",),
        gold_document_ids=("other-doc",),
        gold_chunk_ids=("other-chunk",),
        scoring_parameters={"other": "metadata"},
        lifecycle_logical_fact_id="other-lifecycle",
    )
    assert construct_retrieval_query(mutated) == query


def test_behavior_only_bypasses_retriever_entirely() -> None:
    calls: list[str] = []

    def retriever(text: str) -> list[str]:
        calls.append(text)
        return ["should-never-happen"]

    query, result = retrieve_if_eligible(_example(TaskFamily.behavior_only), retriever)
    assert query.retrieval_eligible is False
    assert query.query_text == ""
    assert result is None
    assert calls == []


def test_eligible_example_invokes_retriever_with_question_only() -> None:
    calls: list[str] = []
    query, result = retrieve_if_eligible(
        _example(TaskFamily.knowledge_only),
        lambda text: calls.append(text) or ["chunk-1"],
    )
    assert query.retrieval_eligible is True
    assert calls == ["What is Nimbus mode alpha?"]
    assert result == ["chunk-1"]
