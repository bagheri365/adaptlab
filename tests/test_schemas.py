import pytest

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    Split,
    SplitType,
    TaskFamily,
)


def make_example(**overrides):
    data = dict(
        example_id="ex-001",
        benchmark_version="0.1",
        task_family=TaskFamily.knowledge_only,
        behavior_type=None,
        difficulty=Difficulty.EASY,
        split=Split.train,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version="v2",
        knowledge_state=KnowledgeState.UNCHANGED,
        evidence_status=EvidenceStatus.PRESENT,
        question="What is the current Nimbus value?",
        expected_output="answer",
        required_record_ids=("REC_V2",),
        required_logical_fact_ids=("REC",),
        gold_document_ids=("doc-1",),
        gold_chunk_ids=("chunk-1",),
        generation_seed=1729,
    )
    data.update(overrides)
    return BenchmarkExample(**data)


def test_valid_behavior_only():
    ex = make_example(
        task_family=TaskFamily.behavior_only,
        behavior_type=BehaviorType.SCHEMA_ADHERENCE,
        knowledge_state=KnowledgeState.NOT_APPLICABLE,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
    )
    assert ex.task_family is TaskFamily.behavior_only


def test_behavior_only_requires_behavior_type():
    with pytest.raises(ValueError, match="require behavior_type"):
        make_example(
            task_family=TaskFamily.behavior_only,
            behavior_type=None,
            knowledge_state=KnowledgeState.NOT_APPLICABLE,
            evidence_status=EvidenceStatus.NOT_APPLICABLE,
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_document_ids=(),
            gold_chunk_ids=(),
        )


def test_behavior_only_requires_not_applicable_evidence_and_knowledge():
    with pytest.raises(ValueError, match="evidence_status=NOT_APPLICABLE"):
        make_example(
            task_family=TaskFamily.behavior_only,
            behavior_type=BehaviorType.CLASSIFICATION_POLICY,
            knowledge_state=KnowledgeState.NOT_APPLICABLE,
            evidence_status=EvidenceStatus.PRESENT,
        )
    with pytest.raises(ValueError, match="knowledge_state=NOT_APPLICABLE"):
        make_example(
            task_family=TaskFamily.behavior_only,
            behavior_type=BehaviorType.CLASSIFICATION_POLICY,
            knowledge_state=KnowledgeState.UNCHANGED,
            evidence_status=EvidenceStatus.NOT_APPLICABLE,
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_document_ids=(),
            gold_chunk_ids=(),
        )


def test_knowledge_only_requires_null_behavior_type():
    with pytest.raises(ValueError, match="behavior_type=None"):
        make_example(behavior_type=BehaviorType.ABSTENTION_BEHAVIOR)


def test_behavior_knowledge_requires_behavior_type():
    with pytest.raises(ValueError, match="require behavior_type"):
        make_example(task_family=TaskFamily.behavior_knowledge)


def test_changed_knowledge_requires_lifecycle_state():
    with pytest.raises(ValueError, match="UNCHANGED, UPDATED, or REMOVED"):
        make_example(
            task_family=TaskFamily.changed_knowledge,
            knowledge_state=KnowledgeState.NOT_APPLICABLE,
        )


@pytest.mark.parametrize(
    "state",
    [KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED],
)
def test_changed_knowledge_accepts_all_lifecycle_states(state):
    ex = make_example(
        task_family=TaskFamily.changed_knowledge,
        knowledge_state=state,
    )
    assert ex.knowledge_state is state


def test_evidence_absent_requires_all_evidence_refs_empty():
    with pytest.raises(ValueError, match="requires empty evidence references"):
        make_example(
            evidence_status=EvidenceStatus.ABSENT,
            gold_document_ids=("doc-1",),
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_chunk_ids=(),
        )

    ex = make_example(
        evidence_status=EvidenceStatus.ABSENT,
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
    )
    assert ex.evidence_status is EvidenceStatus.ABSENT


@pytest.mark.parametrize(
    "field",
    ["required_record_ids", "gold_document_ids", "gold_chunk_ids"],
)
def test_evidence_present_requires_core_gold_refs(field):
    kwargs = {field: ()}
    with pytest.raises(ValueError, match="evidence_status=PRESENT"):
        make_example(**kwargs)


def test_structural_holdout_requires_dimension_and_group():
    with pytest.raises(ValueError, match="holdout_dimension and holdout_group"):
        make_example(split_type=SplitType.structural_holdout)


def test_serialization_round_trip():
    original = make_example()
    restored = BenchmarkExample.from_dict(original.to_dict())
    assert restored == original
    assert original.to_dict()["task_family"] == "knowledge_only"
    assert original.to_dict()["difficulty"] == "EASY"
