from collections import Counter

from adaptlab.benchmark.generate_docs import generate_documents
from adaptlab.benchmark.generate_tasks import generate_tasks
from adaptlab.benchmark.generate_world import generate_world
from adaptlab.domain.enums import BehaviorType, EvidenceStatus, KnowledgeState, TaskFamily


def _fixture(seed: int = 1729):
    world = generate_world(seed)
    documents, chunks = generate_documents(world)
    examples = generate_tasks(world, documents, chunks)
    return world, documents, chunks, examples


def test_generate_tasks_has_five_examples_per_family() -> None:
    _, _, _, examples = _fixture()
    counts = Counter(example.task_family for example in examples)
    assert counts == {
        TaskFamily.behavior_only: 5,
        TaskFamily.knowledge_only: 5,
        TaskFamily.behavior_knowledge: 5,
        TaskFamily.changed_knowledge: 5,
    }


def test_all_task_families_and_behavior_types_are_represented() -> None:
    _, _, _, examples = _fixture()
    assert {example.task_family for example in examples} == set(TaskFamily)
    assert {example.behavior_type for example in examples if example.behavior_type is not None} == set(BehaviorType)


def test_behavior_only_contains_all_answer_facts_and_no_external_evidence() -> None:
    _, _, _, examples = _fixture()
    behavior_examples = [example for example in examples if example.task_family is TaskFamily.behavior_only]
    assert behavior_examples
    assert all(example.evidence_status is EvidenceStatus.NOT_APPLICABLE for example in behavior_examples)
    assert all(not example.required_record_ids for example in behavior_examples)
    assert all(not example.gold_document_ids for example in behavior_examples)
    assert all(not example.gold_chunk_ids for example in behavior_examples)


def test_knowledge_only_present_examples_have_gold_evidence() -> None:
    _, documents, chunks, examples = _fixture()
    document_ids = {document.document_id for document in documents}
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    present = [
        example
        for example in examples
        if example.task_family is TaskFamily.knowledge_only
        and example.evidence_status is EvidenceStatus.PRESENT
    ]
    assert present
    assert all(set(example.gold_document_ids) <= document_ids for example in present)
    assert all(set(example.gold_chunk_ids) <= chunk_ids for example in present)


def test_behavior_knowledge_always_has_behavior_and_external_evidence() -> None:
    _, _, _, examples = _fixture()
    combined = [example for example in examples if example.task_family is TaskFamily.behavior_knowledge]
    assert combined
    assert all(example.behavior_type is not None for example in combined)
    assert all(example.evidence_status is EvidenceStatus.PRESENT for example in combined)
    assert all(example.required_record_ids for example in combined)
    assert all(example.gold_document_ids and example.gold_chunk_ids for example in combined)


def test_changed_knowledge_covers_all_lifecycle_states_and_evidence_absent() -> None:
    _, _, _, examples = _fixture()
    changed = [example for example in examples if example.task_family is TaskFamily.changed_knowledge]
    assert {example.knowledge_state for example in changed} >= {
        KnowledgeState.UNCHANGED,
        KnowledgeState.UPDATED,
        KnowledgeState.REMOVED,
    }
    absent = [example for example in changed if example.evidence_status is EvidenceStatus.ABSENT]
    assert absent
    assert all(not example.required_record_ids for example in absent)
    assert all(not example.required_logical_fact_ids for example in absent)
    assert all(not example.gold_document_ids for example in absent)
    assert all(not example.gold_chunk_ids for example in absent)


def test_generate_tasks_is_deterministic_and_canonically_ordered() -> None:
    _, _, _, first = _fixture(1729)
    _, _, _, second = _fixture(1729)
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]
    ids = [example.example_id for example in first]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_different_seed_generates_valid_examples() -> None:
    world, documents, chunks, examples = _fixture(7331)
    assert world.generation_seed == 7331
    assert len(examples) == 20
    assert all(example.generation_seed == 7331 for example in examples)
    assert documents and chunks


def test_changed_knowledge_updated_answer_uses_v2_value_not_lifecycle_label() -> None:
    world, _, _, examples = _fixture()
    records = {fact.record_id: fact for fact in world.facts}
    updated = next(
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.UPDATED
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    required = records[updated.required_record_ids[0]]
    assert required.version == "v2"
    assert updated.expected_output == required.value
    assert updated.expected_output != KnowledgeState.UPDATED.value


def test_changed_knowledge_unchanged_answer_uses_current_value() -> None:
    world, _, _, examples = _fixture()
    records = {fact.record_id: fact for fact in world.facts}
    unchanged = next(
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.UNCHANGED
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    required = records[unchanged.required_record_ids[0]]
    assert required.version == "v2"
    assert unchanged.expected_output == required.value
    assert unchanged.expected_output != KnowledgeState.UNCHANGED.value


def test_changed_knowledge_removed_answer_uses_retirement_policy() -> None:
    _, _, _, examples = _fixture()
    removed = next(
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.REMOVED
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    assert removed.expected_output == "RETIRED"
    assert removed.expected_output != KnowledgeState.REMOVED.value


def test_changed_knowledge_questions_are_current_knowledge_questions() -> None:
    _, _, _, examples = _fixture()
    changed = [example for example in examples if example.task_family is TaskFamily.changed_knowledge]
    forbidden = ("classify logical fact", "unchanged, updated, or removed")
    assert all("current" in example.question.lower() for example in changed)
    assert all(not any(phrase in example.question.lower() for phrase in forbidden) for example in changed)


def test_evidence_absent_questions_do_not_disclose_missing_evidence() -> None:
    _, _, _, examples = _fixture()
    absent = [example for example in examples if example.evidence_status is EvidenceStatus.ABSENT]
    assert absent
    forbidden = (
        "no evidence",
        "evidence is absent",
        "insufficient evidence",
        "answer insufficient_evidence",
    )
    for example in absent:
        question = example.question.lower()
        assert not any(phrase in question for phrase in forbidden)
        assert example.expected_output == "INSUFFICIENT_EVIDENCE"
        assert not example.required_record_ids
        assert not example.required_logical_fact_ids
        assert not example.gold_document_ids
        assert not example.gold_chunk_ids


def test_changed_knowledge_absent_uses_fact_without_current_v2_record() -> None:
    world, _, _, examples = _fixture()
    absent = next(
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.evidence_status is EvidenceStatus.ABSENT
    )
    assert absent.knowledge_state is KnowledgeState.REMOVED
    classic_records = [fact for fact in world.facts if fact.logical_fact_id == "DEPLOY_CLASSIC_MODE"]
    assert {fact.version for fact in classic_records} == {"v1"}
    assert "classic deployment mode" in absent.question.lower()
