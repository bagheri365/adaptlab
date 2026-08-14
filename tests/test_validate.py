from dataclasses import replace

from adaptlab.benchmark.generate_docs import generate_documents
from adaptlab.benchmark.generate_tasks import generate_tasks
from adaptlab.benchmark.generate_world import generate_world
from adaptlab.benchmark.validate import apply_structural_holdout_rules, validate_fixture
from adaptlab.domain.enums import EvidenceStatus, KnowledgeState, Split, SplitType, TaskFamily


def _fixture(seed: int = 1729):
    world = generate_world(seed)
    documents, chunks = generate_documents(world)
    examples = apply_structural_holdout_rules(world, generate_tasks(world, documents, chunks))
    return world, documents, chunks, examples


def test_valid_fixture_passes() -> None:
    world, documents, chunks, examples = _fixture()
    result = validate_fixture(world, documents, chunks, examples)
    assert result.passed
    assert result.errors == ()
    assert result.statistics["example_count"] == 20


def test_broken_gold_reference_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(i for i, example in enumerate(examples) if example.gold_document_ids)
    broken = replace(examples[index], gold_document_ids=("DOC_DOES_NOT_EXIST",))
    examples[index] = broken
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("unknown gold document IDs" in error for error in result.errors)


def test_invalid_evidence_absent_example_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.ABSENT)
    object.__setattr__(examples[index], "required_record_ids", (world.facts[0].record_id,))
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("evidence_status=ABSENT" in error for error in result.errors)


def test_missing_behavior_type_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    example = next(example for example in examples if example.task_family is TaskFamily.behavior_knowledge)
    object.__setattr__(example, "behavior_type", None)
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("behavior_knowledge requires behavior_type" in error for error in result.errors)


def test_structural_assignment_is_deterministic_and_separate_from_split() -> None:
    world = generate_world(1729)
    documents, chunks = generate_documents(world)
    raw = generate_tasks(world, documents, chunks)
    first = apply_structural_holdout_rules(world, raw)
    second = apply_structural_holdout_rules(world, raw)
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]
    deployment_examples = [
        example for example in first if any(record_id.startswith("DEPLOY_") for record_id in example.required_record_ids)
    ]
    assert deployment_examples
    assert all(example.split is Split.test for example in deployment_examples)
    assert all(example.split_type is SplitType.structural_holdout for example in deployment_examples)
    assert all(example.holdout_dimension == "component_family" for example in deployment_examples)
    assert all(example.holdout_group == "deployments" for example in deployment_examples)


def test_structural_leakage_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(
        i
        for i, example in enumerate(examples)
        if example.holdout_group == "deployments"
    )
    leaking = replace(
        examples[index],
        split=Split.train,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
    )
    examples[index] = leaking
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("leaks structural-test-only component deployments" in error for error in result.errors)


def test_invalid_lifecycle_state_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    example = next(example for example in examples if example.task_family is TaskFamily.changed_knowledge)
    object.__setattr__(example, "knowledge_state", KnowledgeState.NOT_APPLICABLE)
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("invalid lifecycle state" in error for error in result.errors)


def test_non_authoritative_gold_chunk_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    competing = next(chunk for chunk in chunks if not chunk.is_authoritative and not chunk.is_obsolete and chunk.record_ids)
    index = next(
        i
        for i, example in enumerate(examples)
        if example.evidence_status is EvidenceStatus.PRESENT
        and set(example.required_record_ids).issubset(set(competing.record_ids))
    )
    examples[index] = replace(
        examples[index],
        gold_document_ids=(competing.document_id,),
        gold_chunk_ids=(competing.chunk_id,),
    )
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("is not authoritative" in error for error in result.errors)


def test_obsolete_gold_chunk_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    obsolete = next(chunk for chunk in chunks if chunk.is_obsolete)
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    example = examples[index]
    object.__setattr__(example, "required_record_ids", obsolete.record_ids[:1])
    object.__setattr__(example, "required_logical_fact_ids", obsolete.logical_fact_ids[:1])
    object.__setattr__(example, "gold_document_ids", (obsolete.document_id,))
    object.__setattr__(example, "gold_chunk_ids", (obsolete.chunk_id,))
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("is obsolete" in error for error in result.errors)


def test_gold_chunks_must_cover_required_records() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    example = examples[index]
    other = next(
        chunk
        for chunk in chunks
        if chunk.is_authoritative
        and not chunk.is_obsolete
        and set(example.required_record_ids).isdisjoint(set(chunk.record_ids))
    )
    examples[index] = replace(
        example,
        gold_document_ids=(other.document_id,),
        gold_chunk_ids=(other.chunk_id,),
    )
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("not covered by gold chunks" in error for error in result.errors)


def test_changed_knowledge_present_gold_is_current_v2_evidence() -> None:
    world, documents, chunks, examples = _fixture()
    documents_by_id = {document.document_id: document for document in documents}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    records_by_id = {fact.record_id: fact for fact in world.facts}

    changed = [
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.evidence_status is EvidenceStatus.PRESENT
    ]
    assert changed
    for example in changed:
        assert example.knowledge_version == "v2"
        assert all(records_by_id[record_id].version == "v2" for record_id in example.required_record_ids)
        assert all(documents_by_id[document_id].version == "v2" for document_id in example.gold_document_ids)
        assert all(chunks_by_id[chunk_id].version == "v2" for chunk_id in example.gold_chunk_ids)
        assert all(chunks_by_id[chunk_id].is_authoritative for chunk_id in example.gold_chunk_ids)
        assert all(not chunks_by_id[chunk_id].is_obsolete for chunk_id in example.gold_chunk_ids)


def test_changed_knowledge_rejects_obsolete_v1_gold_as_current_evidence() -> None:
    world, documents, chunks, examples = _fixture()
    updated_index = next(
        i
        for i, example in enumerate(examples)
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.UPDATED
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    example = examples[updated_index]
    logical_id = example.required_logical_fact_ids[0]
    obsolete = next(
        chunk
        for chunk in chunks
        if chunk.version == "v1"
        and chunk.is_obsolete
        and logical_id in chunk.logical_fact_ids
    )
    examples[updated_index] = replace(
        example,
        required_record_ids=obsolete.record_ids[:1],
        gold_document_ids=(obsolete.document_id,),
        gold_chunk_ids=(obsolete.chunk_id,),
    )

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("current changed_knowledge task" in error for error in result.errors)


def test_changed_knowledge_expected_output_must_be_supported_by_current_gold() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(
        i
        for i, example in enumerate(examples)
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.UPDATED
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    examples[index] = replace(examples[index], expected_output="WRONG_CURRENT_VALUE")

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("expected_output does not match the authoritative current value" in error for error in result.errors)


def test_required_record_and_logical_fact_identity_mismatch_is_reported() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    example = examples[index]
    wrong_logical_id = next(
        fact.logical_fact_id
        for fact in world.facts
        if fact.logical_fact_id not in example.required_logical_fact_ids
    )
    examples[index] = replace(example, required_logical_fact_ids=(wrong_logical_id,))

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("which is not listed in required_logical_fact_ids" in error for error in result.errors)
    assert any("has no matching required record" in error for error in result.errors)


def test_v2_current_task_rejects_v1_required_record_and_gold_evidence() -> None:
    world, documents, chunks, examples = _fixture()
    obsolete = next(chunk for chunk in chunks if chunk.version == "v1" and chunk.is_obsolete and chunk.record_ids)
    document = next(document for document in documents if document.document_id == obsolete.document_id)
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    example = examples[index]
    examples[index] = replace(
        example,
        knowledge_version="v2",
        required_record_ids=obsolete.record_ids[:1],
        required_logical_fact_ids=obsolete.logical_fact_ids[:1],
        gold_document_ids=(document.document_id,),
        gold_chunk_ids=(obsolete.chunk_id,),
    )

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("knowledge_version=v2 requires current v2 records" in error for error in result.errors)
    assert any("knowledge_version=v2 uses non-v2 gold document" in error for error in result.errors)
    assert any("knowledge_version=v2 uses non-v2 gold chunk" in error for error in result.errors)


def test_changed_knowledge_state_is_checked_against_world_truth() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(
        i
        for i, example in enumerate(examples)
        if example.task_family is TaskFamily.changed_knowledge
        and example.knowledge_state is KnowledgeState.UPDATED
    )
    object.__setattr__(examples[index], "knowledge_state", KnowledgeState.UNCHANGED)

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("does not match world lifecycle UPDATED" in error for error in result.errors)


def test_evidence_absent_changed_knowledge_lifecycle_identity_is_checked() -> None:
    world, documents, chunks, examples = _fixture()
    example = next(
        example
        for example in examples
        if example.task_family is TaskFamily.changed_knowledge
        and example.evidence_status is EvidenceStatus.ABSENT
    )
    assert example.lifecycle_logical_fact_id == "DEPLOY_CLASSIC_MODE"
    object.__setattr__(example, "lifecycle_logical_fact_id", "AUTH_TOKEN_TTL")

    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("does not match world lifecycle UPDATED" in error for error in result.errors)


def test_scoring_rule_rejects_incorrect_fact_value_output() -> None:
    world, documents, chunks, examples = _fixture()
    index = next(
        i for i, example in enumerate(examples)
        if example.task_family is TaskFamily.knowledge_only
        and example.evidence_status is EvidenceStatus.PRESENT
    )
    examples[index] = replace(examples[index], expected_output="DELIBERATELY_WRONG")
    result = validate_fixture(world, documents, chunks, examples)
    assert not result.passed
    assert any("does not match scoring_rule FACT_VALUE" in error for error in result.errors)


def test_generated_examples_have_typed_scoring_rules() -> None:
    from adaptlab.domain.enums import ScoringRule

    _, _, _, examples = _fixture()
    assert all(isinstance(example.scoring_rule, ScoringRule) for example in examples)
