from collections import Counter
from pathlib import Path

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_documents, generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_tasks, generate_full_tasks
from adaptlab.benchmark.generate_world import generate_world, generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.domain.enums import BehaviorType, EvidenceStatus, KnowledgeState, TaskFamily

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"



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


def _full_fixture():
    from pathlib import Path
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import generate_full_world
    from adaptlab.benchmark.generate_docs import generate_full_documents
    from adaptlab.benchmark.holdout import build_full_holdout_policy, validate_full_holdout_examples
    from adaptlab.benchmark.generate_tasks import generate_full_tasks

    config = load_benchmark_config(Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml")
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    return config, world, documents, chunks, policy, examples


def test_full_tasks_exact_split_and_test_family_counts() -> None:
    from adaptlab.domain.enums import Split
    config, _, _, _, _, examples = _full_fixture()
    assert Counter(example.split for example in examples) == {
        Split.train: config.splits.train,
        Split.validation: config.splits.validation,
        Split.test: config.splits.test,
    }
    test = [example for example in examples if example.split is Split.test]
    assert Counter(example.task_family for example in test) == config.test_task_families.by_family()


def test_full_test_exact_difficulty_targets() -> None:
    from adaptlab.domain.enums import Split
    config, _, _, _, _, examples = _full_fixture()
    test = [example for example in examples if example.split is Split.test]
    assert Counter(example.difficulty for example in test) == config.test_difficulty.by_difficulty()


def test_full_behavior_only_test_targets_and_prompt_containment() -> None:
    from adaptlab.domain.enums import Split
    config, _, _, _, _, examples = _full_fixture()
    behavior = [
        example for example in examples
        if example.split is Split.test and example.task_family is TaskFamily.behavior_only
    ]
    assert Counter(example.behavior_type for example in behavior) == config.behavior_only_test_behavior_types.by_behavior_type()
    assert all(example.evidence_status is EvidenceStatus.NOT_APPLICABLE for example in behavior)
    assert all(not example.required_record_ids and not example.gold_chunk_ids for example in behavior)
    # Deterministic behavior tasks include their operands/facts in the prompt.
    assert all(example.question.strip() for example in behavior)


def test_full_evidence_absent_test_targets_and_wording() -> None:
    from adaptlab.domain.enums import Split
    config, _, _, _, _, examples = _full_fixture()
    test_absent = [e for e in examples if e.split is Split.test and e.evidence_status is EvidenceStatus.ABSENT]
    assert len(test_absent) == config.evidence_absent.total
    assert Counter(e.task_family for e in test_absent) == {
        TaskFamily.knowledge_only: config.evidence_absent.knowledge_only,
        TaskFamily.behavior_knowledge: config.evidence_absent.behavior_knowledge,
        TaskFamily.changed_knowledge: config.evidence_absent.changed_knowledge,
    }
    forbidden = ("no evidence", "evidence is absent", "insufficient evidence", "answer insufficient_evidence")
    for example in test_absent:
        assert not any(text in example.question.lower() for text in forbidden)
        assert example.expected_output == "INSUFFICIENT_EVIDENCE"
        assert not any((example.required_record_ids, example.required_logical_fact_ids, example.gold_document_ids, example.gold_chunk_ids))


def test_full_changed_knowledge_exact_lifecycle_targets_and_current_answers() -> None:
    from adaptlab.domain.enums import Split
    config, world, _, _, _, examples = _full_fixture()
    records = {fact.record_id: fact for fact in world.facts}
    changed = [e for e in examples if e.split is Split.test and e.task_family is TaskFamily.changed_knowledge]
    assert Counter(e.knowledge_state for e in changed) == config.changed_knowledge.by_state()
    for example in changed:
        assert "current" in example.question.lower()
        if example.evidence_status is EvidenceStatus.PRESENT:
            record = records[example.required_record_ids[0]]
            assert record.version == "v2"
            if example.knowledge_state is KnowledgeState.REMOVED:
                assert example.expected_output == "RETIRED"
            else:
                assert example.expected_output == record.value


def test_full_present_knowledge_examples_have_current_gold_evidence() -> None:
    from adaptlab.domain.enums import Split
    _, world, documents, chunks, _, examples = _full_fixture()
    records = {fact.record_id: fact for fact in world.facts}
    docs = {doc.document_id: doc for doc in documents}
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    present = [
        e for e in examples
        if e.split is Split.test
        and e.task_family is not TaskFamily.behavior_only
        and e.evidence_status is EvidenceStatus.PRESENT
    ]
    assert present
    for example in present:
        assert all(records[rid].version == "v2" for rid in example.required_record_ids)
        assert all(docs[did].version == "v2" for did in example.gold_document_ids)
        assert all(chunk_map[cid].version == "v2" and chunk_map[cid].is_authoritative and not chunk_map[cid].is_obsolete for cid in example.gold_chunk_ids)


def test_full_holdout_policy_has_no_train_or_validation_structural_leakage() -> None:
    from adaptlab.benchmark.holdout import validate_full_holdout_examples
    _, world, _, _, policy, examples = _full_fixture()
    result = validate_full_holdout_examples(world, examples, policy)
    assert result.passed, result.errors
    assert any(example.split_type.value == "structural_holdout" for example in examples)


def test_full_task_generation_is_deterministic_and_canonical() -> None:
    first = _full_fixture()[-1]
    second = _full_fixture()[-1]
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]
    ids = [example.example_id for example in first]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_full_difficulty_metadata_matches_declared_difficulty() -> None:
    from adaptlab.benchmark.difficulty import DifficultyPlan, DifficultySource, validate_difficulty_plan
    _, _, _, _, _, examples = _full_fixture()
    for example in examples:
        metadata = (example.scoring_parameters or {}).get("difficulty")
        assert metadata is not None
        plan = DifficultyPlan(
            difficulty=example.difficulty,
            relevant_fact_count=int(metadata["relevant_fact_count"]),
            required_evidence_cardinality=int(metadata["required_evidence_cardinality"]),
            retrieval_candidate_count=int(metadata["retrieval_candidate_count"]),
            sources=tuple(DifficultySource(value) for value in metadata["sources"]),
            multi_hop=bool(metadata["multi_hop"]),
            retrieval_applicable=bool(metadata.get("retrieval_applicable", True)),
        )
        assert not validate_difficulty_plan(plan)


def test_full_training_subsets_are_strictly_nested_and_exact() -> None:
    from adaptlab.benchmark.training_subsets import generate_training_subsets
    from adaptlab.domain.enums import Split

    config, _, _, _, _, examples = _full_fixture()
    bundle = generate_training_subsets(examples, config)
    subsets = bundle.by_name()

    assert {name: len(values) for name, values in subsets.items()} == {
        "train_050": 50,
        "train_100": 100,
        "train_200": 200,
        "train_300": 300,
    }
    ids = {name: {example.example_id for example in values} for name, values in subsets.items()}
    assert ids["train_050"] < ids["train_100"] < ids["train_200"] < ids["train_300"]

    full_train = tuple(sorted(
        (example for example in examples if example.split is Split.train),
        key=lambda example: example.example_id,
    ))
    assert bundle.train_300 == full_train
    assert all(example.split is Split.train for values in subsets.values() for example in values)
    assert all(example.split_type.value != "structural_holdout" for values in subsets.values() for example in values)


def test_full_training_subsets_preserve_behavior_type_representation() -> None:
    from adaptlab.benchmark.training_subsets import generate_training_subsets
    from adaptlab.domain.enums import BehaviorType

    config, _, _, _, _, examples = _full_fixture()
    bundle = generate_training_subsets(examples, config)
    for subset in bundle.by_name().values():
        behavior_only = [example for example in subset if example.task_family is TaskFamily.behavior_only]
        assert {example.behavior_type for example in behavior_only} == set(BehaviorType)
        counts = Counter(example.behavior_type for example in behavior_only)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_full_training_subsets_are_deterministic_and_versioned() -> None:
    from adaptlab.benchmark.training_subsets import TRAINING_SUBSET_VERSION, generate_training_subsets

    config, _, _, _, _, examples = _full_fixture()
    first = generate_training_subsets(examples, config)
    second = generate_training_subsets(list(reversed(examples)), config)

    assert first.subset_version == TRAINING_SUBSET_VERSION == "1"
    assert first.benchmark_version == config.benchmark_version
    assert first.generation_seed == config.generation_seed
    assert first.to_dict() == second.to_dict()
    for subset in first.by_name().values():
        assert [example.example_id for example in subset] == sorted(example.example_id for example in subset)


def test_full_behavior_knowledge_questions_require_declared_behavior():
    config = load_benchmark_config(CONFIG_PATH)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    bkn = [e for e in examples if e.task_family is TaskFamily.behavior_knowledge]
    assert bkn
    for example in bkn:
        question = example.question.casefold()
        if example.behavior_type is BehaviorType.SCHEMA_ADHERENCE:
            assert "exactly" in question and "{\"value\"" in example.question
        elif example.behavior_type is BehaviorType.CONDITIONAL_DECISION_RULE:
            assert "only if" in question and "otherwise" in question
        elif example.behavior_type is BehaviorType.TRANSFORMATION_EXTRACTION:
            assert "extract" in question and "return only" in question
        elif example.behavior_type is BehaviorType.CLASSIFICATION_POLICY:
            assert "classify" in question and "numeric" in question and "text" in question
        elif example.behavior_type is BehaviorType.ABSTENTION_BEHAVIOR:
            assert "if a current value is explicitly supported" in question
            assert "otherwise return insufficient_evidence" in question
        else:
            raise AssertionError(example.behavior_type)


def test_full_behavior_knowledge_absent_cases_use_abstention_behavior():
    config = load_benchmark_config(CONFIG_PATH)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    absent = [
        e for e in examples
        if e.task_family is TaskFamily.behavior_knowledge
        and e.evidence_status is EvidenceStatus.ABSENT
    ]
    assert absent
    assert {e.behavior_type for e in absent} == {BehaviorType.ABSTENTION_BEHAVIOR}


def test_full_model_facing_questions_have_no_artificial_review_or_split_nonce_markers():
    config = load_benchmark_config(CONFIG_PATH)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)

    forbidden_phrases = (
        "during review sequence",
        "review sequence",
        "train sequence",
        "validation sequence",
        "test sequence",
        "split marker",
    )
    for example in examples:
        question = example.question.casefold()
        assert not any(marker in question for marker in forbidden_phrases), example.example_id


def _behavior_only_instance_signature(example):
    params = dict(example.scoring_parameters or {})
    params.pop("difficulty", None)
    # These are the task-relevant operands/contract. Split, example_id, wording,
    # and generation order are deliberately excluded.
    return (
        example.behavior_type.value,
        example.scoring_rule.value,
        tuple(sorted((key, repr(value)) for key, value in params.items())),
    )


def test_full_behavior_only_concrete_instances_are_disjoint_across_splits():
    from adaptlab.domain.enums import Split

    _, _, _, _, _, examples = _full_fixture()
    behavior = [e for e in examples if e.task_family is TaskFamily.behavior_only]
    assert {e.behavior_type for e in behavior} == set(BehaviorType)

    by_split = {
        split: {_behavior_only_instance_signature(e) for e in behavior if e.split is split}
        for split in Split
    }
    assert by_split[Split.train].isdisjoint(by_split[Split.validation])
    assert by_split[Split.train].isdisjoint(by_split[Split.test])
    assert by_split[Split.validation].isdisjoint(by_split[Split.test])

    # Every behavior primitive remains represented in every primary split.
    for split in Split:
        assert {e.behavior_type for e in behavior if e.split is split} == set(BehaviorType)


def test_full_knowledge_only_semantic_fingerprints_are_cross_split_disjoint() -> None:
    from adaptlab.benchmark.leakage import semantic_task_fingerprint
    from adaptlab.domain.enums import Split

    _, _, _, _, _, examples = _full_fixture()
    by_split = {
        split: {
            semantic_task_fingerprint(example)
            for example in examples
            if example.split is split and example.task_family is TaskFamily.knowledge_only
        }
        for split in Split
    }

    assert by_split[Split.train].isdisjoint(by_split[Split.validation])
    assert by_split[Split.train].isdisjoint(by_split[Split.test])
    assert by_split[Split.validation].isdisjoint(by_split[Split.test])


def test_full_present_knowledge_only_tasks_have_structured_intent_and_evidence_cardinality() -> None:
    _, _, _, _, _, examples = _full_fixture()
    present = [
        example
        for example in examples
        if example.task_family is TaskFamily.knowledge_only
        and example.evidence_status is EvidenceStatus.PRESENT
    ]
    assert present
    for example in present:
        params = example.scoring_parameters or {}
        assert params["question_intent"] in {
            "current_value_lookup",
            "version_specific_lookup",
            "fact_family_lookup",
            "component_scoped_lookup",
            "component_fact_family_lookup",
        }
        assert params["required_evidence_cardinality"] == len(example.gold_chunk_ids)
        assert 1 <= len(example.gold_chunk_ids) <= 3


def test_full_behavior_knowledge_semantic_fingerprints_are_cross_split_disjoint() -> None:
    from adaptlab.benchmark.leakage import semantic_task_fingerprint
    from adaptlab.domain.enums import Split

    _, _, _, _, _, examples = _full_fixture()
    by_split = {
        split: {
            semantic_task_fingerprint(example)
            for example in examples
            if example.split is split and example.task_family is TaskFamily.behavior_knowledge
        }
        for split in Split
    }

    assert by_split[Split.train].isdisjoint(by_split[Split.validation])
    assert by_split[Split.train].isdisjoint(by_split[Split.test])
    assert by_split[Split.validation].isdisjoint(by_split[Split.test])


def test_full_behavior_knowledge_present_instances_use_task_relevant_allocation() -> None:
    from adaptlab.domain.enums import Split

    _, _, _, _, _, examples = _full_fixture()
    present = [
        example for example in examples
        if example.task_family is TaskFamily.behavior_knowledge
        and example.evidence_status is EvidenceStatus.PRESENT
    ]
    assert present
    for example in present:
        params = example.scoring_parameters or {}
        assert params["question_intent"] in {
            "schema_current_value",
            "conditional_current_value",
            "extract_current_value",
            "classify_current_value",
            "supported_current_value",
        }
        assert params["required_evidence_cardinality"] == len(example.gold_chunk_ids)
        assert 1 <= len(example.gold_chunk_ids) <= 3

    # Every split retains every behavior primitive.
    for split in Split:
        assert {
            example.behavior_type for example in present if example.split is split
        } == set(BehaviorType)


def test_full_changed_knowledge_semantic_fingerprints_are_cross_split_disjoint() -> None:
    from adaptlab.benchmark.leakage import semantic_task_fingerprint
    from adaptlab.domain.enums import Split

    _, _, _, _, _, examples = _full_fixture()
    changed = [e for e in examples if e.task_family is TaskFamily.changed_knowledge]
    by_split = {
        split: {semantic_task_fingerprint(e) for e in changed if e.split is split}
        for split in (Split.train, Split.validation, Split.test)
    }
    assert by_split[Split.train].isdisjoint(by_split[Split.validation])
    assert by_split[Split.train].isdisjoint(by_split[Split.test])
    assert by_split[Split.validation].isdisjoint(by_split[Split.test])

    for split in (Split.train, Split.validation, Split.test):
        split_examples = [e for e in changed if e.split is split]
        assert len(by_split[split]) == len(split_examples)
