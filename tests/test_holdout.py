from __future__ import annotations

from dataclasses import replace

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import (
    COMPONENT_FAMILY_DIMENSION,
    ERROR_FAMILY_DIMENSION,
    apply_full_holdout_policy,
    build_full_holdout_policy,
    error_family_for_fact,
    render_holdout_report,
    validate_full_holdout_examples,
)
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    ScoringRule,
    Split,
    SplitType,
    TaskFamily,
)


def _config_world_policy():
    config = load_benchmark_config("configs/benchmark_v0.0.yaml")
    world = generate_full_world(config)
    return config, world, build_full_holdout_policy(config, world)


def _example_for_record(record, *, example_id: str = "FULL_HOLDOUT_TEST") -> BenchmarkExample:
    return BenchmarkExample(
        example_id=example_id,
        benchmark_version="0.0.0",
        task_family=TaskFamily.knowledge_only,
        behavior_type=None,
        difficulty=Difficulty.EASY,
        split=Split.train,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version=record.version,
        knowledge_state=KnowledgeState.NOT_APPLICABLE,
        evidence_status=EvidenceStatus.ABSENT,
        question="What is the current fictional Nimbus setting?",
        expected_output="INSUFFICIENT_EVIDENCE",
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
        generation_seed=1729,
        scoring_rule=ScoringRule.ABSTENTION,
        lifecycle_logical_fact_id=None,
    )


def _present_example_for_record(record, *, example_id: str) -> BenchmarkExample:
    # Gold document/chunk references are intentionally omitted here because the
    # holdout-policy validator only needs truth references and split metadata.
    example = _example_for_record(record, example_id=example_id)
    object.__setattr__(example, "evidence_status", EvidenceStatus.PRESENT)
    object.__setattr__(example, "required_record_ids", (record.record_id,))
    object.__setattr__(example, "required_logical_fact_ids", (record.logical_fact_id,))
    object.__setattr__(example, "gold_document_ids", ("placeholder-doc",))
    object.__setattr__(example, "gold_chunk_ids", ("placeholder-chunk",))
    object.__setattr__(example, "expected_output", record.value)
    object.__setattr__(example, "scoring_rule", ScoringRule.FACT_VALUE)
    return example


def test_full_holdout_policy_is_deterministic_and_has_two_dimensions() -> None:
    config, world, first = _config_world_policy()
    second = build_full_holdout_policy(config, world)

    assert first.to_dict() == second.to_dict()
    assert first.generation_seed == config.generation_seed
    assert first.component_family.dimension == COMPONENT_FAMILY_DIMENSION
    assert first.error_family.dimension == ERROR_FAMILY_DIMENSION
    assert len(first.component_family.structural_test_groups) >= 1
    assert len(first.error_family.structural_test_groups) >= 1


def test_holdout_dimension_groups_are_disjoint() -> None:
    _, _, policy = _config_world_policy()
    for dimension in (policy.component_family, policy.error_family):
        buckets = [
            set(dimension.train_groups),
            set(dimension.validation_groups),
            set(dimension.iid_test_groups),
            set(dimension.structural_test_groups),
        ]
        combined = set().union(*buckets)
        assert sum(len(bucket) for bucket in buckets) == len(combined)


def test_holdout_report_lists_all_required_group_categories() -> None:
    _, _, policy = _config_world_policy()
    report = render_holdout_report(policy)

    assert "[component_family]" in report
    assert "[error_family]" in report
    assert "train groups:" in report
    assert "validation groups:" in report
    assert "IID test groups:" in report
    assert "structural test groups:" in report


def test_component_structural_group_is_forced_to_structural_test() -> None:
    _, world, policy = _config_world_policy()
    group = policy.component_family.structural_test_groups[0]
    record = next(
        fact for fact in world.facts if fact.component_family == group and fact.version == "v2"
    )
    example = _present_example_for_record(record, example_id="COMP_STRUCT")

    assigned = apply_full_holdout_policy(world, [example], policy)[0]
    assert assigned.split is Split.test
    assert assigned.split_type is SplitType.structural_holdout
    assert assigned.holdout_dimension == COMPONENT_FAMILY_DIMENSION
    assert assigned.holdout_group == group


def test_error_family_structural_group_is_forced_to_structural_test() -> None:
    _, world, policy = _config_world_policy()
    error_group = policy.error_family.structural_test_groups[0]
    record = next(
        fact
        for fact in world.facts
        if fact.version == "v2" and error_family_for_fact(fact) == error_group
    )
    example = _present_example_for_record(record, example_id="ERROR_STRUCT")

    assigned = apply_full_holdout_policy(world, [example], policy)[0]
    assert assigned.split is Split.test
    assert assigned.split_type is SplitType.structural_holdout
    assert assigned.holdout_group in {
        error_group,
        record.component_family,
    }


def test_validator_rejects_structural_component_in_training() -> None:
    _, world, policy = _config_world_policy()
    group = policy.component_family.structural_test_groups[0]
    record = next(
        fact for fact in world.facts if fact.component_family == group and fact.version == "v2"
    )
    leaked = _present_example_for_record(record, example_id="LEAK_TRAIN")

    result = validate_full_holdout_examples(world, [leaked], policy)
    assert not result.passed
    assert any("into train" in error for error in result.errors)


def test_validator_rejects_structural_group_in_validation() -> None:
    _, world, policy = _config_world_policy()
    group = policy.component_family.structural_test_groups[0]
    record = next(
        fact for fact in world.facts if fact.component_family == group and fact.version == "v2"
    )
    leaked = replace(
        _present_example_for_record(record, example_id="LEAK_VALIDATION"),
        split=Split.validation,
    )

    result = validate_full_holdout_examples(world, [leaked], policy)
    assert not result.passed
    assert any("into validation" in error for error in result.errors)


def test_world_seed_must_match_config_before_freezing_holdouts() -> None:
    config = load_benchmark_config("configs/benchmark_v0.0.yaml")
    world = generate_full_world(config)
    object.__setattr__(world, "generation_seed", config.generation_seed + 1)

    try:
        build_full_holdout_policy(config, world)
    except ValueError as exc:
        assert "seed must match" in str(exc)
    else:
        raise AssertionError("expected seed mismatch rejection")
