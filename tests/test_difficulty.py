from __future__ import annotations

from adaptlab.benchmark.difficulty import (
    DifficultyPlan,
    DifficultySource,
    build_difficulty_plan,
    difficulty_metadata,
    validate_difficulty_plan,
)
from adaptlab.domain.enums import Difficulty


def test_easy_plan_has_one_fact_one_chunk_and_direct_wording() -> None:
    plan = build_difficulty_plan(Difficulty.EASY)
    assert plan.relevant_fact_count == 1
    assert plan.required_evidence_cardinality == 1
    assert plan.retrieval_candidate_count == 1
    assert plan.sources == (DifficultySource.DIRECT_WORDING,)
    assert plan.multi_hop is False
    assert validate_difficulty_plan(plan) == ()


def test_medium_plans_have_controlled_retrieval_ambiguity() -> None:
    for variant in range(4):
        plan = build_difficulty_plan(Difficulty.MEDIUM, variant)
        assert plan.relevant_fact_count in (1, 2)
        assert plan.retrieval_candidate_count >= 2
        assert DifficultySource.PARAPHRASED_WORDING in plan.sources
        assert validate_difficulty_plan(plan) == ()


def test_hard_plans_each_have_a_controlled_source() -> None:
    observed = set()
    for variant in range(8):
        plan = build_difficulty_plan(Difficulty.HARD, variant)
        assert validate_difficulty_plan(plan) == ()
        observed.update(plan.sources)
    assert DifficultySource.MULTI_CHUNK_EVIDENCE in observed
    assert DifficultySource.OBSOLETE_CONFLICT in observed
    assert DifficultySource.NEAR_DUPLICATE_DISTRACTORS in observed
    assert DifficultySource.INDIRECT_SEMANTIC_WORDING in observed
    assert DifficultySource.SIMILAR_IDENTIFIERS in observed
    assert DifficultySource.VERSION_DISCRIMINATION in observed
    assert DifficultySource.DETERMINISTIC_INFERENCE in observed


def test_generation_is_deterministic_and_not_random_labeling() -> None:
    for difficulty in Difficulty:
        assert build_difficulty_plan(difficulty, 17) == build_difficulty_plan(difficulty, 17)
        assert difficulty_metadata(difficulty, 17) == difficulty_metadata(difficulty, 17)


def test_invalid_easy_construction_is_rejected() -> None:
    bad = DifficultyPlan(
        difficulty=Difficulty.EASY,
        relevant_fact_count=2,
        required_evidence_cardinality=2,
        retrieval_candidate_count=2,
        sources=(DifficultySource.PARAPHRASED_WORDING,),
        multi_hop=True,
    )
    errors = validate_difficulty_plan(bad)
    assert "EASY requires exactly one relevant fact" in errors
    assert "EASY requires exactly one clear evidence chunk" in errors
    assert "EASY requires direct wording" in errors
    assert "EASY must not require multi-hop reasoning" in errors


def test_invalid_hard_without_controlled_source_is_rejected() -> None:
    bad = DifficultyPlan(
        difficulty=Difficulty.HARD,
        relevant_fact_count=1,
        required_evidence_cardinality=1,
        retrieval_candidate_count=1,
        sources=(DifficultySource.DIRECT_WORDING,),
    )
    assert "HARD requires at least one controlled source of difficulty" in validate_difficulty_plan(bad)


def test_medium_plan_uses_one_or_two_required_gold_chunks() -> None:
    for variant in range(4):
        plan = build_difficulty_plan(Difficulty.MEDIUM, variant)
        assert plan.required_evidence_cardinality in (1, 2)
        assert plan.retrieval_candidate_count >= plan.required_evidence_cardinality
        assert validate_difficulty_plan(plan) == ()


def test_hard_multi_chunk_case_uses_distinct_cardinality_and_candidate_fields() -> None:
    plan = build_difficulty_plan(Difficulty.HARD, 0)
    assert plan.required_evidence_cardinality == 2
    assert plan.retrieval_candidate_count == 4
    assert DifficultySource.MULTI_CHUNK_EVIDENCE in plan.sources
    assert validate_difficulty_plan(plan) == ()
