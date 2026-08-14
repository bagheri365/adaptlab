from dataclasses import replace
from pathlib import Path

import pytest

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_full_tasks
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.benchmark.validate import validate_answer_integrity
from adaptlab.domain.enums import EvidenceStatus, ScoringRule


def _full_fixture():
    config = load_benchmark_config(Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml")
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    return world, documents, chunks, examples


def test_full_benchmark_answer_integrity_passes() -> None:
    world, documents, chunks, examples = _full_fixture()
    result = validate_answer_integrity(world, documents, chunks, examples)
    assert result.passed, result.errors[:10]
    assert result.statistics["scoring_rule_count"] == len(ScoringRule)


@pytest.mark.parametrize("rule", list(ScoringRule))
def test_corrupted_expected_output_fails_for_every_scoring_rule(rule: ScoringRule) -> None:
    world, documents, chunks, examples = _full_fixture()
    index = next(i for i, example in enumerate(examples) if example.scoring_rule is rule)
    corrupted = list(examples)
    corrupted[index] = replace(corrupted[index], expected_output={"corrupted": rule.value})

    result = validate_answer_integrity(world, documents, chunks, corrupted)
    assert not result.passed
    assert any(
        corrupted[index].example_id in error and f"scoring_rule {rule.value}" in error
        for error in result.errors
    )


def test_present_evidence_must_cover_required_truth() -> None:
    world, documents, chunks, examples = _full_fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    current = examples[index]
    unrelated = next(
        chunk for chunk in chunks
        if chunk.is_authoritative and not chunk.is_obsolete
        and set(chunk.record_ids).isdisjoint(current.required_record_ids)
    )
    corrupted = list(examples)
    corrupted[index] = replace(
        current,
        gold_document_ids=(unrelated.document_id,),
        gold_chunk_ids=(unrelated.chunk_id,),
    )
    result = validate_answer_integrity(world, documents, chunks, corrupted)
    assert not result.passed
    assert any("do not cover required records" in error for error in result.errors)


def test_v2_task_rejects_obsolete_v1_evidence() -> None:
    world, documents, chunks, examples = _full_fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    obsolete = next(chunk for chunk in chunks if chunk.version == "v1" and chunk.is_obsolete and chunk.record_ids)
    corrupted = list(examples)
    corrupted[index] = replace(
        examples[index],
        required_record_ids=(obsolete.record_ids[0],),
        required_logical_fact_ids=(obsolete.logical_fact_ids[0],),
        gold_document_ids=(obsolete.document_id,),
        gold_chunk_ids=(obsolete.chunk_id,),
        knowledge_version="v2",
    )
    result = validate_answer_integrity(world, documents, chunks, corrupted)
    assert not result.passed
    assert any("non-v2" in error or "not current authoritative evidence" in error for error in result.errors)


def test_present_evidence_cardinality_metadata_matches_gold_chunks() -> None:
    world, documents, chunks, examples = _full_fixture()
    for example in examples:
        if example.evidence_status is not EvidenceStatus.PRESENT:
            continue
        params = example.scoring_parameters or {}
        difficulty = params.get("difficulty") or {}
        assert params["required_evidence_cardinality"] == len(example.gold_chunk_ids)
        assert difficulty["required_evidence_cardinality"] == len(example.gold_chunk_ids)
        assert difficulty["retrieval_candidate_count"] >= len(example.gold_chunk_ids)

    result = validate_answer_integrity(world, documents, chunks, examples)
    assert result.passed, result.errors[:10]


def test_invalid_required_evidence_cardinality_metadata_fails_validation() -> None:
    world, documents, chunks, examples = _full_fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    current = examples[index]
    params = dict(current.scoring_parameters or {})
    params["required_evidence_cardinality"] = len(current.gold_chunk_ids) + 1
    corrupted = list(examples)
    corrupted[index] = replace(current, scoring_parameters=params)

    result = validate_answer_integrity(world, documents, chunks, corrupted)
    assert not result.passed
    assert any("required_evidence_cardinality" in error for error in result.errors)


def test_invalid_difficulty_cardinality_metadata_fails_validation() -> None:
    world, documents, chunks, examples = _full_fixture()
    index = next(i for i, example in enumerate(examples) if example.evidence_status is EvidenceStatus.PRESENT)
    current = examples[index]
    params = dict(current.scoring_parameters or {})
    difficulty = dict(params["difficulty"])
    difficulty["required_evidence_cardinality"] = len(current.gold_chunk_ids) + 1
    params["difficulty"] = difficulty
    corrupted = list(examples)
    corrupted[index] = replace(current, scoring_parameters=params)

    result = validate_answer_integrity(world, documents, chunks, corrupted)
    assert not result.passed
    assert any("difficulty required_evidence_cardinality" in error for error in result.errors)
