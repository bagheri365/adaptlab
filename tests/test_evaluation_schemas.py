from __future__ import annotations

import json

import pytest

from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    ScoringRule,
    Split,
    SplitType,
    TaskFamily,
)
from adaptlab.evaluation.schemas import (
    EVALUATION_RESULT_SCHEMA_VERSION,
    EVALUATION_RUN_SCHEMA_VERSION,
    FROZEN_BENCHMARK_TAG,
    AdaptationMethod,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunStatus,
    ModelInput,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def make_run(**overrides) -> EvaluationRun:
    values = dict(
        run_id="run-001",
        benchmark_version="0.0.0",
        benchmark_manifest_hash=HASH_A,
        model_id="fake-model",
        model_tag="fake-model",
        model_digest=HASH_B,
        model_revision="fake-model@1",
        provider="fake",
        ollama_version="0.0.0",
        ollama_base_url_policy="http://localhost:11434",
        adaptation_method=AdaptationMethod.PROMPT,
        prompt_version="prompt-v1",
        prompt_hash=HASH_B,
        scorer_version="scorer-v1",
        normalizer_version="normalizer-v1",
        temperature=0.0,
        max_tokens=256,
        seed=1729,
        context_length=4096,
        stream=False,
        think=False,
        started_at="2026-08-14T20:00:00Z",
        completed_at="2026-08-14T20:01:00Z",
        status=EvaluationRunStatus.COMPLETED,
    )
    values.update(overrides)
    return EvaluationRun(**values)


def make_result(**overrides) -> EvaluationResult:
    values = dict(
        example_id="test-0001",
        split=Split.test,
        task_family=TaskFamily.behavior_knowledge,
        difficulty=Difficulty.MEDIUM,
        behavior_type=BehaviorType.CLASSIFICATION_POLICY,
        knowledge_state=KnowledgeState.UNCHANGED,
        evidence_status=EvidenceStatus.PRESENT,
        split_type=SplitType.iid,
        input_hash=HASH_A,
        model_input=ModelInput(system="Follow format.", user="Question?"),
        raw_output="ALLOW",
        normalized_output="ALLOW",
        expected_output="ALLOW",
        score=1.0,
        scoring_rule=ScoringRule.CLASSIFICATION,
        latency_ms=12.5,
        input_tokens=17,
        output_tokens=1,
        provider_error=None,
        retry_count=0,
    )
    values.update(overrides)
    return EvaluationResult(**values)


def test_adaptation_method_vocabulary_is_reserved_but_only_two_are_implemented() -> None:
    assert {member.value for member in AdaptationMethod} == {
        "PROMPT", "ORACLE_CONTEXT", "RAG", "LORA", "LORA_RAG"
    }
    assert make_run(adaptation_method=AdaptationMethod.ORACLE_CONTEXT).adaptation_method is AdaptationMethod.ORACLE_CONTEXT
    with pytest.raises(ValueError, match="only implements PROMPT and ORACLE_CONTEXT"):
        make_run(adaptation_method=AdaptationMethod.RAG)


def test_run_schema_records_frozen_benchmark_identity_and_explicit_versions() -> None:
    run = make_run()
    payload = run.to_dict()
    assert payload["benchmark_version"] == "0.0.0"
    assert payload["benchmark_tag"] == FROZEN_BENCHMARK_TAG == "v0.0-benchmark"
    assert payload["benchmark_manifest_hash"] == HASH_A
    assert payload["model_tag"] == "fake-model"
    assert payload["model_digest"] == HASH_B
    assert payload["ollama_version"] == "0.0.0"
    assert payload["ollama_base_url_policy"] == "http://localhost:11434"
    assert payload["prompt_version"] == "prompt-v1"
    assert payload["prompt_hash"] == HASH_B
    assert payload["scorer_version"] == "scorer-v1"
    assert payload["normalizer_version"] == "normalizer-v1"
    assert payload["context_length"] == 4096
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["schema_version"] == EVALUATION_RUN_SCHEMA_VERSION


def test_run_rejects_non_frozen_benchmark_tag_and_bad_hashes() -> None:
    with pytest.raises(ValueError, match="benchmark_tag"):
        make_run(benchmark_tag="candidate")
    with pytest.raises(ValueError, match="benchmark_manifest_hash"):
        make_run(benchmark_manifest_hash="not-a-hash")
    with pytest.raises(ValueError, match="prompt_hash"):
        make_run(prompt_hash="not-a-hash")


def test_run_serialization_is_deterministic_and_round_trips() -> None:
    run = make_run()
    first = run.to_json_bytes()
    second = run.to_json_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert EvaluationRun.from_dict(json.loads(first)) == run


def test_run_from_legacy_manifest_defaults_missing_runtime_fields() -> None:
    legacy = make_run().to_dict()
    for key in [
        "model_tag",
        "model_digest",
        "ollama_version",
        "ollama_base_url_policy",
        "context_length",
        "stream",
        "think",
    ]:
        legacy.pop(key)
    restored = EvaluationRun.from_dict(legacy)
    assert restored.model_tag is None
    assert restored.model_digest is None
    assert restored.ollama_version is None
    assert restored.ollama_base_url_policy is None
    assert restored.context_length is None
    assert restored.stream is None
    assert restored.think is None


def test_result_contains_required_example_and_provider_fields() -> None:
    result = make_result()
    payload = result.to_dict()
    expected_keys = {
        "example_id", "split", "task_family", "difficulty", "behavior_type",
        "knowledge_state", "evidence_status", "split_type", "input_hash",
        "model_input", "raw_output", "normalized_output", "expected_output",
        "score", "scoring_rule", "latency_ms", "input_tokens", "output_tokens",
        "provider_error", "retry_count", "schema_version",
    }
    assert set(payload) == expected_keys
    assert payload["model_input"] == {"system": "Follow format.", "user": "Question?"}
    assert payload["schema_version"] == EVALUATION_RESULT_SCHEMA_VERSION


def test_result_serialization_is_deterministic_and_round_trips_nested_json() -> None:
    result = make_result(
        normalized_output={"b": 2, "a": [1, 2]},
        expected_output={"a": [1, 2], "b": 2},
    )
    first = result.to_json_bytes()
    second = result.to_json_bytes()
    assert first == second
    assert first.index(b'"a"') < first.index(b'"b"')
    assert EvaluationResult.from_dict(json.loads(first)) == result


def test_result_allows_explicit_provider_failure_without_fabricating_output_or_score() -> None:
    result = make_result(
        raw_output=None,
        normalized_output=None,
        score=None,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        provider_error="PERMANENT_PROVIDER_ERROR",
        retry_count=2,
    )
    assert result.provider_error == "PERMANENT_PROVIDER_ERROR"
    assert result.score is None


def test_result_rejects_invalid_hash_token_counts_and_retry_count() -> None:
    with pytest.raises(ValueError, match="input_hash"):
        make_result(input_hash="abc")
    with pytest.raises(ValueError, match="input_tokens"):
        make_result(input_tokens=-1)
    with pytest.raises(ValueError, match="retry_count"):
        make_result(retry_count=-1)
