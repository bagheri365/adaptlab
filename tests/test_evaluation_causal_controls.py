import json
from copy import deepcopy
from pathlib import Path

import pytest

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.evaluation.causal_controls import (
    CausalControlValidationError,
    derive_rag_control_condition,
    load_condition_config,
    require_causal_controls,
    validate_causal_controls,
)
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.retrieval.frozen_artifact import load_and_verify_frozen_retrieval_artifact

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/generated/v0.0"
PROMPT_CFG = ROOT / "configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml"
ORACLE_CFG = ROOT / "configs/evaluation_conditions/milestone3_ollama_oracle_context_v1.yaml"
PROMPT_PATH = ROOT / "configs/prompts/prompt_v1.yaml"
FROZEN = ROOT / "artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json"


def _examples():
    return [BenchmarkExample.from_dict(x) for x in json.loads((BENCHMARK / "test.json").read_text())]


def _chunks():
    return [DocumentChunk.from_dict(x) for x in json.loads((BENCHMARK / "chunks.json").read_text())]


def _report():
    prompt = load_condition_config(PROMPT_CFG)
    oracle = load_condition_config(ORACLE_CFG)
    return validate_causal_controls(
        examples=_examples(),
        chunks=_chunks(),
        prompt_contract=load_prompt_contract(PROMPT_PATH),
        retrieval_artifact=load_and_verify_frozen_retrieval_artifact(FROZEN),
        prompt_condition=prompt,
        oracle_condition=oracle,
        rag_condition=derive_rag_control_condition(prompt),
    )


def test_canonical_controls_pass_for_all_400_examples():
    report = _report()
    assert report.passed
    assert report.example_count == 400
    assert all(report.condition_checks.values())
    assert sum(row["task_family"] == "behavior_only" for row in report.per_example_checks) == 100
    assert all(row["passed"] for row in report.per_example_checks)


def test_behavior_only_is_byte_identical_and_rag_matches_frozen_artifact():
    report = _report()
    behavior = [row for row in report.per_example_checks if row["task_family"] == "behavior_only"]
    assert behavior and all(row["behavior_only_byte_identical"] for row in behavior)
    assert all(row["rag_chunks_match_frozen_artifact"] for row in report.per_example_checks)


def test_evidence_absent_does_not_require_rag_prompt_equality():
    report = _report()
    absent = [row for row in report.per_example_checks if row["evidence_status"] == "ABSENT"]
    assert absent
    assert all(row["oracle_rag_shared_renderer"] for row in absent)


def test_runtime_drift_fails_and_blocks_analysis():
    prompt = load_condition_config(PROMPT_CFG)
    oracle = load_condition_config(ORACLE_CFG)
    rag = derive_rag_control_condition(prompt)
    rag["request"]["temperature"] = 0.1
    report = validate_causal_controls(
        examples=_examples()[:1], chunks=_chunks(),
        prompt_contract=load_prompt_contract(PROMPT_PATH),
        retrieval_artifact=load_and_verify_frozen_retrieval_artifact(FROZEN),
        prompt_condition=prompt, oracle_condition=oracle, rag_condition=rag,
    )
    assert not report.passed
    assert not report.condition_checks["temperature"]
    with pytest.raises(CausalControlValidationError, match="stop before performance analysis"):
        require_causal_controls(report)


def test_prompt_hash_drift_is_detected():
    prompt = load_condition_config(PROMPT_CFG)
    oracle = load_condition_config(ORACLE_CFG)
    rag = derive_rag_control_condition(prompt)
    rag["prompt"]["prompt_hash"] = "0" * 64
    report = validate_causal_controls(
        examples=_examples()[:1], chunks=_chunks(),
        prompt_contract=load_prompt_contract(PROMPT_PATH),
        retrieval_artifact=load_and_verify_frozen_retrieval_artifact(FROZEN),
        prompt_condition=prompt, oracle_condition=oracle, rag_condition=rag,
    )
    assert not report.condition_checks["prompt_hash"]
    assert not report.passed
