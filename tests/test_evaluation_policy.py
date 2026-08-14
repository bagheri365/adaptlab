from copy import deepcopy
from pathlib import Path

import yaml
import pytest

from adaptlab.benchmark.evaluation_policy import (
    EvaluationPolicyError,
    evaluation_policy_from_mapping,
    load_evaluation_policy,
)


POLICY_PATH = Path("configs/evaluation_policy_v0.0.yaml")


def _raw_policy():
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def test_canonical_evaluation_policy_loads_and_is_precommitted():
    policy = load_evaluation_policy(POLICY_PATH)

    assert policy.benchmark_version == "0.0.0"
    assert policy.status == "PRECOMMITTED_BEFORE_MODEL_RESULTS"
    assert policy.prompt_selection_procedure.selection_split == "validation"
    assert policy.prompt_selection_procedure.test_set_access_during_selection is False
    assert policy.prompt_selection_procedure.sentinel_access_during_selection is False
    assert policy.generalization_guardrail.selection_use == "guardrail_only"
    assert policy.minimum_meaningful_improvement.absolute_percentage_points == 2.0
    assert policy.canonical_lora_seed_policy.confirmatory_seeds == (1729, 2718, 31415)


def test_policy_records_required_precommitment_sections():
    raw = _raw_policy()
    assert {
        "prompt_budget",
        "prompt_selection_procedure",
        "validation_set_usage_budget",
        "allowed_future_prompt_search_space",
        "allowed_future_lora_hyperparameter_search_space",
        "generalization_guardrail",
        "minimum_meaningful_improvement",
        "subgroup_policy",
        "architecture_sensitive_retention_rules",
        "canonical_lora_seed_policy",
    } <= set(raw)


def test_policy_rejects_test_set_access_during_prompt_selection():
    raw = deepcopy(_raw_policy())
    raw["prompt_selection_procedure"]["test_set_access_during_selection"] = True
    with pytest.raises(EvaluationPolicyError, match="test set access"):
        evaluation_policy_from_mapping(raw)


def test_policy_rejects_sentinel_as_selection_data():
    raw = deepcopy(_raw_policy())
    raw["prompt_selection_procedure"]["sentinel_access_during_selection"] = True
    with pytest.raises(EvaluationPolicyError, match="sentinel access"):
        evaluation_policy_from_mapping(raw)


def test_policy_rejects_answer_or_test_content_in_prompt_search_space():
    raw = deepcopy(_raw_policy())
    raw["allowed_future_prompt_search_space"]["benchmark_answer_or_test_content_in_prompts"] = True
    with pytest.raises(EvaluationPolicyError, match="benchmark answers/test content"):
        evaluation_policy_from_mapping(raw)


def test_policy_rejects_architecture_retention_outside_precommitment():
    raw = deepcopy(_raw_policy())
    raw["architecture_sensitive_retention_rules"]["retain_lora_hyperparameters_across_architectures"] = True
    with pytest.raises(EvaluationPolicyError, match="architecture retention rule"):
        evaluation_policy_from_mapping(raw)


def test_policy_rejects_single_seed_canonical_lora_claims():
    raw = deepcopy(_raw_policy())
    raw["canonical_lora_seed_policy"]["single_seed_claims_allowed"] = True
    with pytest.raises(EvaluationPolicyError, match="single-seed"):
        evaluation_policy_from_mapping(raw)
