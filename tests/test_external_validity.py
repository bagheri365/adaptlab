from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from adaptlab.benchmark.external_validity import (
    ExternalValidityPolicyError,
    external_validity_policy_from_mapping,
    load_external_validity_policy,
)


POLICY_PATH = Path("configs/external_validity_v0.0.yaml")


def _raw_policy():
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def test_external_validity_policy_loads_with_precommitted_candidate_rule():
    policy = load_external_validity_policy(POLICY_PATH)
    assert policy.status == "PRECOMMITTED_BEFORE_CANONICAL_MODEL_RESULTS"
    assert policy.benchmark_version == "0.0.0"
    assert tuple(item.benchmark_id for item in policy.candidate_benchmarks) == (
        "IFEval",
        "MMLU-Pro",
        "BBH",
    )
    assert policy.selection_rule.timing == "before_canonical_adaptlab_model_results"
    assert policy.selection_rule.rank_eligible_candidates_by[-1] == "lexicographically_smaller_benchmark_id"


def test_policy_records_all_required_external_validity_sections():
    raw = _raw_policy()
    assert {
        "inclusion_criteria",
        "exclusion_criteria",
        "sample_size_target",
        "scoring_approach",
        "contamination_review",
        "predeclared_directional_transfer_claim",
        "provenance",
    } <= set(raw)


def test_policy_freezes_external_benchmark_out_of_tuning_loop():
    policy = load_external_validity_policy(POLICY_PATH)
    assert policy.scoring_approach.no_prompt_or_hyperparameter_selection_on_external_benchmark is True
    assert policy.scoring_approach.report_absolute_change_from_same_base_model_baseline is True


def test_policy_records_reproducible_sample_size_rule():
    policy = load_external_validity_policy(POLICY_PATH)
    assert policy.sample_size_target.minimum_examples <= policy.sample_size_target.target_examples
    assert policy.sample_size_target.target_examples <= policy.sample_size_target.maximum_examples
    assert policy.sample_size_target.sampling_seed == 1729
    assert "deterministic hash-ranked sample" in policy.sample_size_target.sampling_rule


def test_policy_records_directional_transfer_claim_without_magnitude_equivalence():
    policy = load_external_validity_policy(POLICY_PATH)
    claim = policy.predeclared_directional_transfer_claim
    assert claim.claim_id == "GENERALIZATION_PRESERVATION"
    assert claim.direction_test == "non_negative_change"
    assert claim.magnitude_equivalence_claimed is False


def test_policy_rejects_post_result_selection_timing():
    raw = deepcopy(_raw_policy())
    raw["selection_rule"]["timing"] = "after_nimbus_results"
    with pytest.raises(ExternalValidityPolicyError, match="before canonical AdaptLab model results"):
        external_validity_policy_from_mapping(raw)


def test_policy_rejects_external_benchmark_as_tuning_data():
    raw = deepcopy(_raw_policy())
    raw["scoring_approach"]["no_prompt_or_hyperparameter_selection_on_external_benchmark"] = False
    with pytest.raises(ExternalValidityPolicyError, match="must not be used for prompt or hyperparameter selection"):
        external_validity_policy_from_mapping(raw)


def test_policy_requires_final_manifest_provenance():
    raw = deepcopy(_raw_policy())
    raw["provenance"]["include_in_final_benchmark_manifest"] = False
    with pytest.raises(ExternalValidityPolicyError, match="final benchmark provenance"):
        external_validity_policy_from_mapping(raw)
