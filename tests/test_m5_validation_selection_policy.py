from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptlab.m5 import (
    CANONICAL_TRAINING_CONFIG_HASH,
    DEFAULT_CANDIDATE_IDS,
    DEFAULT_VALIDATION_FAMILIES,
    VALIDATION_SELECTION_VERSION,
    build_validation_selection_policy_artifact,
    validate_validation_selection_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data/generated/v0.0"
TRAINING_CONFIG_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_training_config_v2.json"
EXPERIMENTAL_CONTRACT_HASH = "87dae4d0b51572519f5b82158ef73c2832d8e09c035b0a7e5f1aaa17cb9d5e47"
TRAINING_ISOLATION_AUDIT_HASH = "d189b5aca1d682b951f9997518a3d552b41c6cde80530a5087701a61a6e20878"
TRAINING_FORMATTER_HASH = "b7eefc912755d029d59d2b075b757111cd0c4910d15886f9befbac060b2cfb46"


def _artifact():
    return build_validation_selection_policy_artifact(
        benchmark_dir=BENCHMARK_DIR,
        training_config_path=TRAINING_CONFIG_PATH,
        experimental_contract_hash=EXPERIMENTAL_CONTRACT_HASH,
        training_isolation_audit_hash=TRAINING_ISOLATION_AUDIT_HASH,
        training_formatter_hash=TRAINING_FORMATTER_HASH,
    )


def _valid_manifest(config: dict[str, object]) -> dict[str, object]:
    candidate = config["candidate_budget"]["candidate_records"][2]
    policy = config["validation_selection_policy"]
    frozen_inputs = config["frozen_inputs"]
    return {
        "selection_run_id": "m5-validation-selection-stage1-policy-b-attn-r8",
        "candidate_id": candidate["candidate_id"],
        "stage": candidate["stage"],
        "target_policy": candidate["target_policy"],
        "rank": candidate["rank"],
        "alpha": candidate["alpha"],
        "learning_rate": candidate["learning_rate"],
        "training_duration_iters": candidate["training_duration_iters"],
        "eligible_checkpoint_steps": list(candidate["eligible_checkpoint_steps"]),
        "seed": candidate["seed"],
        "target_modules": list(candidate["target_modules"]),
        "trainable_parameter_count": candidate["trainable_parameter_count"],
        "trainable_percentage_of_total_model_parameters": candidate["trainable_percentage_of_total_model_parameters"],
        "selection_split": "validation",
        "primary_metric_name": "macro_average_exact_match_accuracy",
        "secondary_metric_name": "overall_validation_exact_match_accuracy",
        "tie_breakers": list(policy["tie_breakers"]),
        "checkpoint_selection_rule": policy["checkpoint_policy"]["candidate_checkpoint_rule"],
        "forbidden_signal_attestation": True,
        "training_config_hash": frozen_inputs["training_config_hash"],
        "experimental_contract_hash": frozen_inputs["experimental_contract_hash"],
        "training_isolation_audit_hash": frozen_inputs["training_isolation_audit_hash"],
        "training_formatter_hash": frozen_inputs["training_formatter_hash"],
    }


def _manifest_for_candidate(config: dict[str, object], index: int) -> dict[str, object]:
    candidate = config["candidate_budget"]["candidate_records"][index]
    policy = config["validation_selection_policy"]
    frozen_inputs = config["frozen_inputs"]
    return {
        "selection_run_id": f"m5-validation-selection-{candidate['candidate_id']}",
        "candidate_id": candidate["candidate_id"],
        "stage": candidate["stage"],
        "target_policy": candidate["target_policy"],
        "rank": candidate["rank"],
        "alpha": candidate["alpha"],
        "learning_rate": candidate["learning_rate"],
        "training_duration_iters": candidate["training_duration_iters"],
        "eligible_checkpoint_steps": list(candidate["eligible_checkpoint_steps"]),
        "seed": candidate["seed"],
        "target_modules": list(candidate["target_modules"]),
        "trainable_parameter_count": candidate["trainable_parameter_count"],
        "trainable_percentage_of_total_model_parameters": candidate["trainable_percentage_of_total_model_parameters"],
        "selection_split": "validation",
        "primary_metric_name": "macro_average_exact_match_accuracy",
        "secondary_metric_name": "overall_validation_exact_match_accuracy",
        "tie_breakers": list(policy["tie_breakers"]),
        "checkpoint_selection_rule": policy["checkpoint_policy"]["candidate_checkpoint_rule"],
        "forbidden_signal_attestation": True,
        "training_config_hash": frozen_inputs["training_config_hash"],
        "experimental_contract_hash": frozen_inputs["experimental_contract_hash"],
        "training_isolation_audit_hash": frozen_inputs["training_isolation_audit_hash"],
        "training_formatter_hash": frozen_inputs["training_formatter_hash"],
    }


def test_validation_selection_policy_freezes_metric_and_family_macro_average() -> None:
    artifact = _artifact()
    assert artifact["policy_version"] == VALIDATION_SELECTION_VERSION
    assert artifact["gate"] == "M5_VALIDATION_SELECTION_POLICY_READY"
    assert artifact["validation_selection_policy"]["selection_split"] == "validation"
    assert artifact["validation_selection_policy"]["primary_metric"] == {
        "name": "macro_average_exact_match_accuracy",
        "formula": "mean(exact_match_accuracy for each represented validation task_family)",
        "family_weighting": "uniform_over_represented_families",
        "represented_families": list(DEFAULT_VALIDATION_FAMILIES),
        "represented_family_counts": [
            {"task_family": "behavior_only", "n": 38},
            {"task_family": "behavior_knowledge", "n": 37},
            {"task_family": "changed_knowledge", "n": 37},
            {"task_family": "knowledge_only", "n": 38},
        ],
    }
    assert artifact["validation_selection_policy"]["secondary_metric"] == {
        "name": "overall_validation_exact_match_accuracy",
        "formula": "validation_correct / validation_n",
    }
    assert artifact["validation_selection_policy"]["tie_breakers"] == [
        "smaller_lora_rank",
        "narrower_target_policy_by_trainable_parameter_count",
        "fewer_training_steps",
        "fewer_trainable_parameters",
        "lexical_candidate_id_ascending",
    ]
    assert artifact["validation_selection_policy"]["forbidden_selection_signals"] == [
        "primary_test_accuracy",
        "primary_test_family_scores",
        "updated_test_score",
        "removed_test_score",
        "structural_holdout_test_score",
        "primary_test_rag_results",
        "primary_test_oracle_context_results",
        "milestone_4_failure_categories",
        "generalization_sentinel",
        "qualitative_test_example_inspection",
        "test_set_hyperparameter_tuning",
    ]


def test_validation_selection_policy_predeclares_complete_candidate_budget() -> None:
    artifact = _artifact()
    budget = artifact["candidate_budget"]
    assert budget["total_candidates"] == 6
    assert budget["stage1_candidate_count"] == 4
    assert budget["stage2_candidate_count"] == 2
    assert budget["candidate_ids"] == list(DEFAULT_CANDIDATE_IDS)
    assert budget["excluded_high_rank_scope"] == {
        "POLICY_C_ATTN_MLP_rank8": "excluded",
        "POLICY_C_ATTN_MLP_rank16": "excluded",
        "rationale": (
            "The heavier attention-plus-MLP adapter family is represented only at rank 4 "
            "to keep the candidate budget hardware-realistic and avoid spending most of the "
            "budget on the heaviest adapter configuration."
        ),
    }

    stage1 = [candidate for candidate in budget["candidate_records"] if candidate["stage"] == "stage1"]
    stage2 = [candidate for candidate in budget["candidate_records"] if candidate["stage"] == "stage2"]
    assert len(stage1) == 4
    assert len(stage2) == 2
    assert stage1[0]["candidate_id"] == "S1_POLICY_A_QV_r8_lr1e-05_iters500"
    assert stage1[1]["candidate_id"] == "S1_POLICY_B_ATTN_r4_lr1e-05_iters500"
    assert stage1[2]["candidate_id"] == "S1_POLICY_B_ATTN_r8_lr1e-05_iters500"
    assert stage1[3]["candidate_id"] == "S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500"
    assert stage2[0]["candidate_id"] == "S2_POLICY_B_ATTN_r8_lr5e-06_iters1000"
    assert stage2[1]["candidate_id"] == "S2_POLICY_B_ATTN_r8_lr2e-05_iters250"
    assert stage2[0]["eligible_checkpoint_steps"] == [1000]
    assert stage2[1]["eligible_checkpoint_steps"] == [250]


def test_validation_selection_policy_validator_accepts_frozen_candidate_manifest() -> None:
    artifact = _artifact()
    manifest = _valid_manifest(artifact)
    validate_validation_selection_manifest(manifest, artifact)

    stage2_manifest = _manifest_for_candidate(artifact, 4)
    validate_validation_selection_manifest(stage2_manifest, artifact)


def test_validation_selection_policy_rejects_forbidden_signals_and_drift() -> None:
    artifact = _artifact()
    manifest = _valid_manifest(artifact)

    for field, bad_value in [
        ("selection_split", "test"),
        ("primary_metric_name", "overall_validation_exact_match_accuracy"),
        ("secondary_metric_name", "primary_test_accuracy"),
        ("forbidden_signal_attestation", False),
        ("training_config_hash", "wrong"),
        ("experimental_contract_hash", "wrong"),
        ("tie_breakers", ["lexical_candidate_id_ascending"]),
    ]:
        mutated = dict(manifest)
        mutated[field] = bad_value
        try:
            validate_validation_selection_manifest(mutated, artifact)
        except ValueError:
            continue
        raise AssertionError(f"expected validation to reject drift in {field}")

    forbidden = dict(manifest)
    forbidden["primary_test_accuracy"] = 0.9
    try:
        validate_validation_selection_manifest(forbidden, artifact)
    except ValueError:
        pass
    else:
        raise AssertionError("expected validation to reject forbidden primary-test fields")


def test_validation_selection_policy_is_deterministic_and_hashable(tmp_path: Path) -> None:
    artifact = _artifact()
    second = _artifact()
    assert artifact == second
    path = tmp_path / "m5_validation_selection_policy_v1.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64
