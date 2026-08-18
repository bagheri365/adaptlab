from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptlab.m5.lora_policy import (
    DEFAULT_RANK_CANDIDATES,
    LORA_POLICY_VERSION,
    audit_trainable_parameter_names,
    build_lora_trainable_policy_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_INSPECTION_PATH = ROOT / "artifacts/evaluation/m5/m5_model_module_inspection_v1.json"
RUNTIME_PROVENANCE_PATH = ROOT / "artifacts/evaluation/m5/local_lora_runtime_provenance_v1.json"
EXPERIMENTAL_CONTRACT_HASH = "87dae4d0b51572519f5b82158ef73c2832d8e09c035b0a7e5f1aaa17cb9d5e47"
CONTAMINATION_AUDIT_HASH = "d189b5aca1d682b951f9997518a3d552b41c6cde80530a5087701a61a6e20878"
TRAINING_FORMATTER_HASH = "1d04beb66c4f5d81fdcba9f3a9d9e3cfdb243251766784a4f58dee1a4ce9ca60"
MODULE_INSPECTION_HASH = "63c3c9995162c077950a0373dff346d5622cb525b68e961933f90f958d19eab0"


def _policy_artifact():
    return build_lora_trainable_policy_artifact(
        module_inspection_path=MODULE_INSPECTION_PATH,
        experimental_contract_hash=EXPERIMENTAL_CONTRACT_HASH,
        contamination_audit_hash=CONTAMINATION_AUDIT_HASH,
        training_formatter_hash=TRAINING_FORMATTER_HASH,
        module_inspection_hash=MODULE_INSPECTION_HASH,
        runtime_provenance_path=RUNTIME_PROVENANCE_PATH,
    )


def test_policy_freezes_all_three_candidates_and_all_36_layers() -> None:
    artifact = _policy_artifact()

    assert artifact["policy_version"] == LORA_POLICY_VERSION
    assert artifact["base_parameter_policy"] == {
        "base_parameters_trainable": False,
        "embeddings_trainable": False,
        "lm_head_trainable": False,
        "normalization_trainable": False,
        "bias_trainable": False,
        "adapter_parameters_only_trainable": True,
    }

    layer_policy = artifact["layer_coverage_policy"]
    assert layer_policy["layer_coverage_mode"] == "all_36_transformer_blocks"
    assert layer_policy["num_layers"] == 36
    assert layer_policy["covered_layer_indices"] == list(range(36))

    candidates = {candidate["policy_id"]: candidate for candidate in artifact["candidate_target_policies"]}
    assert set(candidates) == {"POLICY_A_QV", "POLICY_B_ATTN", "POLICY_C_ATTN_MLP"}

    assert candidates["POLICY_A_QV"]["matched_adapter_sites"] == 72
    assert candidates["POLICY_B_ATTN"]["matched_adapter_sites"] == 144
    assert candidates["POLICY_C_ATTN_MLP"]["matched_adapter_sites"] == 252

    assert candidates["POLICY_A_QV"]["expected_adapter_namespace_prefixes"][0] == "model.layers.0.self_attn.q_proj"
    assert "model.embed_tokens" not in json.dumps(candidates["POLICY_A_QV"])
    assert "lm_head" not in json.dumps(candidates["POLICY_A_QV"])
    assert "layernorm" not in json.dumps(candidates["POLICY_A_QV"]).lower()


def test_policy_records_rank_set_alpha_rule_and_counts() -> None:
    artifact = _policy_artifact()
    hyper = artifact["lora_structural_hyperparameters"]
    assert tuple(hyper["rank_candidate_set"]) == DEFAULT_RANK_CANDIDATES
    assert hyper["alpha_rule"] == "alpha = 2 * r"
    assert hyper["dropout"] == 0.0

    counts = {
        candidate["policy_id"]: {row["rank"]: row["adapter_parameter_count"] for row in candidate["trainable_parameter_table"]}
        for candidate in artifact["candidate_target_policies"]
    }
    assert counts["POLICY_A_QV"] == {4: 1916928, 8: 3833856, 16: 7667712}
    assert counts["POLICY_B_ATTN"] == {4: 3833856, 8: 7667712, 16: 15335424}
    assert counts["POLICY_C_ATTN_MLP"] == {4: 10911744, 8: 21823488, 16: 43646976}

    policy_b = next(c for c in artifact["candidate_target_policies"] if c["policy_id"] == "POLICY_B_ATTN")
    assert policy_b["trainable_parameter_table"][1]["trainable_percentage_of_total_model_parameters"] > 0


def test_trainable_parameter_namespace_audit_accepts_only_lora_suffixes() -> None:
    artifact = _policy_artifact()
    candidate = next(c for c in artifact["candidate_target_policies"] if c["policy_id"] == "POLICY_A_QV")
    allowed_prefixes = candidate["expected_adapter_namespace_prefixes"]

    valid_names = [
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.35.self_attn.v_proj.lora_b",
    ]
    audit = audit_trainable_parameter_names(valid_names, allowed_adapter_namespace_prefixes=allowed_prefixes)
    assert audit["passed"] is True
    assert audit["unexpected_trainable_parameter_names"] == []

    invalid_names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.linear.weight",
        "model.embed_tokens.lora_a",
        "lm_head.lora_b",
    ]
    audit = audit_trainable_parameter_names(invalid_names, allowed_adapter_namespace_prefixes=allowed_prefixes)
    assert audit["passed"] is False
    assert set(audit["unexpected_trainable_parameter_names"]) == set(invalid_names)


def test_policy_artifact_is_deterministic_and_hashable(tmp_path: Path) -> None:
    artifact = _policy_artifact()
    again = _policy_artifact()
    assert artifact == again
    assert artifact["policy_hash"]
    path = tmp_path / "m5_lora_trainable_policy_v1.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
