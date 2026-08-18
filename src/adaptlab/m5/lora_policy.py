"""Canonical Milestone 5 LoRA trainable-parameter policy.

This module freezes the structural adapter policy for the canonical Milestone 5
Qwen3 MLX base.  It does not train a model.  It only derives and audits the
allowed adapter namespace, layer coverage, and trainable-parameter counts from
the frozen module-inspection artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json

LORA_POLICY_SCHEMA_VERSION = "m5-lora-trainable-policy-artifact-v1"
LORA_POLICY_VERSION = "m5-lora-trainable-policy-v1"
BENCHMARK_MANIFEST_HASH = "f3933caa5ba4432b9631b989a584d999022a102fe1cbb6b113b51da63ff22b85"
DEFAULT_RANK_CANDIDATES = (4, 8, 16)
DEFAULT_ALPHA_MULTIPLIER = 2
DEFAULT_DROPOUT = 0.0


@dataclass(frozen=True, slots=True)
class TrainablePolicyCandidate:
    """One architecture-supported candidate target policy."""

    policy_id: str
    description: str
    target_module_templates: tuple[str, ...]
    matched_adapter_sites: int
    rank_coefficient: int
    trainable_parameter_table: tuple[dict[str, Any], ...]
    expected_adapter_namespace_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "description": self.description,
            "target_module_templates": list(self.target_module_templates),
            "matched_adapter_sites": self.matched_adapter_sites,
            "rank_coefficient": self.rank_coefficient,
            "trainable_parameter_table": list(self.trainable_parameter_table),
            "expected_adapter_namespace_prefixes": list(self.expected_adapter_namespace_prefixes),
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_text_hash(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _concrete_prefixes(target_templates: Sequence[str], layer_indices: Sequence[int]) -> tuple[str, ...]:
    prefixes: list[str] = []
    for layer_index in layer_indices:
        for template in target_templates:
            prefixes.append(template.format(i=layer_index))
    return tuple(prefixes)


def _adapter_parameter_count(rank_coefficient: int, rank: int) -> int:
    return rank_coefficient * rank


def _trainable_percentage(trainable_parameters: int, total_parameters: int) -> float:
    return (trainable_parameters / total_parameters) * 100.0


def _build_candidate(
    *,
    policy_id: str,
    description: str,
    target_module_templates: Sequence[str],
    module_lookup: dict[str, dict[str, Any]],
    layer_indices: Sequence[int],
    total_parameters: int,
) -> TrainablePolicyCandidate:
    matched_adapter_sites = 0
    rank_coefficient = 0
    for template in target_module_templates:
        info = module_lookup[template]
        matched_adapter_sites += int(info["matched_modules"])
        logical_shape = info["logical_shape"]
        if len(logical_shape) != 2:
            raise ValueError(f"{template} must have a 2D logical shape, got {logical_shape}")
        per_site_coefficient = int(logical_shape[0]) + int(logical_shape[1])
        rank_coefficient += int(info["matched_modules"]) * per_site_coefficient

    trainable_parameter_table = tuple(
        {
            "rank": rank,
            "alpha": DEFAULT_ALPHA_MULTIPLIER * rank,
            "dropout": DEFAULT_DROPOUT,
            "adapter_parameter_count": _adapter_parameter_count(rank_coefficient, rank),
            "trainable_percentage_of_total_model_parameters": _trainable_percentage(
                _adapter_parameter_count(rank_coefficient, rank), total_parameters
            ),
        }
        for rank in DEFAULT_RANK_CANDIDATES
    )
    return TrainablePolicyCandidate(
        policy_id=policy_id,
        description=description,
        target_module_templates=tuple(target_module_templates),
        matched_adapter_sites=matched_adapter_sites,
        rank_coefficient=rank_coefficient,
        trainable_parameter_table=trainable_parameter_table,
        expected_adapter_namespace_prefixes=_concrete_prefixes(target_module_templates, layer_indices),
    )


def audit_trainable_parameter_names(
    trainable_parameter_names: Iterable[str],
    *,
    allowed_adapter_namespace_prefixes: Iterable[str],
    allowed_adapter_leaf_names: Iterable[str] = ("lora_a", "lora_b"),
) -> dict[str, Any]:
    """Check that all trainable parameters stay inside the declared LoRA namespace.

    This is a mechanical namespace check.  It is intended to fail closed if a
    future adapter configuration accidentally makes a base-model parameter
    trainable or introduces a non-LoRA trainable tensor.
    """

    prefixes = tuple(allowed_adapter_namespace_prefixes)
    allowed_leaf_names = set(allowed_adapter_leaf_names)
    allowed: list[str] = []
    unexpected: list[str] = []
    for name in trainable_parameter_names:
        prefix_match = next((p for p in prefixes if name == p or name.startswith(p + ".")), None)
        if prefix_match is None:
            unexpected.append(name)
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in allowed_leaf_names:
            unexpected.append(name)
            continue
        allowed.append(name)
    return {
        "passed": not unexpected,
        "allowed_trainable_parameter_names": allowed,
        "unexpected_trainable_parameter_names": unexpected,
        "allowed_adapter_namespace_prefixes": list(prefixes),
        "allowed_adapter_leaf_names": sorted(allowed_leaf_names),
    }


def build_lora_trainable_policy_artifact(
    *,
    module_inspection_path: str | Path,
    experimental_contract_hash: str,
    contamination_audit_hash: str,
    training_formatter_hash: str,
    module_inspection_hash: str,
    runtime_provenance_path: str | Path,
) -> dict[str, Any]:
    """Build the frozen Milestone 5 LoRA trainable-parameter policy artifact."""

    module_inspection = _load_json(module_inspection_path)
    provenance = _load_json(runtime_provenance_path)

    loaded_base = module_inspection["loaded_base_verification"]
    base_config = loaded_base["model_config"]
    layer_indices = list(range(int(base_config["num_hidden_layers"])))
    total_parameters = int(loaded_base["quantization"]["weight_file_total_parameters"])

    module_lookup = {
        entry["module_pattern"]: entry
        for entry in module_inspection["layer_coverage"]
    }

    candidates = [
        _build_candidate(
            policy_id="POLICY_A_QV",
            description="Attention q_proj and v_proj only.",
            target_module_templates=(
                "model.layers.{i}.self_attn.q_proj",
                "model.layers.{i}.self_attn.v_proj",
            ),
            module_lookup=module_lookup,
            layer_indices=layer_indices,
            total_parameters=total_parameters,
        ),
        _build_candidate(
            policy_id="POLICY_B_ATTN",
            description="Attention q_proj, k_proj, v_proj, and o_proj.",
            target_module_templates=(
                "model.layers.{i}.self_attn.q_proj",
                "model.layers.{i}.self_attn.k_proj",
                "model.layers.{i}.self_attn.v_proj",
                "model.layers.{i}.self_attn.o_proj",
            ),
            module_lookup=module_lookup,
            layer_indices=layer_indices,
            total_parameters=total_parameters,
        ),
        _build_candidate(
            policy_id="POLICY_C_ATTN_MLP",
            description="All attention projections plus gate_proj, up_proj, and down_proj.",
            target_module_templates=(
                "model.layers.{i}.self_attn.q_proj",
                "model.layers.{i}.self_attn.k_proj",
                "model.layers.{i}.self_attn.v_proj",
                "model.layers.{i}.self_attn.o_proj",
                "model.layers.{i}.mlp.gate_proj",
                "model.layers.{i}.mlp.up_proj",
                "model.layers.{i}.mlp.down_proj",
            ),
            module_lookup=module_lookup,
            layer_indices=layer_indices,
            total_parameters=total_parameters,
        ),
    ]

    rank_candidates = list(DEFAULT_RANK_CANDIDATES)
    policy = {
        "schema_version": LORA_POLICY_SCHEMA_VERSION,
        "policy_version": LORA_POLICY_VERSION,
        "gate": "M5_LORA_TRAINABLE_POLICY_READY",
        "frozen_inputs": {
            "experimental_contract_hash": experimental_contract_hash,
            "contamination_audit_hash": contamination_audit_hash,
            "training_formatter_hash": training_formatter_hash,
            "module_inspection_hash": module_inspection_hash,
            "source_lineage": {
                "repository": provenance["source_lineage"]["repository"],
                "revision": provenance["source_lineage"]["revision"],
                "source_manifest_hash": "507f79d4086e495f0852327e79ea6a4daa53afe2beb591a0fd8489dc16fe8397",
                "canonical_mlx_base_identity_hash": module_inspection["frozen_inputs"]["canonical_mlx_base_identity_hash"],
            },
            "runtime_versions": {
                "python_version": provenance["runtime_environment"]["python_version"],
                "macos_version": provenance["runtime_environment"]["macos_version"],
                "machine_architecture": provenance["runtime_environment"]["machine_architecture"],
                "mlx_version": provenance["runtime_environment"]["installed_packages"]["mlx"],
                "mlx_lm_version": provenance["runtime_environment"]["installed_packages"]["mlx-lm"],
            },
            "benchmark_manifest_hash": BENCHMARK_MANIFEST_HASH,
        },
        "base_parameter_policy": {
            "base_parameters_trainable": False,
            "embeddings_trainable": False,
            "lm_head_trainable": False,
            "normalization_trainable": False,
            "bias_trainable": False,
            "adapter_parameters_only_trainable": True,
        },
        "layer_coverage_policy": {
            "layer_coverage_mode": "all_36_transformer_blocks",
            "num_layers": len(layer_indices),
            "covered_layer_indices": layer_indices,
        },
        "lora_structural_hyperparameters": {
            "rank_candidate_set": rank_candidates,
            "alpha_rule": "alpha = 2 * r",
            "dropout": DEFAULT_DROPOUT,
            "bias_policy": "frozen_base_biases; no trainable bias terms introduced by the adapter policy",
        },
        "mechanical_trainable_parameter_audit": {
            "audit_mode": "prefix-and-leaf-name validation against declared adapter namespaces",
            "allowed_trainable_leaf_names": ["lora_a", "lora_b"],
            "unexpected_trainable_parameters_are_fatal": True,
            "base_weight_trainable_parameters_are_fatal": True,
        },
        "candidate_target_policies": [candidate.to_dict() for candidate in candidates],
        "optimization_separation": {
            "structural_policy_frozen": True,
            "later_search_may_vary_only": [
                "target_policy from {POLICY_A_QV, POLICY_B_ATTN, POLICY_C_ATTN_MLP}",
                "rank from {4, 8, 16}",
                "learning_rate",
                "training_duration",
            ],
            "no_new_target_modules_after_validation": True,
            "no_embedding_tuning": True,
            "no_lm_head_tuning": True,
            "no_normalization_tuning": True,
            "no_full_fine_tuning": True,
        },
        "shared_policy_constraints": {
            "same_layer_coverage_for_all_candidates": True,
            "same_dropout_for_all_candidates": True,
            "same_rank_set_for_all_candidates": True,
            "same_base_representation_for_all_candidates": True,
            "same_runtime_for_all_candidates": True,
        },
        "framework_compatibility": {
            "mlx_lm_custom_keys_supported": True,
            "mlx_lm_default_layer_traversal_supported": True,
            "adapter_namespace_can_be_expressed_exactly": True,
        },
        "module_inspection_hash": module_inspection_hash,
    }

    policy["policy_hash"] = sha256_bytes(canonical_json_bytes(policy))
    return policy


def write_lora_trainable_policy_artifact(
    *,
    module_inspection_path: str | Path,
    experimental_contract_hash: str,
    contamination_audit_hash: str,
    training_formatter_hash: str,
    module_inspection_hash: str,
    runtime_provenance_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the frozen Milestone 5 LoRA trainable-parameter policy artifact."""

    artifact = build_lora_trainable_policy_artifact(
        module_inspection_path=module_inspection_path,
        experimental_contract_hash=experimental_contract_hash,
        contamination_audit_hash=contamination_audit_hash,
        training_formatter_hash=training_formatter_hash,
        module_inspection_hash=module_inspection_hash,
        runtime_provenance_path=runtime_provenance_path,
    )
    write_json(Path(output_path), artifact)
    return artifact
