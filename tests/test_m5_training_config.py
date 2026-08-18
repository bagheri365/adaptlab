from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.m5.training_config import (
    DEFAULT_EFFECTIVE_BATCH_SIZE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANK,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TRAINING_ITERS,
    DEFAULT_TARGET_POLICY,
    CANONICAL_BENCHMARK_MANIFEST_HASH,
    CANONICAL_EXPERIMENTAL_CONTRACT_HASH,
    CANONICAL_LORA_POLICY_HASH,
    CANONICAL_MODULE_INSPECTION_HASH,
    CANONICAL_TRAIN_EXAMPLE_IDS_HASH,
    CANONICAL_TRAINING_FORMATTER_HASH,
    CANONICAL_TRAINING_ISOLATION_AUDIT_HASH,
    CANONICAL_SOURCE_MANIFEST_HASH,
    CANONICAL_SOURCE_REPOSITORY,
    CANONICAL_SOURCE_REVISION,
    CANONICAL_MLX_BASE_IDENTITY_HASH,
    DEFAULT_ADAMW_BETAS,
    DEFAULT_ADAMW_EPS,
    DEFAULT_ADAMW_WEIGHT_DECAY,
    DEFAULT_DROPOUT,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_GRADIENT_CLIPPING,
    DEFAULT_MAX_RETAINED_CHECKPOINTS,
    TRAINING_CONFIG_VERSION,
    build_lora_training_config_artifact,
    validate_training_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data/generated/v0.0"
PROMPT_PATH = ROOT / "configs/prompts/prompt_v1.yaml"
MODULE_INSPECTION_PATH = ROOT / "artifacts/evaluation/m5/m5_model_module_inspection_v1.json"
LORA_POLICY_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_trainable_policy_v1.json"
TRAINING_FORMATTER_PATH = ROOT / "artifacts/evaluation/m5/m5_training_formatter_v1.json"
RUNTIME_PROVENANCE_PATH = ROOT / "artifacts/evaluation/m5/local_lora_runtime_provenance_v1.json"
ARTIFACT_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_training_config_v1.json"


def _artifact():
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _train_example_ids() -> list[str]:
    raw = json.loads((BENCHMARK_DIR / "train.json").read_text(encoding="utf-8"))
    return sorted(BenchmarkExample.from_dict(item).example_id for item in raw)


def _valid_manifest(config: dict[str, object]) -> dict[str, object]:
    ids = _train_example_ids()
    default = config["canonical_training_defaults"]
    checkpoint_policy = config["checkpoint_policy"]
    return {
        "run_id": "m5-canonical-policy-b-attn-r8-lr1e-5-iters500",
        "candidate_id": default["candidate_id"],
        "starting_git_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "experimental_contract_hash": CANONICAL_EXPERIMENTAL_CONTRACT_HASH,
        "training_isolation_audit_hash": CANONICAL_TRAINING_ISOLATION_AUDIT_HASH,
        "training_formatter_hash": CANONICAL_TRAINING_FORMATTER_HASH,
        "lora_policy_hash": CANONICAL_LORA_POLICY_HASH,
        "module_inspection_hash": CANONICAL_MODULE_INSPECTION_HASH,
        "benchmark_manifest_hash": CANONICAL_BENCHMARK_MANIFEST_HASH,
        "benchmark_train_split_hash": config["frozen_inputs"]["benchmark_train_split_hash"],
        "source_repository": CANONICAL_SOURCE_REPOSITORY,
        "source_revision": CANONICAL_SOURCE_REVISION,
        "source_manifest_hash": CANONICAL_SOURCE_MANIFEST_HASH,
        "mlx_base_identity_hash": CANONICAL_MLX_BASE_IDENTITY_HASH,
        "tokenizer_identity": config["frozen_inputs"]["tokenizer_identity"],
        "chat_template_identity": config["frozen_inputs"]["tokenizer_identity"]["chat_template_sha256"],
        "framework_versions": config["frozen_inputs"]["runtime_versions"],
        "python_version": "3.12.13",
        "macos_version": "26.5.2",
        "machine_architecture": "arm64",
        "target_policy": DEFAULT_TARGET_POLICY,
        "target_modules": default["target_modules"],
        "layer_coverage": config["provenance_validation_policy"]["frozen_layer_coverage"],
        "rank": DEFAULT_RANK,
        "alpha": 16,
        "dropout": DEFAULT_DROPOUT,
        "total_parameter_count": 8190735360,
        "trainable_parameter_count": default["trainable_parameter_count"],
        "trainable_percentage": default["trainable_percentage_of_total_model_parameters"],
        "optimizer": "adamw",
        "optimizer_hyperparameters": {
            "beta1": DEFAULT_ADAMW_BETAS[0],
            "beta2": DEFAULT_ADAMW_BETAS[1],
            "epsilon": DEFAULT_ADAMW_EPS,
        },
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY,
        "scheduler": "constant",
        "warmup": {"steps": 0, "ratio": None},
        "seed": 1729,
        "gradient_checkpointing": DEFAULT_GRADIENT_CHECKPOINTING,
        "gradient_clipping": DEFAULT_GRADIENT_CLIPPING,
        "micro_batch_size": DEFAULT_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": DEFAULT_EFFECTIVE_BATCH_SIZE,
        "max_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "truncation_audit": {
            "max_observed_sequence_length": 109,
            "assistant_target_truncations": 0,
            "training_record_truncations": 0,
        },
        "training_examples": ids,
        "training_example_ids_hash": CANONICAL_TRAIN_EXAMPLE_IDS_HASH,
        "training_data_hash": config["training_data_provenance"]["training_data_hash"],
        "training_duration_iters": DEFAULT_TRAINING_ITERS,
        "selected_checkpoint_step": DEFAULT_TRAINING_ITERS,
        "checkpoint_policy": {
            "save_every": DEFAULT_SAVE_EVERY,
            "maximum_retained_checkpoints": DEFAULT_MAX_RETAINED_CHECKPOINTS,
            "selected_checkpoint_step": DEFAULT_TRAINING_ITERS,
        },
        "adapter_output_path": "/private/tmp/m5_candidate_adapter",
        "adapter_file_manifest": [
            "adapters.safetensors",
            "adapter_config.json",
            "0000500_adapters.safetensors",
        ],
        "adapter_sha256": "0" * 64,
        "completion_status": "planned",
        "failure_status": None,
        "failure_reason": None,
    }


def test_training_config_freezes_optimizer_batching_sequence_and_search_space() -> None:
    artifact = _artifact()
    assert artifact["training_config_version"] == TRAINING_CONFIG_VERSION
    assert artifact["optimizer_policy"] == {
        "family": "adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "framework_default_note": "Matches mlx.optimizers.AdamW defaults in the frozen MLX package.",
    }
    assert artifact["scheduler_policy"]["type"] == "constant"
    assert artifact["batching_policy"] == {
        "micro_batch_size": DEFAULT_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": DEFAULT_EFFECTIVE_BATCH_SIZE,
        "applies_to_all_candidates": True,
    }
    assert artifact["sequence_length_policy"]["max_sequence_length"] == DEFAULT_MAX_SEQUENCE_LENGTH
    assert artifact["sequence_length_policy"]["training_record_truncations"] == 0
    assert artifact["sequence_length_policy"]["assistant_target_truncations"] == 0
    assert artifact["numeric_and_gradient_policy"]["gradient_clipping"] is None
    assert artifact["numeric_and_gradient_policy"]["gradient_checkpointing"] is DEFAULT_GRADIENT_CHECKPOINTING
    assert artifact["allowed_candidate_dimensions"]["target_policies"] == [
        "POLICY_A_QV",
        "POLICY_B_ATTN",
        "POLICY_C_ATTN_MLP",
    ]
    assert artifact["allowed_candidate_dimensions"]["ranks"] == [4, 8, 16]
    assert artifact["allowed_candidate_dimensions"]["learning_rates"] == [5e-06, 1e-05, 2e-05]


def test_training_config_sequence_length_audit_matches_frozen_train_set() -> None:
    artifact = _artifact()
    audit = artifact["sequence_length_policy"]["token_length_distribution"]
    assert audit["minimum"] == 81
    assert audit["median"] == 93.0
    assert audit["p90"] == 99.1
    assert audit["p95"] == 102.0
    assert audit["p99"] == 106.01
    assert audit["maximum"] == 109
    assert audit["max_sequence_length"] == DEFAULT_MAX_SEQUENCE_LENGTH
    assert audit["truncated_training_records"] == 0
    assert audit["truncated_assistant_targets"] == 0


def test_training_config_validator_accepts_whitelisted_search_variations() -> None:
    artifact = _artifact()
    manifest = _valid_manifest(artifact)
    validate_training_run_manifest(manifest, artifact)

    manifest["rank"] = 16
    manifest["alpha"] = 32
    manifest["learning_rate"] = 2e-5
    manifest["training_duration_iters"] = 1000
    manifest["selected_checkpoint_step"] = 1000
    manifest["candidate_id"] = "POLICY_B_ATTN_r16_lr2e-5_iters1000"
    manifest["trainable_parameter_count"] = 15335424
    manifest["trainable_percentage"] = 0.18722890346197196
    validate_training_run_manifest(manifest, artifact)


def test_training_config_validator_rejects_non_search_drift() -> None:
    artifact = _artifact()
    manifest = _valid_manifest(artifact)

    for field, bad_value in [
        ("source_revision", "wrong"),
        ("source_manifest_hash", "wrong"),
        ("training_formatter_hash", "wrong"),
        ("lora_policy_hash", "wrong"),
        ("optimizer", "adam"),
        ("scheduler", "linear"),
        ("dropout", 0.1),
        ("seed", 1730),
        ("layer_coverage", {"layer_coverage_mode": "last_16_transformer_blocks"}),
        ("benchmark_manifest_hash", "wrong"),
    ]:
        mutated = dict(manifest)
        mutated[field] = bad_value
        try:
            validate_training_run_manifest(mutated, artifact)
        except ValueError:
            continue
        raise AssertionError(f"expected validation to reject drift in {field}")


def test_training_config_is_deterministic_and_hashable(tmp_path: Path) -> None:
    artifact = _artifact()
    second = _artifact()
    assert artifact == second
    digest = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    assert len(digest) == 64
