"""Canonical Milestone 5 LoRA training configuration and provenance policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.m5.model_preflight import resolve_m5_runtime_paths
from adaptlab.m5.training_formatter import build_training_messages

TRAINING_CONFIG_SCHEMA_VERSION = "m5-lora-training-config-artifact-v1"
TRAINING_CONFIG_VERSION = "m5-lora-training-config-v1"

CANONICAL_SOURCE_REPOSITORY = "Qwen/Qwen3-8B-Base"
CANONICAL_SOURCE_REVISION = "7b8a267e13df1a9427e7dfa2691f69a417c58d94"
CANONICAL_SOURCE_MANIFEST_HASH = "507f79d4086e495f0852327e79ea6a4daa53afe2beb591a0fd8489dc16fe8397"
CANONICAL_MLX_BASE_IDENTITY_HASH = "d07ae738ad42baadb62b16115f6b2d90c32fbaa859acc81a4a0a95195e833c80"
CANONICAL_BENCHMARK_MANIFEST_HASH = "f3933caa5ba4432b9631b989a584d999022a102fe1cbb6b113b51da63ff22b85"
CANONICAL_TRAIN_SPLIT_HASH = "c2f49ea90cd171089fb7aa50b503d0d738c5471ea9cd8808d06fd0f02aec908f"
CANONICAL_TRAIN_EXAMPLE_IDS_HASH = "2e5b3bbb1d8a448e54df2b43d6bb5c5e9bfef2812b94e08c1091950a37e22e8c"
CANONICAL_EXPERIMENTAL_CONTRACT_HASH = "87dae4d0b51572519f5b82158ef73c2832d8e09c035b0a7e5f1aaa17cb9d5e47"
CANONICAL_TRAINING_ISOLATION_AUDIT_HASH = "d189b5aca1d682b951f9997518a3d552b41c6cde80530a5087701a61a6e20878"
CANONICAL_TRAINING_FORMATTER_HASH = "b7eefc912755d029d59d2b075b757111cd0c4910d15886f9befbac060b2cfb46"
CANONICAL_LORA_POLICY_HASH = "e0a7cfe5d402150cb0744a6d638e1d03a592f2fa8390fdee890ec96eda272607"
CANONICAL_MODULE_INSPECTION_HASH = "63c3c9995162c077950a0373dff346d5622cb525b68e961933f90f958d19eab0"
CANONICAL_RUNTIME_VERSIONS = {
    "python_version": "3.12.13",
    "macos_version": "26.5.2",
    "machine_architecture": "arm64",
    "mlx_version": "0.32.0",
    "mlx_lm_version": "0.31.3",
}
CANONICAL_TOKENIZER_IDENTITY = {
    "class": "Qwen2Tokenizer",
    "chat_template_present": True,
    "chat_template_sha256": "b2d3c4d81cc138ef3b0c1b8f9c3ac54f15f1ca38ada2eef4c3a8690baece0074",
}
CANONICAL_BASE_QUANTIZATION = {
    "bits": 4,
    "group_size": 64,
    "mode": "affine",
}
CANONICAL_SEED = 1729

DEFAULT_ALPHA_MULTIPLIER = 2
RANK_CANDIDATES = (4, 8, 16)
LEARNING_RATE_CANDIDATES = (5e-6, 1e-5, 2e-5)
TRAINING_DURATION_CANDIDATES = (250, 500, 1000)
DEFAULT_TARGET_POLICY = "POLICY_B_ATTN"
INSPECTION_POLICY_ID_MAP = {
    "POLICY_A_QV": "POLICY_A_Q_V_ONLY",
    "POLICY_B_ATTN": "POLICY_B_ALL_ATTENTION",
    "POLICY_C_ATTN_MLP": "POLICY_C_ATTENTION_PLUS_MLP",
}
DEFAULT_RANK = 8
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_TRAINING_ITERS = 500
DEFAULT_SAVE_EVERY = 100
DEFAULT_MAX_RETAINED_CHECKPOINTS = 10
DEFAULT_MICRO_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_EFFECTIVE_BATCH_SIZE = DEFAULT_MICRO_BATCH_SIZE * DEFAULT_GRADIENT_ACCUMULATION_STEPS
DEFAULT_MAX_SEQUENCE_LENGTH = 128
DEFAULT_ADAMW_BETAS = (0.9, 0.999)
DEFAULT_ADAMW_EPS = 1e-8
DEFAULT_ADAMW_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIPPING = None
DEFAULT_GRADIENT_CHECKPOINTING = False
DEFAULT_DROPOUT = 0.0
DEFAULT_ALPHA_RULE = "alpha = 2 * r"
DEFAULT_PRECISION = {
    "base_representation": "quantized_q4_affine_g64",
    "adapter_dtype": "float32",
    "optimizer_state_dtype": "float32",
}


@dataclass(frozen=True, slots=True)
class SequenceLengthAudit:
    minimum: int
    median: float
    p90: float
    p95: float
    p99: float
    maximum: int
    assistant_target_minimum: int
    assistant_target_median: float
    assistant_target_maximum: int
    max_sequence_length: int
    truncated_training_records: int
    truncated_assistant_targets: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "maximum": self.maximum,
            "assistant_target_minimum": self.assistant_target_minimum,
            "assistant_target_median": self.assistant_target_median,
            "assistant_target_maximum": self.assistant_target_maximum,
            "max_sequence_length": self.max_sequence_length,
            "truncated_training_records": self.truncated_training_records,
            "truncated_assistant_targets": self.truncated_assistant_targets,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sorted_train_examples(benchmark_dir: Path) -> list[BenchmarkExample]:
    train_path = Path(benchmark_dir) / "train.json"
    raw = json.loads(train_path.read_text(encoding="utf-8"))
    examples = sorted((BenchmarkExample.from_dict(item) for item in raw), key=lambda ex: ex.example_id)
    if len(examples) != 300:
        raise ValueError(f"canonical train split must contain 300 examples, found {len(examples)}")
    return examples


def _example_id_hash(example_ids: Sequence[str]) -> str:
    text = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_length_audit(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    runtime_provenance_path: Path,
    source_snapshot_path: Path | None,
    max_sequence_length: int,
) -> tuple[SequenceLengthAudit, dict[str, Any], list[dict[str, Any]]]:
    prompt_contract = load_prompt_contract(prompt_config)
    provenance = _load_json(runtime_provenance_path)
    runtime_paths = resolve_m5_runtime_paths(
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_paths.source_snapshot_path,
        trust_remote_code=True,
    )
    examples = _sorted_train_examples(benchmark_dir)

    full_lengths: list[int] = []
    assistant_lengths: list[int] = []
    records: list[dict[str, Any]] = []
    truncated_training_records = 0
    truncated_assistant_targets = 0

    for example in examples:
        messages = build_training_messages(example=example, prompt_contract=prompt_contract)
        full = tokenizer.apply_chat_template(messages, tools=None, return_dict=False)
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tools=None, add_generation_prompt=True, return_dict=False
        )
        full_len = len(full)
        assistant_len = full_len - len(prefix)
        if full_len > max_sequence_length:
            truncated_training_records += 1
            truncated_assistant_targets += 1
        full_lengths.append(full_len)
        assistant_lengths.append(assistant_len)
        records.append(
            {
                "example_id": example.example_id,
                "full_length": full_len,
                "assistant_target_length": assistant_len,
            }
        )

    audit = SequenceLengthAudit(
        minimum=min(full_lengths),
        median=statistics.median(full_lengths),
        p90=statistics.quantiles(full_lengths, n=10, method="inclusive")[8],
        p95=statistics.quantiles(full_lengths, n=20, method="inclusive")[18],
        p99=statistics.quantiles(full_lengths, n=100, method="inclusive")[98],
        maximum=max(full_lengths),
        assistant_target_minimum=min(assistant_lengths),
        assistant_target_median=statistics.median(assistant_lengths),
        assistant_target_maximum=max(assistant_lengths),
        max_sequence_length=max_sequence_length,
        truncated_training_records=truncated_training_records,
        truncated_assistant_targets=truncated_assistant_targets,
    )
    summary = {
        "tokenizer_identity": provenance["source_lineage"]["tokenizer_identity"],
        "chat_template_sha256": provenance["source_lineage"]["tokenizer_identity"]["chat_template_sha256"],
        "records_analyzed": len(examples),
    }
    return audit, summary, records


def _candidate_policy_from_inspection(module_inspection: Mapping[str, Any], policy_id: str) -> dict[str, Any]:
    lookup_id = INSPECTION_POLICY_ID_MAP.get(policy_id, policy_id)
    for candidate in module_inspection["candidate_target_policies"]:
        if candidate["policy_id"] == lookup_id:
            return candidate
    raise KeyError(lookup_id)


def _canonical_duration_checkpoint_candidates() -> list[dict[str, Any]]:
    return [
        {
            "iters": 250,
            "eligible_checkpoint_steps": [100, 200, 250],
        },
        {
            "iters": 500,
            "eligible_checkpoint_steps": [100, 200, 300, 400, 500],
        },
        {
            "iters": 1000,
            "eligible_checkpoint_steps": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        },
    ]


def build_lora_training_config_artifact(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    module_inspection_path: Path,
    lora_policy_path: Path,
    training_formatter_path: Path,
    runtime_provenance_path: Path,
    source_snapshot_path: Path | None = None,
    experimental_contract_hash: str,
    training_isolation_audit_hash: str,
) -> dict[str, Any]:
    """Build the frozen M5 LoRA training configuration artifact."""

    benchmark_dir = Path(benchmark_dir)
    prompt_config = Path(prompt_config)
    module_inspection_path = Path(module_inspection_path)
    lora_policy_path = Path(lora_policy_path)
    training_formatter_path = Path(training_formatter_path)
    runtime_provenance_path = Path(runtime_provenance_path)

    module_inspection = _load_json(module_inspection_path)
    lora_policy = _load_json(lora_policy_path)
    formatter = _load_json(training_formatter_path)
    provenance = _load_json(runtime_provenance_path)

    token_audit, token_summary, per_example_lengths = _token_length_audit(
        benchmark_dir=benchmark_dir,
        prompt_config=prompt_config,
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
        max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
    )

    train_examples = _sorted_train_examples(benchmark_dir)
    training_example_ids = [example.example_id for example in train_examples]
    training_example_ids_hash = _example_id_hash(training_example_ids)
    training_data_hash = formatter["aggregate_record_manifest_hash"]
    _candidate_policy_from_inspection(module_inspection, DEFAULT_TARGET_POLICY)
    default_policy_candidate = next(
        candidate
        for candidate in lora_policy["candidate_target_policies"]
        if candidate["policy_id"] == DEFAULT_TARGET_POLICY
    )

    canonical_default_candidate = {
        "candidate_id": f"{DEFAULT_TARGET_POLICY}_r{DEFAULT_RANK}_lr{DEFAULT_LEARNING_RATE:.0e}_iters{DEFAULT_TRAINING_ITERS}",
        "target_policy": DEFAULT_TARGET_POLICY,
        "target_modules": default_policy_candidate["target_module_templates"],
        "layer_coverage_mode": "all_36_transformer_blocks",
        "num_layers": 36,
        "covered_layer_indices": list(range(36)),
        "rank": DEFAULT_RANK,
        "alpha": DEFAULT_ALPHA_MULTIPLIER * DEFAULT_RANK,
        "dropout": DEFAULT_DROPOUT,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "training_duration_iters": DEFAULT_TRAINING_ITERS,
        "selected_checkpoint_step": DEFAULT_TRAINING_ITERS,
        "trainable_parameter_count": default_policy_candidate["trainable_parameter_table"][1]["adapter_parameter_count"],
        "trainable_percentage_of_total_model_parameters": default_policy_candidate["trainable_parameter_table"][1]["trainable_percentage_of_total_model_parameters"],
    }

    artifacts = {
        "schema_version": TRAINING_CONFIG_SCHEMA_VERSION,
        "training_config_version": TRAINING_CONFIG_VERSION,
        "gate": "M5_LORA_TRAINING_CONFIG_READY",
        "frozen_inputs": {
            "experimental_contract_hash": experimental_contract_hash,
            "training_isolation_audit_hash": training_isolation_audit_hash,
            "training_formatter_hash": CANONICAL_TRAINING_FORMATTER_HASH,
            "lora_policy_hash": CANONICAL_LORA_POLICY_HASH,
            "module_inspection_hash": CANONICAL_MODULE_INSPECTION_HASH,
            "benchmark_manifest_hash": CANONICAL_BENCHMARK_MANIFEST_HASH,
            "benchmark_train_split_hash": CANONICAL_TRAIN_SPLIT_HASH,
            "training_example_ids_hash": CANONICAL_TRAIN_EXAMPLE_IDS_HASH,
            "source_lineage": {
                "repository": CANONICAL_SOURCE_REPOSITORY,
                "revision": CANONICAL_SOURCE_REVISION,
                "source_manifest_hash": CANONICAL_SOURCE_MANIFEST_HASH,
                "canonical_mlx_base_identity_hash": CANONICAL_MLX_BASE_IDENTITY_HASH,
            },
            "runtime_versions": CANONICAL_RUNTIME_VERSIONS,
            "tokenizer_identity": CANONICAL_TOKENIZER_IDENTITY,
            "base_quantization": CANONICAL_BASE_QUANTIZATION,
        },
        "canonical_training_defaults": canonical_default_candidate,
        "optimizer_policy": {
            "family": "adamw",
            "beta1": DEFAULT_ADAMW_BETAS[0],
            "beta2": DEFAULT_ADAMW_BETAS[1],
            "epsilon": DEFAULT_ADAMW_EPS,
            "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY,
            "framework_default_note": "Matches mlx.optimizers.AdamW defaults in the frozen MLX package.",
        },
        "scheduler_policy": {
            "type": "constant",
            "warmup_policy": "none",
            "warmup_steps": 0,
            "warmup_ratio": None,
            "minimum_learning_rate_policy": "not_applicable_constant_schedule",
            "framework_default_note": "mlx_lm.lora uses a constant LR when lr_schedule is omitted.",
        },
        "batching_policy": {
            "micro_batch_size": DEFAULT_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": DEFAULT_EFFECTIVE_BATCH_SIZE,
            "applies_to_all_candidates": True,
        },
        "sequence_length_policy": {
            "max_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
            "token_length_distribution": token_audit.to_dict(),
            "token_length_audit_method": "Frozen tokenizer + frozen chat template over all 300 TRAIN examples",
            "assistant_target_truncations": token_audit.truncated_assistant_targets,
            "training_record_truncations": token_audit.truncated_training_records,
            "tokenizer_identity": token_summary["tokenizer_identity"],
            "chat_template_sha256": token_summary["chat_template_sha256"],
        },
        "numeric_and_gradient_policy": {
            "precision": DEFAULT_PRECISION,
            "gradient_clipping": DEFAULT_GRADIENT_CLIPPING,
            "gradient_checkpointing": DEFAULT_GRADIENT_CHECKPOINTING,
            "seed_policy": {
                "canonical_seed": CANONICAL_SEED,
                "parameter_initialization_seed": CANONICAL_SEED,
                "data_order_seed": CANONICAL_SEED,
                "framework_rng_seed": CANONICAL_SEED,
                "numpy_rng_seed": CANONICAL_SEED,
            },
            "base_representation": {
                "identity_hash": CANONICAL_MLX_BASE_IDENTITY_HASH,
                "quantization": CANONICAL_BASE_QUANTIZATION,
            },
        },
        "checkpoint_policy": {
            "save_every": DEFAULT_SAVE_EVERY,
            "maximum_retained_checkpoints": DEFAULT_MAX_RETAINED_CHECKPOINTS,
            "final_checkpoint_handling": "always_save_final_adapter_and_keep_predeclared_intermediate_checkpoints",
            "adapter_artifact_format": [
                "adapters.safetensors",
                "adapter_config.json",
                "{step:07d}_adapters.safetensors",
            ],
            "predeclared_duration_checkpoint_candidates": _canonical_duration_checkpoint_candidates(),
        },
        "allowed_candidate_dimensions": {
            "target_policies": ["POLICY_A_QV", "POLICY_B_ATTN", "POLICY_C_ATTN_MLP"],
            "ranks": list(RANK_CANDIDATES),
            "learning_rates": list(LEARNING_RATE_CANDIDATES),
            "training_durations": list(TRAINING_DURATION_CANDIDATES),
        },
        "candidate_trainable_parameter_tables": {
            candidate["policy_id"]: candidate["trainable_parameter_table"]
            for candidate in lora_policy["candidate_target_policies"]
        },
        "training_manifest_schema": {
            "required_fields": [
                "run_id",
                "candidate_id",
                "starting_git_commit",
                "experimental_contract_hash",
                "training_isolation_audit_hash",
                "training_formatter_hash",
                "lora_policy_hash",
                "module_inspection_hash",
                "benchmark_manifest_hash",
                "benchmark_train_split_hash",
                "source_repository",
                "source_revision",
                "source_manifest_hash",
                "mlx_base_identity_hash",
                "tokenizer_identity",
                "chat_template_identity",
                "framework_versions",
                "python_version",
                "macos_version",
                "machine_architecture",
                "target_policy",
                "target_modules",
                "layer_coverage",
                "rank",
                "alpha",
                "dropout",
                "total_parameter_count",
                "trainable_parameter_count",
                "trainable_percentage",
                "optimizer",
                "optimizer_hyperparameters",
                "learning_rate",
                "weight_decay",
                "scheduler",
                "warmup",
                "seed",
                "gradient_checkpointing",
                "gradient_clipping",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "max_sequence_length",
                "truncation_audit",
                "training_examples",
                "training_example_ids_hash",
                "training_data_hash",
                "training_duration_iters",
                "selected_checkpoint_step",
                "checkpoint_policy",
                "adapter_output_path",
                "adapter_file_manifest",
                "adapter_sha256",
                "completion_status",
                "failure_status",
                "failure_reason",
            ],
            "optional_observational_fields": [
                "start_runtime_timestamp",
                "end_runtime_timestamp",
                "training_duration_seconds",
                "peak_memory_gb",
            ],
        },
        "provenance_validation_policy": {
            "reject_on_non_search_field_drift": True,
            "reject_on_unexpected_fields": True,
            "frozen_layer_coverage": {
                "layer_coverage_mode": "all_36_transformer_blocks",
                "num_layers": 36,
                "covered_layer_indices": list(range(36)),
            },
            "frozen_fields": {
                "experimental_contract_hash": experimental_contract_hash,
                "training_isolation_audit_hash": training_isolation_audit_hash,
                "training_formatter_hash": CANONICAL_TRAINING_FORMATTER_HASH,
                "lora_policy_hash": CANONICAL_LORA_POLICY_HASH,
                "module_inspection_hash": CANONICAL_MODULE_INSPECTION_HASH,
                "benchmark_manifest_hash": CANONICAL_BENCHMARK_MANIFEST_HASH,
                "benchmark_train_split_hash": CANONICAL_TRAIN_SPLIT_HASH,
                "training_example_ids_hash": CANONICAL_TRAIN_EXAMPLE_IDS_HASH,
                "source_repository": CANONICAL_SOURCE_REPOSITORY,
                "source_revision": CANONICAL_SOURCE_REVISION,
                "source_manifest_hash": CANONICAL_SOURCE_MANIFEST_HASH,
                "mlx_base_identity_hash": CANONICAL_MLX_BASE_IDENTITY_HASH,
                "layer_coverage": {
                    "layer_coverage_mode": "all_36_transformer_blocks",
                    "num_layers": 36,
                    "covered_layer_indices": list(range(36)),
                },
                "optimizer": "adamw",
                "optimizer_hyperparameters": {
                    "beta1": DEFAULT_ADAMW_BETAS[0],
                    "beta2": DEFAULT_ADAMW_BETAS[1],
                    "epsilon": DEFAULT_ADAMW_EPS,
                },
                "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY,
                "scheduler": "constant",
                "warmup": {"steps": 0, "ratio": None},
                "micro_batch_size": DEFAULT_MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": DEFAULT_EFFECTIVE_BATCH_SIZE,
                "max_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
                "dropout": DEFAULT_DROPOUT,
                "seed": CANONICAL_SEED,
                "gradient_checkpointing": DEFAULT_GRADIENT_CHECKPOINTING,
                "gradient_clipping": DEFAULT_GRADIENT_CLIPPING,
            },
            "whitelisted_search_fields": {
                "target_policy": ["POLICY_A_QV", "POLICY_B_ATTN", "POLICY_C_ATTN_MLP"],
                "rank": list(RANK_CANDIDATES),
                "learning_rate": list(LEARNING_RATE_CANDIDATES),
                "training_duration_iters": list(TRAINING_DURATION_CANDIDATES),
                "selected_checkpoint_step": [100, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
            },
            "target_policy_to_modules": {
                "POLICY_A_QV": [
                    "model.layers.{i}.self_attn.q_proj",
                    "model.layers.{i}.self_attn.v_proj",
                ],
                "POLICY_B_ATTN": [
                    "model.layers.{i}.self_attn.q_proj",
                    "model.layers.{i}.self_attn.k_proj",
                    "model.layers.{i}.self_attn.v_proj",
                    "model.layers.{i}.self_attn.o_proj",
                ],
                "POLICY_C_ATTN_MLP": [
                    "model.layers.{i}.self_attn.q_proj",
                    "model.layers.{i}.self_attn.k_proj",
                    "model.layers.{i}.self_attn.v_proj",
                    "model.layers.{i}.self_attn.o_proj",
                    "model.layers.{i}.mlp.gate_proj",
                    "model.layers.{i}.mlp.up_proj",
                    "model.layers.{i}.mlp.down_proj",
                ],
            },
        },
        "training_data_provenance": {
            "training_example_count": len(train_examples),
            "training_example_ids_hash": training_example_ids_hash,
            "training_data_hash": training_data_hash,
            "training_examples_source": "frozen TRAIN split only; ordered by example_id",
            "training_examples_preview": training_example_ids[:5],
        },
        "default_candidate_provenance": {
            "target_policy": DEFAULT_TARGET_POLICY,
            "target_modules": default_policy_candidate["target_module_templates"],
            "rank": DEFAULT_RANK,
            "alpha": 2 * DEFAULT_RANK,
            "dropout": DEFAULT_DROPOUT,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "training_duration_iters": DEFAULT_TRAINING_ITERS,
            "selected_checkpoint_step": DEFAULT_TRAINING_ITERS,
            "trainable_parameter_count": default_policy_candidate["trainable_parameter_table"][1]["adapter_parameter_count"],
            "trainable_percentage": default_policy_candidate["trainable_parameter_table"][1]["trainable_percentage_of_total_model_parameters"],
        },
        "training_data_audit": per_example_lengths,
        "provenance_drift_examples": {
            "wrong_source_revision": "reject",
            "wrong_mlx_base_hash": "reject",
            "wrong_formatter_hash": "reject",
            "wrong_target_policy_definition": "reject",
            "wrong_dropout": "reject",
            "wrong_layer_coverage": "reject",
            "wrong_seed": "reject",
            "wrong_optimizer": "reject",
            "wrong_scheduler": "reject",
        },
    }

    artifacts["config_hash"] = sha256_bytes(canonical_json_bytes(artifacts))
    return artifacts


def validate_training_run_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Reject any candidate run whose frozen fields drift from the config."""

    policy = config["provenance_validation_policy"]
    if policy.get("reject_on_unexpected_fields") is not True:
        raise ValueError("provenance validation policy must reject unexpected fields")

    required_fields = set(config["training_manifest_schema"]["required_fields"])
    allowed_optional = set(config["training_manifest_schema"].get("optional_observational_fields", ()))
    frozen_fields = policy["frozen_fields"]
    search_fields = policy["whitelisted_search_fields"]
    target_policy_to_modules = policy["target_policy_to_modules"]

    missing = required_fields - set(manifest)
    if missing:
        raise ValueError(f"training manifest is missing required fields: {sorted(missing)}")

    allowed_fields = required_fields | allowed_optional
    unknown = set(manifest) - allowed_fields
    if unknown:
        raise ValueError(f"training manifest has unexpected fields: {sorted(unknown)}")

    for key, expected in frozen_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"training manifest drifted for frozen field {key}")

    if manifest.get("layer_coverage") != policy["frozen_layer_coverage"]:
        raise ValueError("training manifest layer coverage drifted from the frozen policy")

    for key, allowed in search_fields.items():
        if key not in manifest:
            raise ValueError(f"training manifest is missing search field {key}")
        if manifest[key] not in allowed:
            raise ValueError(f"training manifest search field {key} is outside the whitelist")

    target_policy = manifest["target_policy"]
    expected_modules = target_policy_to_modules.get(target_policy)
    if expected_modules is None:
        raise ValueError(f"unknown target policy {target_policy}")
    if list(manifest["target_modules"]) != expected_modules:
        raise ValueError("training manifest target_modules do not match the declared target policy")

    duration = int(manifest["training_duration_iters"])
    if duration not in TRAINING_DURATION_CANDIDATES:
        raise ValueError("training duration is outside the whitelist")
    selected_step = int(manifest["selected_checkpoint_step"])
    eligible_steps = {
        row["iters"]: set(row["eligible_checkpoint_steps"])
        for row in config["checkpoint_policy"]["predeclared_duration_checkpoint_candidates"]
    }[duration]
    if selected_step not in eligible_steps:
        raise ValueError("selected checkpoint step is not eligible for the chosen training duration")

    if int(manifest["max_sequence_length"]) != DEFAULT_MAX_SEQUENCE_LENGTH:
        raise ValueError("max sequence length drifted from the frozen policy")

    trunc = manifest.get("truncation_audit", {})
    if trunc.get("assistant_target_truncations") != 0:
        raise ValueError("assistant target truncation must remain zero")

    if manifest.get("optimizer") != "adamw":
        raise ValueError("optimizer drifted from the frozen policy")
    if manifest.get("scheduler") != "constant":
        raise ValueError("scheduler drifted from the frozen policy")

    rank = int(manifest["rank"])
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is outside the whitelist")

    alpha = int(manifest["alpha"])
    if alpha != DEFAULT_ALPHA_MULTIPLIER * rank:
        raise ValueError("alpha must satisfy the frozen scaling rule alpha = 2 * rank")

    candidate_tables = config["candidate_trainable_parameter_tables"]
    target_table = candidate_tables[target_policy]
    matching_row = None
    for row in target_table:
        if int(row["rank"]) == rank:
            matching_row = row
            break
    if matching_row is None:
        raise ValueError("trainable-parameter table has no entry for the declared rank")

    if int(manifest["trainable_parameter_count"]) != int(matching_row["adapter_parameter_count"]):
        raise ValueError("trainable parameter count drifted from the frozen table")
    if float(manifest["trainable_percentage"]) != float(
        matching_row["trainable_percentage_of_total_model_parameters"]
    ):
        raise ValueError("trainable percentage drifted from the frozen table")


def write_lora_training_config_artifact(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    module_inspection_path: Path,
    lora_policy_path: Path,
    training_formatter_path: Path,
    runtime_provenance_path: Path,
    source_snapshot_path: Path | None = None,
    experimental_contract_hash: str,
    training_isolation_audit_hash: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write the frozen training config artifact and return it."""

    artifact = build_lora_training_config_artifact(
        benchmark_dir=benchmark_dir,
        prompt_config=prompt_config,
        module_inspection_path=module_inspection_path,
        lora_policy_path=lora_policy_path,
        training_formatter_path=training_formatter_path,
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
        experimental_contract_hash=experimental_contract_hash,
        training_isolation_audit_hash=training_isolation_audit_hash,
    )
    write_json(Path(output_path), artifact)
    return artifact
