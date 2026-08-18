from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.tuner.datasets import load_local_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.domain.enums import Split
from adaptlab.evaluation.runner import load_benchmark_split
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.m5.lora_policy import audit_trainable_parameter_names
from adaptlab.m5.model_preflight import (
    preflight_m5_local_model,
    format_m5_local_model_preflight_failure,
)
from adaptlab.m5.smoke_training import (
    _flatten_trainable_parameters,
    _trainable_parameter_names,
    _train_one_smoke_cycle,
    _load_model_and_tokenizer,
    _snapshot_parameter_tree,
    _hash_bytes,
)
from adaptlab.m5.training_formatter import build_training_messages
from adaptlab.m5.training_config import (
    CANONICAL_SEED,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _candidate_from_policy(selection_policy: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = selection_policy["candidate_budget"]["candidate_records"]
    matches = [row for row in rows if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate_id {candidate_id!r} is not uniquely present in frozen budget")
    return matches[0]


def _materialize_full_train(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    output_dir: Path,
) -> int:
    examples = sorted(
        load_benchmark_split(benchmark_dir, Split.train),
        key=lambda ex: ex.example_id,
    )
    if len(examples) != 300:
        raise ValueError(f"expected 300 TRAIN examples, found {len(examples)}")

    prompt_contract = load_prompt_contract(prompt_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"

    lines = []
    for example in examples:
        messages = build_training_messages(
            example=example,
            prompt_contract=prompt_contract,
        )
        lines.append(
            json.dumps(
                {"messages": messages},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(examples)


def run_candidate_training(
    *,
    candidate_id: str,
    benchmark_dir: Path,
    prompt_config: Path,
    runtime_provenance_path: Path,
    lora_policy_path: Path,
    selection_policy_path: Path,
    output_dir: Path,
    source_snapshot_path: Path | None = None,
    mlx_base_path: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    prompt_config = Path(prompt_config)
    runtime_provenance_path = Path(runtime_provenance_path)
    lora_policy_path = Path(lora_policy_path)
    selection_policy_path = Path(selection_policy_path)
    output_dir = Path(output_dir)

    selection_policy = _load_json(selection_policy_path)
    lora_policy = _load_json(lora_policy_path)
    candidate = _candidate_from_policy(selection_policy, candidate_id)

    preflight = preflight_m5_local_model(
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
        mlx_base_path=mlx_base_path,
    )
    if not preflight.ready:
        raise ValueError(format_m5_local_model_preflight_failure(preflight))

    mlx_model_path = preflight.runtime_paths.mlx_base_path

    model, tokenizer, model_config = _load_model_and_tokenizer(mlx_model_path)
    if model_config.get("model_type") != "qwen3":
        raise ValueError("loaded base is not canonical Qwen3 MLX lineage")

    mx.random.seed(CANONICAL_SEED)
    np.random.seed(CANONICAL_SEED)

    model.freeze()

    frozen_target_modules = list(candidate["target_modules"])
    mlx_lora_keys = []
    prefix = "model.layers.{i}."
    for module in frozen_target_modules:
        if not module.startswith(prefix):
            raise ValueError(
                f"unexpected frozen target module template: {module!r}"
            )
        mlx_lora_keys.append(module[len(prefix):])

    mlx_lora_scale = float(candidate["alpha"]) / int(candidate["rank"])

    lora_config = {
        "rank": int(candidate["rank"]),
        "scale": mlx_lora_scale,
        "dropout": 0.0,
        "keys": mlx_lora_keys,
    }

    linear_to_lora_layers(
        model,
        36,
        lora_config,
        use_dora=False,
    )
    mx.eval(model.parameters())

    trainable_names = _trainable_parameter_names(model)
    trainable_count = sum(
        value.size for value in _flatten_trainable_parameters(model).values()
    )

    policy = next(
        row
        for row in lora_policy["candidate_target_policies"]
        if row["policy_id"] == candidate["target_policy"]
    )

    audit = audit_trainable_parameter_names(
        trainable_names,
        allowed_adapter_namespace_prefixes=policy["expected_adapter_namespace_prefixes"],
    )
    if not audit["passed"]:
        raise ValueError(
            f"unexpected trainable parameters: "
            f"{audit['unexpected_trainable_parameter_names']}"
        )

    expected_trainable = int(candidate["trainable_parameter_count"])
    if trainable_count != expected_trainable:
        raise ValueError(
            f"trainable parameter count mismatch: "
            f"expected {expected_trainable}, got {trainable_count}"
        )

    # Sample only base parameters that are never LoRA targets under any
    # frozen M5 policy. Targeted Linear modules are wrapped by MLX LoRA,
    # which changes their parameter paths even though their base weights
    # remain frozen.
    frozen_snapshot_names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.17.self_attn.q_norm.weight",
        "model.layers.35.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    base_before = _snapshot_parameter_tree(model, frozen_snapshot_names)

    adapter_before = _flatten_trainable_parameters(model)
    adapter_before_hashes = {
        name: _hash_bytes(value)
        for name, value in adapter_before.items()
    }

    dataset_dir = output_dir / "dataset"
    train_count = _materialize_full_train(
        benchmark_dir=benchmark_dir,
        prompt_config=prompt_config,
        output_dir=dataset_dir,
    )

    data_args = types.SimpleNamespace(mask_prompt=True)
    train_set, valid_set, test_set = load_local_dataset(
        dataset_dir,
        tokenizer,
        data_args,
    )

    if len(valid_set) != 0 or len(test_set) != 0:
        raise ValueError("candidate dataset must contain TRAIN only")

    training_metrics = _train_one_smoke_cycle(
        model=model,
        train_set=train_set,
        batch_size=DEFAULT_MICRO_BATCH_SIZE,
        iters=int(candidate["training_duration_iters"]),
        grad_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        max_seq_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        learning_rate=float(candidate["learning_rate"]),
    )

    adapter_after = _flatten_trainable_parameters(model)
    changed_adapter_names = [
        name
        for name in sorted(adapter_before_hashes)
        if adapter_before_hashes[name] != _hash_bytes(adapter_after[name])
    ]

    if not changed_adapter_names:
        raise ValueError("adapter parameters did not change during training")

    base_after = _snapshot_parameter_tree(model, frozen_snapshot_names)
    base_immutable = all(
        base_before[name].sha256 == base_after[name].sha256
        for name in frozen_snapshot_names
    )

    if not base_immutable:
        raise ValueError("sampled frozen base parameters changed")

    if not training_metrics["loss_finite"]:
        raise ValueError("TRAINING_FAILED: non-finite training loss")
    if not training_metrics["gradient_finite"]:
        raise ValueError("TRAINING_FAILED: non-finite gradients")
    if not training_metrics["optimizer_step_applied"]:
        raise ValueError("TRAINING_FAILED: optimizer never applied an update")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the final trained adapter in the MLX-compatible format already
    # exercised by the M5 smoke path.
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = _flatten_trainable_parameters(model)
    adapter_file = adapter_dir / "adapters.safetensors"
    final_step = int(candidate["training_duration_iters"])
    checkpoint_file = adapter_dir / f"{final_step:07d}_adapters.safetensors"

    mx.save_safetensors(str(adapter_file), adapter_weights)
    mx.save_safetensors(str(checkpoint_file), adapter_weights)

    adapter_config = {
        "model": str(mlx_model_path),
        "train": True,
        "data": str(dataset_dir),
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "mask_prompt": True,
        "num_layers": 36,
        "batch_size": DEFAULT_MICRO_BATCH_SIZE,
        "iters": final_step,
        "val_batches": -1,
        "learning_rate": float(candidate["learning_rate"]),
        "steps_per_report": final_step,
        "steps_per_eval": final_step,
        "grad_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        "resume_adapter_file": None,
        "adapter_path": str(adapter_dir),
        "save_every": final_step,
        "test": False,
        "test_batches": -1,
        "max_seq_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "config": None,
        "grad_checkpoint": False,
        "clear_cache_threshold": 0,
        "lr_schedule": None,
        "lora_parameters": {
            "keys": list(mlx_lora_keys),
            "rank": int(candidate["rank"]),
            "dropout": 0.0,
            "scale": mlx_lora_scale,
        },
        "report_to": None,
        "project_name": None,
        "seed": CANONICAL_SEED,
    }
    save_config(adapter_config, adapter_dir / "adapter_config.json")

    adapter_files = []
    for file_path in (
        adapter_file,
        adapter_dir / "adapter_config.json",
        checkpoint_file,
    ):
        adapter_files.append(
            {
                "file_name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_bytes(file_path.read_bytes()),
            }
        )

    # Location-independent aggregate identity.
    adapter_hash = sha256_bytes(
        canonical_json_bytes({"files": adapter_files})
    )

    loss_trace = {
        "candidate_id": candidate_id,
        "declared_iterations": final_step,
        "recorded_loss_count": len(training_metrics["losses"]),
        "losses": training_metrics["losses"],
    }
    loss_trace_path = output_dir / "training_loss_trace.json"
    loss_trace_path.write_text(
        json.dumps(loss_trace, indent=2) + "\n",
        encoding="utf-8",
    )

    result = {
        "candidate_id": candidate_id,
        "target_policy": candidate["target_policy"],
        "target_modules": candidate["target_modules"],
        "mlx_attachment_keys": mlx_lora_keys,
        "rank": candidate["rank"],
        "alpha": candidate["alpha"],
        "dropout": 0.0,
        "learning_rate": candidate["learning_rate"],
        "declared_iterations": final_step,
        "completed_iterations": len(training_metrics["losses"]),
        "seed": CANONICAL_SEED,
        "training_example_count": train_count,
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_audit": audit,
        "finite_loss": training_metrics["loss_finite"],
        "finite_gradients": training_metrics["gradient_finite"],
        "optimizer_step_applied": training_metrics["optimizer_step_applied"],
        "changed_adapter_parameter_count": len(changed_adapter_names),
        "base_weight_immutability": base_immutable,
        "adapter_dir": str(adapter_dir),
        "adapter_file_manifest": adapter_files,
        "adapter_hash": adapter_hash,
        "loss_trace_path": str(loss_trace_path),
        "status": "COMPLETED",
    }

    if result["completed_iterations"] != final_step:
        raise ValueError(
            f"TRAINING_FAILED: expected {final_step} losses/iterations, "
            f"recorded {result['completed_iterations']}"
        )

    result_path = output_dir / "training_result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    return result
