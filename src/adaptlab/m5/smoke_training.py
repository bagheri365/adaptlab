"""Canonical Milestone 5 LoRA smoke-training runner.

This module performs a disposable infrastructure-only smoke run. It reuses the
frozen Milestone 5 formatter, source lineage, LoRA policy, and selection runner
without launching any canonical candidate run.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import types
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split
from adaptlab.evaluation.inputs import canonical_model_input_bytes, construct_model_input, construct_rag_model_input
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.runner import load_benchmark_split, load_chunks, verify_frozen_benchmark
from adaptlab.evaluation.schemas import AdaptationMethod, ModelInput
from adaptlab.evaluation.scoring import score_output
from adaptlab.m5.lora_policy import audit_trainable_parameter_names
from adaptlab.m5.model_preflight import (
    CANONICAL_MLX_BASE_SEMANTIC_TENSOR_HASH,
    format_m5_local_model_preflight_failure,
    preflight_m5_local_model,
)
from adaptlab.m5.selection_runner import (
    SELECTION_DECISION_FILENAME,
    ValidationSelectionCandidateResult,
    ValidationSelectionExampleResult,
    ValidationSelectionStatus,
    build_validation_selection_decision,
)
from adaptlab.m5.validation_selection import DEFAULT_VALIDATION_FAMILIES
from adaptlab.m5.training_config import (
    CANONICAL_BASE_QUANTIZATION,
    CANONICAL_MLX_BASE_IDENTITY_HASH,
    CANONICAL_RUNTIME_VERSIONS,
    CANONICAL_SEED,
    CANONICAL_SOURCE_MANIFEST_HASH,
    CANONICAL_SOURCE_REPOSITORY,
    CANONICAL_SOURCE_REVISION,
    CANONICAL_TRAIN_EXAMPLE_IDS_HASH,
    CANONICAL_TRAIN_SPLIT_HASH,
    CANONICAL_TRAINING_FORMATTER_HASH,
    CANONICAL_TRAINING_ISOLATION_AUDIT_HASH,
    DEFAULT_ADAMW_BETAS,
    DEFAULT_ADAMW_EPS,
    DEFAULT_ADAMW_WEIGHT_DECAY,
    DEFAULT_DROPOUT,
    DEFAULT_EFFECTIVE_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_GRADIENT_CLIPPING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TARGET_POLICY,
    DEFAULT_TRAINING_ITERS,
    TRAINING_CONFIG_SCHEMA_VERSION,
    TRAINING_CONFIG_VERSION,
)
from adaptlab.m5.training_formatter import (
    build_training_record,
    build_training_messages,
    canonical_assistant_target,
)

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset, load_local_dataset
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load, save_config


SMOKE_RUN_ID = "M5_SMOKE_ONLY"
SMOKE_SCHEMA_VERSION = "m5-lora-training-smoke-artifact-v1"
SMOKE_VERSION = "m5-lora-training-smoke-v1"
SMOKE_TARGET_POLICY = "POLICY_A_QV"
SMOKE_RANK = 4
SMOKE_ALPHA = 8
SMOKE_DROPOUT = 0.0
SMOKE_EXPECTED_TRAINABLE_PARAMETER_COUNT = 1916928
SMOKE_TRAINING_FORMATTER_HASH = "1d04beb66c4f5d81fdcba9f3a9d9e3cfdb243251766784a4f58dee1a4ce9ca60"
SMOKE_SUBSET_SIZE = 4
SMOKE_ITERS = 8
SMOKE_BATCH_SIZE = 1
SMOKE_GRAD_ACCUMULATION_STEPS = 8
SMOKE_MAX_TOKENS = 8
SMOKE_GENERATION_TEMPERATURE = 0.0
SMOKE_GENERATION_TOP_P = 1.0
SMOKE_GENERATION_TOP_K = 0
SMOKE_GENERATION_SEED = CANONICAL_SEED
SMOKE_NEUTRAL_PROMPT = "Reply with the single word: smoke."
SMOKE_NEUTRAL_SYSTEM = "You are a deterministic infrastructure smoke tester."
SMOKE_RAG_VALIDATION_EXAMPLE_ID = "FULL_VA_BKN_001"
SMOKE_FAMILIES = ("behavior_only", "behavior_knowledge", "changed_knowledge", "knowledge_only")
CANONICAL_TRAINING_CONFIG_HASH_V2 = "7555d16029fc006b02d51307414d1ba3fe859974b9564e86cdfa4e2c01006f69"
CANONICAL_TRAINING_CONFIG_HASH_V3 = "4eaa4e003deb7e7d601934fa8bf8a8689b17292b99c6eb9f545978a38735177d"


@dataclass(frozen=True, slots=True)
class SmokeTensorSnapshot:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_bytes(value: Any) -> str:
    try:
        return hashlib.sha256(bytes(value)).hexdigest()
    except Exception:
        arr = np.asarray(value)
        return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _canonical_dict_hash(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_snapshot_files(snapshot_path: Path, file_hashes: Iterable[dict[str, Any]]) -> None:
    for item in file_hashes:
        file_path = snapshot_path / item["file"]
        digest = sha256_bytes(file_path.read_bytes())
        if digest != item["sha256"]:
            raise ValueError(f"snapshot hash mismatch for {item['file']}")


def _select_train_subset(examples: list[BenchmarkExample], *, subset_size: int) -> list[BenchmarkExample]:
    if subset_size % len(SMOKE_FAMILIES) != 0:
        raise ValueError("smoke subset size must be divisible by the number of task families")
    ordered = sorted(examples, key=lambda ex: ex.example_id)
    buckets: dict[str, list[BenchmarkExample]] = {family: [] for family in SMOKE_FAMILIES}
    for example in ordered:
        buckets[str(example.task_family.value)].append(example)
    per_family = subset_size // len(SMOKE_FAMILIES)
    selected: list[BenchmarkExample] = []
    for family in SMOKE_FAMILIES:
        family_examples = buckets[family]
        if len(family_examples) < per_family:
            raise ValueError(f"train split does not contain enough examples for {family}")
        selected.extend(family_examples[:per_family])
    selected = sorted(selected, key=lambda ex: ex.example_id)
    if len(selected) != subset_size:
        raise ValueError("smoke training subset selection drifted")
    return selected


def _materialize_training_subset(
    *,
    examples: list[BenchmarkExample],
    prompt_config: Path,
    training_formatter_hash: str,
    output_dir: Path,
) -> dict[str, Any]:
    prompt_contract = load_prompt_contract(prompt_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train.jsonl"
    messages_records: list[dict[str, Any]] = []
    formatter_records: list[dict[str, Any]] = []
    ids: list[str] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record_payload, record = build_training_record(example=example, prompt_contract=prompt_contract)
            handle.write(json.dumps(record_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            ids.append(example.example_id)
            messages_records.append({"example_id": example.example_id, "messages": record_payload["messages"]})
            formatter_records.append({"example_id": example.example_id, **record.to_dict()})
            if record.input_message_hash != construct_model_input(
                example=example,
                method=AdaptationMethod.PROMPT,
                prompt_contract=prompt_contract,
            ).input_hash:
                raise ValueError("smoke training formatter does not align with the PROMPT evaluation path")
    subset_hash = sha256_bytes(canonical_json_bytes(ids))
    return {
        "subset_size": len(ids),
        "sorted_example_ids": ids,
        "subset_id_hash": subset_hash,
        "jsonl_path": str(jsonl_path),
        "records": messages_records,
        "formatter_records": formatter_records,
        "formatter_hash": training_formatter_hash,
        "prompt_hash": prompt_contract.prompt_hash,
    }


def _snapshot_parameter_tree(model: nn.Module, names: Iterable[str]) -> dict[str, SmokeTensorSnapshot]:
    params = dict(tree_flatten(model.parameters(), destination={}))
    snapshots: dict[str, SmokeTensorSnapshot] = {}
    for name in names:
        value = params[name]
        snapshots[name] = SmokeTensorSnapshot(
            name=name,
            shape=tuple(int(dim) for dim in getattr(value, "shape", ())),
            dtype=str(getattr(value, "dtype", type(value).__name__)),
            sha256=_hash_bytes(value),
        )
    return snapshots


def _flatten_trainable_parameters(model: nn.Module) -> dict[str, Any]:
    return dict(tree_flatten(model.trainable_parameters(), destination={}))


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    return sorted(_flatten_trainable_parameters(model))


def _m5_causal_loss(model, batch, lengths):
    """Causal LM loss with exact prompt masking and no padded target leakage.

    ``lengths[:, 0]`` is the first supervised token position.
    ``lengths[:, 1]`` is the exclusive sequence length.

    For a sequence of length L, valid causal targets are positions 1..L-1,
    so the upper bound must be ``steps < L`` rather than ``steps <= L``.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(
        steps >= lengths[:, 0:1],
        steps < lengths[:, 1:],
    )

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def _train_one_smoke_cycle(
    *,
    model: nn.Module,
    train_set,
    batch_size: int,
    iters: int,
    grad_accumulation_steps: int,
    max_seq_length: int,
    learning_rate: float,
) -> dict[str, Any]:
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        betas=list(DEFAULT_ADAMW_BETAS),
        eps=DEFAULT_ADAMW_EPS,
        weight_decay=DEFAULT_ADAMW_WEIGHT_DECAY,
    )
    loss_and_grad = nn.value_and_grad(model, _m5_causal_loss)
    batches = iterate_batches(
        dataset=CacheDataset(train_set),
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        loop=True,
        seed=SMOKE_GENERATION_SEED,
    )
    losses: list[float] = []
    finite_loss = True
    finite_grad = True
    update_applied = False
    grad_accum = None
    for step in range(1, iters + 1):
        batch = next(batches)
        (loss_value, toks), grad = loss_and_grad(model, *batch)
        mx.eval(loss_value, toks)
        loss_float = float(loss_value)
        losses.append(loss_float)
        finite_loss = bool(finite_loss and np.isfinite(loss_float))
        flat_grad = tree_flatten(grad, destination={})
        step_finite = bool(all(np.isfinite(np.asarray(value)).all() for value in flat_grad.values()))
        finite_grad = bool(finite_grad and step_finite)
        if grad_accum is None:
            grad_accum = grad
        else:
            grad_accum = tree_map(lambda left, right: left + right, grad_accum, grad)
        if step % grad_accumulation_steps == 0:
            update_grad = tree_map(lambda x: x / grad_accumulation_steps, grad_accum)
            optimizer.update(model, update_grad)
            grad_accum = None
            update_applied = bool(True)
    # Match native mlx-lm trainer semantics: incomplete gradient
    # accumulation at the end of a run is not applied.
    if grad_accum is not None:
        grad_accum = None
    mx.eval(model.parameters())
    return {
        "losses": losses,
        "loss_finite": finite_loss,
        "gradient_finite": finite_grad,
        "optimizer_initialized": True,
        "optimizer_step_applied": update_applied,
    }


def _save_adapter_artifact(model: nn.Module, adapter_dir: Path, smoke_args: dict[str, Any]) -> dict[str, Any]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = _flatten_trainable_parameters(model)
    adapter_file = adapter_dir / "adapters.safetensors"
    checkpoint_file = adapter_dir / f"{SMOKE_ITERS:07d}_adapters.safetensors"
    mx.save_safetensors(str(adapter_file), adapter_weights)
    mx.save_safetensors(str(checkpoint_file), adapter_weights)
    save_config(smoke_args, adapter_dir / "adapter_config.json")
    files = []
    for path in [adapter_file, adapter_dir / "adapter_config.json", checkpoint_file]:
        files.append(
            {
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    aggregate = _canonical_dict_hash({"files": files, "adapter_dir": str(adapter_dir)})
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_file_manifest": files,
        "adapter_hash": aggregate,
        "adapter_file_names": [entry["file_name"] for entry in files],
    }


def _generate_text(model: nn.Module, tokenizer, *, system: str, user: str) -> dict[str, Any]:
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=None,
        return_dict=False,
    )
    prompt_hash = sha256_bytes(canonical_json_bytes({"system": system, "user": user}))
    sampler = make_sampler(
        SMOKE_GENERATION_TEMPERATURE,
        top_p=SMOKE_GENERATION_TOP_P,
        top_k=SMOKE_GENERATION_TOP_K,
    )
    text = generate(
        model,
        tokenizer,
        prompt_tokens,
        verbose=False,
        max_tokens=SMOKE_MAX_TOKENS,
        sampler=sampler,
    )
    return {
        "input_hash": prompt_hash,
        "prompt_tokens": prompt_tokens,
        "raw_output": text,
    }


def _load_model_and_tokenizer(model_path: Path):
    model, tokenizer, config = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": True},
        return_config=True,
    )
    return model, tokenizer, config


def _freeze_smoke_candidate_result(
    *,
    validation_path: Path,
    prompt_config: Path,
    training_config: dict[str, Any],
) -> ValidationSelectionCandidateResult:
    examples = sorted(load_benchmark_split(validation_path.parent, Split.validation), key=lambda ex: ex.example_id)
    prompt_contract = load_prompt_contract(prompt_config)
    results = []
    for example in examples:
        scored = score_output(example, canonical_assistant_target(example.expected_output))
        results.append(
            ValidationSelectionExampleResult(
                example_id=example.example_id,
                task_family=example.task_family.value,
                raw_output=canonical_assistant_target(example.expected_output),
                normalized_output=scored.normalized_output,
                correct=True,
            )
        )
    per_family = {}
    for family in DEFAULT_VALIDATION_FAMILIES:
        family_results = [r for r in results if r.task_family == family]
        correct = sum(1 for r in family_results if r.correct)
        n = len(family_results)
        frac = Fraction(correct, n)
        per_family[family] = {
            "n": n,
            "correct": correct,
            "accuracy": float(frac),
            "accuracy_fraction": {"numerator": frac.numerator, "denominator": frac.denominator},
        }
    overall = Fraction(sum(1 for r in results if r.correct), len(results))
    macro = sum(Fraction(v["correct"], v["n"]) for v in per_family.values()) / len(per_family)
    aggregate = {
        "overall_correct": len(results),
        "overall_accuracy": float(overall),
        "overall_accuracy_fraction": {"numerator": overall.numerator, "denominator": overall.denominator},
        "per_family": per_family,
        "macro_family_accuracy": float(macro),
        "macro_family_accuracy_fraction": {"numerator": macro.numerator, "denominator": macro.denominator},
    }
    candidate = ValidationSelectionCandidateResult(
        candidate_id=SMOKE_RUN_ID,
        checkpoint_id="ckpt-0000008",
        checkpoint_iteration=SMOKE_ITERS,
        candidate_manifest_hash="0" * 64,
        candidate_search_hash=_canonical_dict_hash({"candidate_budget": "smoke_only"}),
        selection_policy_hash=_canonical_dict_hash({"selection_policy": "smoke_only"}),
        adapter_hash=_canonical_dict_hash({"adapter": "smoke_only"}),
        base_identity_hash=training_config["numeric_and_gradient_policy"]["base_representation"]["identity_hash"],
        validation_split_hash=sha256_bytes((validation_path).read_bytes()),
        validation_example_ids_hash=sha256_bytes(canonical_json_bytes([ex.example_id for ex in examples])),
        source_repository=training_config["frozen_inputs"]["source_lineage"]["repository"],
        source_revision=training_config["frozen_inputs"]["source_lineage"]["revision"],
        source_manifest_hash=training_config["frozen_inputs"]["source_lineage"]["source_manifest_hash"],
        training_formatter_hash=training_config["frozen_inputs"]["training_formatter_hash"],
        lora_policy_hash=training_config["frozen_inputs"]["lora_policy_hash"],
        training_config_hash=training_config["config_hash"],
        seed=training_config["numeric_and_gradient_policy"]["seed_policy"]["canonical_seed"],
        dropout=training_config["default_candidate_provenance"]["dropout"],
        optimizer=training_config["optimizer_policy"]["family"],
        scheduler=training_config["scheduler_policy"]["type"],
        batching=training_config["batching_policy"],
        layer_coverage=training_config["provenance_validation_policy"]["frozen_layer_coverage"],
        rank=SMOKE_RANK,
        target_policy=SMOKE_TARGET_POLICY,
        alpha=SMOKE_ALPHA,
        learning_rate=DEFAULT_LEARNING_RATE,
        training_duration_iters=SMOKE_ITERS,
        eligible_checkpoint_steps=(SMOKE_ITERS,),
        target_modules=("model.layers.{i}.self_attn.q_proj", "model.layers.{i}.self_attn.v_proj"),
        trainable_parameter_count=SMOKE_EXPECTED_TRAINABLE_PARAMETER_COUNT,
        training_steps=SMOKE_ITERS,
        n_total=len(results),
        per_example_results=tuple(results),
        aggregate=aggregate,
        completion_status=ValidationSelectionStatus.VALID,
        provider_runtime_failure_count=0,
        failure_reason=None,
    )
    manifest_hash = sha256_bytes(canonical_json_bytes({
        **candidate.to_dict(),
        "candidate_manifest_hash": "0" * 64,
    }))
    return ValidationSelectionCandidateResult.from_dict(
        {**candidate.to_dict(), "candidate_manifest_hash": manifest_hash}
    )


def run_m5_lora_smoke(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    runtime_provenance_path: Path,
    training_formatter_path: Path,
    training_config_path: Path,
    lora_policy_path: Path,
    module_inspection_path: Path,
    selection_policy_path: Path,
    smoke_output_dir: Path,
    smoke_artifact_path: Path,
    retrieval_artifact_path: Path,
    source_snapshot_path: Path | None = None,
    mlx_base_path: Path | None = None,
    sample_validation_example_id: str = SMOKE_RAG_VALIDATION_EXAMPLE_ID,
    subset_size: int = SMOKE_SUBSET_SIZE,
) -> dict[str, Any]:
    """Run the disposable smoke test and write the frozen artifact."""

    benchmark_dir = Path(benchmark_dir)
    prompt_config = Path(prompt_config)
    runtime_provenance_path = Path(runtime_provenance_path)
    training_formatter_path = Path(training_formatter_path)
    training_config_path = Path(training_config_path)
    lora_policy_path = Path(lora_policy_path)
    module_inspection_path = Path(module_inspection_path)
    selection_policy_path = Path(selection_policy_path)
    smoke_output_dir = Path(smoke_output_dir)
    smoke_artifact_path = Path(smoke_artifact_path)
    retrieval_artifact_path = Path(retrieval_artifact_path)
    source_snapshot_path = Path(source_snapshot_path) if source_snapshot_path is not None else None
    mlx_base_path = Path(mlx_base_path) if mlx_base_path is not None else None

    manifest, benchmark_manifest_hash = verify_frozen_benchmark(benchmark_dir)
    training_config = _load_json(training_config_path)
    formatter_artifact = _load_json(training_formatter_path)
    lora_policy = _load_json(lora_policy_path)
    module_inspection = _load_json(module_inspection_path)
    runtime = _load_json(runtime_provenance_path)

    if training_config["config_hash"] not in {
        "011e2246ec866b4e9fb58ba5dec8baac9946b242d4ec42cd39a5fba4c36b4c56",
        CANONICAL_TRAINING_CONFIG_HASH_V2,
        CANONICAL_TRAINING_CONFIG_HASH_V3,
    }:
        raise ValueError("training config hash drifted from the frozen canonical value")
    if formatter_artifact["formatter_hash"] != SMOKE_TRAINING_FORMATTER_HASH:
        raise ValueError("formatter hash drifted from the frozen canonical value")
    if runtime["source_lineage"]["repository"] != CANONICAL_SOURCE_REPOSITORY:
        raise ValueError("source repository drifted from the frozen canonical value")
    if runtime["source_lineage"]["revision"] != CANONICAL_SOURCE_REVISION:
        raise ValueError("source revision drifted from the frozen canonical value")
    base_identity_hash = training_config["numeric_and_gradient_policy"]["base_representation"]["identity_hash"]
    if base_identity_hash not in {
        CANONICAL_MLX_BASE_IDENTITY_HASH,
        CANONICAL_MLX_BASE_SEMANTIC_TENSOR_HASH,
    }:
        raise ValueError("base identity hash drifted from the frozen canonical value")
    if training_config["frozen_inputs"]["training_formatter_hash"] != CANONICAL_TRAINING_FORMATTER_HASH:
        raise ValueError("training formatter frozen input hash drifted")
    if training_config["frozen_inputs"]["training_isolation_audit_hash"] != CANONICAL_TRAINING_ISOLATION_AUDIT_HASH:
        raise ValueError("training isolation audit hash drifted")
    if training_config["frozen_inputs"]["benchmark_train_split_hash"] != CANONICAL_TRAIN_SPLIT_HASH:
        raise ValueError("training split hash drifted")
    if training_config["frozen_inputs"]["training_example_ids_hash"] != CANONICAL_TRAIN_EXAMPLE_IDS_HASH:
        raise ValueError("training example-ID hash drifted")

    preflight = preflight_m5_local_model(
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
        mlx_base_path=mlx_base_path,
    )
    if not preflight.ready:
        raise ValueError(format_m5_local_model_preflight_failure(preflight))

    snapshot_path = preflight.runtime_paths.source_snapshot_path
    mlx_model_path = preflight.runtime_paths.mlx_base_path
    _verify_snapshot_files(snapshot_path, runtime["source_lineage"]["file_hashes"])

    train_examples = load_benchmark_split(benchmark_dir, Split.train)
    smoke_examples = _select_train_subset(train_examples, subset_size=subset_size)
    subset_manifest = _materialize_training_subset(
        examples=smoke_examples,
        prompt_config=prompt_config,
        training_formatter_hash=SMOKE_TRAINING_FORMATTER_HASH,
        output_dir=smoke_output_dir / "train_subset",
    )

    smoke_data_dir = smoke_output_dir / "dataset"
    smoke_data_dir.mkdir(parents=True, exist_ok=True)
    (smoke_data_dir / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"messages": record["messages"]}, ensure_ascii=False, separators=(",", ":"))
            for record in subset_manifest["records"]
        )
        + "\n",
        encoding="utf-8",
    )

    model, tokenizer, model_config = _load_model_and_tokenizer(mlx_model_path)
    if model_config.get("model_type") != "qwen3":
        raise ValueError("loaded base is not the frozen Qwen3 MLX lineage")
    mx.random.seed(CANONICAL_SEED)
    np.random.seed(CANONICAL_SEED)
    model.freeze()
    lora_config = {"rank": SMOKE_RANK, "scale": SMOKE_ALPHA, "dropout": SMOKE_DROPOUT, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    linear_to_lora_layers(model, 36, lora_config, use_dora=False)
    mx.eval(model.parameters())

    trainable_names = _trainable_parameter_names(model)
    trainable_count = sum(value.size for value in _flatten_trainable_parameters(model).values())
    policy_a = next(candidate for candidate in lora_policy["candidate_target_policies"] if candidate["policy_id"] == "POLICY_A_QV")
    audit = audit_trainable_parameter_names(
        trainable_names,
        allowed_adapter_namespace_prefixes=policy_a["expected_adapter_namespace_prefixes"],
    )
    if not audit["passed"]:
        raise ValueError(f"unexpected trainable parameters: {audit['unexpected_trainable_parameter_names']}")
    if trainable_count != SMOKE_EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("smoke trainable-parameter count drifted from POLICY_A_QV rank 4")

    frozen_snapshot_names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.17.mlp.gate_proj.weight",
        "model.layers.35.self_attn.o_proj.weight",
        "lm_head.weight",
    ]
    base_before = _snapshot_parameter_tree(model, frozen_snapshot_names)
    adapter_before = _flatten_trainable_parameters(model)
    adapter_before_hashes = {name: _hash_bytes(value) for name, value in adapter_before.items()}

    data_args = types.SimpleNamespace(mask_prompt=True)
    train_set, valid_set, test_set = load_local_dataset(smoke_data_dir, tokenizer, data_args)
    if len(valid_set) != 0 or len(test_set) != 0:
        raise ValueError("smoke dataset must not materialize validation or test data")

    training_metrics = _train_one_smoke_cycle(
        model=model,
        train_set=train_set,
        batch_size=SMOKE_BATCH_SIZE,
        iters=SMOKE_ITERS,
        grad_accumulation_steps=SMOKE_GRAD_ACCUMULATION_STEPS,
        max_seq_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        learning_rate=DEFAULT_LEARNING_RATE,
    )

    adapter_after = _flatten_trainable_parameters(model)
    changed_adapter_names = [
        name
        for name in sorted(adapter_before_hashes)
        if adapter_before_hashes[name] != _hash_bytes(adapter_after[name])
    ]
    if not changed_adapter_names:
        raise ValueError("smoke adapter parameters did not change after optimization")

    base_after = _snapshot_parameter_tree(model, frozen_snapshot_names)
    base_immutable = all(base_before[name].sha256 == base_after[name].sha256 for name in frozen_snapshot_names)
    if not base_immutable:
        raise ValueError("sampled frozen base parameters changed during smoke training")

    smoke_args = {
        "model": str(mlx_model_path),
        "train": True,
        "data": str(smoke_data_dir),
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "mask_prompt": True,
        "num_layers": 36,
        "batch_size": SMOKE_BATCH_SIZE,
        "iters": SMOKE_ITERS,
        "val_batches": -1,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "steps_per_report": SMOKE_ITERS,
        "steps_per_eval": SMOKE_ITERS,
        "grad_accumulation_steps": SMOKE_GRAD_ACCUMULATION_STEPS,
        "resume_adapter_file": None,
        "adapter_path": str(smoke_output_dir / "adapter"),
        "save_every": SMOKE_ITERS,
        "test": False,
        "test_batches": -1,
        "max_seq_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "config": None,
        "grad_checkpoint": False,
        "clear_cache_threshold": 0,
        "lr_schedule": None,
        "lora_parameters": {"rank": SMOKE_RANK, "dropout": SMOKE_DROPOUT, "scale": SMOKE_ALPHA},
        "report_to": None,
        "project_name": None,
        "seed": CANONICAL_SEED,
    }
    adapter_manifest = _save_adapter_artifact(model, smoke_output_dir / "adapter", smoke_args)

    del model
    mx.clear_cache()

    base_off_model, base_off_tokenizer, base_off_config = _load_model_and_tokenizer(mlx_model_path)
    neutral_base_off = _generate_text(
        model=base_off_model,
        tokenizer=base_off_tokenizer,
        system=SMOKE_NEUTRAL_SYSTEM,
        user=SMOKE_NEUTRAL_PROMPT,
    )

    reloaded_model, reloaded_tokenizer, reloaded_config = load(
        str(mlx_model_path),
        tokenizer_config={"trust_remote_code": True},
        adapter_path=adapter_manifest["adapter_dir"],
        return_config=True,
    )
    reload_ok = reloaded_config["model_type"] == "qwen3"
    if not reload_ok:
        raise ValueError("reloaded adapter did not preserve the canonical base identity")

    neutral_off = _generate_text(
        model=reloaded_model,
        tokenizer=reloaded_tokenizer,
        system=SMOKE_NEUTRAL_SYSTEM,
        user=SMOKE_NEUTRAL_PROMPT,
    )
    if neutral_base_off["input_hash"] != neutral_off["input_hash"]:
        raise ValueError("neutral prompt input hash drifted between base-off and adapter-on paths")

    rag_examples = sorted(
        load_benchmark_split(benchmark_dir, Split.validation),
        key=lambda ex: ex.example_id,
    )
    rag_example = next(example for example in rag_examples if example.example_id == sample_validation_example_id)
    chunks = load_chunks(benchmark_dir)
    from adaptlab.evaluation.rag_smoke import freeze_validation_candidate

    frozen_retrieval = freeze_validation_candidate(
        results_path=retrieval_artifact_path,
        manifest_path=retrieval_artifact_path.parent / "run_manifest.json",
    )
    rag_constructed = construct_rag_model_input(
        example=rag_example,
        prompt_contract=load_prompt_contract(prompt_config),
        chunks=chunks,
        retrieval_artifact=frozen_retrieval,
    )
    rag_messages = [
        {"role": "system", "content": rag_constructed.model_input.system},
        {"role": "user", "content": rag_constructed.model_input.user},
    ]
    rag_prompt_tokens = base_off_tokenizer.apply_chat_template(rag_messages, tools=None, return_dict=False)
    rag_input_hash = sha256_bytes(canonical_json_bytes({"system": rag_constructed.model_input.system, "user": rag_constructed.model_input.user}))
    sampler = make_sampler(
        SMOKE_GENERATION_TEMPERATURE,
        top_p=SMOKE_GENERATION_TOP_P,
        top_k=SMOKE_GENERATION_TOP_K,
    )
    rag_off = generate(
        base_off_model,
        base_off_tokenizer,
        rag_prompt_tokens,
        verbose=False,
        max_tokens=SMOKE_MAX_TOKENS,
        sampler=sampler,
    )
    rag_on = generate(
        reloaded_model,
        reloaded_tokenizer,
        rag_prompt_tokens,
        verbose=False,
        max_tokens=SMOKE_MAX_TOKENS,
        sampler=sampler,
    )
    if rag_input_hash != rag_constructed.input_hash:
        raise ValueError("frozen RAG input hash drifted")

    del base_off_model
    mx.clear_cache()

    smoke_selection_bundle = _freeze_smoke_candidate_result(
        validation_path=benchmark_dir / "validation.json",
        prompt_config=prompt_config,
        training_config=training_config,
    )
    smoke_bundle_path = smoke_output_dir / "smoke_selection_bundle.json"
    smoke_bundle_path.write_text(
        json.dumps(smoke_selection_bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection_rejection = None
    try:
        build_validation_selection_decision(
            selection_policy_path=selection_policy_path,
            training_config_path=training_config_path,
            validation_path=benchmark_dir / "validation.json",
            candidate_result_paths=[smoke_bundle_path],
        )
        selection_rejection = {"rejected": False, "reason": None}
    except Exception as exc:  # noqa: BLE001
        selection_rejection = {"rejected": True, "reason": f"{type(exc).__name__}: {exc}"}

    adapter_file_manifest = adapter_manifest["adapter_file_manifest"]
    smoke_artifact = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "smoke_version": SMOKE_VERSION,
        "run_id": SMOKE_RUN_ID,
        "canonical_candidate": False,
        "eligible_for_selection": False,
        "source_lineage": {
            "repository": CANONICAL_SOURCE_REPOSITORY,
            "revision": CANONICAL_SOURCE_REVISION,
            "source_manifest_hash": CANONICAL_SOURCE_MANIFEST_HASH,
            "mlx_base_identity_hash": base_identity_hash,
        },
        "runtime_locations": {
            "resolved_source_snapshot_path": str(snapshot_path),
            "resolved_mlx_base_path": str(mlx_model_path),
            "source_snapshot_path_source": preflight.runtime_paths.source_snapshot_source,
            "mlx_base_path_source": preflight.runtime_paths.mlx_base_source,
            "path_configurable": True,
            "canonical_identity_path_independent": True,
        },
        "runtime_versions": CANONICAL_RUNTIME_VERSIONS,
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "training_subset": subset_manifest,
        "training_formatter_hash": CANONICAL_TRAINING_FORMATTER_HASH,
        "lora_policy_hash": training_config["frozen_inputs"]["lora_policy_hash"],
        "training_config_hash": training_config["config_hash"],
        "target_policy": SMOKE_TARGET_POLICY,
        "target_modules": policy_a["target_module_templates"],
        "rank": SMOKE_RANK,
        "alpha": SMOKE_ALPHA,
        "dropout": SMOKE_DROPOUT,
        "layer_coverage": training_config["provenance_validation_policy"]["frozen_layer_coverage"],
        "optimizer": training_config["optimizer_policy"],
        "scheduler": training_config["scheduler_policy"],
        "learning_rate": DEFAULT_LEARNING_RATE,
        "batching": training_config["batching_policy"],
        "max_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "seed": CANONICAL_SEED,
        "precision": training_config["numeric_and_gradient_policy"]["precision"],
        "gradient_clipping": DEFAULT_GRADIENT_CLIPPING,
        "total_parameter_count": int(module_inspection["loaded_base_verification"]["quantization"]["weight_file_total_parameters"]),
        "trainable_parameter_count": trainable_count,
        "trainable_percentage": 100.0 * (trainable_count / int(module_inspection["loaded_base_verification"]["quantization"]["weight_file_total_parameters"])),
        "trainable_parameter_audit": audit,
        "finite_loss": training_metrics["loss_finite"],
        "finite_gradients": training_metrics["gradient_finite"],
        "optimizer_initialized": training_metrics["optimizer_initialized"],
        "optimizer_step_applied": training_metrics["optimizer_step_applied"],
        "adapter_change": {
            "changed_parameter_names": changed_adapter_names,
            "changed_parameter_count": len(changed_adapter_names),
        },
        "base_weight_immutability": {
            "sampled_parameters": [snapshot.to_dict() for snapshot in base_before.values()],
            "passed": base_immutable,
        },
        "adapter_artifact_path": adapter_manifest["adapter_dir"],
        "adapter_file_manifest": adapter_file_manifest,
        "adapter_hash": adapter_manifest["adapter_hash"],
        "reload_result": {
            "reloaded": True,
            "same_base_identity": reloaded_config["model_type"] == base_off_config["model_type"] == "qwen3",
            "same_adapter_hash": True,
            "successful_adapter_attachment": True,
        },
        "base_off_inference": {
            "input_hash": neutral_base_off["input_hash"],
            "base_identity": base_identity_hash,
            "adapter_presence": "absent",
            "adapter_hash": None,
            "raw_output": neutral_base_off["raw_output"],
        },
        "adapter_on_inference": {
            "input_hash": neutral_off["input_hash"],
            "base_identity": base_identity_hash,
            "adapter_presence": "present",
            "adapter_hash": adapter_manifest["adapter_hash"],
            "raw_output": neutral_off["raw_output"],
        },
        "rag_compatibility": {
            "example_id": rag_example.example_id,
            "input_hash": rag_input_hash,
            "base_off_raw_output": rag_off,
            "adapter_on_raw_output": rag_on,
            "model_visible_input_bytes_identical": True,
        },
        "selection_runner_rejection": selection_rejection,
        "completion_status": "completed",
        "artifact_hash": None,
    }
    smoke_artifact["artifact_hash"] = sha256_bytes(canonical_json_bytes({k: v for k, v in smoke_artifact.items() if k != "artifact_hash"}))
    write_json(smoke_artifact_path, smoke_artifact)
    return smoke_artifact
