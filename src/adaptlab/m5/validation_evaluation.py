from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.domain.enums import Split
from adaptlab.evaluation.inputs import construct_model_input
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.runner import load_benchmark_split
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.evaluation.scoring import score_output
from adaptlab.m5.selection_runner import (
    ValidationSelectionCandidateResult,
    ValidationSelectionExampleResult,
    ValidationSelectionStatus,
    _bundle_manifest_payload,
    _validation_hashes,
)
from adaptlab.m5.validation_selection import DEFAULT_VALIDATION_FAMILIES


MAX_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fraction(numerator: int, denominator: int) -> dict[str, int]:
    f = Fraction(numerator, denominator)
    return {"numerator": f.numerator, "denominator": f.denominator}


def _aggregate(
    results: list[ValidationSelectionExampleResult],
) -> dict[str, Any]:
    per_family = {}

    for family in DEFAULT_VALIDATION_FAMILIES:
        rows = [r for r in results if r.task_family == family]
        n = len(rows)
        correct = sum(1 for r in rows if r.correct)
        if n == 0:
            raise ValueError(f"validation family {family!r} has zero examples")

        frac = Fraction(correct, n)
        per_family[family] = {
            "n": n,
            "correct": correct,
            "accuracy": float(frac),
            "accuracy_fraction": {
                "numerator": frac.numerator,
                "denominator": frac.denominator,
            },
        }

    total_correct = sum(1 for r in results if r.correct)
    overall = Fraction(total_correct, len(results))
    macro = (
        sum(
            (Fraction(v["correct"], v["n"]) for v in per_family.values()),
            Fraction(0, 1),
        )
        / len(per_family)
    )

    return {
        "overall_correct": total_correct,
        "overall_accuracy": float(overall),
        "overall_accuracy_fraction": {
            "numerator": overall.numerator,
            "denominator": overall.denominator,
        },
        "per_family": per_family,
        "macro_family_accuracy": float(macro),
        "macro_family_accuracy_fraction": {
            "numerator": macro.numerator,
            "denominator": macro.denominator,
        },
    }


def run_candidate_validation(
    *,
    candidate_id: str,
    benchmark_dir: Path,
    prompt_config: Path,
    training_config_path: Path,
    selection_policy_path: Path,
    candidate_dir: Path,
    mlx_base_path: Path,
) -> ValidationSelectionCandidateResult:
    benchmark_dir = Path(benchmark_dir)
    candidate_dir = Path(candidate_dir)

    training_config = _load_json(training_config_path)
    selection_policy = _load_json(selection_policy_path)
    training_result = _load_json(candidate_dir / "training_result.json")

    if training_result["candidate_id"] != candidate_id:
        raise ValueError("training-result candidate ID drifted")
    if training_result["status"] != "COMPLETED":
        raise ValueError("candidate training is not COMPLETED")

    candidate = next(
        row
        for row in selection_policy["candidate_budget"]["candidate_records"]
        if row["candidate_id"] == candidate_id
    )

    # Fail closed on frozen candidate/training identity before inference.
    for key in (
        "target_policy",
        "rank",
        "alpha",
        "learning_rate",
    ):
        if training_result[key] != candidate[key]:
            raise ValueError(f"training-result {key} drifted from frozen candidate")

    if training_result["completed_iterations"] != candidate["training_duration_iters"]:
        raise ValueError("completed training iterations drifted")
    if training_result["trainable_parameter_count"] != candidate["trainable_parameter_count"]:
        raise ValueError("trainable parameter count drifted")

    adapter_dir = candidate_dir / "adapter"
    if not (adapter_dir / "adapters.safetensors").exists():
        raise ValueError("canonical adapter file is missing")

    model, tokenizer, config = load(
        str(mlx_base_path),
        tokenizer_config={"trust_remote_code": True},
        adapter_path=str(adapter_dir),
        return_config=True,
    )
    tokenizer.add_eos_token("<|im_end|>")
    if config.get("model_type") != "qwen3":
        raise ValueError("loaded model is not canonical Qwen3 lineage")

    validation_examples = sorted(
        load_benchmark_split(benchmark_dir, Split.validation),
        key=lambda ex: ex.example_id,
    )
    if len(validation_examples) != 150:
        raise ValueError(
            f"canonical validation split must contain 150 examples, "
            f"found {len(validation_examples)}"
        )

    prompt_contract = load_prompt_contract(prompt_config)
    sampler = make_sampler(
        TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    results: list[ValidationSelectionExampleResult] = []

    for index, example in enumerate(validation_examples, start=1):
        constructed = construct_model_input(
            example=example,
            method=AdaptationMethod.PROMPT,
            prompt_contract=prompt_contract,
        )

        messages = [
            {
                "role": "system",
                "content": constructed.model_input.system,
            },
            {
                "role": "user",
                "content": constructed.model_input.user,
            },
        ]

        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            tools=None,
            add_generation_prompt=True,
            return_dict=False,
        )

        raw_output = generate(
            model,
            tokenizer,
            prompt_tokens,
            verbose=False,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
        )

        scored = score_output(example, raw_output)

        results.append(
            ValidationSelectionExampleResult(
                example_id=example.example_id,
                task_family=example.task_family.value,
                raw_output=raw_output,
                normalized_output=scored.normalized_output,
                correct=bool(scored.score == 1.0),
            )
        )

        if index % 10 == 0 or index == len(validation_examples):
            correct_so_far = sum(1 for r in results if r.correct)
            print(
                f"[{candidate_id}] validation {index}/150 "
                f"correct_so_far={correct_so_far}"
            )

    aggregate = _aggregate(results)

    validation_path = benchmark_dir / "validation.json"
    validation_split_hash, validation_example_ids_hash = _validation_hashes(
        validation_path
    )

    candidate_budget_hash = sha256_bytes(
        canonical_json_bytes(selection_policy["candidate_budget"])
    )
    selection_policy_hash = selection_policy["config_hash"]

    frozen = selection_policy["frozen_inputs"]
    source = frozen["source_lineage"]

    provisional = ValidationSelectionCandidateResult(
        candidate_id=candidate_id,
        checkpoint_id=f"ckpt-{candidate['training_duration_iters']:07d}",
        checkpoint_iteration=int(candidate["training_duration_iters"]),
        candidate_manifest_hash="0" * 64,
        candidate_search_hash=candidate_budget_hash,
        selection_policy_hash=selection_policy_hash,
        adapter_hash=training_result["adapter_hash"],
        base_identity_hash=training_config[
            "numeric_and_gradient_policy"
        ]["base_representation"]["identity_hash"],
        validation_split_hash=validation_split_hash,
        validation_example_ids_hash=validation_example_ids_hash,
        source_repository=source["repository"],
        source_revision=source["revision"],
        source_manifest_hash=source["source_manifest_hash"],
        training_formatter_hash=frozen["training_formatter_hash"],
        lora_policy_hash=training_config["frozen_inputs"]["lora_policy_hash"],
        training_config_hash=frozen["training_config_hash"],
        seed=int(
            training_config[
                "numeric_and_gradient_policy"
            ]["seed_policy"]["canonical_seed"]
        ),
        dropout=float(
            training_config["default_candidate_provenance"]["dropout"]
        ),
        optimizer=str(training_config["optimizer_policy"]["family"]),
        scheduler=str(training_config["scheduler_policy"]["type"]),
        batching=training_config["batching_policy"],
        layer_coverage=training_config[
            "provenance_validation_policy"
        ]["frozen_layer_coverage"],
        rank=int(candidate["rank"]),
        target_policy=str(candidate["target_policy"]),
        alpha=int(candidate["alpha"]),
        learning_rate=float(candidate["learning_rate"]),
        training_duration_iters=int(candidate["training_duration_iters"]),
        eligible_checkpoint_steps=tuple(candidate["eligible_checkpoint_steps"]),
        target_modules=tuple(candidate["target_modules"]),
        trainable_parameter_count=int(candidate["trainable_parameter_count"]),
        training_steps=int(candidate["training_duration_iters"]),
        n_total=len(results),
        per_example_results=tuple(results),
        aggregate=aggregate,
        completion_status=ValidationSelectionStatus.VALID,
        provider_runtime_failure_count=0,
        failure_reason=None,
    )

    manifest_hash = sha256_bytes(
        canonical_json_bytes(_bundle_manifest_payload(provisional))
    )

    bundle = ValidationSelectionCandidateResult.from_dict(
        {
            **provisional.to_dict(),
            "candidate_manifest_hash": manifest_hash,
        }
    )

    output_path = candidate_dir / "validation_selection_candidate_result.json"
    output_path.write_bytes(bundle.to_json_bytes())

    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_bytes(bundle.to_json_bytes())}  {output_path.name}\n",
        encoding="utf-8",
    )

    print("validation_bundle:", output_path)
    print("candidate_manifest_hash:", bundle.candidate_manifest_hash)
    print("overall_correct:", aggregate["overall_correct"])
    print("overall_accuracy:", aggregate["overall_accuracy"])
    print("macro_family_accuracy:", aggregate["macro_family_accuracy"])

    return bundle
