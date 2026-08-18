from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.evaluation.scoring import score_output
from adaptlab.m5.selection_runner import (
    SELECTION_DECISION_FILENAME,
    ValidationSelectionCandidateResult,
    ValidationSelectionExampleResult,
    ValidationSelectionStatus,
    _validate_candidate_bundle_against_frozen_policy,
    _bundle_manifest_payload,
    build_validation_selection_decision,
    load_validation_selection_candidate_result,
    write_validation_selection_decision_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = ROOT / "data/generated/v0.0/validation.json"
SELECTION_POLICY_PATH = ROOT / "artifacts/evaluation/m5/m5_validation_selection_policy_v1.json"
SELECTION_POLICY_PATH_V2 = ROOT / "artifacts/evaluation/m5/m5_validation_selection_policy_v2.json"
SELECTION_POLICY_PATH_V3 = ROOT / "artifacts/evaluation/m5/m5_validation_selection_policy_v3.json"
TRAINING_CONFIG_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_training_config_v1.json"
TRAINING_CONFIG_PATH_V2 = ROOT / "artifacts/evaluation/m5/m5_lora_training_config_v2.json"


def _load_validation_examples() -> list[BenchmarkExample]:
    return sorted(
        (BenchmarkExample.from_dict(item) for item in json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))),
        key=lambda ex: ex.example_id,
    )


def _policy(path: Path = SELECTION_POLICY_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_config(path: Path = TRAINING_CONFIG_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _correct_raw_output(example: BenchmarkExample) -> str:
    expected = example.expected_output
    if isinstance(expected, str):
        return expected
    return json.dumps(expected, ensure_ascii=False)


def _wrong_raw_output(example: BenchmarkExample) -> str:
    expected = example.expected_output
    if isinstance(expected, str):
        return "__wrong__"
    return json.dumps({"wrong": example.example_id}, ensure_ascii=False)


def _make_example_results(*, wrong_example_ids: set[str] | None = None) -> tuple[ValidationSelectionExampleResult, ...]:
    wrong_example_ids = wrong_example_ids or set()
    results: list[ValidationSelectionExampleResult] = []
    for example in _load_validation_examples():
        raw_output = _wrong_raw_output(example) if example.example_id in wrong_example_ids else _correct_raw_output(example)
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
    return tuple(results)


def _aggregate(results: tuple[ValidationSelectionExampleResult, ...]) -> dict[str, object]:
    family_order = ("behavior_only", "behavior_knowledge", "changed_knowledge", "knowledge_only")
    per_family: dict[str, dict[str, object]] = {}
    total_correct = sum(1 for result in results if result.correct)
    for family in family_order:
        subset = [result for result in results if result.task_family == family]
        correct = sum(1 for result in subset if result.correct)
        n = len(subset)
        fraction = Fraction(correct, n)
        per_family[family] = {
            "n": n,
            "correct": correct,
            "accuracy": float(fraction),
            "accuracy_fraction": {"numerator": fraction.numerator, "denominator": fraction.denominator},
        }
    macro = sum(Fraction(item["correct"], item["n"]) for item in per_family.values()) / len(per_family)
    overall = Fraction(total_correct, len(results))
    return {
        "overall_correct": total_correct,
        "overall_accuracy": float(overall),
        "overall_accuracy_fraction": {"numerator": overall.numerator, "denominator": overall.denominator},
        "per_family": per_family,
        "macro_family_accuracy": float(macro),
        "macro_family_accuracy_fraction": {"numerator": macro.numerator, "denominator": macro.denominator},
    }


def _smoke_candidate_bundle(
    *,
    policy_path: Path = SELECTION_POLICY_PATH_V3,
    training_config_path: Path = TRAINING_CONFIG_PATH_V2,
) -> ValidationSelectionCandidateResult:
    policy = _policy(policy_path)
    training_config = _training_config(training_config_path)
    results = tuple(
        ValidationSelectionExampleResult(
            example_id=example.example_id,
            task_family=example.task_family.value,
            raw_output=example.expected_output if isinstance(example.expected_output, str) else json.dumps(example.expected_output, ensure_ascii=False),
            normalized_output=score_output(
                example,
                example.expected_output if isinstance(example.expected_output, str) else json.dumps(example.expected_output, ensure_ascii=False),
            ).normalized_output,
            correct=True,
        )
        for example in _load_validation_examples()
    )
    per_family = {}
    for family in ("behavior_only", "behavior_knowledge", "changed_knowledge", "knowledge_only"):
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
        candidate_id="M5_SMOKE_ONLY",
        checkpoint_id="ckpt-0000008",
        checkpoint_iteration=8,
        candidate_manifest_hash="0" * 64,
        candidate_search_hash=sha256_bytes(canonical_json_bytes(policy["candidate_budget"])),
        selection_policy_hash=policy["config_hash"],
        adapter_hash=hashlib.sha256(b"M5_SMOKE_ONLY").hexdigest(),
        base_identity_hash=training_config["numeric_and_gradient_policy"]["base_representation"]["identity_hash"],
        validation_split_hash=sha256_bytes(VALIDATION_PATH.read_bytes()),
        validation_example_ids_hash=sha256_bytes(canonical_json_bytes([example.example_id for example in _load_validation_examples()])),
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
        rank=4,
        target_policy="POLICY_A_QV",
        alpha=8,
        learning_rate=1e-05,
        training_duration_iters=8,
        eligible_checkpoint_steps=(8,),
        target_modules=("model.layers.{i}.self_attn.q_proj", "model.layers.{i}.self_attn.v_proj"),
        trainable_parameter_count=1916928,
        training_steps=8,
        n_total=len(results),
        per_example_results=results,
        aggregate=aggregate,
        completion_status=ValidationSelectionStatus.VALID,
        provider_runtime_failure_count=0,
        failure_reason=None,
    )
    manifest_hash = sha256_bytes(canonical_json_bytes(_bundle_manifest_payload(candidate)))
    return replace(candidate, candidate_manifest_hash=manifest_hash)


def _bundle(
    candidate_id: str,
    *,
    policy_path: Path = SELECTION_POLICY_PATH,
    training_config_path: Path = TRAINING_CONFIG_PATH,
    status: ValidationSelectionStatus = ValidationSelectionStatus.VALID,
    wrong_example_ids: set[str] | None = None,
    checkpoint_iteration: int | None = None,
    training_steps: int | None = None,
    provider_runtime_failure_count: int = 0,
) -> ValidationSelectionCandidateResult:
    policy = _policy(policy_path)
    training_config = _training_config(training_config_path)
    candidate = next(row for row in policy["candidate_budget"]["candidate_records"] if row["candidate_id"] == candidate_id)
    results = _make_example_results(wrong_example_ids=wrong_example_ids)
    if checkpoint_iteration is None:
        checkpoint_iteration = int(candidate["eligible_checkpoint_steps"][0])
    if training_steps is None:
        training_steps = int(candidate["training_duration_iters"])
    bundle = ValidationSelectionCandidateResult(
        candidate_id=candidate_id,
        checkpoint_id=f"ckpt-{checkpoint_iteration:07d}",
        checkpoint_iteration=checkpoint_iteration,
        candidate_manifest_hash="0" * 64,
        candidate_search_hash=sha256_bytes(canonical_json_bytes(policy["candidate_budget"])),
        selection_policy_hash=policy["config_hash"],
        adapter_hash=hashlib.sha256(candidate_id.encode("utf-8")).hexdigest(),
        base_identity_hash=training_config["numeric_and_gradient_policy"]["base_representation"]["identity_hash"],
        validation_split_hash=sha256_bytes(VALIDATION_PATH.read_bytes()),
        validation_example_ids_hash=sha256_bytes(canonical_json_bytes([example.example_id for example in _load_validation_examples()])),
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
        rank=int(candidate["rank"]),
        target_policy=str(candidate["target_policy"]),
        alpha=int(candidate["alpha"]),
        learning_rate=float(candidate["learning_rate"]),
        training_duration_iters=int(candidate["training_duration_iters"]),
        eligible_checkpoint_steps=tuple(candidate["eligible_checkpoint_steps"]),
        target_modules=tuple(candidate["target_modules"]),
        trainable_parameter_count=int(candidate["trainable_parameter_count"]),
        training_steps=training_steps,
        n_total=len(results),
        per_example_results=results,
        aggregate=_aggregate(results),
        completion_status=status,
        provider_runtime_failure_count=provider_runtime_failure_count,
        failure_reason=None,
    )
    manifest_hash = sha256_bytes(canonical_json_bytes(_bundle_manifest_payload(bundle)))
    return replace(bundle, candidate_manifest_hash=manifest_hash)


def _write_candidate_bundles(tmp_path: Path, bundles: list[ValidationSelectionCandidateResult]) -> list[Path]:
    paths: list[Path] = []
    for index, bundle in enumerate(bundles):
        path = tmp_path / f"bundle_{index}.json"
        path.write_bytes(bundle.to_json_bytes())
        paths.append(path)
    return paths


def test_selection_runner_requires_full_frozen_candidate_budget(tmp_path: Path) -> None:
    bundles = [_bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500")]
    paths = _write_candidate_bundles(tmp_path, bundles)
    with pytest.raises(ValueError, match="cover the frozen candidate budget exactly"):
        build_validation_selection_decision(
            selection_policy_path=SELECTION_POLICY_PATH,
            training_config_path=TRAINING_CONFIG_PATH,
            validation_path=VALIDATION_PATH,
            candidate_result_paths=paths,
        )


def test_selection_runner_ranks_valid_bundles_by_frozen_tie_breakers(tmp_path: Path) -> None:
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.TRAINING_FAILED),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    decision = build_validation_selection_decision(
        selection_policy_path=SELECTION_POLICY_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        validation_path=VALIDATION_PATH,
        candidate_result_paths=paths,
    )
    assert decision.selected_candidate_id == "S1_POLICY_B_ATTN_r4_lr1e-05_iters500"
    assert decision.selected_candidate_summary.rank == 4
    assert decision.selected_candidate_summary.macro_family_accuracy == 1.0


def test_selection_runner_uses_narrower_target_policy_when_rank_ties(tmp_path: Path) -> None:
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.TRAINING_FAILED),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    decision = build_validation_selection_decision(
        selection_policy_path=SELECTION_POLICY_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        validation_path=VALIDATION_PATH,
        candidate_result_paths=paths,
    )
    assert decision.selected_candidate_id == "S1_POLICY_A_QV_r8_lr1e-05_iters500"


def test_selection_runner_prefers_fewer_training_steps_when_other_factors_tie(tmp_path: Path) -> None:
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500", status=ValidationSelectionStatus.VALID, training_steps=500),
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.VALID, training_steps=1000),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.TRAINING_FAILED),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    decision = build_validation_selection_decision(
        selection_policy_path=SELECTION_POLICY_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        validation_path=VALIDATION_PATH,
        candidate_result_paths=paths,
    )
    assert decision.selected_candidate_id == "S1_POLICY_B_ATTN_r8_lr1e-05_iters500"


def test_selection_runner_excludes_ineligible_statuses_even_with_identical_scores(tmp_path: Path) -> None:
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.TRAINING_FAILED),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    decision = build_validation_selection_decision(
        selection_policy_path=SELECTION_POLICY_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        validation_path=VALIDATION_PATH,
        candidate_result_paths=paths,
    )
    assert decision.selected_candidate_id == "S1_POLICY_B_ATTN_r4_lr1e-05_iters500"


def test_selection_runner_rejects_stale_selection_policy_version(tmp_path: Path) -> None:
    bundles = [
        _bundle(
            "S1_POLICY_A_QV_r8_lr1e-05_iters500",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
        _bundle(
            "S1_POLICY_B_ATTN_r4_lr1e-05_iters500",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
        _bundle(
            "S1_POLICY_B_ATTN_r8_lr1e-05_iters500",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
        _bundle(
            "S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
        _bundle(
            "S2_POLICY_B_ATTN_r8_lr5e-06_iters1000",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
        _bundle(
            "S2_POLICY_B_ATTN_r8_lr2e-05_iters250",
            policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
        ),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    with pytest.raises(ValueError, match="selection policy version drifted"):
        build_validation_selection_decision(
            selection_policy_path=SELECTION_POLICY_PATH_V2,
            training_config_path=TRAINING_CONFIG_PATH_V2,
            validation_path=VALIDATION_PATH,
            candidate_result_paths=paths,
        )


def test_selection_runner_accepts_valid_frozen_candidate_identity(tmp_path: Path) -> None:
    policy = _policy(SELECTION_POLICY_PATH_V3)
    training_config = _training_config(TRAINING_CONFIG_PATH_V2)
    bundle = _bundle(
        "S1_POLICY_B_ATTN_r8_lr1e-05_iters500",
        policy_path=SELECTION_POLICY_PATH_V3,
        training_config_path=TRAINING_CONFIG_PATH_V2,
    )
    summary, result_hash = _validate_candidate_bundle_against_frozen_policy(
        policy=policy,
        training_config=training_config,
        validation_examples=_load_validation_examples(),
        validation_example_ids={example.example_id for example in _load_validation_examples()},
        validation_split_hash=sha256_bytes(VALIDATION_PATH.read_bytes()),
        validation_example_ids_hash=sha256_bytes(canonical_json_bytes([example.example_id for example in _load_validation_examples()])),
        candidate_budget_hash=sha256_bytes(canonical_json_bytes(policy["candidate_budget"])),
        selection_policy_hash=policy["config_hash"],
        bundle=bundle,
    )
    assert summary.candidate_id == "S1_POLICY_B_ATTN_r8_lr1e-05_iters500"
    assert result_hash == sha256_bytes(bundle.to_json_bytes())


def test_selection_runner_rejects_smoke_candidate_as_undeclared(tmp_path: Path) -> None:
    policy = _policy(SELECTION_POLICY_PATH_V3)
    training_config = _training_config(TRAINING_CONFIG_PATH_V2)
    smoke_bundle = _smoke_candidate_bundle(
        policy_path=SELECTION_POLICY_PATH_V3,
        training_config_path=TRAINING_CONFIG_PATH_V2,
    )
    smoke_path = tmp_path / "smoke_bundle.json"
    smoke_path.write_bytes(smoke_bundle.to_json_bytes())
    with pytest.raises(ValueError, match="candidate_id M5_SMOKE_ONLY is not in the frozen canonical candidate budget"):
        _validate_candidate_bundle_against_frozen_policy(
            policy=policy,
            training_config=training_config,
            validation_examples=_load_validation_examples(),
            validation_example_ids={example.example_id for example in _load_validation_examples()},
            validation_split_hash=sha256_bytes(VALIDATION_PATH.read_bytes()),
            validation_example_ids_hash=sha256_bytes(canonical_json_bytes([example.example_id for example in _load_validation_examples()])),
            candidate_budget_hash=sha256_bytes(canonical_json_bytes(policy["candidate_budget"])),
            selection_policy_hash=policy["config_hash"],
            bundle=load_validation_selection_candidate_result(smoke_path),
        )


def test_selection_runner_rejects_forbidden_fields_and_hash_drift(tmp_path: Path) -> None:
    bundle = _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500")
    payload = bundle.to_dict()
    payload["primary_test_accuracy"] = 0.9
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        load_validation_selection_candidate_result(bad_path)

    drifted = replace(bundle, candidate_search_hash="1" * 64)
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        drifted,
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)
    with pytest.raises(ValueError, match="candidate-search hash"):
        build_validation_selection_decision(
            selection_policy_path=SELECTION_POLICY_PATH,
            training_config_path=TRAINING_CONFIG_PATH,
            validation_path=VALIDATION_PATH,
            candidate_result_paths=paths,
        )


def test_selection_runner_writes_decision_and_sidecar(tmp_path: Path) -> None:
    bundles = [
        _bundle("S1_POLICY_A_QV_r8_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S1_POLICY_B_ATTN_r4_lr1e-05_iters500", status=ValidationSelectionStatus.VALID),
        _bundle("S1_POLICY_B_ATTN_r8_lr1e-05_iters500", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500", status=ValidationSelectionStatus.RESOURCE_INFEASIBLE),
        _bundle("S2_POLICY_B_ATTN_r8_lr5e-06_iters1000", status=ValidationSelectionStatus.TRAINING_FAILED),
        _bundle("S2_POLICY_B_ATTN_r8_lr2e-05_iters250", status=ValidationSelectionStatus.TRAINING_FAILED),
    ]
    paths = _write_candidate_bundles(tmp_path, bundles)

    output_path = tmp_path / SELECTION_DECISION_FILENAME
    decision = write_validation_selection_decision_artifact(
        selection_policy_path=SELECTION_POLICY_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        validation_path=VALIDATION_PATH,
        candidate_result_paths=paths,
        output_path=output_path,
    )
    assert output_path.exists()
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    assert sidecar.exists()
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {output_path.name}\n"
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded["selected_candidate_id"] == decision.selected_candidate_id
    assert reloaded["selection_policy_hash"] == decision.selection_policy_hash
