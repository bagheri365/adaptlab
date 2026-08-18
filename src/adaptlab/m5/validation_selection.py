"""Canonical Milestone 5 validation-selection policy.

This module freezes the validation-only candidate selection contract for the
canonical Milestone 5 LoRA experiment.  It does not train or score any model.
It only predeclares the admissible selection metric, tie-break order, and the
small fixed candidate budget that may be evaluated on the frozen validation
split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json

VALIDATION_SELECTION_SCHEMA_VERSION = "m5-validation-selection-policy-artifact-v1"
VALIDATION_SELECTION_VERSION_V1 = "m5-validation-selection-policy-v1"
CANONICAL_TRAINING_CONFIG_HASH = "011e2246ec866b4e9fb58ba5dec8baac9946b242d4ec42cd39a5fba4c36b4c56"
CANONICAL_TRAINING_CONFIG_HASH_V2 = "7555d16029fc006b02d51307414d1ba3fe859974b9564e86cdfa4e2c01006f69"
CANONICAL_TRAINING_CONFIG_HASH_V3 = "4eaa4e003deb7e7d601934fa8bf8a8689b17292b99c6eb9f545978a38735177d"
CANONICAL_TRAINING_CONFIG_HASHES = {
    CANONICAL_TRAINING_CONFIG_HASH,
    CANONICAL_TRAINING_CONFIG_HASH_V2,
    CANONICAL_TRAINING_CONFIG_HASH_V3,
}
VALIDATION_SELECTION_VERSION = "m5-validation-selection-policy-v3"
CANONICAL_VALIDATION_SELECTION_POLICY_VERSIONS = {
    VALIDATION_SELECTION_VERSION_V1,
    VALIDATION_SELECTION_VERSION,
}

DEFAULT_VALIDATION_FAMILIES = (
    "behavior_only",
    "behavior_knowledge",
    "changed_knowledge",
    "knowledge_only",
)
DEFAULT_CANDIDATE_IDS = (
    "S1_POLICY_A_QV_r8_lr1e-05_iters500",
    "S1_POLICY_B_ATTN_r4_lr1e-05_iters500",
    "S1_POLICY_B_ATTN_r8_lr1e-05_iters500",
    "S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500",
    "S2_POLICY_B_ATTN_r8_lr5e-06_iters1000",
    "S2_POLICY_B_ATTN_r8_lr2e-05_iters250",
)


@dataclass(frozen=True, slots=True)
class ValidationSelectionCandidate:
    """One predeclared validation-eligible LoRA candidate."""

    candidate_id: str
    stage: str
    target_policy: str
    rank: int
    alpha: int
    learning_rate: float
    training_duration_iters: int
    eligible_checkpoint_steps: tuple[int, ...]
    seed: int
    target_modules: tuple[str, ...]
    trainable_parameter_count: int
    trainable_percentage_of_total_model_parameters: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "target_policy": self.target_policy,
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "training_duration_iters": self.training_duration_iters,
            "eligible_checkpoint_steps": list(self.eligible_checkpoint_steps),
            "seed": self.seed,
            "target_modules": list(self.target_modules),
            "trainable_parameter_count": self.trainable_parameter_count,
            "trainable_percentage_of_total_model_parameters": self.trainable_percentage_of_total_model_parameters,
            "rationale": self.rationale,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sorted_validation_examples(benchmark_dir: Path) -> list[dict[str, Any]]:
    validation_path = Path(benchmark_dir) / "validation.json"
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = sorted(raw, key=lambda item: item["example_id"])
    if len(examples) != 150:
        raise ValueError(f"canonical validation split must contain 150 examples, found {len(examples)}")
    return examples


def _example_id_hash(example_ids: Sequence[str]) -> str:
    text = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family_counts(examples: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        family = str(example["task_family"])
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_lookup(config: Mapping[str, Any], policy_id: str) -> tuple[list[str], dict[int, dict[str, Any]]]:
    module_templates = config["provenance_validation_policy"]["target_policy_to_modules"][policy_id]
    table = {
        int(row["rank"]): row
        for row in config["candidate_trainable_parameter_tables"][policy_id]
    }
    return list(module_templates), table


def _build_candidate(
    *,
    config: Mapping[str, Any],
    candidate_id: str,
    stage: str,
    target_policy: str,
    rank: int,
    learning_rate: float,
    training_duration_iters: int,
    rationale: str,
) -> ValidationSelectionCandidate:
    module_templates, table = _candidate_lookup(config, target_policy)
    row = table.get(rank)
    if row is None:
        raise ValueError(f"no frozen trainable-parameter row for {target_policy} rank {rank}")
    return ValidationSelectionCandidate(
        candidate_id=candidate_id,
        stage=stage,
        target_policy=target_policy,
        rank=rank,
        alpha=2 * rank,
        learning_rate=learning_rate,
        training_duration_iters=training_duration_iters,
        eligible_checkpoint_steps=(training_duration_iters,),
        seed=1729,
        target_modules=tuple(module_templates),
        trainable_parameter_count=int(row["adapter_parameter_count"]),
        trainable_percentage_of_total_model_parameters=float(row["trainable_percentage_of_total_model_parameters"]),
        rationale=rationale,
    )


def _candidate_budget(config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        _build_candidate(
            config=config,
            candidate_id="S1_POLICY_A_QV_r8_lr1e-05_iters500",
            stage="stage1",
            target_policy="POLICY_A_QV",
            rank=8,
            learning_rate=1e-5,
            training_duration_iters=500,
            rationale=(
                "Narrow attention-only adapter baseline; paired against broader policies "
                "under the same rank, learning rate, and duration."
            ),
        ),
        _build_candidate(
            config=config,
            candidate_id="S1_POLICY_B_ATTN_r4_lr1e-05_iters500",
            stage="stage1",
            target_policy="POLICY_B_ATTN",
            rank=4,
            learning_rate=1e-5,
            training_duration_iters=500,
            rationale=(
                "Lower-rank attention-only control to test whether the broader policy "
                "needs extra capacity before any learning-rate or duration variation."
            ),
        ),
        _build_candidate(
            config=config,
            candidate_id="S1_POLICY_B_ATTN_r8_lr1e-05_iters500",
            stage="stage1",
            target_policy="POLICY_B_ATTN",
            rank=8,
            learning_rate=1e-5,
            training_duration_iters=500,
            rationale=(
                "Canonical medium-capacity attention-only candidate used as the primary "
                "reference point for stage-1 comparisons."
            ),
        ),
        _build_candidate(
            config=config,
            candidate_id="S1_POLICY_C_ATTN_MLP_r4_lr1e-05_iters500",
            stage="stage1",
            target_policy="POLICY_C_ATTN_MLP",
            rank=4,
            learning_rate=1e-5,
            training_duration_iters=500,
            rationale=(
                "Broader attention-plus-MLP adapter at the lowest rank to probe whether "
                "the extra scope is useful without spending budget on heavier ranks."
            ),
        ),
        _build_candidate(
            config=config,
            candidate_id="S2_POLICY_B_ATTN_r8_lr5e-06_iters1000",
            stage="stage2",
            target_policy="POLICY_B_ATTN",
            rank=8,
            learning_rate=5e-6,
            training_duration_iters=1000,
            rationale=(
                "Predeclared lower-learning-rate and longer-duration variant to test a "
                "small optimization change without tuning based on validation outcomes."
            ),
        ),
        _build_candidate(
            config=config,
            candidate_id="S2_POLICY_B_ATTN_r8_lr2e-05_iters250",
            stage="stage2",
            target_policy="POLICY_B_ATTN",
            rank=8,
            learning_rate=2e-5,
            training_duration_iters=250,
            rationale=(
                "Predeclared higher-learning-rate and shorter-duration variant to test a "
                "small optimization change without tuning based on validation outcomes."
            ),
        ),
    ]

    return {
        "total_candidates": len(candidates),
        "stage1_candidate_count": sum(1 for candidate in candidates if candidate.stage == "stage1"),
        "stage2_candidate_count": sum(1 for candidate in candidates if candidate.stage == "stage2"),
        "stage1_stage2_precommitment": (
            "Stage-2 candidates are fixed before any Stage-1 result is observed; "
            "no later candidate may be added, removed, or reweighted based on validation scores."
        ),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "candidate_records": [candidate.to_dict() for candidate in candidates],
        "excluded_high_rank_scope": {
            "POLICY_C_ATTN_MLP_rank8": "excluded",
            "POLICY_C_ATTN_MLP_rank16": "excluded",
            "rationale": (
                "The heavier attention-plus-MLP adapter family is represented only at rank 4 "
                "to keep the candidate budget hardware-realistic and avoid spending most of the "
                "budget on the heaviest adapter configuration."
            ),
        },
    }


def build_validation_selection_policy_artifact(
    *,
    benchmark_dir: Path,
    training_config_path: Path,
    experimental_contract_hash: str,
    training_isolation_audit_hash: str,
    training_formatter_hash: str,
) -> dict[str, Any]:
    """Build the frozen validation-selection policy artifact."""

    benchmark_dir = Path(benchmark_dir)
    training_config_path = Path(training_config_path)

    training_config = _load_json(training_config_path)
    if training_config.get("config_hash") not in CANONICAL_TRAINING_CONFIG_HASHES:
        raise ValueError("training config hash drifted from the frozen canonical hash")
    if training_config.get("gate") != "M5_LORA_TRAINING_CONFIG_READY":
        raise ValueError("training config artifact is not the frozen canonical training config")

    validation_examples = _sorted_validation_examples(benchmark_dir)
    validation_example_ids = [example["example_id"] for example in validation_examples]
    validation_family_counts = _family_counts(validation_examples)

    represented_families = []
    for family in DEFAULT_VALIDATION_FAMILIES:
        count = validation_family_counts.get(family)
        if count is None:
            raise ValueError(f"canonical validation split is missing required family {family}")
        represented_families.append({"task_family": family, "n": count})

    artifact = {
        "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
        "policy_version": VALIDATION_SELECTION_VERSION,
        "gate": "M5_VALIDATION_SELECTION_POLICY_READY",
        "frozen_inputs": {
            "experimental_contract_hash": experimental_contract_hash,
            "training_isolation_audit_hash": training_isolation_audit_hash,
            "training_formatter_hash": training_formatter_hash,
            "training_config_hash": training_config["config_hash"],
            "training_config_version": training_config["training_config_version"],
            "training_config_file_hash": sha256_bytes(training_config_path.read_bytes()),
            "benchmark_manifest_hash": training_config["frozen_inputs"]["benchmark_manifest_hash"],
            "validation_split_file_hash": sha256_bytes((benchmark_dir / "validation.json").read_bytes()),
            "validation_example_ids_hash": _example_id_hash(validation_example_ids),
            "source_lineage": training_config["frozen_inputs"]["source_lineage"],
            "runtime_versions": training_config["frozen_inputs"]["runtime_versions"],
        },
        "validation_selection_policy": {
            "selection_split": "validation",
            "selection_scope": "validation_only",
            "primary_metric": {
                "name": "macro_average_exact_match_accuracy",
                "formula": "mean(exact_match_accuracy for each represented validation task_family)",
                "family_weighting": "uniform_over_represented_families",
                "represented_families": [item["task_family"] for item in represented_families],
                "represented_family_counts": represented_families,
            },
            "secondary_metric": {
                "name": "overall_validation_exact_match_accuracy",
                "formula": "validation_correct / validation_n",
            },
            "tie_breakers": [
                "smaller_lora_rank",
                "narrower_target_policy_by_trainable_parameter_count",
                "fewer_training_steps",
                "fewer_trainable_parameters",
                "lexical_candidate_id_ascending",
            ],
            "checkpoint_policy": {
                "candidate_checkpoint_rule": "one_predeclared_final_checkpoint_per_candidate",
                "checkpoint_selection_use": "validation_metric_only",
                "checkpoint_selection_tie_breakers": [
                    "smaller_lora_rank",
                    "narrower_target_policy_by_trainable_parameter_count",
                    "fewer_training_steps",
                    "fewer_trainable_parameters",
                    "lexical_candidate_id_ascending",
                ],
            },
            "forbidden_selection_signals": [
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
            ],
            "selection_metric_guardrails": {
                "no_proxy_for_missing_family": True,
                "no_test_set_access": True,
                "no_sentinel_access": True,
                "no_primary_test_score_use": True,
                "no_qualitative_test_tuning": True,
            },
        },
        "candidate_budget": _candidate_budget(training_config),
        "candidate_manifest_schema": {
            "required_fields": [
                "selection_run_id",
                "candidate_id",
                "stage",
                "target_policy",
                "rank",
                "alpha",
                "learning_rate",
                "training_duration_iters",
                "eligible_checkpoint_steps",
                "seed",
                "target_modules",
                "trainable_parameter_count",
                "trainable_percentage_of_total_model_parameters",
                "selection_split",
                "primary_metric_name",
                "secondary_metric_name",
                "tie_breakers",
                "checkpoint_selection_rule",
                "forbidden_signal_attestation",
                "training_config_hash",
                "experimental_contract_hash",
                "training_isolation_audit_hash",
                "training_formatter_hash",
            ],
            "optional_observational_fields": [
                "start_runtime_timestamp",
                "end_runtime_timestamp",
                "validation_duration_seconds",
                "peak_memory_gb",
            ],
        },
    }
    artifact["config_hash"] = sha256_bytes(canonical_json_bytes(artifact))
    return artifact


def validate_validation_selection_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Reject candidate-selection manifests that drift from the frozen policy."""

    policy = config["validation_selection_policy"]
    if policy.get("selection_split") != "validation":
        raise ValueError("validation selection policy must use the validation split")

    required_fields = set(config["candidate_manifest_schema"]["required_fields"])
    allowed_optional = set(config["candidate_manifest_schema"].get("optional_observational_fields", ()))
    missing = required_fields - set(manifest)
    if missing:
        raise ValueError(f"selection manifest is missing required fields: {sorted(missing)}")

    allowed_fields = required_fields | allowed_optional
    unknown = set(manifest) - allowed_fields
    if unknown:
        raise ValueError(f"selection manifest has unexpected fields: {sorted(unknown)}")

    if manifest.get("selection_split") != "validation":
        raise ValueError("selection manifest must be validation-only")
    if manifest.get("primary_metric_name") != "macro_average_exact_match_accuracy":
        raise ValueError("selection manifest must use the frozen primary metric")
    if manifest.get("secondary_metric_name") != "overall_validation_exact_match_accuracy":
        raise ValueError("selection manifest must use the frozen secondary metric")
    if list(manifest.get("tie_breakers", ())) != list(policy["tie_breakers"]):
        raise ValueError("selection manifest tie-breakers drifted from the frozen policy")
    if manifest.get("checkpoint_selection_rule") != policy["checkpoint_policy"]["candidate_checkpoint_rule"]:
        raise ValueError("selection manifest checkpoint policy drifted from the frozen policy")
    if manifest.get("forbidden_signal_attestation") is not True:
        raise ValueError("selection manifest must attest that forbidden signals were not used")

    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in config["candidate_budget"]["candidate_records"]}
    candidate_id = manifest["candidate_id"]
    candidate = candidate_lookup.get(candidate_id)
    if candidate is None:
        raise ValueError(f"unknown validation-selection candidate {candidate_id}")

    frozen_candidate_fields = (
        "stage",
        "target_policy",
        "rank",
        "alpha",
        "learning_rate",
        "training_duration_iters",
        "seed",
    )
    for key in frozen_candidate_fields:
        if manifest.get(key) != candidate.get(key):
            raise ValueError(f"selection manifest drifted for frozen field {key}")

    frozen_inputs = config["frozen_inputs"]
    frozen_manifest_fields = {
        "training_config_hash": frozen_inputs["training_config_hash"],
        "experimental_contract_hash": frozen_inputs["experimental_contract_hash"],
        "training_isolation_audit_hash": frozen_inputs["training_isolation_audit_hash"],
        "training_formatter_hash": frozen_inputs["training_formatter_hash"],
    }
    for key, expected in frozen_manifest_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"selection manifest drifted for frozen field {key}")

    if list(manifest["eligible_checkpoint_steps"]) != list(candidate["eligible_checkpoint_steps"]):
        raise ValueError("eligible checkpoint steps drifted from the frozen candidate budget")
    if list(manifest["target_modules"]) != list(candidate["target_modules"]):
        raise ValueError("target modules drifted from the frozen candidate budget")
    if int(manifest["trainable_parameter_count"]) != int(candidate["trainable_parameter_count"]):
        raise ValueError("trainable parameter count drifted from the frozen candidate budget")
    if float(manifest["trainable_percentage_of_total_model_parameters"]) != float(
        candidate["trainable_percentage_of_total_model_parameters"]
    ):
        raise ValueError("trainable percentage drifted from the frozen candidate budget")

    rank = int(manifest["rank"])
    if rank not in (4, 8, 16):
        raise ValueError("selection candidate rank is outside the canonical whitelist")
    if int(manifest["alpha"]) != 2 * rank:
        raise ValueError("selection candidate alpha must satisfy alpha = 2 * rank")


def write_validation_selection_policy_artifact(
    *,
    benchmark_dir: Path,
    training_config_path: Path,
    experimental_contract_hash: str,
    training_isolation_audit_hash: str,
    training_formatter_hash: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write the frozen validation-selection policy artifact and return it."""

    artifact = build_validation_selection_policy_artifact(
        benchmark_dir=benchmark_dir,
        training_config_path=training_config_path,
        experimental_contract_hash=experimental_contract_hash,
        training_isolation_audit_hash=training_isolation_audit_hash,
        training_formatter_hash=training_formatter_hash,
    )
    write_json(Path(output_path), artifact)
    return artifact
