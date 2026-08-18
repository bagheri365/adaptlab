"""Canonical Milestone 5 validation-selection runner.

This module mechanically enforces the frozen validation-selection policy over
candidate result bundles that are already computed. It does not train models,
invoke retrieval, or perform inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.evaluation.scoring import score_output
from adaptlab.m5.validation_selection import (
    CANONICAL_VALIDATION_SELECTION_POLICY_VERSIONS,
    VALIDATION_SELECTION_VERSION,
)

SELECTION_DECISION_SCHEMA_VERSION = "m5-validation-selection-decision-v1"
SELECTION_RUNNER_VERSION = "m5-validation-selection-runner-v1"
SELECTION_DECISION_FILENAME = "lora_selection_decision_v1.json"

_FORBIDDEN_TOP_LEVEL_FIELDS = {
    "primary_test_accuracy",
    "primary_test_family_scores",
    "updated_test_score",
    "removed_test_score",
    "structural_holdout_test_score",
    "primary_test_rag_results",
    "primary_test_oracle_context_results",
    "milestone_4_failure_categories",
    "generalization_sentinel",
    "sentinel_score",
    "test_task_family_metrics",
}

_EXACT_EXAMPLE_RESULT_FIELDS = {
    "example_id",
    "task_family",
    "raw_output",
    "normalized_output",
    "correct",
}

_EXACT_AGGREGATE_FIELDS = {
    "overall_correct",
    "overall_accuracy",
    "overall_accuracy_fraction",
    "per_family",
    "macro_family_accuracy",
    "macro_family_accuracy_fraction",
}

_EXACT_DECISION_FIELDS = {
    "schema_version",
    "runner_version",
    "selection_run_id",
    "selection_policy_hash",
    "candidate_search_hash",
    "validation_split_hash",
    "validation_example_ids_hash",
    "candidate_budget_hash",
    "candidate_result_hashes",
    "status_counts",
    "candidate_count",
    "valid_candidate_count",
    "ineligible_candidate_count",
    "candidate_rankings",
    "selected_candidate_id",
    "selected_checkpoint_id",
    "selected_checkpoint_iteration",
    "selected_candidate_manifest_hash",
    "selected_candidate_result_hash",
    "selected_candidate_summary",
}

_HASH_FIELDS = {
    "adapter_hash",
    "base_identity_hash",
    "candidate_manifest_hash",
    "candidate_search_hash",
    "selection_policy_hash",
    "validation_split_hash",
    "validation_example_ids_hash",
    "source_manifest_hash",
    "training_formatter_hash",
    "lora_policy_hash",
    "training_config_hash",
}


def _require_sha256(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


class ValidationSelectionStatus(str, Enum):
    VALID = "VALID"
    RESOURCE_INFEASIBLE = "RESOURCE_INFEASIBLE"
    TRAINING_FAILED = "TRAINING_FAILED"
    VALIDATION_INCOMPLETE = "VALIDATION_INCOMPLETE"
    PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sorted_validation_examples(validation_path: Path) -> list[BenchmarkExample]:
    examples = [BenchmarkExample.from_dict(item) for item in _load_json(validation_path)]
    examples = sorted(examples, key=lambda ex: ex.example_id)
    if len(examples) != 150:
        raise ValueError(f"canonical validation split must contain 150 examples, found {len(examples)}")
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.task_family.value] = counts.get(example.task_family.value, 0) + 1
    expected = {
        "behavior_only": 38,
        "behavior_knowledge": 37,
        "changed_knowledge": 37,
        "knowledge_only": 38,
    }
    if counts != expected:
        raise ValueError(f"validation family counts drifted from the frozen split: {counts}")
    return examples


def _validation_hashes(validation_path: Path) -> tuple[str, str]:
    examples = _sorted_validation_examples(validation_path)
    validation_split_hash = sha256_bytes(validation_path.read_bytes())
    validation_example_ids_hash = sha256_bytes(
        canonical_json_bytes([example.example_id for example in examples])
    )
    return validation_split_hash, validation_example_ids_hash


def _family_order() -> tuple[str, ...]:
    return ("behavior_only", "behavior_knowledge", "changed_knowledge", "knowledge_only")


def _require_exact_keys(payload: Mapping[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{context} has unexpected fields: {sorted(unknown)}")


def _canonical_fraction(numerator: int, denominator: int) -> dict[str, int]:
    fraction = Fraction(numerator, denominator)
    return {"numerator": fraction.numerator, "denominator": fraction.denominator}


def _fraction_value(numerator: int, denominator: int) -> Fraction:
    return Fraction(numerator, denominator)


@dataclass(frozen=True, slots=True)
class ValidationSelectionExampleResult:
    example_id: str
    task_family: str
    raw_output: str | None
    normalized_output: Any
    correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(self.task_family, str) or not self.task_family.strip():
            raise ValueError("task_family must be a non-empty string")
        if self.raw_output is not None and not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string or None")
        if not isinstance(self.correct, bool):
            raise ValueError("correct must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task_family": self.task_family,
            "raw_output": self.raw_output,
            "normalized_output": self.normalized_output,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationSelectionExampleResult":
        _require_exact_keys(data, _EXACT_EXAMPLE_RESULT_FIELDS, context="example result")
        return cls(
            example_id=data["example_id"],
            task_family=data["task_family"],
            raw_output=data["raw_output"],
            normalized_output=data["normalized_output"],
            correct=data["correct"],
        )


@dataclass(frozen=True, slots=True)
class ValidationSelectionCandidateResult:
    candidate_id: str
    checkpoint_id: str
    checkpoint_iteration: int
    candidate_manifest_hash: str
    candidate_search_hash: str
    selection_policy_hash: str
    adapter_hash: str
    base_identity_hash: str
    validation_split_hash: str
    validation_example_ids_hash: str
    source_repository: str
    source_revision: str
    source_manifest_hash: str
    training_formatter_hash: str
    lora_policy_hash: str
    training_config_hash: str
    seed: int
    dropout: float
    optimizer: str
    scheduler: str
    batching: Mapping[str, Any]
    layer_coverage: Mapping[str, Any]
    rank: int
    target_policy: str
    alpha: int
    learning_rate: float
    training_duration_iters: int
    eligible_checkpoint_steps: tuple[int, ...]
    target_modules: tuple[str, ...]
    trainable_parameter_count: int
    training_steps: int
    n_total: int
    per_example_results: tuple[ValidationSelectionExampleResult, ...]
    aggregate: Mapping[str, Any]
    completion_status: ValidationSelectionStatus
    provider_runtime_failure_count: int
    failure_reason: str | None = None
    schema_version: str = SELECTION_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "checkpoint_id",
            "candidate_manifest_hash",
            "candidate_search_hash",
            "selection_policy_hash",
            "adapter_hash",
            "base_identity_hash",
            "validation_split_hash",
            "validation_example_ids_hash",
            "source_repository",
            "source_revision",
            "source_manifest_hash",
            "training_formatter_hash",
            "lora_policy_hash",
            "training_config_hash",
            "optimizer",
            "scheduler",
            "target_policy",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in _HASH_FIELDS:
            _require_sha256(field_name, getattr(self, field_name))
        for field_name in (
            "checkpoint_iteration",
            "seed",
            "rank",
            "alpha",
            "training_duration_iters",
            "training_steps",
            "n_total",
            "trainable_parameter_count",
            "provider_runtime_failure_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("dropout", "learning_rate"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative number")
        if not isinstance(self.batching, Mapping):
            raise ValueError("batching must be a mapping")
        if not isinstance(self.layer_coverage, Mapping):
            raise ValueError("layer_coverage must be a mapping")
        if not isinstance(self.aggregate, Mapping):
            raise ValueError("aggregate must be a mapping")
        if set(self.aggregate) != _EXACT_AGGREGATE_FIELDS:
            raise ValueError(f"aggregate must contain exactly {sorted(_EXACT_AGGREGATE_FIELDS)}")
        if not isinstance(self.per_example_results, tuple):
            raise ValueError("per_example_results must be a tuple")
        if len(self.per_example_results) != self.n_total:
            raise ValueError("n_total must equal the number of per-example results")
        if self.eligible_checkpoint_steps != tuple(sorted(set(self.eligible_checkpoint_steps))):
            raise ValueError("eligible_checkpoint_steps must be unique and ascending")
        if any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in self.eligible_checkpoint_steps):
            raise ValueError("eligible_checkpoint_steps must contain non-negative integers")
        if self.completion_status not in set(ValidationSelectionStatus):
            raise ValueError("completion_status must be a canonical validation-selection status")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise ValueError("failure_reason must be a string or None")
        if self.schema_version != SELECTION_DECISION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SELECTION_DECISION_SCHEMA_VERSION!r}")
        family_order = _family_order()
        per_family = self.aggregate["per_family"]
        if not isinstance(per_family, Mapping):
            raise ValueError("aggregate per_family must be a mapping")
        if set(per_family) != set(family_order):
            raise ValueError(f"aggregate per_family must contain exactly {list(family_order)}")
        family_metrics: dict[str, dict[str, Any]] = dict(per_family)
        total_correct = 0
        for family in family_order:
            family_results = [result for result in self.per_example_results if result.task_family == family]
            family_n = len(family_results)
            family_correct = sum(1 for result in family_results if result.correct)
            metric = family_metrics[family]
            if metric["n"] != family_n:
                raise ValueError(f"aggregate family count drifted for {family}")
            if metric["correct"] != family_correct:
                raise ValueError(f"aggregate family correctness drifted for {family}")
            if metric["accuracy_fraction"] != _canonical_fraction(family_correct, family_n):
                raise ValueError(f"aggregate family accuracy fraction drifted for {family}")
            if metric["accuracy"] != float(Fraction(family_correct, family_n)):
                raise ValueError(f"aggregate family accuracy drifted for {family}")
            total_correct += family_correct
        overall_fraction = _canonical_fraction(total_correct, self.n_total)
        if self.aggregate["overall_correct"] != total_correct:
            raise ValueError("aggregate overall_correct drifted")
        if self.aggregate["overall_accuracy_fraction"] != overall_fraction:
            raise ValueError("aggregate overall_accuracy_fraction drifted")
        if self.aggregate["overall_accuracy"] != float(Fraction(total_correct, self.n_total)):
            raise ValueError("aggregate overall_accuracy drifted")
        macro_fraction = _macro_family_accuracy_fraction(family_metrics)
        if self.aggregate["macro_family_accuracy_fraction"] != _canonical_fraction(
            macro_fraction.numerator, macro_fraction.denominator
        ):
            raise ValueError("aggregate macro_family_accuracy_fraction drifted")
        if self.aggregate["macro_family_accuracy"] != float(macro_fraction):
            raise ValueError("aggregate macro_family_accuracy drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_hash": self.adapter_hash,
            "aggregate": dict(self.aggregate),
            "alpha": self.alpha,
            "base_identity_hash": self.base_identity_hash,
            "batching": dict(self.batching),
            "candidate_id": self.candidate_id,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "candidate_search_hash": self.candidate_search_hash,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_iteration": self.checkpoint_iteration,
            "completion_status": self.completion_status.value,
            "dropout": self.dropout,
            "eligible_checkpoint_steps": list(self.eligible_checkpoint_steps),
            "failure_reason": self.failure_reason,
            "layer_coverage": dict(self.layer_coverage),
            "learning_rate": self.learning_rate,
            "lora_policy_hash": self.lora_policy_hash,
            "n_total": self.n_total,
            "optimizer": self.optimizer,
            "per_example_results": [result.to_dict() for result in self.per_example_results],
            "provider_runtime_failure_count": self.provider_runtime_failure_count,
            "rank": self.rank,
            "schema_version": self.schema_version,
            "scheduler": self.scheduler,
            "seed": self.seed,
            "selection_policy_hash": self.selection_policy_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "target_modules": list(self.target_modules),
            "target_policy": self.target_policy,
            "training_config_hash": self.training_config_hash,
            "training_duration_iters": self.training_duration_iters,
            "training_formatter_hash": self.training_formatter_hash,
            "training_steps": self.training_steps,
            "trainable_parameter_count": self.trainable_parameter_count,
            "validation_example_ids_hash": self.validation_example_ids_hash,
            "validation_split_hash": self.validation_split_hash,
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationSelectionCandidateResult":
        allowed = {
            "adapter_hash",
            "aggregate",
            "alpha",
            "base_identity_hash",
            "batching",
            "candidate_id",
            "candidate_manifest_hash",
            "candidate_search_hash",
            "checkpoint_id",
            "checkpoint_iteration",
            "completion_status",
            "dropout",
            "eligible_checkpoint_steps",
            "failure_reason",
            "layer_coverage",
            "learning_rate",
            "lora_policy_hash",
            "n_total",
            "optimizer",
            "per_example_results",
            "provider_runtime_failure_count",
            "rank",
            "schema_version",
            "scheduler",
            "seed",
            "selection_policy_hash",
            "source_manifest_hash",
            "source_repository",
            "source_revision",
            "target_modules",
            "target_policy",
            "training_config_hash",
            "training_duration_iters",
            "training_formatter_hash",
            "training_steps",
            "trainable_parameter_count",
            "validation_example_ids_hash",
            "validation_split_hash",
        }
        _require_exact_keys(data, allowed, context="candidate result bundle")
        results = tuple(ValidationSelectionExampleResult.from_dict(item) for item in data["per_example_results"])
        return cls(
            candidate_id=data["candidate_id"],
            checkpoint_id=data["checkpoint_id"],
            checkpoint_iteration=data["checkpoint_iteration"],
            candidate_manifest_hash=data["candidate_manifest_hash"],
            candidate_search_hash=data["candidate_search_hash"],
            selection_policy_hash=data["selection_policy_hash"],
            adapter_hash=data["adapter_hash"],
            base_identity_hash=data["base_identity_hash"],
            validation_split_hash=data["validation_split_hash"],
            validation_example_ids_hash=data["validation_example_ids_hash"],
            source_repository=data["source_repository"],
            source_revision=data["source_revision"],
            source_manifest_hash=data["source_manifest_hash"],
            training_formatter_hash=data["training_formatter_hash"],
            lora_policy_hash=data["lora_policy_hash"],
            training_config_hash=data["training_config_hash"],
            seed=data["seed"],
            dropout=data["dropout"],
            optimizer=data["optimizer"],
            scheduler=data["scheduler"],
            batching=dict(data["batching"]),
            layer_coverage=dict(data["layer_coverage"]),
            rank=data["rank"],
            target_policy=data["target_policy"],
            alpha=data["alpha"],
            learning_rate=data["learning_rate"],
            training_duration_iters=data["training_duration_iters"],
            eligible_checkpoint_steps=tuple(data["eligible_checkpoint_steps"]),
            target_modules=tuple(data["target_modules"]),
            trainable_parameter_count=data["trainable_parameter_count"],
            training_steps=data["training_steps"],
            n_total=data["n_total"],
            per_example_results=results,
            aggregate=dict(data["aggregate"]),
            completion_status=ValidationSelectionStatus(data["completion_status"]),
            provider_runtime_failure_count=data["provider_runtime_failure_count"],
            failure_reason=data.get("failure_reason"),
            schema_version=str(data.get("schema_version", SELECTION_DECISION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class ValidationSelectionCandidateSummary:
    candidate_id: str
    checkpoint_id: str
    checkpoint_iteration: int
    status: str
    valid_for_selection: bool
    rank: int
    target_policy: str
    trainable_parameter_count: int
    training_steps: int
    n_total: int
    overall_correct: int
    overall_accuracy_fraction: dict[str, int]
    overall_accuracy: float
    per_family: dict[str, dict[str, Any]]
    macro_family_accuracy_fraction: dict[str, int]
    macro_family_accuracy: float
    candidate_manifest_hash: str
    candidate_result_hash: str
    provider_runtime_failure_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_iteration": self.checkpoint_iteration,
            "status": self.status,
            "valid_for_selection": self.valid_for_selection,
            "rank": self.rank,
            "target_policy": self.target_policy,
            "trainable_parameter_count": self.trainable_parameter_count,
            "training_steps": self.training_steps,
            "n_total": self.n_total,
            "overall_correct": self.overall_correct,
            "overall_accuracy_fraction": dict(self.overall_accuracy_fraction),
            "overall_accuracy": self.overall_accuracy,
            "per_family": self.per_family,
            "macro_family_accuracy_fraction": dict(self.macro_family_accuracy_fraction),
            "macro_family_accuracy": self.macro_family_accuracy,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "candidate_result_hash": self.candidate_result_hash,
            "provider_runtime_failure_count": self.provider_runtime_failure_count,
        }


@dataclass(frozen=True, slots=True)
class ValidationSelectionDecision:
    selection_run_id: str
    selection_policy_hash: str
    candidate_search_hash: str
    validation_split_hash: str
    validation_example_ids_hash: str
    candidate_budget_hash: str
    candidate_result_hashes: dict[str, str]
    status_counts: dict[str, int]
    candidate_count: int
    valid_candidate_count: int
    ineligible_candidate_count: int
    candidate_rankings: tuple[ValidationSelectionCandidateSummary, ...]
    selected_candidate_id: str
    selected_checkpoint_id: str
    selected_checkpoint_iteration: int
    selected_candidate_manifest_hash: str
    selected_candidate_result_hash: str
    selected_candidate_summary: ValidationSelectionCandidateSummary
    schema_version: str = SELECTION_DECISION_SCHEMA_VERSION
    runner_version: str = SELECTION_RUNNER_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "selection_run_id",
            "selection_policy_hash",
            "candidate_search_hash",
            "validation_split_hash",
            "validation_example_ids_hash",
            "candidate_budget_hash",
            "selected_candidate_id",
            "selected_checkpoint_id",
            "selected_candidate_manifest_hash",
            "selected_candidate_result_hash",
            "schema_version",
            "runner_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "candidate_count",
            "valid_candidate_count",
            "ineligible_candidate_count",
            "selected_checkpoint_iteration",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.candidate_result_hashes, dict):
            raise ValueError("candidate_result_hashes must be a mapping")
        if not isinstance(self.status_counts, dict):
            raise ValueError("status_counts must be a mapping")
        if tuple(self.candidate_rankings) != tuple(sorted(self.candidate_rankings, key=_summary_sort_key)):
            raise ValueError("candidate_rankings must be supplied in deterministic selection order")
        if self.schema_version != SELECTION_DECISION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SELECTION_DECISION_SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_budget_hash": self.candidate_budget_hash,
            "candidate_count": self.candidate_count,
            "candidate_rankings": [summary.to_dict() for summary in self.candidate_rankings],
            "candidate_result_hashes": dict(self.candidate_result_hashes),
            "candidate_search_hash": self.candidate_search_hash,
            "ineligible_candidate_count": self.ineligible_candidate_count,
            "runner_version": self.runner_version,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_manifest_hash": self.selected_candidate_manifest_hash,
            "selected_candidate_result_hash": self.selected_candidate_result_hash,
            "selected_candidate_summary": self.selected_candidate_summary.to_dict(),
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "selected_checkpoint_iteration": self.selected_checkpoint_iteration,
            "selection_policy_hash": self.selection_policy_hash,
            "selection_run_id": self.selection_run_id,
            "status_counts": dict(self.status_counts),
            "valid_candidate_count": self.valid_candidate_count,
            "validation_example_ids_hash": self.validation_example_ids_hash,
            "validation_split_hash": self.validation_split_hash,
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _bundle_manifest_payload(bundle: ValidationSelectionCandidateResult) -> dict[str, Any]:
    return {
        "adapter_hash": bundle.adapter_hash,
        "alpha": bundle.alpha,
        "base_identity_hash": bundle.base_identity_hash,
        "batching": dict(bundle.batching),
        "candidate_id": bundle.candidate_id,
        "checkpoint_id": bundle.checkpoint_id,
        "checkpoint_iteration": bundle.checkpoint_iteration,
        "candidate_search_hash": bundle.candidate_search_hash,
        "dropout": bundle.dropout,
        "eligible_checkpoint_steps": list(bundle.eligible_checkpoint_steps),
        "layer_coverage": dict(bundle.layer_coverage),
        "learning_rate": bundle.learning_rate,
        "lora_policy_hash": bundle.lora_policy_hash,
        "optimizer": bundle.optimizer,
        "rank": bundle.rank,
        "scheduler": bundle.scheduler,
        "seed": bundle.seed,
        "selection_policy_hash": bundle.selection_policy_hash,
        "source_manifest_hash": bundle.source_manifest_hash,
        "source_repository": bundle.source_repository,
        "source_revision": bundle.source_revision,
        "target_modules": list(bundle.target_modules),
        "target_policy": bundle.target_policy,
        "training_config_hash": bundle.training_config_hash,
        "training_duration_iters": bundle.training_duration_iters,
        "training_formatter_hash": bundle.training_formatter_hash,
        "training_steps": bundle.training_steps,
        "trainable_parameter_count": bundle.trainable_parameter_count,
        "validation_example_ids_hash": bundle.validation_example_ids_hash,
        "validation_split_hash": bundle.validation_split_hash,
    }


def _expected_candidate_fields(policy: Mapping[str, Any], bundle: ValidationSelectionCandidateResult) -> dict[str, Any]:
    frozen = policy["frozen_inputs"]
    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in policy["candidate_budget"]["candidate_records"]}
    candidate = candidate_lookup.get(bundle.candidate_id)
    if candidate is None:
        raise ValueError(f"candidate_id {bundle.candidate_id!r} is not present in the frozen candidate budget")
    return {
        "candidate": candidate,
        "frozen_inputs": frozen,
        "candidate_budget_hash": sha256_bytes(canonical_json_bytes(policy["candidate_budget"])),
    }


def _exact_family_metrics(per_example_results: Sequence[ValidationSelectionExampleResult], validation_examples: Sequence[BenchmarkExample]) -> tuple[dict[str, dict[str, Any]], int, int, dict[str, int], dict[str, int]]:
    expected_by_id = {example.example_id: example for example in validation_examples}
    family_order = _family_order()
    family_counts = {family: 0 for family in family_order}
    family_correct = {family: 0 for family in family_order}
    total_correct = 0
    seen: set[str] = set()

    for result in per_example_results:
        if result.example_id in seen:
            raise ValueError(f"duplicate validation example_id in bundle: {result.example_id}")
        seen.add(result.example_id)
        example = expected_by_id.get(result.example_id)
        if example is None:
            raise ValueError(f"unknown validation example_id in bundle: {result.example_id}")
        if result.task_family != example.task_family.value:
            raise ValueError(f"task_family mismatch for example {result.example_id}")
        if result.raw_output is None:
            raise ValueError(f"VALID result bundle must include raw_output for example {result.example_id}")
        scored = score_output(example, result.raw_output)
        if result.normalized_output != scored.normalized_output:
            raise ValueError(f"normalized output drift for example {result.example_id}")
        if result.correct != bool(scored.score == 1.0):
            raise ValueError(f"correct flag drift for example {result.example_id}")
        family = result.task_family
        family_counts[family] += 1
        if result.correct:
            family_correct[family] += 1
            total_correct += 1

    family_metrics: dict[str, dict[str, Any]] = {}
    for family in family_order:
        n = family_counts[family]
        correct = family_correct[family]
        family_metrics[family] = {
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else None,
            "accuracy_fraction": _canonical_fraction(correct, n) if n else None,
        }
    return family_metrics, total_correct, len(per_example_results), family_counts, family_correct


def _macro_family_accuracy_fraction(family_metrics: Mapping[str, Mapping[str, Any]]) -> Fraction:
    values = []
    for family in _family_order():
        metric = family_metrics[family]
        if metric["n"] == 0:
            raise ValueError(f"validation family {family} is missing from the bundle")
        values.append(_fraction_value(metric["correct"], metric["n"]))
    return sum(values, Fraction(0, 1)) / len(values)


def _candidate_summary(
    *,
    bundle: ValidationSelectionCandidateResult,
    candidate_manifest_hash: str,
    candidate_result_hash: str,
    validation_examples: Sequence[BenchmarkExample],
) -> ValidationSelectionCandidateSummary:
    family_metrics, total_correct, n_total, _, _ = _exact_family_metrics(bundle.per_example_results, validation_examples)
    macro_fraction = _macro_family_accuracy_fraction(family_metrics)
    overall_fraction = _fraction_value(total_correct, n_total) if n_total else Fraction(0, 1)
    valid = bundle.completion_status is ValidationSelectionStatus.VALID
    return ValidationSelectionCandidateSummary(
        candidate_id=bundle.candidate_id,
        checkpoint_id=bundle.checkpoint_id,
        checkpoint_iteration=bundle.checkpoint_iteration,
        status=bundle.completion_status.value,
        valid_for_selection=valid,
        rank=bundle.rank,
        target_policy=bundle.target_policy,
        trainable_parameter_count=bundle.trainable_parameter_count,
        training_steps=bundle.training_steps,
        n_total=n_total,
        overall_correct=total_correct,
        overall_accuracy_fraction=_canonical_fraction(total_correct, n_total) if n_total else {"numerator": 0, "denominator": 1},
        overall_accuracy=float(overall_fraction),
        per_family=family_metrics,
        macro_family_accuracy_fraction=_canonical_fraction(macro_fraction.numerator, macro_fraction.denominator),
        macro_family_accuracy=float(macro_fraction),
        candidate_manifest_hash=candidate_manifest_hash,
        candidate_result_hash=candidate_result_hash,
        provider_runtime_failure_count=bundle.provider_runtime_failure_count,
    )


def _rank_key(summary: ValidationSelectionCandidateSummary) -> tuple[Any, ...]:
    macro = Fraction(summary.macro_family_accuracy_fraction["numerator"], summary.macro_family_accuracy_fraction["denominator"])
    overall = Fraction(summary.overall_accuracy_fraction["numerator"], summary.overall_accuracy_fraction["denominator"])
    return (
        -macro,
        -overall,
        summary.rank,
        summary.trainable_parameter_count,
        summary.training_steps,
        summary.trainable_parameter_count,
        summary.candidate_id,
    )


def _summary_sort_key(summary: ValidationSelectionCandidateSummary) -> tuple[Any, ...]:
    if summary.valid_for_selection:
        return (0, *_rank_key(summary))
    status_priority = {
        ValidationSelectionStatus.PROTOCOL_VIOLATION.value: 0,
        ValidationSelectionStatus.RESOURCE_INFEASIBLE.value: 1,
        ValidationSelectionStatus.TRAINING_FAILED.value: 2,
        ValidationSelectionStatus.VALIDATION_INCOMPLETE.value: 3,
    }
    return (
        1,
        status_priority.get(summary.status, 99),
        summary.candidate_id,
        summary.checkpoint_iteration,
    )


def load_validation_selection_candidate_result(path: str | Path) -> ValidationSelectionCandidateResult:
    return ValidationSelectionCandidateResult.from_dict(_load_json(path))


def _validate_candidate_bundle_against_frozen_policy(
    *,
    policy: Mapping[str, Any],
    training_config: Mapping[str, Any],
    validation_examples: list[BenchmarkExample],
    validation_example_ids: set[str],
    validation_split_hash: str,
    validation_example_ids_hash: str,
    candidate_budget_hash: str,
    selection_policy_hash: str,
    bundle: ValidationSelectionCandidateResult,
) -> tuple[ValidationSelectionCandidateSummary, str]:
    candidate_budget_ids = {candidate["candidate_id"] for candidate in policy["candidate_budget"]["candidate_records"]}
    if bundle.candidate_id not in candidate_budget_ids:
        raise ValueError(f"candidate_id {bundle.candidate_id} is not in the frozen canonical candidate budget")

    if bundle.candidate_search_hash != candidate_budget_hash:
        raise ValueError("candidate-search hash drifted from the frozen candidate budget")
    if bundle.selection_policy_hash != selection_policy_hash:
        raise ValueError("selection-policy hash drifted from the frozen policy")
    if bundle.validation_split_hash != validation_split_hash:
        raise ValueError("validation split hash drifted from the frozen validation split")
    if bundle.validation_example_ids_hash != validation_example_ids_hash:
        raise ValueError("validation example-ID-set hash drifted from the frozen validation split")

    expected = _expected_candidate_fields(policy, bundle)
    candidate = expected["candidate"]
    policy_lineage = expected["frozen_inputs"]["source_lineage"]
    if bundle.source_repository != policy_lineage["repository"]:
        raise ValueError("bundle source repository drifted from the frozen training config")
    if bundle.source_revision != policy_lineage["revision"]:
        raise ValueError("bundle source revision drifted from the frozen training config")
    if bundle.source_manifest_hash != policy_lineage["source_manifest_hash"]:
        raise ValueError("bundle source manifest hash drifted from the frozen training config")
    if bundle.training_formatter_hash != expected["frozen_inputs"]["training_formatter_hash"]:
        raise ValueError("bundle training formatter hash drifted from the frozen training config")
    if bundle.lora_policy_hash != training_config["frozen_inputs"]["lora_policy_hash"]:
        raise ValueError("bundle LoRA policy hash drifted from the frozen training config")
    if bundle.training_config_hash != expected["frozen_inputs"]["training_config_hash"]:
        raise ValueError("bundle training config hash drifted from the frozen training config")
    if bundle.base_identity_hash != training_config["numeric_and_gradient_policy"]["base_representation"]["identity_hash"]:
        raise ValueError("bundle base identity hash drifted from the frozen training config")
    if bundle.seed != int(training_config["numeric_and_gradient_policy"]["seed_policy"]["canonical_seed"]):
        raise ValueError("bundle seed drifted from the frozen training config")
    if bundle.dropout != float(training_config["default_candidate_provenance"]["dropout"]):
        raise ValueError("bundle dropout drifted from the frozen training config")
    if bundle.optimizer != str(training_config["optimizer_policy"]["family"]):
        raise ValueError("bundle optimizer drifted from the frozen training config")
    if bundle.scheduler != str(training_config["scheduler_policy"]["type"]):
        raise ValueError("bundle scheduler drifted from the frozen training config")
    if dict(bundle.batching) != dict(training_config["batching_policy"]):
        raise ValueError("bundle batching policy drifted from the frozen training config")
    if dict(bundle.layer_coverage) != dict(training_config["provenance_validation_policy"]["frozen_layer_coverage"]):
        raise ValueError("bundle layer coverage drifted from the frozen training config")

    if bundle.rank != int(candidate["rank"]):
        raise ValueError("bundle rank drifted from the frozen candidate budget")
    if bundle.target_policy != str(candidate["target_policy"]):
        raise ValueError("bundle target policy drifted from the frozen candidate budget")
    if bundle.alpha != int(candidate["alpha"]):
        raise ValueError("bundle alpha drifted from the frozen candidate budget")
    if bundle.learning_rate != float(candidate["learning_rate"]):
        raise ValueError("bundle learning rate drifted from the frozen candidate budget")
    if bundle.training_duration_iters != int(candidate["training_duration_iters"]):
        raise ValueError("bundle training duration drifted from the frozen candidate budget")
    if tuple(bundle.eligible_checkpoint_steps) != tuple(candidate["eligible_checkpoint_steps"]):
        raise ValueError("bundle eligible checkpoint steps drifted from the frozen candidate budget")
    if bundle.training_steps != bundle.training_duration_iters:
        raise ValueError("training steps must equal the declared training duration for canonical bundles")
    if bundle.checkpoint_iteration not in bundle.eligible_checkpoint_steps:
        raise ValueError("checkpoint iteration is not eligible under the frozen policy")
    if bundle.n_total != len(bundle.per_example_results):
        raise ValueError("n_total must equal the number of per-example results")
    if bundle.completion_status is ValidationSelectionStatus.VALID and bundle.n_total != len(validation_examples):
        raise ValueError("VALID bundles must contain the complete validation split")
    if bundle.provider_runtime_failure_count < 0:
        raise ValueError("provider/runtime failure count must be non-negative")
    if bundle.completion_status is ValidationSelectionStatus.VALID and bundle.provider_runtime_failure_count != 0:
        raise ValueError("VALID bundles must not report provider/runtime failures")
    if len({result.example_id for result in bundle.per_example_results}) != len(bundle.per_example_results):
        raise ValueError("per-example results contain duplicate example IDs")
    if {result.example_id for result in bundle.per_example_results} != validation_example_ids:
        raise ValueError("bundle does not cover the frozen validation example-ID set exactly")
    if bundle.completion_status is ValidationSelectionStatus.VALIDATION_INCOMPLETE:
        raise ValueError("validation-incomplete bundles cannot participate in canonical selection")

    manifest_payload = _bundle_manifest_payload(bundle)
    expected_manifest_hash = sha256_bytes(canonical_json_bytes(manifest_payload))
    if expected_manifest_hash != bundle.candidate_manifest_hash:
        raise ValueError("candidate manifest hash does not match the declared candidate payload")

    summary = _candidate_summary(
        bundle=bundle,
        candidate_manifest_hash=bundle.candidate_manifest_hash,
        candidate_result_hash=sha256_bytes(bundle.to_json_bytes()),
        validation_examples=validation_examples,
    )
    return summary, sha256_bytes(bundle.to_json_bytes())


def build_validation_selection_decision(
    *,
    selection_policy_path: str | Path,
    training_config_path: str | Path,
    validation_path: str | Path,
    candidate_result_paths: Sequence[str | Path],
) -> ValidationSelectionDecision:
    selection_policy_path = Path(selection_policy_path)
    training_config_path = Path(training_config_path)
    validation_path = Path(validation_path)
    policy = _load_json(selection_policy_path)
    training_config = _load_json(training_config_path)
    if policy.get("config_hash") is None:
        raise ValueError("selection policy artifact is missing config_hash")
    if policy.get("gate") != "M5_VALIDATION_SELECTION_POLICY_READY":
        raise ValueError("selection policy is not frozen-ready")
    if policy.get("policy_version") not in CANONICAL_VALIDATION_SELECTION_POLICY_VERSIONS:
        raise ValueError("selection policy version drifted")
    expected_training_config_hash = policy["frozen_inputs"]["training_config_hash"]
    if training_config.get("config_hash") != expected_training_config_hash:
        raise ValueError("selection policy training config hash drifted")

    validation_examples = _sorted_validation_examples(validation_path)
    validation_example_ids = {example.example_id for example in validation_examples}
    validation_split_hash, validation_example_ids_hash = _validation_hashes(validation_path)
    candidate_budget_hash = sha256_bytes(canonical_json_bytes(policy["candidate_budget"]))
    selection_policy_hash = str(policy["config_hash"])

    bundles = [load_validation_selection_candidate_result(path) for path in candidate_result_paths]
    if not bundles:
        raise ValueError("at least one candidate result bundle is required")
    bundle_ids = {bundle.candidate_id for bundle in bundles}
    candidate_budget_ids = {candidate["candidate_id"] for candidate in policy["candidate_budget"]["candidate_records"]}
    if bundle_ids != candidate_budget_ids:
        missing = sorted(candidate_budget_ids - bundle_ids)
        extra = sorted(bundle_ids - candidate_budget_ids)
        if extra:
            raise ValueError(f"candidate_id {extra[0]} is not in the frozen canonical candidate budget")
        raise ValueError(
            "candidate result bundles must cover the frozen candidate budget exactly "
            f"(missing={missing}, extra={extra})"
        )
    if len(bundle_ids) != len(bundles):
        raise ValueError("candidate result bundles contain duplicate candidate IDs")

    summaries: list[ValidationSelectionCandidateSummary] = []
    result_hashes: dict[str, str] = {}
    status_counts: dict[str, int] = {}
    for bundle in bundles:
        summary, result_hash = _validate_candidate_bundle_against_frozen_policy(
            policy=policy,
            training_config=training_config,
            validation_examples=validation_examples,
            validation_example_ids=validation_example_ids,
            validation_split_hash=validation_split_hash,
            validation_example_ids_hash=validation_example_ids_hash,
            candidate_budget_hash=candidate_budget_hash,
            selection_policy_hash=selection_policy_hash,
            bundle=bundle,
        )
        result_hashes[bundle.candidate_id] = result_hash
        status_counts[bundle.completion_status.value] = status_counts.get(bundle.completion_status.value, 0) + 1
        summaries.append(summary)

    valid_summaries = [summary for summary in summaries if summary.valid_for_selection]
    if not valid_summaries:
        raise ValueError("no VALID candidate bundles were provided")

    ranked = sorted(valid_summaries, key=_rank_key)
    selected = ranked[0]

    run_identity = {
        "candidate_budget_hash": candidate_budget_hash,
        "candidate_result_hashes": result_hashes,
        "candidate_search_hash": candidate_budget_hash,
        "selection_policy_hash": selection_policy_hash,
        "validation_example_ids_hash": validation_example_ids_hash,
        "validation_split_hash": validation_split_hash,
        "runner_version": SELECTION_RUNNER_VERSION,
        "selected_candidate_id": selected.candidate_id,
        "selected_checkpoint_id": selected.checkpoint_id,
        "selected_checkpoint_iteration": selected.checkpoint_iteration,
    }
    selection_run_id = "m5-validation-selection-" + sha256_bytes(canonical_json_bytes(run_identity))[:16]

    candidate_rankings = tuple(sorted(summaries, key=_summary_sort_key))

    decision = ValidationSelectionDecision(
        selection_run_id=selection_run_id,
        selection_policy_hash=selection_policy_hash,
        candidate_search_hash=candidate_budget_hash,
        validation_split_hash=validation_split_hash,
        validation_example_ids_hash=validation_example_ids_hash,
        candidate_budget_hash=candidate_budget_hash,
        candidate_result_hashes=result_hashes,
        status_counts=status_counts,
        candidate_count=len(bundles),
        valid_candidate_count=len(valid_summaries),
        ineligible_candidate_count=len(bundles) - len(valid_summaries),
        candidate_rankings=candidate_rankings,
        selected_candidate_id=selected.candidate_id,
        selected_checkpoint_id=selected.checkpoint_id,
        selected_checkpoint_iteration=selected.checkpoint_iteration,
        selected_candidate_manifest_hash=selected.candidate_manifest_hash,
        selected_candidate_result_hash=selected.candidate_result_hash,
        selected_candidate_summary=selected,
    )
    return decision


def write_validation_selection_decision_artifact(
    *,
    selection_policy_path: str | Path,
    training_config_path: str | Path,
    validation_path: str | Path,
    candidate_result_paths: Sequence[str | Path],
    output_path: str | Path,
) -> ValidationSelectionDecision:
    decision = build_validation_selection_decision(
        selection_policy_path=selection_policy_path,
        training_config_path=training_config_path,
        validation_path=validation_path,
        candidate_result_paths=candidate_result_paths,
    )
    output_path = Path(output_path)
    data = write_json(output_path, decision.to_dict())
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_bytes(data)}  {output_path.name}\n",
        encoding="utf-8",
    )
    return decision
