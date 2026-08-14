"""Typed precommitment policy for future AdaptLab model evaluation.

This module records the search/selection boundaries that must exist before
canonical model results are observed.  It does not implement prompting, LoRA,
or model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class EvaluationPolicyError(ValueError):
    """Raised when the v0.0 evaluation precommitment is malformed."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationPolicyError(f"{name} must be a mapping")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationPolicyError(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise EvaluationPolicyError(f"{name} must be a positive number")
    return float(value)


@dataclass(frozen=True, slots=True)
class PromptBudget:
    max_candidate_templates_per_task_family: int
    max_total_candidate_templates: int
    max_few_shot_examples: int


@dataclass(frozen=True, slots=True)
class PromptSelectionProcedure:
    selection_split: str
    primary_objective: str
    tie_breakers: tuple[str, ...]
    test_set_access_during_selection: bool
    sentinel_access_during_selection: bool


@dataclass(frozen=True, slots=True)
class ValidationUsageBudget:
    max_full_validation_passes_per_method: int
    max_validation_passes_per_candidate: int
    subgroup_analysis_during_selection: str


@dataclass(frozen=True, slots=True)
class GeneralizationGuardrail:
    artifact: str
    maximum_absolute_degradation_percentage_points: float
    selection_use: str


@dataclass(frozen=True, slots=True)
class MinimumMeaningfulImprovement:
    metric: str
    absolute_percentage_points: float


@dataclass(frozen=True, slots=True)
class CanonicalLoraSeedPolicy:
    tuning_seed: int
    confirmatory_seeds: tuple[int, ...]
    canonical_report: str
    single_seed_claims_allowed: bool


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    policy_name: str
    policy_version: str
    benchmark_version: str
    status: str
    prompt_budget: PromptBudget
    prompt_selection_procedure: PromptSelectionProcedure
    validation_set_usage_budget: ValidationUsageBudget
    allowed_future_prompt_search_space: Mapping[str, Any]
    allowed_future_lora_hyperparameter_search_space: Mapping[str, Any]
    generalization_guardrail: GeneralizationGuardrail
    minimum_meaningful_improvement: MinimumMeaningfulImprovement
    subgroup_policy: Mapping[str, Any]
    architecture_sensitive_retention_rules: Mapping[str, Any]
    canonical_lora_seed_policy: CanonicalLoraSeedPolicy

    def validate(self) -> None:
        if self.status != "PRECOMMITTED_BEFORE_MODEL_RESULTS":
            raise EvaluationPolicyError(
                "status must be PRECOMMITTED_BEFORE_MODEL_RESULTS"
            )
        if self.prompt_selection_procedure.selection_split != "validation":
            raise EvaluationPolicyError("prompt selection must use validation split")
        if self.prompt_selection_procedure.test_set_access_during_selection:
            raise EvaluationPolicyError("test set access during prompt selection is prohibited")
        if self.prompt_selection_procedure.sentinel_access_during_selection:
            raise EvaluationPolicyError("sentinel access during prompt selection is prohibited")

        budget = self.prompt_budget
        for name, value in (
            ("max_candidate_templates_per_task_family", budget.max_candidate_templates_per_task_family),
            ("max_total_candidate_templates", budget.max_total_candidate_templates),
            ("max_few_shot_examples", budget.max_few_shot_examples),
            ("max_full_validation_passes_per_method", self.validation_set_usage_budget.max_full_validation_passes_per_method),
            ("max_validation_passes_per_candidate", self.validation_set_usage_budget.max_validation_passes_per_candidate),
        ):
            _non_negative_int(value, name)
        if budget.max_total_candidate_templates < budget.max_candidate_templates_per_task_family:
            raise EvaluationPolicyError(
                "max_total_candidate_templates cannot be smaller than the per-family budget"
            )

        _positive_number(
            self.generalization_guardrail.maximum_absolute_degradation_percentage_points,
            "generalization guardrail threshold",
        )
        _positive_number(
            self.minimum_meaningful_improvement.absolute_percentage_points,
            "minimum meaningful improvement",
        )
        if self.generalization_guardrail.selection_use != "guardrail_only":
            raise EvaluationPolicyError("generalization sentinel must be guardrail_only")

        prompt_space = self.allowed_future_prompt_search_space
        if prompt_space.get("benchmark_answer_or_test_content_in_prompts") is not False:
            raise EvaluationPolicyError("benchmark answers/test content must be prohibited in prompts")

        lora_space = self.allowed_future_lora_hyperparameter_search_space
        for key in ("ranks", "alphas", "dropouts", "learning_rates", "epochs", "target_modules"):
            if key not in lora_space:
                raise EvaluationPolicyError(f"LoRA search space is missing {key}")

        retention = self.architecture_sensitive_retention_rules
        required_retention = {
            "compare_adaptation_methods_within_same_base_model": True,
            "retain_prompt_choice_across_architectures": False,
            "retain_lora_hyperparameters_across_architectures": False,
            "retune_only_within_precommitted_search_space": True,
        }
        for key, expected in required_retention.items():
            if retention.get(key) is not expected:
                raise EvaluationPolicyError(f"invalid architecture retention rule: {key}")

        seeds = self.canonical_lora_seed_policy
        if seeds.tuning_seed not in seeds.confirmatory_seeds:
            raise EvaluationPolicyError("LoRA tuning seed must be one of the confirmatory seeds")
        if len(set(seeds.confirmatory_seeds)) != len(seeds.confirmatory_seeds):
            raise EvaluationPolicyError("LoRA confirmatory seeds must be unique")
        if seeds.single_seed_claims_allowed:
            raise EvaluationPolicyError("single-seed canonical LoRA claims must be disabled")


DEFAULT_EVALUATION_POLICY = Path("configs/evaluation_policy_v0.0.yaml")


def evaluation_policy_from_mapping(raw: Mapping[str, Any]) -> EvaluationPolicy:
    expected = {
        "policy_name",
        "policy_version",
        "benchmark_version",
        "status",
        "prompt_budget",
        "prompt_selection_procedure",
        "validation_set_usage_budget",
        "allowed_future_prompt_search_space",
        "allowed_future_lora_hyperparameter_search_space",
        "generalization_guardrail",
        "minimum_meaningful_improvement",
        "subgroup_policy",
        "architecture_sensitive_retention_rules",
        "canonical_lora_seed_policy",
    }
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise EvaluationPolicyError(f"policy is missing fields: {sorted(missing)}")
    if unknown:
        raise EvaluationPolicyError(f"policy has unknown fields: {sorted(unknown)}")

    prompt_budget = _mapping(raw["prompt_budget"], "prompt_budget")
    selection = _mapping(raw["prompt_selection_procedure"], "prompt_selection_procedure")
    validation = _mapping(raw["validation_set_usage_budget"], "validation_set_usage_budget")
    guardrail = _mapping(raw["generalization_guardrail"], "generalization_guardrail")
    improvement = _mapping(raw["minimum_meaningful_improvement"], "minimum_meaningful_improvement")
    seeds = _mapping(raw["canonical_lora_seed_policy"], "canonical_lora_seed_policy")

    policy = EvaluationPolicy(
        policy_name=str(raw["policy_name"]),
        policy_version=str(raw["policy_version"]),
        benchmark_version=str(raw["benchmark_version"]),
        status=str(raw["status"]),
        prompt_budget=PromptBudget(
            max_candidate_templates_per_task_family=_non_negative_int(prompt_budget.get("max_candidate_templates_per_task_family"), "prompt_budget.max_candidate_templates_per_task_family"),
            max_total_candidate_templates=_non_negative_int(prompt_budget.get("max_total_candidate_templates"), "prompt_budget.max_total_candidate_templates"),
            max_few_shot_examples=_non_negative_int(prompt_budget.get("max_few_shot_examples"), "prompt_budget.max_few_shot_examples"),
        ),
        prompt_selection_procedure=PromptSelectionProcedure(
            selection_split=str(selection.get("selection_split")),
            primary_objective=str(selection.get("primary_objective")),
            tie_breakers=tuple(str(item) for item in selection.get("tie_breakers", ())),
            test_set_access_during_selection=bool(selection.get("test_set_access_during_selection")),
            sentinel_access_during_selection=bool(selection.get("sentinel_access_during_selection")),
        ),
        validation_set_usage_budget=ValidationUsageBudget(
            max_full_validation_passes_per_method=_non_negative_int(validation.get("max_full_validation_passes_per_method"), "validation_set_usage_budget.max_full_validation_passes_per_method"),
            max_validation_passes_per_candidate=_non_negative_int(validation.get("max_validation_passes_per_candidate"), "validation_set_usage_budget.max_validation_passes_per_candidate"),
            subgroup_analysis_during_selection=str(validation.get("subgroup_analysis_during_selection")),
        ),
        allowed_future_prompt_search_space=dict(_mapping(raw["allowed_future_prompt_search_space"], "allowed_future_prompt_search_space")),
        allowed_future_lora_hyperparameter_search_space=dict(_mapping(raw["allowed_future_lora_hyperparameter_search_space"], "allowed_future_lora_hyperparameter_search_space")),
        generalization_guardrail=GeneralizationGuardrail(
            artifact=str(guardrail.get("artifact")),
            maximum_absolute_degradation_percentage_points=_positive_number(guardrail.get("maximum_absolute_degradation_percentage_points"), "generalization_guardrail.maximum_absolute_degradation_percentage_points"),
            selection_use=str(guardrail.get("selection_use")),
        ),
        minimum_meaningful_improvement=MinimumMeaningfulImprovement(
            metric=str(improvement.get("metric")),
            absolute_percentage_points=_positive_number(improvement.get("absolute_percentage_points"), "minimum_meaningful_improvement.absolute_percentage_points"),
        ),
        subgroup_policy=dict(_mapping(raw["subgroup_policy"], "subgroup_policy")),
        architecture_sensitive_retention_rules=dict(_mapping(raw["architecture_sensitive_retention_rules"], "architecture_sensitive_retention_rules")),
        canonical_lora_seed_policy=CanonicalLoraSeedPolicy(
            tuning_seed=_non_negative_int(seeds.get("tuning_seed"), "canonical_lora_seed_policy.tuning_seed"),
            confirmatory_seeds=tuple(_non_negative_int(seed, "canonical_lora_seed_policy.confirmatory_seeds") for seed in seeds.get("confirmatory_seeds", ())),
            canonical_report=str(seeds.get("canonical_report")),
            single_seed_claims_allowed=bool(seeds.get("single_seed_claims_allowed")),
        ),
    )
    policy.validate()
    return policy


def load_evaluation_policy(path: str | Path = DEFAULT_EVALUATION_POLICY) -> EvaluationPolicy:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationPolicyError(f"could not read evaluation policy {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EvaluationPolicyError(f"invalid YAML in evaluation policy {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise EvaluationPolicyError("evaluation policy root must be a mapping")
    return evaluation_policy_from_mapping(raw)
