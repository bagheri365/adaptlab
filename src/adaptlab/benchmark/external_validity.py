"""Typed external-validity benchmark-selection precommitment.

This module records how a future external benchmark will be selected before
canonical AdaptLab model results are observed. It does not download, run, or
score any external benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ExternalValidityPolicyError(ValueError):
    """Raised when the external-validity precommitment is malformed."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalValidityPolicyError(f"{name} must be a mapping")
    return value


def _non_empty_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ExternalValidityPolicyError(f"{name} must be a non-empty list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ExternalValidityPolicyError(f"{name} must not contain empty values")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExternalValidityPolicyError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkCandidate:
    benchmark_id: str
    public_reference: str
    capability_focus: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionRule:
    timing: str
    eligible_only_if_all: tuple[str, ...]
    rank_eligible_candidates_by: tuple[str, ...]
    no_eligible_candidate_action: str


@dataclass(frozen=True, slots=True)
class SampleSizeTarget:
    target_examples: int
    minimum_examples: int
    maximum_examples: int
    sampling_rule: str
    sampling_seed: int


@dataclass(frozen=True, slots=True)
class ScoringApproach:
    primary_metric: str
    fallback_metric: str
    aggregation: str
    report_absolute_change_from_same_base_model_baseline: bool
    no_prompt_or_hyperparameter_selection_on_external_benchmark: bool


@dataclass(frozen=True, slots=True)
class ContaminationReview:
    timing: str
    checks: tuple[str, ...]
    dispositions: tuple[str, ...]
    exclusion_on_direct_project_leakage: bool


@dataclass(frozen=True, slots=True)
class DirectionalTransferClaim:
    claim_id: str
    statement: str
    direction_test: str
    magnitude_equivalence_claimed: bool
    subgroup_transfer_is_exploratory: bool


@dataclass(frozen=True, slots=True)
class ExternalValidityPolicy:
    policy_name: str
    policy_version: str
    benchmark_version: str
    status: str
    candidate_benchmarks: tuple[ExternalBenchmarkCandidate, ...]
    selection_rule: SelectionRule
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    sample_size_target: SampleSizeTarget
    scoring_approach: ScoringApproach
    contamination_review: ContaminationReview
    predeclared_directional_transfer_claim: DirectionalTransferClaim
    provenance: Mapping[str, Any]

    def validate(self) -> None:
        if self.status != "PRECOMMITTED_BEFORE_CANONICAL_MODEL_RESULTS":
            raise ExternalValidityPolicyError(
                "status must be PRECOMMITTED_BEFORE_CANONICAL_MODEL_RESULTS"
            )
        if len(self.candidate_benchmarks) < 2:
            raise ExternalValidityPolicyError("at least two candidate benchmarks are required")
        ids = tuple(candidate.benchmark_id for candidate in self.candidate_benchmarks)
        if len(set(ids)) != len(ids):
            raise ExternalValidityPolicyError("candidate benchmark IDs must be unique")
        if self.selection_rule.timing != "before_canonical_adaptlab_model_results":
            raise ExternalValidityPolicyError("benchmark selection must be frozen before canonical AdaptLab model results")
        if not self.selection_rule.eligible_only_if_all or not self.selection_rule.rank_eligible_candidates_by:
            raise ExternalValidityPolicyError("selection rule must define eligibility and deterministic ranking")
        if self.selection_rule.rank_eligible_candidates_by[-1] != "lexicographically_smaller_benchmark_id":
            raise ExternalValidityPolicyError("selection rule must end with a deterministic lexical tie-breaker")

        sample = self.sample_size_target
        if not (sample.minimum_examples <= sample.target_examples <= sample.maximum_examples):
            raise ExternalValidityPolicyError("sample-size target must fall between minimum and maximum")
        _positive_int(sample.sampling_seed, "sample_size_target.sampling_seed")

        scoring = self.scoring_approach
        if not scoring.report_absolute_change_from_same_base_model_baseline:
            raise ExternalValidityPolicyError("external scoring must compare against the same base-model baseline")
        if not scoring.no_prompt_or_hyperparameter_selection_on_external_benchmark:
            raise ExternalValidityPolicyError("external benchmark must not be used for prompt or hyperparameter selection")

        review = self.contamination_review
        if review.timing != "after_model_identity_is_fixed_but_before_external_evaluation":
            raise ExternalValidityPolicyError("contamination review timing is not precommitted correctly")
        if not review.exclusion_on_direct_project_leakage:
            raise ExternalValidityPolicyError("direct project leakage must exclude an external candidate")

        claim = self.predeclared_directional_transfer_claim
        if claim.direction_test != "non_negative_change":
            raise ExternalValidityPolicyError("directional transfer must use the predeclared non-negative change test")
        if claim.magnitude_equivalence_claimed:
            raise ExternalValidityPolicyError("external validity precommitment must not claim magnitude equivalence")

        provenance = self.provenance
        if provenance.get("include_in_final_benchmark_manifest") is not True:
            raise ExternalValidityPolicyError("external-validity policy must be included in final benchmark provenance")
        if provenance.get("policy_artifact") != "configs/external_validity_v0.0.yaml":
            raise ExternalValidityPolicyError("external-validity provenance artifact path is invalid")


DEFAULT_EXTERNAL_VALIDITY_POLICY = Path("configs/external_validity_v0.0.yaml")


def external_validity_policy_from_mapping(raw: Mapping[str, Any]) -> ExternalValidityPolicy:
    expected = {
        "policy_name",
        "policy_version",
        "benchmark_version",
        "status",
        "candidate_benchmarks",
        "selection_rule",
        "inclusion_criteria",
        "exclusion_criteria",
        "sample_size_target",
        "scoring_approach",
        "contamination_review",
        "predeclared_directional_transfer_claim",
        "provenance",
    }
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise ExternalValidityPolicyError(f"policy is missing fields: {sorted(missing)}")
    if unknown:
        raise ExternalValidityPolicyError(f"policy has unknown fields: {sorted(unknown)}")

    raw_candidates = raw["candidate_benchmarks"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ExternalValidityPolicyError("candidate_benchmarks must be a non-empty list")
    candidates = tuple(
        ExternalBenchmarkCandidate(
            benchmark_id=str(_mapping(item, "candidate benchmark").get("benchmark_id", "")).strip(),
            public_reference=str(_mapping(item, "candidate benchmark").get("public_reference", "")).strip(),
            capability_focus=_non_empty_tuple(_mapping(item, "candidate benchmark").get("capability_focus"), "candidate capability_focus"),
        )
        for item in raw_candidates
    )
    if any(not item.benchmark_id or not item.public_reference for item in candidates):
        raise ExternalValidityPolicyError("candidate benchmark ID/reference must be non-empty")

    selection = _mapping(raw["selection_rule"], "selection_rule")
    sample = _mapping(raw["sample_size_target"], "sample_size_target")
    scoring = _mapping(raw["scoring_approach"], "scoring_approach")
    review = _mapping(raw["contamination_review"], "contamination_review")
    claim = _mapping(raw["predeclared_directional_transfer_claim"], "predeclared_directional_transfer_claim")

    policy = ExternalValidityPolicy(
        policy_name=str(raw["policy_name"]),
        policy_version=str(raw["policy_version"]),
        benchmark_version=str(raw["benchmark_version"]),
        status=str(raw["status"]),
        candidate_benchmarks=candidates,
        selection_rule=SelectionRule(
            timing=str(selection.get("timing")),
            eligible_only_if_all=_non_empty_tuple(selection.get("eligible_only_if_all"), "selection_rule.eligible_only_if_all"),
            rank_eligible_candidates_by=_non_empty_tuple(selection.get("rank_eligible_candidates_by"), "selection_rule.rank_eligible_candidates_by"),
            no_eligible_candidate_action=str(selection.get("no_eligible_candidate_action")),
        ),
        inclusion_criteria=_non_empty_tuple(raw["inclusion_criteria"], "inclusion_criteria"),
        exclusion_criteria=_non_empty_tuple(raw["exclusion_criteria"], "exclusion_criteria"),
        sample_size_target=SampleSizeTarget(
            target_examples=_positive_int(sample.get("target_examples"), "sample_size_target.target_examples"),
            minimum_examples=_positive_int(sample.get("minimum_examples"), "sample_size_target.minimum_examples"),
            maximum_examples=_positive_int(sample.get("maximum_examples"), "sample_size_target.maximum_examples"),
            sampling_rule=str(sample.get("sampling_rule")),
            sampling_seed=_positive_int(sample.get("sampling_seed"), "sample_size_target.sampling_seed"),
        ),
        scoring_approach=ScoringApproach(
            primary_metric=str(scoring.get("primary_metric")),
            fallback_metric=str(scoring.get("fallback_metric")),
            aggregation=str(scoring.get("aggregation")),
            report_absolute_change_from_same_base_model_baseline=bool(scoring.get("report_absolute_change_from_same_base_model_baseline")),
            no_prompt_or_hyperparameter_selection_on_external_benchmark=bool(scoring.get("no_prompt_or_hyperparameter_selection_on_external_benchmark")),
        ),
        contamination_review=ContaminationReview(
            timing=str(review.get("timing")),
            checks=_non_empty_tuple(review.get("checks"), "contamination_review.checks"),
            dispositions=_non_empty_tuple(review.get("dispositions"), "contamination_review.dispositions"),
            exclusion_on_direct_project_leakage=bool(review.get("exclusion_on_direct_project_leakage")),
        ),
        predeclared_directional_transfer_claim=DirectionalTransferClaim(
            claim_id=str(claim.get("claim_id")),
            statement=str(claim.get("statement")),
            direction_test=str(claim.get("direction_test")),
            magnitude_equivalence_claimed=bool(claim.get("magnitude_equivalence_claimed")),
            subgroup_transfer_is_exploratory=bool(claim.get("subgroup_transfer_is_exploratory")),
        ),
        provenance=dict(_mapping(raw["provenance"], "provenance")),
    )
    policy.validate()
    return policy


def load_external_validity_policy(
    path: str | Path = DEFAULT_EXTERNAL_VALIDITY_POLICY,
) -> ExternalValidityPolicy:
    policy_path = Path(path)
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ExternalValidityPolicyError("external-validity policy root must be a mapping")
    return external_validity_policy_from_mapping(raw)
