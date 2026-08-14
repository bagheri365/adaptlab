"""Deterministic structural-holdout policy for the full v0.0 benchmark.

The policy is constructed from the frozen benchmark configuration and the
expanded Nimbus world *before* full task generation.  It keeps dataset split
eligibility separate from structural-holdout metadata and supports independent
holdouts for component families and error families.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Iterable, Literal

from adaptlab.benchmark.config import BenchmarkConfig
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split, SplitType
from adaptlab.domain.world import NimbusFact, NimbusWorld

COMPONENT_FAMILY_DIMENSION = "component_family"
ERROR_FAMILY_DIMENSION = "error_family"

GroupRole = Literal["train", "validation", "iid_test", "structural_test"]


@dataclass(frozen=True, slots=True)
class HoldoutDimensionPolicy:
    """Frozen split eligibility for one structural grouping dimension."""

    dimension: str
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    iid_test_groups: tuple[str, ...]
    structural_test_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            self.train_groups
            + self.validation_groups
            + self.iid_test_groups
            + self.structural_test_groups
        )
        if len(groups) != len(set(groups)):
            raise ValueError(f"{self.dimension} holdout groups must be disjoint")
        if not self.structural_test_groups:
            raise ValueError(f"{self.dimension} requires at least one structural-test group")

    def role_for(self, group: str) -> GroupRole | None:
        if group in self.train_groups:
            return "train"
        if group in self.validation_groups:
            return "validation"
        if group in self.iid_test_groups:
            return "iid_test"
        if group in self.structural_test_groups:
            return "structural_test"
        return None


@dataclass(frozen=True, slots=True)
class FullHoldoutPolicy:
    """Frozen full-benchmark holdout policy across all supported dimensions."""

    policy_version: str
    generation_seed: int
    component_family: HoldoutDimensionPolicy
    error_family: HoldoutDimensionPolicy

    def dimension(self, name: str) -> HoldoutDimensionPolicy:
        if name == COMPONENT_FAMILY_DIMENSION:
            return self.component_family
        if name == ERROR_FAMILY_DIMENSION:
            return self.error_family
        raise KeyError(f"unknown holdout dimension: {name}")

    def to_dict(self) -> dict[str, object]:
        def encode(value: HoldoutDimensionPolicy) -> dict[str, object]:
            return {
                "dimension": value.dimension,
                "train_groups": list(value.train_groups),
                "validation_groups": list(value.validation_groups),
                "iid_test_groups": list(value.iid_test_groups),
                "structural_test_groups": list(value.structural_test_groups),
            }

        return {
            "policy_version": self.policy_version,
            "generation_seed": self.generation_seed,
            "component_family": encode(self.component_family),
            "error_family": encode(self.error_family),
        }


@dataclass(frozen=True, slots=True)
class HoldoutValidationResult:
    passed: bool
    errors: tuple[str, ...]


def error_family_for_fact(fact: NimbusFact) -> str | None:
    """Return the deterministic error-family group for an error-code fact."""

    if fact.fact_type != "error_code":
        return None
    return f"{fact.component_family}_errors"


def _make_dimension_policy(
    dimension: str,
    groups: Iterable[str],
    *,
    rng: random.Random,
    structural_count: int = 2,
) -> HoldoutDimensionPolicy:
    ordered = sorted(set(groups))
    if len(ordered) < structural_count + 3:
        raise ValueError(f"{dimension} requires at least {structural_count + 3} groups")

    shuffled = list(ordered)
    rng.shuffle(shuffled)
    structural = sorted(shuffled[:structural_count])
    validation = sorted(shuffled[structural_count : structural_count + 1])
    iid_test = sorted(shuffled[structural_count + 1 : structural_count + 2])
    train = sorted(shuffled[structural_count + 2 :])

    return HoldoutDimensionPolicy(
        dimension=dimension,
        train_groups=tuple(train),
        validation_groups=tuple(validation),
        iid_test_groups=tuple(iid_test),
        structural_test_groups=tuple(structural),
    )


def build_full_holdout_policy(
    config: BenchmarkConfig,
    world: NimbusWorld,
) -> FullHoldoutPolicy:
    """Freeze deterministic component/error holdouts before task generation."""

    if world.generation_seed != config.generation_seed:
        raise ValueError("world generation seed must match benchmark configuration")

    component_groups = sorted({fact.component_family for fact in world.facts})
    error_groups = sorted(
        {
            group
            for fact in world.facts
            if (group := error_family_for_fact(fact)) is not None
        }
    )

    # Separate RNG streams make each dimension stable if another dimension's
    # implementation changes while preserving the canonical config seed.
    component_rng = random.Random(f"{config.generation_seed}:component_family")
    error_rng = random.Random(f"{config.generation_seed}:error_family")

    component_policy = _make_dimension_policy(
        COMPONENT_FAMILY_DIMENSION, component_groups, rng=component_rng
    )
    error_policy = _make_dimension_policy(
        ERROR_FAMILY_DIMENSION, error_groups, rng=error_rng
    )

    return FullHoldoutPolicy(
        policy_version="full-v0.0-structural-holdout-v1",
        generation_seed=config.generation_seed,
        component_family=component_policy,
        error_family=error_policy,
    )


def render_holdout_report(policy: FullHoldoutPolicy) -> str:
    """Render the frozen policy as a stable human-readable report."""

    lines = [
        "AdaptLab Nimbus Full Benchmark Holdout Report",
        f"Policy version: {policy.policy_version}",
        f"Generation seed: {policy.generation_seed}",
        "",
    ]
    for dimension in (policy.component_family, policy.error_family):
        lines.extend(
            [
                f"[{dimension.dimension}]",
                f"train groups: {', '.join(dimension.train_groups)}",
                f"validation groups: {', '.join(dimension.validation_groups)}",
                f"IID test groups: {', '.join(dimension.iid_test_groups)}",
                f"structural test groups: {', '.join(dimension.structural_test_groups)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _fact_groups_for_example(
    example: BenchmarkExample,
    records_by_id: dict[str, NimbusFact],
) -> tuple[set[str], set[str]]:
    components: set[str] = set()
    error_groups: set[str] = set()
    for record_id in example.required_record_ids:
        fact = records_by_id.get(record_id)
        if fact is None:
            continue
        components.add(fact.component_family)
        error_group = error_family_for_fact(fact)
        if error_group is not None:
            error_groups.add(error_group)
    return components, error_groups


def structural_holdout_for_example(
    example: BenchmarkExample,
    world: NimbusWorld,
    policy: FullHoldoutPolicy,
) -> tuple[str, str] | None:
    """Return the structural holdout dimension/group applicable to an example.

    Component-family holdouts take precedence because they are broader.  Error
    family holdouts then apply to error-code examples from otherwise eligible
    components.
    """

    records_by_id = {fact.record_id: fact for fact in world.facts}
    components, error_groups = _fact_groups_for_example(example, records_by_id)

    component_hits = components & set(policy.component_family.structural_test_groups)
    if component_hits:
        return COMPONENT_FAMILY_DIMENSION, sorted(component_hits)[0]

    error_hits = error_groups & set(policy.error_family.structural_test_groups)
    if error_hits:
        return ERROR_FAMILY_DIMENSION, sorted(error_hits)[0]
    return None


def apply_full_holdout_policy(
    world: NimbusWorld,
    examples: Iterable[BenchmarkExample],
    policy: FullHoldoutPolicy,
) -> list[BenchmarkExample]:
    """Apply an already-frozen full policy to generated examples.

    Full task generation should preferentially choose eligible groups using the
    policy before constructing examples.  This function remains useful for
    validation fixtures and deterministic post-construction annotation.
    """

    assigned: list[BenchmarkExample] = []
    records_by_id = {fact.record_id: fact for fact in world.facts}
    for example in examples:
        structural = structural_holdout_for_example(example, world, policy)
        if structural is not None:
            dimension, group = structural
            assigned.append(
                replace(
                    example,
                    split=Split.test,
                    split_type=SplitType.structural_holdout,
                    holdout_dimension=dimension,
                    holdout_group=group,
                )
            )
            continue

        components, error_groups = _fact_groups_for_example(example, records_by_id)
        roles: set[GroupRole] = set()
        for component in components:
            role = policy.component_family.role_for(component)
            if role is not None:
                roles.add(role)
        for error_group in error_groups:
            role = policy.error_family.role_for(error_group)
            if role is not None:
                roles.add(role)

        if "validation" in roles:
            split = Split.validation
        elif "iid_test" in roles:
            split = Split.test
        else:
            split = Split.train
        assigned.append(
            replace(
                example,
                split=split,
                split_type=SplitType.iid,
                holdout_dimension=None,
                holdout_group=None,
            )
        )

    return sorted(assigned, key=lambda item: item.example_id)


def validate_full_holdout_examples(
    world: NimbusWorld,
    examples: Iterable[BenchmarkExample],
    policy: FullHoldoutPolicy,
) -> HoldoutValidationResult:
    """Reject structural-test leakage and invalid full-policy annotations."""

    records_by_id = {fact.record_id: fact for fact in world.facts}
    errors: list[str] = []
    for example in examples:
        prefix = f"example {example.example_id}"
        components, error_groups = _fact_groups_for_example(example, records_by_id)

        structural_component_hits = components & set(
            policy.component_family.structural_test_groups
        )
        structural_error_hits = error_groups & set(policy.error_family.structural_test_groups)
        any_structural = bool(structural_component_hits or structural_error_hits)

        if any_structural and example.split in (Split.train, Split.validation):
            groups = sorted(structural_component_hits | structural_error_hits)
            errors.append(
                f"{prefix} leaks structural-test group(s) {', '.join(groups)} "
                f"into {example.split.value}"
            )

        if example.split_type is SplitType.structural_holdout:
            if example.split is not Split.test:
                errors.append(f"{prefix} structural_holdout examples must use split=test")
            if example.holdout_dimension not in (
                COMPONENT_FAMILY_DIMENSION,
                ERROR_FAMILY_DIMENSION,
            ):
                errors.append(f"{prefix} has invalid holdout_dimension {example.holdout_dimension!r}")
                continue
            dimension_policy = policy.dimension(example.holdout_dimension)
            if example.holdout_group not in dimension_policy.structural_test_groups:
                errors.append(
                    f"{prefix} holdout_group {example.holdout_group!r} is not a frozen "
                    f"structural-test group for {example.holdout_dimension}"
                )
        elif any_structural and example.split is Split.test:
            errors.append(
                f"{prefix} uses a structural-test group but is not marked structural_holdout"
            )

    return HoldoutValidationResult(passed=not errors, errors=tuple(errors))
