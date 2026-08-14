"""Typed configuration for the full AdaptLab Nimbus benchmark."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from adaptlab.domain.enums import BehaviorType, Difficulty, KnowledgeState, TaskFamily


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark configuration violates the frozen contract."""


T = TypeVar("T")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkConfigError(f"{name} must be a mapping")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkConfigError(f"{name} must be an integer")
    if value < 0:
        raise BenchmarkConfigError(f"{name} must be non-negative")
    return value


def _count_dataclass(cls: type[T], raw: Any, name: str) -> T:
    data = _mapping(raw, name)
    expected = {field.name for field in fields(cls)}
    unknown = set(data) - expected
    missing = expected - set(data)
    if missing:
        raise BenchmarkConfigError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise BenchmarkConfigError(f"{name} has unknown fields: {sorted(unknown)}")
    values = {
        field.name: _non_negative_int(data[field.name], f"{name}.{field.name}")
        for field in fields(cls)
    }
    return cls(**values)


@dataclass(frozen=True)
class SplitTargets:
    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class SentinelTargets:
    count: int


@dataclass(frozen=True)
class TaskFamilyTargets:
    behavior_only: int
    knowledge_only: int
    behavior_knowledge: int
    changed_knowledge: int

    def by_family(self) -> dict[TaskFamily, int]:
        return {TaskFamily(name): value for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class DifficultyTargets:
    EASY: int
    MEDIUM: int
    HARD: int

    def by_difficulty(self) -> dict[Difficulty, int]:
        return {Difficulty[name]: value for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class ChangedKnowledgeTargets:
    UNCHANGED: int
    UPDATED: int
    REMOVED: int

    def by_state(self) -> dict[KnowledgeState, int]:
        return {KnowledgeState[name]: value for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class EvidenceAbsentTargets:
    total: int
    knowledge_only: int
    behavior_knowledge: int
    changed_knowledge: int


@dataclass(frozen=True)
class CorpusTargets:
    total_chunks: int
    current_authoritative: int
    obsolete_versioned: int
    competing_near_duplicate: int
    domain_distractor: int


@dataclass(frozen=True)
class BehaviorTypeTargets:
    SCHEMA_ADHERENCE: int
    CONDITIONAL_DECISION_RULE: int
    TRANSFORMATION_EXTRACTION: int
    CLASSIFICATION_POLICY: int
    ABSTENTION_BEHAVIOR: int

    def by_behavior_type(self) -> dict[BehaviorType, int]:
        return {BehaviorType[name]: value for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_name: str
    benchmark_version: str
    generation_seed: int
    world_schema_version: str
    splits: SplitTargets
    generalization_sentinel: SentinelTargets
    test_task_families: TaskFamilyTargets
    test_difficulty: DifficultyTargets
    changed_knowledge: ChangedKnowledgeTargets
    evidence_absent: EvidenceAbsentTargets
    corpus: CorpusTargets
    behavior_only_test_behavior_types: BehaviorTypeTargets

    def validate(self) -> None:
        if not self.benchmark_name.strip():
            raise BenchmarkConfigError("benchmark_name must not be empty")
        if not self.benchmark_version.strip():
            raise BenchmarkConfigError("benchmark_version must not be empty")
        if not self.world_schema_version.strip():
            raise BenchmarkConfigError("world_schema_version must not be empty")
        _non_negative_int(self.generation_seed, "generation_seed")

        family_total = sum(self.test_task_families.__dict__.values())
        if family_total != self.splits.test:
            raise BenchmarkConfigError(
                "test_task_families total must equal splits.test "
                f"({family_total} != {self.splits.test})"
            )

        difficulty_total = sum(self.test_difficulty.__dict__.values())
        if difficulty_total != self.splits.test:
            raise BenchmarkConfigError(
                "test_difficulty total must equal splits.test "
                f"({difficulty_total} != {self.splits.test})"
            )

        changed_total = sum(self.changed_knowledge.__dict__.values())
        if changed_total != self.test_task_families.changed_knowledge:
            raise BenchmarkConfigError(
                "changed_knowledge total must equal the changed_knowledge test target "
                f"({changed_total} != {self.test_task_families.changed_knowledge})"
            )

        absent_subtotal = (
            self.evidence_absent.knowledge_only
            + self.evidence_absent.behavior_knowledge
            + self.evidence_absent.changed_knowledge
        )
        if absent_subtotal != self.evidence_absent.total:
            raise BenchmarkConfigError(
                "evidence_absent subtotals must equal evidence_absent.total "
                f"({absent_subtotal} != {self.evidence_absent.total})"
            )

        for family_name in ("knowledge_only", "behavior_knowledge", "changed_knowledge"):
            absent_count = getattr(self.evidence_absent, family_name)
            family_count = getattr(self.test_task_families, family_name)
            if absent_count > family_count:
                raise BenchmarkConfigError(
                    f"evidence_absent.{family_name} cannot exceed its test task-family target"
                )

        behavior_total = sum(self.behavior_only_test_behavior_types.__dict__.values())
        if behavior_total != self.test_task_families.behavior_only:
            raise BenchmarkConfigError(
                "behavior-only behavior-type targets must equal the behavior_only test target "
                f"({behavior_total} != {self.test_task_families.behavior_only})"
            )

        corpus_categories = (
            self.corpus.current_authoritative
            + self.corpus.obsolete_versioned
            + self.corpus.competing_near_duplicate
            + self.corpus.domain_distractor
        )
        if corpus_categories != self.corpus.total_chunks:
            raise BenchmarkConfigError(
                "corpus category counts must equal corpus.total_chunks "
                f"({corpus_categories} != {self.corpus.total_chunks})"
            )


DEFAULT_FULL_BENCHMARK_CONFIG = Path("configs/benchmark_v0.0.yaml")


def benchmark_config_from_mapping(raw: Mapping[str, Any]) -> BenchmarkConfig:
    required_top_level = {
        "benchmark_name",
        "benchmark_version",
        "generation_seed",
        "world_schema_version",
        "splits",
        "generalization_sentinel",
        "test_task_families",
        "test_difficulty",
        "changed_knowledge",
        "evidence_absent",
        "corpus",
        "behavior_only_test_behavior_types",
    }
    missing = required_top_level - set(raw)
    unknown = set(raw) - required_top_level
    if missing:
        raise BenchmarkConfigError(f"configuration is missing fields: {sorted(missing)}")
    if unknown:
        raise BenchmarkConfigError(f"configuration has unknown fields: {sorted(unknown)}")

    config = BenchmarkConfig(
        benchmark_name=str(raw["benchmark_name"]),
        benchmark_version=str(raw["benchmark_version"]),
        generation_seed=_non_negative_int(raw["generation_seed"], "generation_seed"),
        world_schema_version=str(raw["world_schema_version"]),
        splits=_count_dataclass(SplitTargets, raw["splits"], "splits"),
        generalization_sentinel=_count_dataclass(
            SentinelTargets, raw["generalization_sentinel"], "generalization_sentinel"
        ),
        test_task_families=_count_dataclass(
            TaskFamilyTargets, raw["test_task_families"], "test_task_families"
        ),
        test_difficulty=_count_dataclass(
            DifficultyTargets, raw["test_difficulty"], "test_difficulty"
        ),
        changed_knowledge=_count_dataclass(
            ChangedKnowledgeTargets, raw["changed_knowledge"], "changed_knowledge"
        ),
        evidence_absent=_count_dataclass(
            EvidenceAbsentTargets, raw["evidence_absent"], "evidence_absent"
        ),
        corpus=_count_dataclass(CorpusTargets, raw["corpus"], "corpus"),
        behavior_only_test_behavior_types=_count_dataclass(
            BehaviorTypeTargets,
            raw["behavior_only_test_behavior_types"],
            "behavior_only_test_behavior_types",
        ),
    )
    config.validate()
    return config


def load_benchmark_config(path: str | Path = DEFAULT_FULL_BENCHMARK_CONFIG) -> BenchmarkConfig:
    """Load and validate a full benchmark configuration from YAML."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkConfigError(f"could not read benchmark config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise BenchmarkConfigError(f"invalid YAML in benchmark config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise BenchmarkConfigError("benchmark configuration root must be a mapping")
    return benchmark_config_from_mapping(raw)
