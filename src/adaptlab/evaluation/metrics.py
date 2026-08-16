"""Deterministic aggregation for AdaptLab evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    SplitType,
    TaskFamily,
)
from adaptlab.evaluation.schemas import EvaluationResult

METRICS_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class AccuracyMetric:
    """Accuracy for one group.

    ``n`` counts scored examples only. Provider failures and other unresolved
    results have ``score=None`` and therefore are not silently converted to
    incorrect predictions.
    """

    n: int
    accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"n": self.n}
        if self.accuracy is not None:
            data["accuracy"] = self.accuracy
        return data


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Canonical metric report grouped by analysis role."""

    primary: dict[str, Any]
    confirmatory: dict[str, dict[str, AccuracyMetric]]
    exploratory: dict[str, dict[str, AccuracyMetric]]
    schema_version: str = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary": {
                key: value.to_dict() if isinstance(value, AccuracyMetric) else value
                for key, value in self.primary.items()
            },
            "confirmatory": {
                dimension: {key: metric.to_dict() for key, metric in groups.items()}
                for dimension, groups in self.confirmatory.items()
            },
            "exploratory": {
                dimension: {key: metric.to_dict() for key, metric in groups.items()}
                for dimension, groups in self.exploratory.items()
            },
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def human_summary(self) -> str:
        """Return a compact deterministic text summary.

        Empty groups are shown as ``n=0`` with no percentage, per the reporting
        contract.
        """

        lines = [f"overall: {_format_metric(self.primary['overall_accuracy'])}"]
        for role in ("confirmatory", "exploratory"):
            section = getattr(self, role)
            for dimension, groups in section.items():
                rendered = ", ".join(
                    f"{name} {_format_metric(metric)}" for name, metric in groups.items()
                )
                lines.append(f"{role} {dimension}: {rendered}")
        return "\n".join(lines) + "\n"


T = TypeVar("T")


def _scored(results: Iterable[EvaluationResult]) -> list[EvaluationResult]:
    return [result for result in results if result.score is not None]


def _accuracy(results: Iterable[EvaluationResult]) -> AccuracyMetric:
    scored = _scored(results)
    if not scored:
        return AccuracyMetric(n=0, accuracy=None)
    return AccuracyMetric(
        n=len(scored),
        accuracy=sum(float(result.score) for result in scored) / len(scored),
    )


def _group_metrics(
    results: list[EvaluationResult],
    values: Iterable[T],
    getter: Callable[[EvaluationResult], T | None],
    label: Callable[[T], str],
) -> dict[str, AccuracyMetric]:
    return {
        label(value): _accuracy(result for result in results if getter(result) == value)
        for value in values
    }


def aggregate_metrics(results: Iterable[EvaluationResult]) -> AggregateMetrics:
    """Aggregate evaluation results deterministically over predeclared groups."""

    ordered = sorted(list(results), key=lambda result: result.example_id)

    # Overall accuracy is the sole primary metric at this milestone. Task family
    # and difficulty are predeclared confirmatory breakdowns; finer diagnostic
    # dimensions remain exploratory until a later analysis prompt predeclares
    # specific paired comparisons.
    primary = {"overall_accuracy": _accuracy(ordered)}

    confirmatory = {
        "task_family": _group_metrics(
            ordered, TaskFamily, lambda result: result.task_family, lambda value: value.value
        ),
        "difficulty": _group_metrics(
            ordered, Difficulty, lambda result: result.difficulty, lambda value: value.value
        ),
    }

    exploratory = {
        "behavior_type": _group_metrics(
            ordered,
            BehaviorType,
            lambda result: result.behavior_type,
            lambda value: value.value,
        ),
        "knowledge_state": _group_metrics(
            ordered,
            KnowledgeState,
            lambda result: result.knowledge_state,
            lambda value: value.value,
        ),
        "evidence_status": _group_metrics(
            ordered,
            EvidenceStatus,
            lambda result: result.evidence_status,
            lambda value: value.value,
        ),
        "split_type": _group_metrics(
            ordered,
            SplitType,
            lambda result: result.split_type,
            lambda value: value.value.upper(),
        ),
    }

    return AggregateMetrics(
        primary=primary,
        confirmatory=confirmatory,
        exploratory=exploratory,
    )


def _format_metric(metric: AccuracyMetric) -> str:
    if metric.accuracy is None:
        return f"n={metric.n}"
    return f"n={metric.n} accuracy={metric.accuracy:.3f}"
