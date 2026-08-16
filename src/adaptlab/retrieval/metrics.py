"""Deterministic retrieval-only metrics for AdaptLab Milestone 4.

These metrics apply only to retrieval-eligible examples with PRESENT evidence.
Behavior-only and evidence-ABSENT examples are deliberately excluded from
retrieval-quality denominators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.retrieval.schemas import RetrievalResult

PRIMARY_CUTOFFS = (1, 3, 5)
METRIC_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class RetrievalMetricValues:
    """Per-example retrieval metrics computed from ranked chunk IDs."""

    any_gold_at_1: bool
    any_gold_at_3: bool
    any_gold_at_5: bool
    any_gold_at_k: bool
    all_required_gold_at_1: bool
    all_required_gold_at_3: bool
    all_required_gold_at_5: bool
    all_required_gold_at_k: bool
    gold_recall_at_1: float
    gold_recall_at_3: float
    gold_recall_at_5: float
    gold_recall_at_k: float
    first_gold_reciprocal_rank: float


def _at_cutoff(
    ranked_chunk_ids: tuple[str, ...],
    gold: frozenset[str],
    required: frozenset[str],
    cutoff: int,
) -> tuple[bool, bool, float]:
    observed = frozenset(ranked_chunk_ids[:cutoff])
    any_gold = bool(observed & gold)
    all_required = required.issubset(observed)
    recall = len(observed & required) / len(required)
    return any_gold, all_required, recall


def compute_retrieval_metrics(
    candidate_chunk_ids: Iterable[str],
    gold_chunk_ids: Iterable[str],
    required_gold_chunk_ids: Iterable[str],
    *,
    top_k: int,
) -> RetrievalMetricValues:
    """Compute deterministic multi-chunk retrieval metrics for one example.

    ANY_GOLD and reciprocal rank use the complete permitted gold set.
    ALL_REQUIRED_GOLD and GOLD_RECALL use the required-gold set, which makes
    partial retrieval mechanically distinguishable for multi-chunk examples.
    """

    ranked = tuple(candidate_chunk_ids)
    gold = frozenset(gold_chunk_ids)
    required = frozenset(required_gold_chunk_ids)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if len(ranked) > top_k:
        raise ValueError("candidate count cannot exceed top_k")
    if len(set(ranked)) != len(ranked):
        raise ValueError("candidate_chunk_ids must be unique")
    if not gold:
        raise ValueError("evidence-present retrieval metrics require at least one gold chunk")
    if not required:
        raise ValueError("evidence-present retrieval metrics require at least one required_gold chunk")
    if not required.issubset(gold):
        raise ValueError("required_gold_chunk_ids must be a subset of gold_chunk_ids")

    values: dict[int, tuple[bool, bool, float]] = {
        cutoff: _at_cutoff(ranked, gold, required, cutoff) for cutoff in PRIMARY_CUTOFFS
    }
    at_k = _at_cutoff(ranked, gold, required, top_k)

    first_rank = next((rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in gold), None)
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank

    return RetrievalMetricValues(
        any_gold_at_1=values[1][0],
        any_gold_at_3=values[3][0],
        any_gold_at_5=values[5][0],
        any_gold_at_k=at_k[0],
        all_required_gold_at_1=values[1][1],
        all_required_gold_at_3=values[3][1],
        all_required_gold_at_5=values[5][1],
        all_required_gold_at_k=at_k[1],
        gold_recall_at_1=values[1][2],
        gold_recall_at_3=values[3][2],
        gold_recall_at_5=values[5][2],
        gold_recall_at_k=at_k[2],
        first_gold_reciprocal_rank=reciprocal_rank,
    )


def with_retrieval_metrics(result: RetrievalResult) -> RetrievalResult:
    """Return a result with retrieval-quality metrics populated when applicable.

    Ineligible and non-PRESENT examples retain ``None`` metrics rather than being
    assigned fake zeros that could leak into retrieval-quality denominators.
    """

    metric_fields = (
        "any_gold_at_1", "any_gold_at_3", "any_gold_at_5", "any_gold_at_k",
        "all_required_gold_at_1", "all_required_gold_at_3", "all_required_gold_at_5",
        "all_required_gold_at_k", "gold_recall_at_1", "gold_recall_at_3",
        "gold_recall_at_5", "gold_recall_at_k", "first_gold_reciprocal_rank",
    )
    if (
        not result.retrieval_eligible
        or result.task_family is TaskFamily.behavior_only
        or result.evidence_status is not EvidenceStatus.PRESENT
    ):
        return replace(result, **{name: None for name in metric_fields})

    metrics = compute_retrieval_metrics(
        result.candidate_chunk_ids,
        result.gold_chunk_ids,
        result.required_gold_chunk_ids,
        top_k=result.top_k,
    )
    return replace(result, **metrics.__dict__)


@dataclass(frozen=True)
class RetrievalMetricRow:
    dimension: str
    value: str
    n: int
    any_gold_at_1: float
    any_gold_at_3: float
    any_gold_at_5: float
    all_required_gold_at_1: float
    all_required_gold_at_3: float
    all_required_gold_at_5: float
    gold_recall_at_1: float
    gold_recall_at_3: float
    gold_recall_at_5: float
    mrr_first_gold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "n": self.n,
            "ANY_GOLD@1": self.any_gold_at_1,
            "ANY_GOLD@3": self.any_gold_at_3,
            "ANY_GOLD@5": self.any_gold_at_5,
            "ALL_REQUIRED_GOLD@1": self.all_required_gold_at_1,
            "ALL_REQUIRED_GOLD@3": self.all_required_gold_at_3,
            "ALL_REQUIRED_GOLD@5": self.all_required_gold_at_5,
            "GOLD_RECALL@1": self.gold_recall_at_1,
            "GOLD_RECALL@3": self.gold_recall_at_3,
            "GOLD_RECALL@5": self.gold_recall_at_5,
            "MRR_FIRST_GOLD": self.mrr_first_gold,
        }


@dataclass(frozen=True)
class RetrievalMetricsReport:
    rows: tuple[RetrievalMetricRow, ...]
    schema_version: str = METRIC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "denominator_policy": "retrieval_eligible AND evidence_status=PRESENT; behavior_only excluded",
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_text(self) -> str:
        headers = (
            "dimension", "value", "n", "ANY@1", "ANY@3", "ANY@5",
            "ALL@1", "ALL@3", "ALL@5", "RECALL@1", "RECALL@3", "RECALL@5", "MRR",
        )
        lines = [
            "Retrieval quality (eligible, evidence-PRESENT only; behavior_only excluded)",
            " | ".join(headers),
            " | ".join("---" for _ in headers),
        ]
        for row in self.rows:
            lines.append(" | ".join((
                row.dimension,
                row.value,
                str(row.n),
                f"{row.any_gold_at_1:.6f}",
                f"{row.any_gold_at_3:.6f}",
                f"{row.any_gold_at_5:.6f}",
                f"{row.all_required_gold_at_1:.6f}",
                f"{row.all_required_gold_at_3:.6f}",
                f"{row.all_required_gold_at_5:.6f}",
                f"{row.gold_recall_at_1:.6f}",
                f"{row.gold_recall_at_3:.6f}",
                f"{row.gold_recall_at_5:.6f}",
                f"{row.mrr_first_gold:.6f}",
            )))
        return "\n".join(lines) + "\n"


def _mean_bool(values: Iterable[bool]) -> float:
    seq = tuple(values)
    return sum(1.0 if value else 0.0 for value in seq) / len(seq)


def _mean_float(values: Iterable[float]) -> float:
    seq = tuple(values)
    return sum(seq) / len(seq)


def _row(dimension: str, value: str, results: tuple[RetrievalResult, ...]) -> RetrievalMetricRow:
    # Recompute mechanically so reports do not trust stale/pre-populated metric fields.
    computed = tuple(with_retrieval_metrics(result) for result in results)
    return RetrievalMetricRow(
        dimension=dimension,
        value=value,
        n=len(computed),
        any_gold_at_1=_mean_bool(result.any_gold_at_1 for result in computed if result.any_gold_at_1 is not None),
        any_gold_at_3=_mean_bool(result.any_gold_at_3 for result in computed if result.any_gold_at_3 is not None),
        any_gold_at_5=_mean_bool(result.any_gold_at_5 for result in computed if result.any_gold_at_5 is not None),
        all_required_gold_at_1=_mean_bool(result.all_required_gold_at_1 for result in computed if result.all_required_gold_at_1 is not None),
        all_required_gold_at_3=_mean_bool(result.all_required_gold_at_3 for result in computed if result.all_required_gold_at_3 is not None),
        all_required_gold_at_5=_mean_bool(result.all_required_gold_at_5 for result in computed if result.all_required_gold_at_5 is not None),
        gold_recall_at_1=_mean_float(result.gold_recall_at_1 for result in computed if result.gold_recall_at_1 is not None),
        gold_recall_at_3=_mean_float(result.gold_recall_at_3 for result in computed if result.gold_recall_at_3 is not None),
        gold_recall_at_5=_mean_float(result.gold_recall_at_5 for result in computed if result.gold_recall_at_5 is not None),
        mrr_first_gold=_mean_float(
            result.first_gold_reciprocal_rank
            for result in computed
            if result.first_gold_reciprocal_rank is not None
        ),
    )


def summarize_retrieval_metrics(results: Iterable[RetrievalResult]) -> RetrievalMetricsReport:
    """Aggregate primary retrieval metrics overall and by required dimensions."""

    eligible = tuple(
        result for result in results
        if result.retrieval_eligible
        and result.task_family is not TaskFamily.behavior_only
        and result.evidence_status is EvidenceStatus.PRESENT
    )
    if not eligible:
        return RetrievalMetricsReport(rows=())

    rows: list[RetrievalMetricRow] = [_row("overall", "ALL", eligible)]
    dimensions = (
        ("task_family", lambda r: r.task_family.value),
        ("difficulty", lambda r: r.difficulty.value),
        ("knowledge_state", lambda r: r.knowledge_state.value),
        ("split_type", lambda r: r.split_type.value),
    )
    for dimension, getter in dimensions:
        values = sorted({getter(result) for result in eligible})
        for value in values:
            group = tuple(result for result in eligible if getter(result) == value)
            rows.append(_row(dimension, value, group))
    return RetrievalMetricsReport(rows=tuple(rows))
