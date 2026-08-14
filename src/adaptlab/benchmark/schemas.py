from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    Split,
    SplitType,
    TaskFamily,
    ScoringRule,
)


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    benchmark_version: str
    task_family: TaskFamily
    behavior_type: BehaviorType | None
    difficulty: Difficulty
    split: Split
    split_type: SplitType
    holdout_dimension: str | None
    holdout_group: str | None
    knowledge_version: str | None
    knowledge_state: KnowledgeState
    evidence_status: EvidenceStatus
    question: str
    expected_output: Any
    required_record_ids: tuple[str, ...]
    required_logical_fact_ids: tuple[str, ...]
    gold_document_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    generation_seed: int
    scoring_rule: ScoringRule | None = None
    scoring_parameters: dict[str, Any] | None = None
    lifecycle_logical_fact_id: str | None = None

    def __post_init__(self) -> None:
        self._validate_nonempty_strings()
        self._validate_task_family_invariants()
        self._validate_evidence_invariants()
        self._validate_holdout_fields()

    def _validate_nonempty_strings(self) -> None:
        for field_name in ("example_id", "benchmark_version", "question"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def _validate_task_family_invariants(self) -> None:
        if self.task_family is TaskFamily.behavior_only:
            if self.behavior_type is None:
                raise ValueError("behavior_only examples require behavior_type")
            if self.evidence_status is not EvidenceStatus.NOT_APPLICABLE:
                raise ValueError(
                    "behavior_only examples require evidence_status=NOT_APPLICABLE"
                )
            if self.knowledge_state is not KnowledgeState.NOT_APPLICABLE:
                raise ValueError(
                    "behavior_only examples require knowledge_state=NOT_APPLICABLE"
                )

        elif self.task_family is TaskFamily.knowledge_only:
            if self.behavior_type is not None:
                raise ValueError("knowledge_only examples require behavior_type=None")
            if self.evidence_status not in (
                EvidenceStatus.PRESENT,
                EvidenceStatus.ABSENT,
            ):
                raise ValueError(
                    "knowledge_only examples require evidence_status=PRESENT or ABSENT"
                )

        elif self.task_family is TaskFamily.behavior_knowledge:
            if self.behavior_type is None:
                raise ValueError("behavior_knowledge examples require behavior_type")
            if self.evidence_status not in (
                EvidenceStatus.PRESENT,
                EvidenceStatus.ABSENT,
            ):
                raise ValueError(
                    "behavior_knowledge examples require evidence_status=PRESENT or ABSENT"
                )

        elif self.task_family is TaskFamily.changed_knowledge:
            if self.knowledge_state not in (
                KnowledgeState.UNCHANGED,
                KnowledgeState.UPDATED,
                KnowledgeState.REMOVED,
            ):
                raise ValueError(
                    "changed_knowledge examples require knowledge_state="
                    "UNCHANGED, UPDATED, or REMOVED"
                )
            if self.evidence_status not in (
                EvidenceStatus.PRESENT,
                EvidenceStatus.ABSENT,
            ):
                raise ValueError(
                    "changed_knowledge examples require evidence_status=PRESENT or ABSENT"
                )

    def _validate_evidence_invariants(self) -> None:
        if self.evidence_status is EvidenceStatus.ABSENT:
            nonempty = [
                name
                for name in (
                    "required_record_ids",
                    "required_logical_fact_ids",
                    "gold_document_ids",
                    "gold_chunk_ids",
                )
                if getattr(self, name)
            ]
            if nonempty:
                raise ValueError(
                    "evidence_status=ABSENT requires empty evidence references: "
                    + ", ".join(nonempty)
                )

        elif self.evidence_status is EvidenceStatus.PRESENT:
            if not self.required_record_ids:
                raise ValueError(
                    "evidence_status=PRESENT requires required_record_ids to be non-empty"
                )
            if not self.gold_document_ids:
                raise ValueError(
                    "evidence_status=PRESENT requires gold_document_ids to be non-empty"
                )
            if not self.gold_chunk_ids:
                raise ValueError(
                    "evidence_status=PRESENT requires gold_chunk_ids to be non-empty"
                )

    def _validate_holdout_fields(self) -> None:
        if self.split_type is SplitType.structural_holdout:
            if not self.holdout_dimension or not self.holdout_group:
                raise ValueError(
                    "structural_holdout examples require holdout_dimension and holdout_group"
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "task_family",
            "behavior_type",
            "difficulty",
            "split",
            "split_type",
            "knowledge_state",
            "evidence_status",
            "scoring_rule",
        ):
            value = data[key]
            data[key] = value.value if value is not None else None
        for key in (
            "required_record_ids",
            "required_logical_fact_ids",
            "gold_document_ids",
            "gold_chunk_ids",
        ):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkExample":
        return cls(
            example_id=data["example_id"],
            benchmark_version=data["benchmark_version"],
            task_family=TaskFamily(data["task_family"]),
            behavior_type=(
                BehaviorType(data["behavior_type"])
                if data.get("behavior_type") is not None
                else None
            ),
            difficulty=Difficulty(data["difficulty"]),
            split=Split(data["split"]),
            split_type=SplitType(data["split_type"]),
            holdout_dimension=data.get("holdout_dimension"),
            holdout_group=data.get("holdout_group"),
            knowledge_version=data.get("knowledge_version"),
            knowledge_state=KnowledgeState(data["knowledge_state"]),
            evidence_status=EvidenceStatus(data["evidence_status"]),
            question=data["question"],
            expected_output=data["expected_output"],
            required_record_ids=tuple(data.get("required_record_ids", ())),
            required_logical_fact_ids=tuple(data.get("required_logical_fact_ids", ())),
            gold_document_ids=tuple(data.get("gold_document_ids", ())),
            gold_chunk_ids=tuple(data.get("gold_chunk_ids", ())),
            generation_seed=int(data["generation_seed"]),
            scoring_rule=(ScoringRule(data["scoring_rule"]) if data.get("scoring_rule") is not None else None),
            scoring_parameters=data.get("scoring_parameters"),
            lifecycle_logical_fact_id=data.get("lifecycle_logical_fact_id"),
        )
