"""Deterministic sampling and serialization helpers for documented human audits."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import BehaviorType, Difficulty, EvidenceStatus, KnowledgeState, SplitType, TaskFamily

HUMAN_AUDIT_VERSION = "2"


@dataclass(frozen=True, slots=True)
class HumanAuditReview:
    example_id: str
    review_result: str
    notes: str
    correction_required: bool
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HumanAuditQueueRecord:
    example_id: str
    question: str
    expected_output: str
    task_family: str
    behavior_type: str | None
    difficulty: str
    knowledge_state: str
    evidence_status: str
    required_records: tuple[str, ...]
    gold_documents: tuple[str, ...]
    gold_chunks: tuple[str, ...]
    gold_evidence_text: tuple[str, ...] = ()
    structured_truth: tuple[dict[str, object], ...] = ()
    review_status: str = "PENDING_HUMAN_REVIEW"
    review_notes: str = ""
    review_checks: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["required_records"] = list(self.required_records)
        value["gold_documents"] = list(self.gold_documents)
        value["gold_chunks"] = list(self.gold_chunks)
        value["gold_evidence_text"] = list(self.gold_evidence_text)
        value["structured_truth"] = list(self.structured_truth)
        return value


@dataclass(frozen=True, slots=True)
class HumanReviewQueueArtifact:
    audit_version: str
    sample_size: int
    coverage: dict[str, dict[str, int]]
    reviews: tuple[HumanAuditQueueRecord, ...]
    summary: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": self.audit_version,
            "sample_size": self.sample_size,
            "coverage": self.coverage,
            "reviews": [review.to_dict() for review in self.reviews],
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class HumanAuditArtifact:
    audit_version: str
    sample_size: int
    coverage: dict[str, dict[str, int]]
    reviews: tuple[HumanAuditReview, ...]
    summary: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": self.audit_version,
            "sample_size": self.sample_size,
            "coverage": self.coverage,
            "reviews": [review.to_dict() for review in self.reviews],
            "summary": self.summary,
        }


def _coverage(examples: Sequence[BenchmarkExample]) -> dict[str, dict[str, int]]:
    fields = {
        "task_family": [example.task_family.value for example in examples],
        "difficulty": [example.difficulty.value for example in examples],
        "split_type": [example.split_type.value for example in examples],
        "evidence_status": [example.evidence_status.value for example in examples],
        "knowledge_state": [example.knowledge_state.value for example in examples],
        "behavior_type": [example.behavior_type.value if example.behavior_type else "NONE" for example in examples],
    }
    return {name: dict(sorted(Counter(values).items())) for name, values in fields.items()}


def _feature_set(example: BenchmarkExample) -> set[tuple[str, str]]:
    values = {
        "task_family": example.task_family.value,
        "difficulty": example.difficulty.value,
        "split_type": example.split_type.value,
        "evidence_status": example.evidence_status.value,
        "knowledge_state": example.knowledge_state.value,
        "behavior_type": example.behavior_type.value if example.behavior_type else "NONE",
    }
    return set(values.items())


def select_human_audit_sample(
    examples: Iterable[BenchmarkExample], *, sample_size: int = 50
) -> list[BenchmarkExample]:
    """Choose a deterministic stratified sample covering the required human-review strata.

    Changed-knowledge coverage is deliberately stronger than generic set coverage:
    the queue includes at least five UNCHANGED, five UPDATED, and five REMOVED
    examples whenever the benchmark contains that many.
    """
    pool = sorted(examples, key=lambda example: example.example_id)
    if sample_size < 40 or sample_size > 60:
        raise ValueError("human audit sample_size must be between 40 and 60")
    if len(pool) < sample_size:
        raise ValueError("not enough examples for requested human audit sample")

    chosen: list[BenchmarkExample] = []
    chosen_ids: set[str] = set()

    def add(example: BenchmarkExample) -> None:
        if example.example_id not in chosen_ids:
            chosen.append(example)
            chosen_ids.add(example.example_id)

    # First guarantee meaningful changed-knowledge lifecycle coverage.
    for state in (KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED):
        candidates = [
            example
            for example in pool
            if example.task_family is TaskFamily.changed_knowledge and example.knowledge_state is state
        ]
        for example in candidates[: min(5, len(candidates))]:
            add(example)

    required: set[tuple[str, str]] = {
        *(("task_family", value.value) for value in TaskFamily),
        *(("difficulty", value.value) for value in Difficulty),
        ("split_type", SplitType.iid.value),
        ("split_type", SplitType.structural_holdout.value),
        ("evidence_status", EvidenceStatus.PRESENT.value),
        ("evidence_status", EvidenceStatus.ABSENT.value),
        *(("behavior_type", value.value) for value in BehaviorType),
    }
    covered = set().union(*(_feature_set(item) for item in chosen)) if chosen else set()
    uncovered = required - covered
    remaining = [item for item in pool if item.example_id not in chosen_ids]

    while uncovered and len(chosen) < sample_size:
        best = max(
            remaining,
            key=lambda example: (
                len(_feature_set(example) & uncovered),
                -sum(1 for item in chosen if item.task_family is example.task_family),
                example.example_id,
            ),
        )
        add(best)
        remaining.remove(best)
        uncovered -= _feature_set(best)

    if uncovered:
        raise ValueError(f"unable to cover required audit strata: {sorted(uncovered)}")

    while len(chosen) < sample_size:
        task_counts = Counter(item.task_family for item in chosen)
        diff_counts = Counter(item.difficulty for item in chosen)
        split_counts = Counter(item.split_type for item in chosen)
        evidence_counts = Counter(item.evidence_status for item in chosen)
        best = min(
            remaining,
            key=lambda example: (
                task_counts[example.task_family],
                diff_counts[example.difficulty],
                split_counts[example.split_type],
                evidence_counts[example.evidence_status],
                example.example_id,
            ),
        )
        add(best)
        remaining.remove(best)

    return sorted(chosen, key=lambda example: example.example_id)


def build_pending_human_review_queue(
    sampled_examples: Sequence[BenchmarkExample],
    *,
    chunk_text_by_id: dict[str, str] | None = None,
    structured_truth_by_example_id: dict[str, tuple[dict[str, object], ...]] | None = None,
) -> HumanReviewQueueArtifact:
    """Create an honest review queue without fabricating human PASS judgments."""
    pending = "PENDING_HUMAN_REVIEW"
    chunk_text_by_id = chunk_text_by_id or {}
    structured_truth_by_example_id = structured_truth_by_example_id or {}
    checklist_names = (
        "question_is_clear",
        "expected_output_is_correct",
        "answer_follows_structured_truth",
        "evidence_sufficient_when_present",
        "evidence_genuinely_absent_when_absent",
        "task_family_label_correct",
        "behavior_type_label_correct",
        "difficulty_construction_reasonable",
        "changed_knowledge_semantics_correct",
        "no_accidental_shortcut_invalidates_task",
    )
    reviews = tuple(
        HumanAuditQueueRecord(
            example_id=example.example_id,
            question=example.question,
            expected_output=example.expected_output,
            task_family=example.task_family.value,
            behavior_type=example.behavior_type.value if example.behavior_type else None,
            difficulty=example.difficulty.value,
            knowledge_state=example.knowledge_state.value,
            evidence_status=example.evidence_status.value,
            required_records=tuple(example.required_record_ids),
            gold_documents=tuple(example.gold_document_ids),
            gold_chunks=tuple(example.gold_chunk_ids),
            gold_evidence_text=tuple(chunk_text_by_id.get(chunk_id, "") for chunk_id in example.gold_chunk_ids),
            structured_truth=structured_truth_by_example_id.get(example.example_id, ()),
            review_status=pending,
            review_notes="",
            review_checks={name: pending for name in checklist_names},
        )
        for example in sorted(sampled_examples, key=lambda item: item.example_id)
    )
    return HumanReviewQueueArtifact(
        audit_version=HUMAN_AUDIT_VERSION,
        sample_size=len(sampled_examples),
        coverage=_coverage(sampled_examples),
        reviews=reviews,
        summary={
            "passed": 0,
            "failed": 0,
            "correction_required": 0,
            "pending_human_review": len(reviews),
        },
    )


def write_human_review_queue(artifact: HumanReviewQueueArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_human_audit_artifact(
    sampled_examples: Sequence[BenchmarkExample],
    reviews: Sequence[HumanAuditReview],
) -> HumanAuditArtifact:
    by_id = {example.example_id for example in sampled_examples}
    review_ids = {review.example_id for review in reviews}
    if len(review_ids) != len(reviews):
        raise ValueError("human audit review example IDs must be unique")
    if by_id != review_ids:
        raise ValueError("human audit reviews must cover the sampled examples exactly")
    ordered_reviews = tuple(sorted(reviews, key=lambda review: review.example_id))
    result_counts = Counter(review.review_result for review in ordered_reviews)
    return HumanAuditArtifact(
        audit_version=HUMAN_AUDIT_VERSION,
        sample_size=len(sampled_examples),
        coverage=_coverage(sampled_examples),
        reviews=ordered_reviews,
        summary={
            "passed": result_counts.get("PASS", 0),
            "failed": result_counts.get("FAIL", 0),
            "correction_required": sum(review.correction_required for review in ordered_reviews),
        },
    )


def write_human_audit_artifact(artifact: HumanAuditArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


HUMAN_REVIEW_DECISIONS = {"PASS", "FAIL", "CORRECTION_REQUIRED"}


def load_human_review_queue(path: Path) -> dict[str, object]:
    """Load a human-review queue artifact without changing review decisions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    reviews = data.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("human audit artifact must contain a reviews list")
    for review in reviews:
        if not isinstance(review, dict) or not review.get("example_id"):
            raise ValueError("every human audit review must contain example_id")
        status = review.get("review_status", "PENDING_HUMAN_REVIEW")
        if status not in HUMAN_REVIEW_DECISIONS | {"PENDING_HUMAN_REVIEW"}:
            raise ValueError(f"invalid human review status: {status}")
    return data


def _recompute_review_summary(data: dict[str, object]) -> None:
    reviews = data["reviews"]
    counts = Counter(review.get("review_status", "PENDING_HUMAN_REVIEW") for review in reviews)
    data["summary"] = {
        "passed": counts.get("PASS", 0),
        "failed": counts.get("FAIL", 0),
        "correction_required": counts.get("CORRECTION_REQUIRED", 0),
        "pending_human_review": counts.get("PENDING_HUMAN_REVIEW", 0),
    }


def update_human_review_record(
    path: Path,
    example_id: str,
    decision: str,
    notes: str = "",
) -> dict[str, object]:
    """Persist one genuine human review decision atomically enough for CLI use."""
    decision = decision.upper()
    if decision not in HUMAN_REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(HUMAN_REVIEW_DECISIONS)}")
    path = Path(path)
    data = load_human_review_queue(path)
    matches = [review for review in data["reviews"] if review["example_id"] == example_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one human-review record for {example_id}")
    record = matches[0]
    record["review_status"] = decision
    record["review_notes"] = notes
    # Checklist details are intentionally left for the human reviewer; a top-level
    # decision must never be expanded into fabricated automated checklist passes.
    _recompute_review_summary(data)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
    return data


def pending_human_review_records(path: Path) -> list[dict[str, object]]:
    data = load_human_review_queue(path)
    return [
        review for review in data["reviews"]
        if review.get("review_status", "PENDING_HUMAN_REVIEW") == "PENDING_HUMAN_REVIEW"
    ]


def validate_completed_human_audit(path: Path) -> dict[str, object]:
    """Validate a genuinely completed human-review queue and return coverage/counts.

    Completion is based only on recorded human decisions. Automated checklist fields
    are not interpreted as human judgments.
    """
    data = load_human_review_queue(path)
    reviews = data["reviews"]
    counts = Counter(review.get("review_status", "PENDING_HUMAN_REVIEW") for review in reviews)
    pending = counts.get("PENDING_HUMAN_REVIEW", 0)
    if pending:
        raise ValueError(f"human audit is incomplete: {pending} pending reviews")
    failed = counts.get("FAIL", 0)
    correction = counts.get("CORRECTION_REQUIRED", 0)

    coverage_fields = (
        "task_family",
        "difficulty",
        "knowledge_state",
        "evidence_status",
        "behavior_type",
    )
    coverage: dict[str, dict[str, int]] = {}
    for field in coverage_fields:
        values = [review.get(field) if review.get(field) is not None else "NONE" for review in reviews]
        coverage[field] = dict(sorted(Counter(values).items()))

    # split_type was added to the queue later in the readiness-fix pass when available.
    if all("split_type" in review for review in reviews):
        coverage["split_type"] = dict(sorted(Counter(review["split_type"] for review in reviews).items()))
    else:
        # Preserve the canonical sampled coverage already stored in the artifact.
        split_cov = data.get("coverage", {}).get("split_type", {})
        coverage["split_type"] = dict(sorted(split_cov.items()))

    return {
        "complete": failed == 0 and correction == 0,
        "passed": counts.get("PASS", 0),
        "failed": failed,
        "correction_required": correction,
        "pending_human_review": pending,
        "coverage": coverage,
    }


def finalize_completed_human_audit(path: Path) -> tuple[dict[str, object], str]:
    """Finalize a completed human audit and write a non-recursive SHA-256 sidecar."""
    import hashlib

    path = Path(path)
    result = validate_completed_human_audit(path)
    data = load_human_review_queue(path)
    data["complete"] = bool(result["complete"])
    data["summary"] = {
        "passed": result["passed"],
        "failed": result["failed"],
        "correction_required": result["correction_required"],
        "pending_human_review": result["pending_human_review"],
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return result, digest
