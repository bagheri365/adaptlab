"""Deterministic retrieval-failure audits for Milestone 4.

Failure labels are assigned only when they can be derived mechanically from the
frozen retrieval result plus frozen corpus/example-visible text/provenance.  No
LLM judge or semantic guess is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import EvidenceStatus
from adaptlab.retrieval.schemas import RetrievalResult
from adaptlab.retrieval.version_metrics import with_version_diagnostics

FAILURE_AUDIT_SCHEMA_VERSION = "1"
_IDENTIFIER_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*|[A-Z][A-Z0-9]*_[A-Z0-9_-]+)\b")


class RetrievalFailureCategory(str, Enum):
    GOLD_OUTSIDE_TOP_K = "GOLD_OUTSIDE_TOP_K"
    PARTIAL_GOLD = "PARTIAL_GOLD"
    OBSOLETE_ONLY = "OBSOLETE_ONLY"
    WRONG_VERSION_TOP1 = "WRONG_VERSION_TOP1"
    NEAR_DUPLICATE_DISTRACTOR = "NEAR_DUPLICATE_DISTRACTOR"
    SAME_COMPONENT_DISTRACTOR = "SAME_COMPONENT_DISTRACTOR"
    IDENTIFIER_SHORTCUT = "IDENTIFIER_SHORTCUT"
    NO_GOLD_EXISTS = "NO_GOLD_EXISTS"


def _identifiers(text: str) -> frozenset[str]:
    return frozenset(match.group(0) for match in _IDENTIFIER_RE.finditer(text))


def _corpus_maps(chunks: tuple[DocumentChunk, ...]) -> tuple[dict[str, DocumentChunk], Counter[str]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("frozen corpus chunk IDs must be unique")
    identifier_counts: Counter[str] = Counter()
    for chunk in chunks:
        identifier_counts.update(_identifiers(chunk.content))
    return by_id, identifier_counts


@dataclass(frozen=True)
class RetrievalFailureAudit:
    example_id: str
    task_family: str
    difficulty: str
    knowledge_state: str
    split_type: str
    categories: tuple[str, ...]
    retrieved_gold_chunk_ids: tuple[str, ...]
    missing_required_gold_chunk_ids: tuple[str, ...]
    distractor_chunk_ids: tuple[str, ...]
    schema_version: str = FAILURE_AUDIT_SCHEMA_VERSION

    @property
    def has_failure(self) -> bool:
        return bool(self.categories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "task_family": self.task_family,
            "difficulty": self.difficulty,
            "knowledge_state": self.knowledge_state,
            "split_type": self.split_type,
            "categories": list(self.categories),
            "retrieved_gold_chunk_ids": list(self.retrieved_gold_chunk_ids),
            "missing_required_gold_chunk_ids": list(self.missing_required_gold_chunk_ids),
            "distractor_chunk_ids": list(self.distractor_chunk_ids),
        }


def audit_retrieval_failure(
    result: RetrievalResult,
    chunks: Iterable[DocumentChunk],
) -> RetrievalFailureAudit:
    """Mechanically label one retrieval result without forcing a causal label."""
    chunk_tuple = tuple(chunks)
    chunks_by_id, identifier_counts = _corpus_maps(chunk_tuple)
    unknown = set(result.candidate_chunk_ids) - set(chunks_by_id)
    if unknown:
        raise ValueError(f"retrieved chunks missing from frozen corpus: {sorted(unknown)!r}")

    categories: set[RetrievalFailureCategory] = set()
    retrieved = tuple(result.candidate_chunk_ids)
    gold = frozenset(result.gold_chunk_ids)
    required = frozenset(result.required_gold_chunk_ids)
    retrieved_gold = tuple(chunk_id for chunk_id in retrieved if chunk_id in gold)
    missing_required = tuple(sorted(required - set(retrieved)))
    distractors = tuple(chunk_id for chunk_id in retrieved if chunk_id not in gold)

    if result.retrieval_eligible and result.evidence_status is EvidenceStatus.ABSENT:
        categories.add(RetrievalFailureCategory.NO_GOLD_EXISTS)
    elif result.retrieval_eligible and result.evidence_status is EvidenceStatus.PRESENT:
        required_retrieved = required.intersection(retrieved)
        if not retrieved_gold:
            categories.add(RetrievalFailureCategory.GOLD_OUTSIDE_TOP_K)
        if required_retrieved and required_retrieved != required:
            categories.add(RetrievalFailureCategory.PARTIAL_GOLD)

        # Version labels reuse Prompt 6's mechanical frozen-provenance logic.
        versioned = with_version_diagnostics(result, chunk_tuple)
        if versioned.obsolete_only_retrieved is True:
            categories.add(RetrievalFailureCategory.OBSOLETE_ONLY)
        if versioned.wrong_version_top1 is True:
            categories.add(RetrievalFailureCategory.WRONG_VERSION_TOP1)

        # Cause-like distractor labels are only attached to an actual retrieval
        # deficiency (not-all-required), avoiding labeling harmless extra context
        # as a retrieval failure.
        retrieval_deficient = bool(required) and bool(missing_required)
        if retrieval_deficient and distractors:
            gold_chunks = [chunks_by_id[cid] for cid in gold if cid in chunks_by_id]
            gold_logical_ids = set().union(*(set(c.logical_fact_ids) for c in gold_chunks)) if gold_chunks else set()
            gold_components = {c.component_family for c in gold_chunks}

            distractor_objs = [chunks_by_id[cid] for cid in distractors]
            if any(
                (not chunk.is_authoritative)
                and (not chunk.is_obsolete)
                and bool(gold_logical_ids.intersection(chunk.logical_fact_ids))
                for chunk in distractor_objs
            ):
                categories.add(RetrievalFailureCategory.NEAR_DUPLICATE_DISTRACTOR)

            if gold_components and any(chunk.component_family in gold_components for chunk in distractor_objs):
                categories.add(RetrievalFailureCategory.SAME_COMPONENT_DISTRACTOR)

            query_ids = _identifiers(result.query_text)
            unique_query_ids = {identifier for identifier in query_ids if identifier_counts[identifier] == 1}
            if unique_query_ids and any(
                unique_query_ids.intersection(_identifiers(chunk.content)) for chunk in distractor_objs
            ):
                categories.add(RetrievalFailureCategory.IDENTIFIER_SHORTCUT)

    ordered = tuple(category.value for category in RetrievalFailureCategory if category in categories)
    return RetrievalFailureAudit(
        example_id=result.example_id,
        task_family=result.task_family.value,
        difficulty=result.difficulty.value,
        knowledge_state=result.knowledge_state.value,
        split_type=result.split_type.value,
        categories=ordered,
        retrieved_gold_chunk_ids=retrieved_gold,
        missing_required_gold_chunk_ids=missing_required,
        distractor_chunk_ids=distractors,
    )


@dataclass(frozen=True)
class RetrievalFailureGroupRow:
    dimension: str
    value: str
    n: int
    failure_example_count: int
    category_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "n": self.n,
            "failure_example_count": self.failure_example_count,
            "category_counts": dict(sorted(self.category_counts.items())),
        }


@dataclass(frozen=True)
class RetrievalFailureAuditReport:
    examples: tuple[RetrievalFailureAudit, ...]
    groups: tuple[RetrievalFailureGroupRow, ...]
    schema_version: str = FAILURE_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "examples": [row.to_dict() for row in self.examples],
            "groups": [row.to_dict() for row in self.groups],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_text(self) -> str:
        lines = ["Deterministic retrieval failure audit"]
        for row in self.groups:
            counts = ", ".join(f"{name}={count}" for name, count in sorted(row.category_counts.items())) or "none"
            lines.append(
                f"{row.dimension}={row.value}: n={row.n} failures={row.failure_example_count}; {counts}"
            )
        return "\n".join(lines) + "\n"


def summarize_retrieval_failures(
    results: Iterable[RetrievalResult],
    chunks: Iterable[DocumentChunk],
) -> RetrievalFailureAuditReport:
    chunk_tuple = tuple(chunks)
    audits = tuple(sorted(
        (audit_retrieval_failure(result, chunk_tuple) for result in results),
        key=lambda row: row.example_id,
    ))
    dimensions = ("task_family", "difficulty", "knowledge_state", "split_type")
    groups: list[RetrievalFailureGroupRow] = []
    for dimension in dimensions:
        grouped: dict[str, list[RetrievalFailureAudit]] = defaultdict(list)
        for audit in audits:
            grouped[getattr(audit, dimension)].append(audit)
        for value in sorted(grouped):
            rows = grouped[value]
            counts: Counter[str] = Counter(category for row in rows for category in row.categories)
            groups.append(RetrievalFailureGroupRow(
                dimension=dimension,
                value=value,
                n=len(rows),
                failure_example_count=sum(row.has_failure for row in rows),
                category_counts=dict(counts),
            ))
    return RetrievalFailureAuditReport(examples=audits, groups=tuple(groups))
