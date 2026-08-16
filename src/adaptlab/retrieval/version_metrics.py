"""Mechanical version-aware retrieval diagnostics for frozen Nimbus provenance."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import EvidenceStatus, KnowledgeState, TaskFamily
from adaptlab.retrieval.schemas import RetrievalResult

VERSION_METRIC_SCHEMA_VERSION = "1"
_APPLICABLE_STATES = (KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED)


def _relevant_obsolete_ids(result: RetrievalResult, chunks_by_id: dict[str, DocumentChunk]) -> frozenset[str]:
    """Find obsolete evidence for the same frozen logical facts as current gold.

    The current/gold side comes only from benchmark-provided gold chunk IDs.  The
    obsolete side comes only from frozen corpus provenance (`logical_fact_ids` and
    `is_obsolete`), never from text matching or an LLM judge.  This also handles
    REMOVED examples correctly: their current gold is retirement/current-state
    evidence, not an assumed replacement value.
    """
    logical_ids: set[str] = set()
    for chunk_id in result.gold_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"gold chunk {chunk_id!r} is missing from frozen corpus")
        logical_ids.update(chunk.logical_fact_ids)
    return frozenset(
        chunk.chunk_id
        for chunk in chunks_by_id.values()
        if chunk.is_obsolete and logical_ids.intersection(chunk.logical_fact_ids)
    )


def with_version_diagnostics(
    result: RetrievalResult,
    chunks: Iterable[DocumentChunk],
) -> RetrievalResult:
    """Populate version diagnostics when the frozen benchmark semantics permit it."""
    fields = {
        "wrong_version_top1": None,
        "current_gold_retrieved": None,
        "obsolete_only_retrieved": None,
        "current_and_obsolete_retrieved": None,
    }
    if (
        not result.retrieval_eligible
        or result.task_family is not TaskFamily.changed_knowledge
        or result.knowledge_state not in _APPLICABLE_STATES
        or result.evidence_status is not EvidenceStatus.PRESENT
    ):
        return replace(result, **fields)

    chunk_tuple = tuple(chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunk_tuple}
    if not chunk_tuple:
        raise ValueError("frozen corpus must not be empty")
    if len(chunks_by_id) != len(chunk_tuple):
        raise ValueError("frozen corpus chunk IDs must be unique")

    obsolete_ids = _relevant_obsolete_ids(result, chunks_by_id)
    retrieved = frozenset(result.candidate_chunk_ids)
    current = bool(retrieved.intersection(result.gold_chunk_ids))
    obsolete = bool(retrieved.intersection(obsolete_ids))
    top1 = result.candidate_chunk_ids[0] if result.candidate_chunk_ids else None
    return replace(
        result,
        current_gold_retrieved=current,
        obsolete_only_retrieved=obsolete and not current,
        current_and_obsolete_retrieved=current and obsolete,
        wrong_version_top1=top1 in obsolete_ids if top1 is not None else False,
    )


@dataclass(frozen=True)
class VersionDiagnosticRow:
    knowledge_state: str
    n: int
    current_gold_retrieved_count: int
    current_gold_retrieved_rate: float
    obsolete_only_retrieved_count: int
    obsolete_only_retrieved_rate: float
    current_and_obsolete_retrieved_count: int
    current_and_obsolete_retrieved_rate: float
    wrong_version_top1_count: int
    wrong_version_top1_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_state": self.knowledge_state,
            "n": self.n,
            "CURRENT_GOLD_RETRIEVED": {"count": self.current_gold_retrieved_count, "rate": self.current_gold_retrieved_rate},
            "OBSOLETE_ONLY_RETRIEVED": {"count": self.obsolete_only_retrieved_count, "rate": self.obsolete_only_retrieved_rate},
            "CURRENT_AND_OBSOLETE_RETRIEVED": {"count": self.current_and_obsolete_retrieved_count, "rate": self.current_and_obsolete_retrieved_rate},
            "WRONG_VERSION_TOP1": {"count": self.wrong_version_top1_count, "rate": self.wrong_version_top1_rate},
        }


@dataclass(frozen=True)
class VersionDiagnosticsReport:
    rows: tuple[VersionDiagnosticRow, ...]
    schema_version: str = VERSION_METRIC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "rows": [row.to_dict() for row in self.rows]}

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_text(self) -> str:
        lines = ["Version-aware retrieval diagnostics (evidence-PRESENT changed_knowledge only)"]
        for row in self.rows:
            lines.append(
                f"{row.knowledge_state}: n={row.n} "
                f"current={row.current_gold_retrieved_count}/{row.n} "
                f"obsolete_only={row.obsolete_only_retrieved_count}/{row.n} "
                f"current_and_obsolete={row.current_and_obsolete_retrieved_count}/{row.n} "
                f"wrong_version_top1={row.wrong_version_top1_count}/{row.n}"
            )
        return "\n".join(lines) + "\n"


def summarize_version_diagnostics(
    results: Iterable[RetrievalResult], chunks: Iterable[DocumentChunk]
) -> VersionDiagnosticsReport:
    chunk_tuple = tuple(chunks)
    diagnosed = tuple(with_version_diagnostics(result, chunk_tuple) for result in results)
    rows: list[VersionDiagnosticRow] = []
    for state in _APPLICABLE_STATES:
        group = tuple(r for r in diagnosed if r.knowledge_state is state and r.current_gold_retrieved is not None)
        if not group:
            continue
        n = len(group)
        counts = {
            "current": sum(r.current_gold_retrieved is True for r in group),
            "obsolete_only": sum(r.obsolete_only_retrieved is True for r in group),
            "both": sum(r.current_and_obsolete_retrieved is True for r in group),
            "wrong_top1": sum(r.wrong_version_top1 is True for r in group),
        }
        rows.append(VersionDiagnosticRow(
            knowledge_state=state.value, n=n,
            current_gold_retrieved_count=counts["current"], current_gold_retrieved_rate=counts["current"] / n,
            obsolete_only_retrieved_count=counts["obsolete_only"], obsolete_only_retrieved_rate=counts["obsolete_only"] / n,
            current_and_obsolete_retrieved_count=counts["both"], current_and_obsolete_retrieved_rate=counts["both"] / n,
            wrong_version_top1_count=counts["wrong_top1"], wrong_version_top1_rate=counts["wrong_top1"] / n,
        ))
    return VersionDiagnosticsReport(rows=tuple(rows))
