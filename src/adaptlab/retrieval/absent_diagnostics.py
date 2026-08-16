"""Diagnostics for retrieval-eligible examples with no sufficient gold evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import EvidenceStatus
from adaptlab.retrieval.schemas import RetrievalResult

ABSENT_DIAGNOSTIC_SCHEMA_VERSION = "1"


def with_absent_diagnostics(result: RetrievalResult) -> RetrievalResult:
    """Populate non-gold retrieval diagnostics for evidence-ABSENT examples.

    No relevance threshold or confidence heuristic is introduced here. In
    particular, ``wrongly_high_confidence`` remains ``None`` until a later
    mechanically defined policy explicitly freezes such a threshold.
    """

    if not result.retrieval_eligible or result.evidence_status is not EvidenceStatus.ABSENT:
        return replace(
            result,
            top1_chunk_id=None,
            top1_score=None,
            top_k_chunk_ids=(),
            score_margin_top1_top2=None,
            retrieval_returned_any_context=None,
            wrongly_high_confidence=None,
        )

    ids = tuple(result.candidate_chunk_ids[: result.top_k])
    scores = tuple(result.candidate_scores[: result.top_k])
    top1_chunk_id = ids[0] if ids else None
    top1_score = scores[0] if scores else None
    margin = scores[0] - scores[1] if len(scores) >= 2 else None
    return replace(
        result,
        top1_chunk_id=top1_chunk_id,
        top1_score=top1_score,
        top_k_chunk_ids=ids,
        score_margin_top1_top2=margin,
        retrieval_returned_any_context=bool(ids),
        wrongly_high_confidence=None,
    )


@dataclass(frozen=True)
class AbsentDiagnosticRow:
    example_id: str
    top1_chunk_id: str | None
    top1_score: float | None
    top_k_chunk_ids: tuple[str, ...]
    score_margin_top1_top2: float | None
    retrieval_returned_any_context: bool
    wrongly_high_confidence: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "top1_chunk_id": self.top1_chunk_id,
            "top1_score": self.top1_score,
            "top_k_chunk_ids": list(self.top_k_chunk_ids),
            "score_margin_top1_top2": self.score_margin_top1_top2,
            "retrieval_returned_any_context": self.retrieval_returned_any_context,
            "wrongly_high_confidence": self.wrongly_high_confidence,
        }


@dataclass(frozen=True)
class AbsentDiagnosticsReport:
    rows: tuple[AbsentDiagnosticRow, ...]
    schema_version: str = ABSENT_DIAGNOSTIC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": "retrieval_eligible AND evidence_status=ABSENT",
            "note": "Returned context is unverified; no gold-quality metric is implied.",
            "confidence_policy": "not_defined",
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_text(self) -> str:
        lines = [
            "Evidence-ABSENT retrieval diagnostics (returned context is unverified)",
            "example_id | top1_chunk_id | top1_score | top_k_chunk_ids | margin_top1_top2 | returned_any_context",
            "--- | --- | --- | --- | --- | ---",
        ]
        for row in self.rows:
            lines.append(" | ".join((
                row.example_id,
                row.top1_chunk_id or "",
                "" if row.top1_score is None else f"{row.top1_score:.6f}",
                ",".join(row.top_k_chunk_ids),
                "" if row.score_margin_top1_top2 is None else f"{row.score_margin_top1_top2:.6f}",
                "true" if row.retrieval_returned_any_context else "false",
            )))
        return "\n".join(lines) + "\n"


def summarize_absent_diagnostics(results: Iterable[RetrievalResult]) -> AbsentDiagnosticsReport:
    rows: list[AbsentDiagnosticRow] = []
    for result in sorted(results, key=lambda item: item.example_id):
        if not result.retrieval_eligible or result.evidence_status is not EvidenceStatus.ABSENT:
            continue
        diagnostic = with_absent_diagnostics(result)
        rows.append(AbsentDiagnosticRow(
            example_id=diagnostic.example_id,
            top1_chunk_id=diagnostic.top1_chunk_id,
            top1_score=diagnostic.top1_score,
            top_k_chunk_ids=diagnostic.top_k_chunk_ids,
            score_margin_top1_top2=diagnostic.score_margin_top1_top2,
            retrieval_returned_any_context=bool(diagnostic.retrieval_returned_any_context),
            wrongly_high_confidence=diagnostic.wrongly_high_confidence,
        ))
    return AbsentDiagnosticsReport(rows=tuple(rows))
