"""Deterministic anti-confounding diagnostics for the Nimbus benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from adaptlab.benchmark.documents import Document
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    DocumentStyle,
    EvidenceStatus,
    KnowledgeState,
    SplitType,
    TaskFamily,
)
from adaptlab.domain.world import NimbusWorld

NOT_APPLICABLE = "NOT_APPLICABLE"
MULTIPLE = "MULTIPLE"


class ImbalanceDisposition(str, Enum):
    """Required disposition labels for material full-benchmark associations."""

    ACCEPTED_BY_DESIGN = "ACCEPTED_BY_DESIGN"
    CORRECTED_BEFORE_FREEZE = "CORRECTED_BEFORE_FREEZE"
    EXCLUDED_FROM_CANONICAL_CLAIM = "EXCLUDED_FROM_CANONICAL_CLAIM"


@dataclass(frozen=True, slots=True)
class AssociationFinding:
    table: str
    row: str
    dominant_columns: tuple[str, ...]
    count: int
    total: int
    share: float
    disposition: ImbalanceDisposition
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "row": self.row,
            "dominant_columns": list(self.dominant_columns),
            "count": self.count,
            "total": self.total,
            "share": self.share,
            "disposition": self.disposition.value,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    tables: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]
    summary: dict[str, Any]
    material_imbalances: tuple[AssociationFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": self.tables,
            "warnings": list(self.warnings),
            "summary": self.summary,
            "material_imbalances": [finding.to_dict() for finding in self.material_imbalances],
        }


def _component_for_example(world: NimbusWorld, example: BenchmarkExample) -> str:
    record_components = {
        fact.component_family
        for fact in world.facts
        if fact.record_id in example.required_record_ids
    }
    logical_components = {
        fact.component_family
        for fact in world.facts
        if fact.logical_fact_id in example.required_logical_fact_ids
    }
    components = record_components | logical_components
    if not components:
        return NOT_APPLICABLE
    if len(components) == 1:
        return next(iter(components))
    return MULTIPLE


def _fact_family_for_example(world: NimbusWorld, example: BenchmarkExample) -> str:
    logical_ids = set(example.required_logical_fact_ids)
    if example.lifecycle_logical_fact_id:
        logical_ids.add(example.lifecycle_logical_fact_id)
    families = {
        fact.fact_type
        for fact in world.facts
        if fact.record_id in example.required_record_ids or fact.logical_fact_id in logical_ids
    }
    if not families:
        return NOT_APPLICABLE
    if len(families) == 1:
        return next(iter(families))
    return MULTIPLE


def _cross_tab(
    rows: Sequence[str],
    columns: Sequence[str],
    observations: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    counts = Counter(observations)
    cells = {
        row: {column: counts[(row, column)] for column in columns}
        for row in rows
    }
    empty = [
        {"row": row, "column": column}
        for row in rows
        for column in columns
        if counts[(row, column)] == 0
    ]
    return {
        "rows": list(rows),
        "columns": list(columns),
        "counts": cells,
        "empty_categories": empty,
    }


def _concentration_warnings(name: str, table: dict[str, Any]) -> list[str]:
    """Flag strong row-wise concentration without treating it as a failure."""

    warnings: list[str] = []
    columns: list[str] = table["columns"]
    counts: dict[str, dict[str, int]] = table["counts"]
    for row, row_counts in counts.items():
        total = sum(row_counts.values())
        if total < 4:
            continue
        peak = max(row_counts.values(), default=0)
        if peak == 0:
            continue
        share = peak / total
        if share >= 0.80:
            peak_columns = sorted(column for column in columns if row_counts[column] == peak)
            warnings.append(
                f"{name}: row {row!r} is severely concentrated in "
                f"{', '.join(repr(column) for column in peak_columns)} "
                f"({peak}/{total}, {share:.0%})"
            )
    return warnings


def _design_disposition(table: str, row: str, dominant_columns: tuple[str, ...]) -> tuple[ImbalanceDisposition, str]:
    """Assign a deterministic pre-freeze disposition to a material association."""

    # These associations are direct consequences of benchmark contracts rather
    # than accidental proxies. They are retained and documented explicitly.
    if table == "task_family_x_component_family" and row == TaskFamily.behavior_only.value:
        return (
            ImbalanceDisposition.ACCEPTED_BY_DESIGN,
            "Behavior-only tasks intentionally require no Nimbus world fact, so component_family is not applicable.",
        )
    if table == "knowledge_state_x_component_family" and row == KnowledgeState.NOT_APPLICABLE.value:
        return (
            ImbalanceDisposition.ACCEPTED_BY_DESIGN,
            "Knowledge-state metadata is intentionally not applicable outside changed-knowledge tasks.",
        )
    if table == "evidence_status_x_difficulty" and row == EvidenceStatus.NOT_APPLICABLE.value:
        return (
            ImbalanceDisposition.ACCEPTED_BY_DESIGN,
            "Behavior-only tasks intentionally use NOT_APPLICABLE evidence status; difficulty is constructed independently.",
        )
    if table in {"task_family_x_split_type", "difficulty_x_split_type"} and "structural_holdout" in dominant_columns:
        return (
            ImbalanceDisposition.ACCEPTED_BY_DESIGN,
            "Structural-holdout frequency follows the frozen holdout policy and is not selected from model performance.",
        )
    return (
        ImbalanceDisposition.EXCLUDED_FROM_CANONICAL_CLAIM,
        "Strong association is retained for review; no claim of independence is made for this factor pair unless corrected before freeze.",
    )


def _material_findings(tables: dict[str, dict[str, Any]], *, minimum_total: int = 20, threshold: float = 0.80) -> tuple[AssociationFinding, ...]:
    findings: list[AssociationFinding] = []
    for table_name in sorted(tables):
        table = tables[table_name]
        columns: list[str] = table["columns"]
        for row in table["rows"]:
            row_counts: dict[str, int] = table["counts"][row]
            total = sum(row_counts.values())
            if total < minimum_total:
                continue
            peak = max(row_counts.values(), default=0)
            if peak == 0:
                continue
            share = peak / total
            if share < threshold:
                continue
            dominant = tuple(sorted(column for column in columns if row_counts[column] == peak))
            disposition, explanation = _design_disposition(table_name, row, dominant)
            findings.append(
                AssociationFinding(
                    table=table_name,
                    row=row,
                    dominant_columns=dominant,
                    count=peak,
                    total=total,
                    share=round(share, 6),
                    disposition=disposition,
                    explanation=explanation,
                )
            )
    return tuple(findings)


def audit_benchmark(
    world: NimbusWorld,
    documents: Iterable[Document],
    examples: Iterable[BenchmarkExample],
    *,
    full_scale: bool = False,
) -> AuditResult:
    """Return deterministic cross-tab diagnostics for a benchmark fixture.

    Imbalance is diagnostic only. For ``full_scale=True`` material associations
    receive one required disposition label so they can be reviewed before the
    benchmark freeze without requiring cosmetic balance.
    """

    documents = list(documents)
    examples = list(examples)
    documents_by_id = {document.document_id: document for document in documents}

    task_families = [item.value for item in TaskFamily]
    difficulties = [item.value for item in Difficulty]
    split_types = [item.value for item in SplitType]
    knowledge_states = [item.value for item in KnowledgeState]
    behavior_types = [item.value for item in BehaviorType] + [NOT_APPLICABLE]
    evidence_statuses = [item.value for item in EvidenceStatus]
    document_styles = [item.value for item in DocumentStyle]
    components = sorted({fact.component_family for fact in world.facts}) + [NOT_APPLICABLE, MULTIPLE]
    fact_families = sorted({fact.fact_type for fact in world.facts}) + [NOT_APPLICABLE, MULTIPLE]

    component_by_example = {
        example.example_id: _component_for_example(world, example) for example in examples
    }
    fact_family_by_example = {
        example.example_id: _fact_family_for_example(world, example) for example in examples
    }

    tables: dict[str, dict[str, Any]] = {
        "task_family_x_difficulty": _cross_tab(
            task_families,
            difficulties,
            ((example.task_family.value, example.difficulty.value) for example in examples),
        ),
        "task_family_x_component_family": _cross_tab(
            task_families,
            components,
            (
                (example.task_family.value, component_by_example[example.example_id])
                for example in examples
            ),
        ),
        "task_family_x_split_type": _cross_tab(
            task_families,
            split_types,
            ((example.task_family.value, example.split_type.value) for example in examples),
        ),
        "difficulty_x_split_type": _cross_tab(
            difficulties,
            split_types,
            ((example.difficulty.value, example.split_type.value) for example in examples),
        ),
        "knowledge_state_x_component_family": _cross_tab(
            knowledge_states,
            components,
            (
                (example.knowledge_state.value, component_by_example[example.example_id])
                for example in examples
            ),
        ),
        "behavior_type_x_difficulty": _cross_tab(
            behavior_types,
            difficulties,
            (
                (
                    example.behavior_type.value if example.behavior_type else NOT_APPLICABLE,
                    example.difficulty.value,
                )
                for example in examples
            ),
        ),
        "behavior_type_x_component_family": _cross_tab(
            behavior_types,
            components,
            (
                (
                    example.behavior_type.value if example.behavior_type else NOT_APPLICABLE,
                    component_by_example[example.example_id],
                )
                for example in examples
            ),
        ),
        "evidence_status_x_difficulty": _cross_tab(
            evidence_statuses,
            difficulties,
            ((example.evidence_status.value, example.difficulty.value) for example in examples),
        ),
        "document_style_x_task_family": _cross_tab(
            document_styles,
            task_families,
            (
                (documents_by_id[document_id].document_style.value, example.task_family.value)
                for example in examples
                for document_id in example.gold_document_ids
                if document_id in documents_by_id
            ),
        ),
    }

    if full_scale:
        tables["knowledge_state_x_fact_family"] = _cross_tab(
            knowledge_states,
            fact_families,
            (
                (example.knowledge_state.value, fact_family_by_example[example.example_id])
                for example in examples
            ),
        )

    warnings: list[str] = []
    for name in sorted(tables):
        warnings.extend(_concentration_warnings(name, tables[name]))

    material_imbalances = _material_findings(tables) if full_scale else ()
    if full_scale:
        for finding in material_imbalances:
            warnings.append(
                f"{finding.table}: material association for row {finding.row!r} "
                f"({finding.count}/{finding.total}, {finding.share:.0%}); "
                f"disposition={finding.disposition.value}"
            )

    empty_cell_count = sum(len(table["empty_categories"]) for table in tables.values())
    warnings = sorted(set(warnings))
    disposition_counts = Counter(f.disposition.value for f in material_imbalances)
    summary = {
        "table_count": len(tables),
        "example_count": len(examples),
        "document_count": len(documents),
        "empty_cell_count": empty_cell_count,
        "concentration_warning_count": len(warnings),
        "prototype_balance_required": False,
        "full_scale": full_scale,
        "material_imbalance_count": len(material_imbalances),
        "material_imbalance_dispositions": dict(sorted(disposition_counts.items())),
    }
    return AuditResult(
        tables=tables,
        warnings=tuple(warnings),
        summary=summary,
        material_imbalances=material_imbalances,
    )


def write_audit_artifact(result: AuditResult, path: str | Path) -> Path:
    """Write a stable machine-readable audit artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
