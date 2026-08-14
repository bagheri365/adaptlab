"""Deterministic anti-confounding diagnostics for the Nimbus benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class AuditResult:
    tables: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]
    summary: dict[str, Any]


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


def audit_benchmark(
    world: NimbusWorld,
    documents: Iterable[Document],
    examples: Iterable[BenchmarkExample],
) -> AuditResult:
    """Return deterministic cross-tab diagnostics for a benchmark fixture.

    Prototype imbalance is diagnostic only. Empty cells and severe concentration
    become warnings/metadata, never benchmark failures.
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

    component_by_example = {
        example.example_id: _component_for_example(world, example) for example in examples
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

    warnings: list[str] = []
    for name in sorted(tables):
        warnings.extend(_concentration_warnings(name, tables[name]))

    empty_cell_count = sum(len(table["empty_categories"]) for table in tables.values())
    warnings = sorted(set(warnings))
    summary = {
        "table_count": len(tables),
        "example_count": len(examples),
        "document_count": len(documents),
        "empty_cell_count": empty_cell_count,
        "concentration_warning_count": len(warnings),
        "prototype_balance_required": False,
    }
    return AuditResult(tables=tables, warnings=tuple(warnings), summary=summary)
