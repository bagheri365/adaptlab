"""Fixture validation and deterministic structural-holdout assignment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
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
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import FactStatus, NimbusWorld

HOLDOUT_DIMENSION = "component_family"
STRUCTURAL_TEST_ONLY = frozenset({"deployments"})
TRAIN_ELIGIBLE = frozenset({"authentication"})
TRAIN_OR_VALIDATION_ELIGIBLE = frozenset({"projects"})


@dataclass(frozen=True, slots=True)
class ValidationResult:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, Any]


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _component_for_example(
    example: BenchmarkExample,
    record_to_component: dict[str, str],
    logical_to_components: dict[str, set[str]],
) -> str | None:
    components = {
        record_to_component[record_id]
        for record_id in example.required_record_ids
        if record_id in record_to_component
    }
    for logical_id in example.required_logical_fact_ids:
        components.update(logical_to_components.get(logical_id, set()))
    if len(components) == 1:
        return next(iter(components))
    return None



def _derive_scoring_output(
    example: BenchmarkExample,
    records_by_id: dict[str, object],
    facts_by_logical_id: dict[str, dict[str, object]],
) -> tuple[bool, Any]:
    """Mechanically derive supported gold answers from structured truth/typed metadata."""

    rule = example.scoring_rule
    if rule is None:
        return False, None

    required_records = [
        records_by_id[record_id]
        for record_id in example.required_record_ids
        if record_id in records_by_id
    ]

    if rule is ScoringRule.FACT_VALUE:
        if len(required_records) != 1:
            return False, None
        return True, required_records[0].value

    if rule is ScoringRule.RETIRED_STATUS:
        lifecycle_id = example.lifecycle_logical_fact_id
        if lifecycle_id and lifecycle_id in facts_by_logical_id:
            versions = facts_by_logical_id[lifecycle_id]
            v1_fact = versions.get("v1")
            if v1_fact is not None and classify_knowledge_state(v1_fact, versions.get("v2")) is KnowledgeState.REMOVED:
                return True, "RETIRED"
        if len(required_records) == 1 and required_records[0].status is FactStatus.RETIRED:
            return True, "RETIRED"
        return False, None

    if rule is ScoringRule.ABSTENTION:
        if example.evidence_status is EvidenceStatus.ABSENT and not any((
            example.required_record_ids,
            example.required_logical_fact_ids,
            example.gold_document_ids,
            example.gold_chunk_ids,
        )):
            return True, "INSUFFICIENT_EVIDENCE"
        return False, None

    if len(required_records) != 1:
        return False, None
    record = required_records[0]
    params = example.scoring_parameters or {}

    if rule is ScoringRule.STRUCTURED_EXTRACTION:
        if params.get("mode") == "scalar":
            return True, record.value
        output_key = params.get("output_key")
        if isinstance(output_key, str):
            value = record.value
            if params.get("coerce") == "int":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    return False, None
            return True, {output_key: value}
        return False, None

    if rule is ScoringRule.CONDITIONAL_RULE:
        candidate = params.get("candidate")
        operator = params.get("operator")
        if candidate is None or operator != "lte":
            return False, None
        try:
            condition = float(candidate) <= float(record.value)
        except (TypeError, ValueError):
            return False, None
        return True, params.get("true_output") if condition else params.get("false_output")

    if rule is ScoringRule.CLASSIFICATION:
        threshold = params.get("threshold")
        operator = params.get("operator")
        if threshold is None or operator != "gt":
            return False, None
        try:
            condition = float(record.value) > float(threshold)
        except (TypeError, ValueError):
            return False, None
        return True, params.get("true_output") if condition else params.get("false_output")

    return False, None

def apply_structural_holdout_rules(
    world: NimbusWorld,
    examples: Iterable[BenchmarkExample],
) -> list[BenchmarkExample]:
    """Apply deterministic component-family split eligibility rules.

    Dataset split and structural-holdout status remain separate fields. Examples
    whose required truth belongs to ``deployments`` are structural-test-only;
    ``projects`` examples are assigned to validation; ``authentication`` and
    component-free behavior examples remain IID train examples.
    """

    records_by_id = {fact.record_id: fact for fact in world.facts}
    record_to_component = {fact.record_id: fact.component_family for fact in world.facts}
    logical_to_components: dict[str, set[str]] = {}
    for fact in world.facts:
        logical_to_components.setdefault(fact.logical_fact_id, set()).add(fact.component_family)

    assigned: list[BenchmarkExample] = []
    for example in examples:
        component = _component_for_example(example, record_to_component, logical_to_components)
        if component in STRUCTURAL_TEST_ONLY:
            assigned.append(
                replace(
                    example,
                    split=Split.test,
                    split_type=SplitType.structural_holdout,
                    holdout_dimension=HOLDOUT_DIMENSION,
                    holdout_group=component,
                )
            )
        elif component in TRAIN_OR_VALIDATION_ELIGIBLE:
            assigned.append(
                replace(
                    example,
                    split=Split.validation,
                    split_type=SplitType.iid,
                    holdout_dimension=None,
                    holdout_group=None,
                )
            )
        else:
            assigned.append(
                replace(
                    example,
                    split=Split.train,
                    split_type=SplitType.iid,
                    holdout_dimension=None,
                    holdout_group=None,
                )
            )
    assigned.sort(key=lambda example: example.example_id)
    return assigned


def validate_fixture(
    world: NimbusWorld,
    documents: Iterable[Document],
    chunks: Iterable[DocumentChunk],
    examples: Iterable[BenchmarkExample],
    *,
    expected_generation_seed: int | None = None,
) -> ValidationResult:
    """Validate benchmark invariants without repairing invalid input."""

    documents = list(documents)
    chunks = list(chunks)
    examples = list(examples)
    errors: list[str] = []
    warnings: list[str] = []

    record_ids = [fact.record_id for fact in world.facts]
    logical_ids = {fact.logical_fact_id for fact in world.facts}
    document_ids = [document.document_id for document in documents]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    example_ids = [example.example_id for example in examples]

    for label, values in (
        ("record_id", record_ids),
        ("document_id", document_ids),
        ("chunk_id", chunk_ids),
        ("example_id", example_ids),
    ):
        dupes = _duplicates(values)
        if dupes:
            errors.append(f"duplicate {label} values: {', '.join(dupes)}")

    record_id_set = set(record_ids)
    document_id_set = set(document_ids)
    chunk_id_set = set(chunk_ids)
    documents_by_id = {document.document_id: document for document in documents}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    records_by_id = {fact.record_id: fact for fact in world.facts}
    record_to_component = {fact.record_id: fact.component_family for fact in world.facts}
    logical_to_components: dict[str, set[str]] = {}
    facts_by_logical_id: dict[str, dict[str, object]] = {}
    for fact in world.facts:
        logical_to_components.setdefault(fact.logical_fact_id, set()).add(fact.component_family)
        facts_by_logical_id.setdefault(fact.logical_fact_id, {})[fact.version] = fact

    for document in documents:
        missing = sorted(set(document.record_ids) - record_id_set)
        if missing:
            errors.append(
                f"document {document.document_id} references unknown record IDs: {', '.join(missing)}"
            )
        missing_logical = sorted(set(document.logical_fact_ids) - logical_ids)
        if missing_logical:
            errors.append(
                f"document {document.document_id} references unknown logical fact IDs: "
                + ", ".join(missing_logical)
            )

    for chunk in chunks:
        if chunk.document_id not in document_id_set:
            errors.append(
                f"chunk {chunk.chunk_id} belongs to unknown document {chunk.document_id}"
            )
        missing = sorted(set(chunk.record_ids) - record_id_set)
        if missing:
            errors.append(
                f"chunk {chunk.chunk_id} references unknown record IDs: {', '.join(missing)}"
            )
        missing_logical = sorted(set(chunk.logical_fact_ids) - logical_ids)
        if missing_logical:
            errors.append(
                f"chunk {chunk.chunk_id} references unknown logical fact IDs: "
                + ", ".join(missing_logical)
            )

    expected_seed = world.generation_seed if expected_generation_seed is None else expected_generation_seed

    for example in examples:
        prefix = f"example {example.example_id}"

        enum_fields = (
            ("task_family", example.task_family, TaskFamily),
            ("difficulty", example.difficulty, Difficulty),
            ("split", example.split, Split),
            ("split_type", example.split_type, SplitType),
            ("knowledge_state", example.knowledge_state, KnowledgeState),
            ("evidence_status", example.evidence_status, EvidenceStatus),
        )
        if example.behavior_type is not None:
            enum_fields += (("behavior_type", example.behavior_type, BehaviorType),)
        if example.scoring_rule is not None:
            enum_fields += (("scoring_rule", example.scoring_rule, ScoringRule),)
        for field_name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                errors.append(f"{prefix} has invalid {field_name}: {value!r}")

        missing_records = sorted(set(example.required_record_ids) - record_id_set)
        if missing_records:
            errors.append(f"{prefix} references unknown required record IDs: {', '.join(missing_records)}")
        missing_logical = sorted(set(example.required_logical_fact_ids) - logical_ids)
        if missing_logical:
            errors.append(
                f"{prefix} references unknown required logical fact IDs: {', '.join(missing_logical)}"
            )

        required_logical_set = set(example.required_logical_fact_ids)
        for record_id in example.required_record_ids:
            record = records_by_id.get(record_id)
            if record is not None and record.logical_fact_id not in required_logical_set:
                errors.append(
                    f"{prefix} required record {record_id} belongs to logical fact "
                    f"{record.logical_fact_id}, which is not listed in required_logical_fact_ids"
                )
        for logical_id in example.required_logical_fact_ids:
            related_required = [
                records_by_id[record_id]
                for record_id in example.required_record_ids
                if record_id in records_by_id
                and records_by_id[record_id].logical_fact_id == logical_id
            ]
            if not related_required:
                errors.append(
                    f"{prefix} required logical fact {logical_id} has no matching required record"
                )

        missing_docs = sorted(set(example.gold_document_ids) - document_id_set)
        if missing_docs:
            errors.append(f"{prefix} references unknown gold document IDs: {', '.join(missing_docs)}")
        missing_chunks = sorted(set(example.gold_chunk_ids) - chunk_id_set)
        if missing_chunks:
            errors.append(f"{prefix} references unknown gold chunk IDs: {', '.join(missing_chunks)}")

        gold_chunks = [
            chunks_by_id[chunk_id]
            for chunk_id in example.gold_chunk_ids
            if chunk_id in chunks_by_id
        ]
        for chunk in gold_chunks:
            if chunk.document_id not in example.gold_document_ids:
                errors.append(
                    f"{prefix} gold chunk {chunk.chunk_id} belongs to {chunk.document_id}, "
                    "which is not listed in gold_document_ids"
                )
            if not chunk.is_authoritative:
                errors.append(f"{prefix} gold chunk {chunk.chunk_id} is not authoritative")
            if chunk.is_obsolete:
                errors.append(f"{prefix} gold chunk {chunk.chunk_id} is obsolete")

            parent = documents_by_id.get(chunk.document_id)
            if parent is not None:
                if chunk.version != parent.version:
                    errors.append(
                        f"{prefix} gold chunk {chunk.chunk_id} version does not match "
                        f"parent document {parent.document_id}"
                    )
                if chunk.component_family != parent.component_family:
                    errors.append(
                        f"{prefix} gold chunk {chunk.chunk_id} component_family does not match "
                        f"parent document {parent.document_id}"
                    )
                if chunk.document_style is not parent.document_style:
                    errors.append(
                        f"{prefix} gold chunk {chunk.chunk_id} document_style does not match "
                        f"parent document {parent.document_id}"
                    )

        if example.evidence_status is EvidenceStatus.PRESENT:
            gold_chunk_record_ids = {
                record_id for chunk in gold_chunks for record_id in chunk.record_ids
            }
            uncovered_records = sorted(
                set(example.required_record_ids) - gold_chunk_record_ids
            )
            if uncovered_records:
                errors.append(
                    f"{prefix} required record IDs are not covered by gold chunks: "
                    + ", ".join(uncovered_records)
                )

            gold_chunk_logical_ids = {
                logical_id for chunk in gold_chunks for logical_id in chunk.logical_fact_ids
            }
            uncovered_logical = sorted(
                set(example.required_logical_fact_ids) - gold_chunk_logical_ids
            )
            if uncovered_logical:
                errors.append(
                    f"{prefix} required logical fact IDs are not covered by gold chunks: "
                    + ", ".join(uncovered_logical)
                )

        if example.knowledge_version == "v2" and example.evidence_status is EvidenceStatus.PRESENT:
            for record_id in example.required_record_ids:
                record = records_by_id.get(record_id)
                if record is not None and record.version != "v2":
                    errors.append(
                        f"{prefix} knowledge_version=v2 requires current v2 records; "
                        f"{record.record_id} is {record.version}"
                    )
            for document_id in example.gold_document_ids:
                document = documents_by_id.get(document_id)
                if document is not None and document.version != "v2":
                    errors.append(
                        f"{prefix} knowledge_version=v2 uses non-v2 gold document "
                        f"{document.document_id}"
                    )
            for chunk in gold_chunks:
                if chunk.version != "v2":
                    errors.append(
                        f"{prefix} knowledge_version=v2 uses non-v2 gold chunk {chunk.chunk_id}"
                    )

        if example.evidence_status is EvidenceStatus.ABSENT and any(
            (
                example.required_record_ids,
                example.required_logical_fact_ids,
                example.gold_document_ids,
                example.gold_chunk_ids,
            )
        ):
            errors.append(f"{prefix} has evidence_status=ABSENT but contains evidence references")

        if example.task_family is TaskFamily.behavior_only:
            if example.evidence_status is not EvidenceStatus.NOT_APPLICABLE:
                errors.append(f"{prefix} behavior_only requires NOT_APPLICABLE evidence status")
            if example.behavior_type is None:
                errors.append(f"{prefix} behavior_only requires behavior_type")

        if example.task_family is TaskFamily.behavior_knowledge and example.behavior_type is None:
            errors.append(f"{prefix} behavior_knowledge requires behavior_type")

        if example.task_family is TaskFamily.changed_knowledge and example.knowledge_state not in {
            KnowledgeState.UNCHANGED,
            KnowledgeState.UPDATED,
            KnowledgeState.REMOVED,
        }:
            errors.append(f"{prefix} changed_knowledge has invalid lifecycle state")

        if example.task_family is TaskFamily.changed_knowledge:
            lifecycle_logical_id = example.lifecycle_logical_fact_id
            if lifecycle_logical_id is None and len(example.required_logical_fact_ids) == 1:
                lifecycle_logical_id = example.required_logical_fact_ids[0]
            if lifecycle_logical_id is None:
                errors.append(f"{prefix} changed_knowledge lacks lifecycle logical-fact identity metadata")
            elif lifecycle_logical_id not in facts_by_logical_id:
                errors.append(
                    f"{prefix} lifecycle logical fact {lifecycle_logical_id} does not exist in world"
                )
            else:
                versions = facts_by_logical_id[lifecycle_logical_id]
                v1_fact = versions.get("v1")
                v2_fact = versions.get("v2")
                if v1_fact is None:
                    errors.append(
                        f"{prefix} lifecycle logical fact {lifecycle_logical_id} lacks v1 authority"
                    )
                else:
                    actual_state = classify_knowledge_state(v1_fact, v2_fact)
                    if example.knowledge_state is not actual_state:
                        errors.append(
                            f"{prefix} knowledge_state={example.knowledge_state.value} does not match "
                            f"world lifecycle {actual_state.value} for {lifecycle_logical_id}"
                        )
                if example.required_logical_fact_ids and lifecycle_logical_id not in example.required_logical_fact_ids:
                    errors.append(
                        f"{prefix} lifecycle logical fact {lifecycle_logical_id} is inconsistent with "
                        "required_logical_fact_ids"
                    )

        if (
            example.task_family is TaskFamily.changed_knowledge
            and example.evidence_status is EvidenceStatus.PRESENT
        ):
            required_records = [
                records_by_id[record_id]
                for record_id in example.required_record_ids
                if record_id in records_by_id
            ]
            if example.knowledge_version == "v2":
                for record in required_records:
                    if record.version != "v2":
                        errors.append(
                            f"{prefix} current changed_knowledge task requires v2 records; "
                            f"{record.record_id} is {record.version}"
                        )
                for document_id in example.gold_document_ids:
                    document = documents_by_id.get(document_id)
                    if document is not None and document.version != "v2":
                        errors.append(
                            f"{prefix} current changed_knowledge task uses non-v2 gold document "
                            f"{document.document_id}"
                        )
                for chunk in gold_chunks:
                    if chunk.version != "v2":
                        errors.append(
                            f"{prefix} current changed_knowledge task uses non-v2 gold chunk "
                            f"{chunk.chunk_id}"
                        )

            if len(required_records) == 1:
                record = required_records[0]
                if example.knowledge_state is KnowledgeState.REMOVED:
                    if record.status is not FactStatus.RETIRED:
                        errors.append(
                            f"{prefix} REMOVED current-answer task lacks an authoritative "
                            "retirement record"
                        )
                    if example.expected_output != "RETIRED":
                        errors.append(
                            f"{prefix} REMOVED current-answer output is not supported by "
                            "the retirement evidence"
                        )
                elif example.expected_output != record.value:
                    errors.append(
                        f"{prefix} expected_output does not match the authoritative current "
                        f"value in {record.record_id}"
                    )

        if example.scoring_rule is None:
            errors.append(f"{prefix} is missing scoring_rule")
        else:
            can_check, derived_output = _derive_scoring_output(
                example, records_by_id, facts_by_logical_id
            )
            if can_check and example.expected_output != derived_output:
                errors.append(
                    f"{prefix} expected_output does not match scoring_rule "
                    f"{example.scoring_rule.value}: derived {derived_output!r}"
                )

        if example.generation_seed != expected_seed:
            errors.append(
                f"{prefix} uses generation_seed={example.generation_seed}; expected {expected_seed}"
            )

        component = _component_for_example(example, record_to_component, logical_to_components)
        if component in STRUCTURAL_TEST_ONLY:
            if example.split is Split.train:
                errors.append(
                    f"{prefix} leaks structural-test-only component {component} into training"
                )
            if not (
                example.split is Split.test
                and example.split_type is SplitType.structural_holdout
                and example.holdout_dimension == HOLDOUT_DIMENSION
                and example.holdout_group == component
            ):
                errors.append(
                    f"{prefix} for structural-test-only component {component} must be "
                    "test/structural_holdout with component_family holdout metadata"
                )

        if example.split_type is SplitType.structural_holdout:
            if example.split is not Split.test:
                errors.append(f"{prefix} structural_holdout examples must use split=test")
            if example.holdout_dimension != HOLDOUT_DIMENSION:
                errors.append(
                    f"{prefix} structural_holdout requires holdout_dimension={HOLDOUT_DIMENSION}"
                )
            if example.holdout_group not in STRUCTURAL_TEST_ONLY:
                errors.append(f"{prefix} has invalid structural holdout group {example.holdout_group!r}")

    statistics = {
        "record_count": len(record_ids),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "example_count": len(examples),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
    return ValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        statistics=statistics,
    )
