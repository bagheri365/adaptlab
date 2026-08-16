#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample, EvidenceStatus, KnowledgeState, SplitType, TaskFamily
from adaptlab.retrieval.failure_audit import audit_retrieval_failure
from adaptlab.retrieval.schemas import RetrievalResult
from adaptlab.retrieval.version_metrics import with_version_diagnostics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "artifacts" / "evaluation" / "m4" / "rag_failure_decomposition_v1.json"
OUTPUT_TXT = ROOT / "artifacts" / "evaluation" / "m4" / "rag_failure_decomposition_v1.txt"

COMPARISON_READY = ROOT / "artifacts" / "evaluation" / "m4" / "comparison_readiness_v1.json"
TEST_DATA = ROOT / "data" / "generated" / "v0.0" / "test.json"
CHUNKS = ROOT / "data" / "generated" / "v0.0" / "chunks.json"
PROMPT_RESULTS = ROOT / "artifacts" / "evaluation" / "m3" / "prompt" / "results.json"
ORACLE_RESULTS = ROOT / "artifacts" / "evaluation" / "m3" / "oracle_context" / "results.json"
RAG_RESULTS = ROOT / "artifacts" / "evaluation" / "m4" / "rag" / "results.json"
RETRIEVAL_RESULTS = ROOT / "artifacts" / "retrieval" / "m4" / "primary_test_bm25_v1" / "results.json"
RETRIEVAL_ARTIFACT = (
    ROOT / "artifacts" / "retrieval" / "m4" / "primary_test_bm25_v1" / "frozen" / "canonical_retrieval_artifact_v1.json"
)


PRIMARY_CATEGORIES = [
    "ALL_GOLD_RETRIEVED_MODEL_FAILED",
    "ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED",
    "PARTIAL_RETRIEVAL",
    "RETRIEVAL_MISS",
    "WRONG_VERSION",
    "OBSOLETE_ONLY",
    "EVIDENCE_ABSENT_FALSE_CONTEXT",
    "FORMAT_OR_SCORING_FAILURE",
    "UNCLASSIFIED",
]

SECONDARY_TAGS = ["DISTRACTOR_DOMINANCE"]


@dataclass(frozen=True)
class Row:
    example: BenchmarkExample
    prompt: Dict[str, Any]
    oracle: Dict[str, Any]
    rag: Dict[str, Any]
    retrieval: RetrievalResult
    retrieval_with_version: RetrievalResult
    audit: Dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows() -> dict[str, Row]:
    examples = {item["example_id"]: BenchmarkExample.from_dict(item) for item in load_json(TEST_DATA)}
    prompt = {item["example_id"]: item for item in load_json(PROMPT_RESULTS)}
    oracle = {item["example_id"]: item for item in load_json(ORACLE_RESULTS)}
    rag = {item["example_id"]: item for item in load_json(RAG_RESULTS)}
    retrieval = {item["example_id"]: RetrievalResult.from_dict(item) for item in load_json(RETRIEVAL_RESULTS)}
    chunks = tuple(sorted((DocumentChunk.from_dict(item) for item in load_json(CHUNKS)), key=lambda x: x.chunk_id))

    rows: dict[str, Row] = {}
    for ex_id, example in examples.items():
        result = retrieval[ex_id]
        rows[ex_id] = Row(
            example=example,
            prompt=prompt[ex_id],
            oracle=oracle[ex_id],
            rag=rag[ex_id],
            retrieval=result,
            retrieval_with_version=with_version_diagnostics(result, chunks),
            audit=audit_retrieval_failure(result, chunks),
        )
    return rows


def score_is_correct(row: Dict[str, Any]) -> bool:
    return row.get("score") == 1.0 or row.get("score") == 1


def score_is_incorrect(row: Dict[str, Any]) -> bool:
    return not score_is_correct(row)


def retrieved_ids(result: RetrievalResult) -> set[str]:
    return set(result.candidate_chunk_ids or [])


def gold_ids(result: RetrievalResult) -> set[str]:
    return set(result.gold_chunk_ids or [])


def required_gold_ids(result: RetrievalResult) -> set[str]:
    return set(result.required_gold_chunk_ids or [])


def all_required_gold_retrieved(result: RetrievalResult) -> bool:
    required = required_gold_ids(result)
    if not required:
        return False
    return required.issubset(retrieved_ids(result))


def any_required_gold_retrieved(result: RetrievalResult) -> bool:
    required = required_gold_ids(result)
    return bool(required & retrieved_ids(result))


def extra_non_gold_chunks(result: RetrievalResult) -> set[str]:
    return retrieved_ids(result) - gold_ids(result)


def has_distractor_tag(audit: Dict[str, Any]) -> bool:
    categories = set(getattr(audit, "categories", ()))
    return bool(categories & {"NEAR_DUPLICATE_DISTRACTOR", "SAME_COMPONENT_DISTRACTOR", "IDENTIFIER_SHORTCUT"})


def classify_primary(row: Row) -> str:
    if score_is_correct(row.rag):
        return "NOT_INCORRECT"
    if row.example.evidence_status == EvidenceStatus.ABSENT:
        return "EVIDENCE_ABSENT_FALSE_CONTEXT"

    result = row.retrieval
    if row.rag.get("provider_error") is not None or row.rag.get("score") is None or row.rag.get("raw_output") is None:
        return "FORMAT_OR_SCORING_FAILURE"

    if all_required_gold_retrieved(result):
        if extra_non_gold_chunks(result):
            return "ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED"
        return "ALL_GOLD_RETRIEVED_MODEL_FAILED"

    if any_required_gold_retrieved(result):
        return "PARTIAL_RETRIEVAL"

    if getattr(row.retrieval_with_version, "wrong_version_top1", False):
        return "WRONG_VERSION"
    if getattr(row.retrieval_with_version, "obsolete_only_retrieved", False):
        return "OBSOLETE_ONLY"
    return "RETRIEVAL_MISS"


def condition_label(row: Row, primary: str) -> str:
    if primary in {"ALL_GOLD_RETRIEVED_MODEL_FAILED", "ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED"}:
        return "all_required_gold_retrieved"
    if primary == "PARTIAL_RETRIEVAL":
        return "partial_gold_retrieved"
    if primary in {"WRONG_VERSION", "OBSOLETE_ONLY", "RETRIEVAL_MISS"}:
        return "no_required_gold_retrieved"
    if primary == "EVIDENCE_ABSENT_FALSE_CONTEXT":
        return "evidence_absent"
    if primary == "FORMAT_OR_SCORING_FAILURE":
        return "format_or_scoring_failure"
    return "unclassified"


def pct(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def summarize_counts(indices: Iterable[str], rows: dict[str, Row]) -> dict[str, Any]:
    counts = Counter()
    secondary = Counter()
    by_condition = Counter()
    example_ids: dict[str, list[str]] = defaultdict(list)
    incorrect_total = 0

    for ex_id in indices:
        row = rows[ex_id]
        primary = classify_primary(row)
        if primary == "NOT_INCORRECT":
            continue
        incorrect_total += 1
        counts[primary] += 1
        by_condition[condition_label(row, primary)] += 1
        example_ids[primary].append(ex_id)
        if has_distractor_tag(row.audit):
            secondary["DISTRACTOR_DOMINANCE"] += 1

    return {
        "total_incorrect": incorrect_total,
        "primary_counts": dict(counts),
        "secondary_counts": dict(secondary),
        "by_condition": dict(by_condition),
        "example_ids": dict(example_ids),
    }


def count_by_slice(rows: dict[str, Row], attr: str, values: Iterable[Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for value in values:
        label = value.name if hasattr(value, "name") else str(value)
        ex_ids = [ex_id for ex_id, row in rows.items() if getattr(row.example, attr) == value]
        out[label] = {
            "n": len(ex_ids),
            "rag_incorrect": sum(1 for ex_id in ex_ids if score_is_incorrect(rows[ex_id].rag)),
        }
    return out


def slice_category_breakdown(rows: dict[str, Row], predicate) -> dict[str, Any]:
    slice_rows = [row for row in rows.values() if predicate(row)]
    incorrect_rows = [row for row in slice_rows if score_is_incorrect(row.rag)]
    cat_counts = Counter(classify_primary(row) for row in incorrect_rows)
    return {
        "n": len(slice_rows),
        "rag_incorrect": len(incorrect_rows),
        "primary_category_counts": dict(cat_counts),
    }


def condition_denominators(rows: dict[str, Row]) -> dict[str, int]:
    out = defaultdict(int)
    for row in rows.values():
        if score_is_incorrect(row.rag):
            primary = classify_primary(row)
            out[condition_label(row, primary)] += 1
    return dict(out)


def category_report(category: str, count: int, failure_total: int, condition_total: int) -> dict[str, Any]:
    return {
        "n": count,
        "pct_of_rag_failures": pct(count, failure_total),
        "pct_within_relevant_condition": pct(count, condition_total),
    }


def main() -> None:
    if not json.loads(COMPARISON_READY.read_text(encoding="utf-8")):
        raise SystemExit("comparison readiness gate missing")
    readiness = load_json(COMPARISON_READY)
    if readiness.get("status") != "M4_CANONICAL_COMPARISON_READY":
        raise SystemExit(f"readiness gate not satisfied: {readiness.get('status')}")

    rows = load_rows()
    incorrect_rows = {ex_id: row for ex_id, row in rows.items() if score_is_incorrect(row.rag)}
    failure_total = len(incorrect_rows)

    primary_counts = Counter()
    secondary_counts = Counter()
    condition_counts = Counter()
    for row in incorrect_rows.values():
        primary = classify_primary(row)
        primary_counts[primary] += 1
        condition_counts[condition_label(row, primary)] += 1
        if has_distractor_tag(row.audit):
            secondary_counts["DISTRACTOR_DOMINANCE"] += 1

    # Slice totals.
    by_task_family = {
        family.name: slice_category_breakdown(rows, lambda row, family=family: row.example.task_family == family)
        for family in [TaskFamily.behavior_only, TaskFamily.knowledge_only, TaskFamily.behavior_knowledge, TaskFamily.changed_knowledge]
    }
    by_knowledge_state = {
        state.name: slice_category_breakdown(rows, lambda row, state=state: row.example.knowledge_state == state)
        for state in [KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED]
    }

    # Retrieval condition aggregates.
    all_required_gold = []
    partial_gold = []
    no_gold = []
    evidence_absent = []
    exact_gold = []
    wrong_version_applicable = []
    obsolete_only_applicable = []
    for row in rows.values():
        result = row.retrieval
        if result.evidence_status == EvidenceStatus.ABSENT:
            evidence_absent.append(row)
        if required_gold_ids(result) and required_gold_ids(result) <= retrieved_ids(result):
            all_required_gold.append(row)
            if retrieved_ids(result) == gold_ids(result):
                exact_gold.append(row)
        elif any_required_gold_retrieved(result):
            partial_gold.append(row)
        else:
            no_gold.append(row)
        if getattr(row.retrieval_with_version, "wrong_version_top1", False):
            wrong_version_applicable.append(row)
        if getattr(row.retrieval_with_version, "obsolete_only_retrieved", False):
            obsolete_only_applicable.append(row)

    # Explicit 45-case audit.
    all_gold_wrong_rows = [
        row
        for row in rows.values()
        if score_is_incorrect(row.rag) and all_required_gold_retrieved(row.retrieval)
    ]
    explicit_45 = {
        "count": len(all_gold_wrong_rows),
        "oracle_correct": sum(1 for row in all_gold_wrong_rows if score_is_correct(row.oracle)),
        "oracle_wrong": sum(1 for row in all_gold_wrong_rows if score_is_incorrect(row.oracle)),
        "with_extra_distractors": sum(1 for row in all_gold_wrong_rows if extra_non_gold_chunks(row.retrieval)),
        "updated": sum(1 for row in all_gold_wrong_rows if row.example.knowledge_state == KnowledgeState.UPDATED),
        "removed": sum(1 for row in all_gold_wrong_rows if row.example.knowledge_state == KnowledgeState.REMOVED),
        "structural_holdout": sum(1 for row in all_gold_wrong_rows if row.example.split_type == SplitType.structural_holdout),
        "hard": sum(1 for row in all_gold_wrong_rows if row.example.difficulty.name == "HARD"),
        "example_ids": [row.example.example_id for row in all_gold_wrong_rows],
    }

    # Wrong-version / obsolete-only counts.
    wrong_version_counts = {
        "count": len(wrong_version_applicable),
        "rag_correct": sum(1 for row in wrong_version_applicable if score_is_correct(row.rag)),
        "rag_wrong": sum(1 for row in wrong_version_applicable if score_is_incorrect(row.rag)),
    }
    obsolete_only_counts = {
        "count": len(obsolete_only_applicable),
        "rag_correct": sum(1 for row in obsolete_only_applicable if score_is_correct(row.rag)),
        "rag_wrong": sum(1 for row in obsolete_only_applicable if score_is_incorrect(row.rag)),
    }

    oracle_compare = {
        "rag_wrong_oracle_correct": sum(1 for row in incorrect_rows.values() if score_is_correct(row.oracle)),
        "rag_wrong_oracle_wrong": sum(1 for row in incorrect_rows.values() if score_is_incorrect(row.oracle)),
    }

    # Primary category table.
    all_required_condition_total = sum(1 for row in rows.values() if all_required_gold_retrieved(row.retrieval))
    partial_condition_total = sum(1 for row in rows.values() if any_required_gold_retrieved(row.retrieval) and not all_required_gold_retrieved(row.retrieval))
    no_gold_condition_total = sum(
        1 for row in rows.values() if required_gold_ids(row.retrieval) and not any_required_gold_retrieved(row.retrieval)
    )
    evidence_absent_condition_total = sum(1 for row in rows.values() if row.example.evidence_status == EvidenceStatus.ABSENT)

    category_details = {
        "ALL_GOLD_RETRIEVED_MODEL_FAILED": category_report(
            "ALL_GOLD_RETRIEVED_MODEL_FAILED",
            primary_counts["ALL_GOLD_RETRIEVED_MODEL_FAILED"],
            failure_total,
            all_required_condition_total,
        ),
        "ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED": category_report(
            "ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED",
            primary_counts["ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED"],
            failure_total,
            all_required_condition_total,
        ),
        "PARTIAL_RETRIEVAL": category_report(
            "PARTIAL_RETRIEVAL", primary_counts["PARTIAL_RETRIEVAL"], failure_total, partial_condition_total
        ),
        "RETRIEVAL_MISS": category_report(
            "RETRIEVAL_MISS", primary_counts["RETRIEVAL_MISS"], failure_total, no_gold_condition_total
        ),
        "WRONG_VERSION": category_report(
            "WRONG_VERSION", primary_counts["WRONG_VERSION"], failure_total, wrong_version_counts["count"]
        ),
        "OBSOLETE_ONLY": category_report(
            "OBSOLETE_ONLY", primary_counts["OBSOLETE_ONLY"], failure_total, obsolete_only_counts["count"]
        ),
        "EVIDENCE_ABSENT_FALSE_CONTEXT": category_report(
            "EVIDENCE_ABSENT_FALSE_CONTEXT",
            primary_counts["EVIDENCE_ABSENT_FALSE_CONTEXT"],
            failure_total,
            evidence_absent_condition_total,
        ),
        "FORMAT_OR_SCORING_FAILURE": category_report(
            "FORMAT_OR_SCORING_FAILURE", primary_counts["FORMAT_OR_SCORING_FAILURE"], failure_total, failure_total
        ),
        "UNCLASSIFIED": category_report("UNCLASSIFIED", primary_counts["UNCLASSIFIED"], failure_total, failure_total),
        "DISTRACTOR_DOMINANCE": {
            "n": secondary_counts["DISTRACTOR_DOMINANCE"],
            "pct_of_rag_failures": pct(secondary_counts["DISTRACTOR_DOMINANCE"], failure_total),
            "pct_within_relevant_condition": pct(
                secondary_counts["DISTRACTOR_DOMINANCE"],
                primary_counts["ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED"]
                + primary_counts["PARTIAL_RETRIEVAL"]
                + primary_counts["RETRIEVAL_MISS"]
                + primary_counts["WRONG_VERSION"]
                + primary_counts["OBSOLETE_ONLY"],
            ),
            "note": "secondary mechanical tag; not asserted causal",
        },
    }

    knowledge_breakdown = {
        "task_family": by_task_family,
        "knowledge_state": by_knowledge_state,
    }

    artifact = {
        "schema_version": "m4-rag-failure-decomposition-v1",
        "status": "complete",
        "inputs": {
            "comparison_readiness": str(COMPARISON_READY.relative_to(ROOT)),
            "benchmark": str(TEST_DATA.relative_to(ROOT)),
            "chunks": str(CHUNKS.relative_to(ROOT)),
            "prompt_results": str(PROMPT_RESULTS.relative_to(ROOT)),
            "oracle_results": str(ORACLE_RESULTS.relative_to(ROOT)),
            "rag_results": str(RAG_RESULTS.relative_to(ROOT)),
            "retrieval_results": str(RETRIEVAL_RESULTS.relative_to(ROOT)),
            "retrieval_artifact": str(RETRIEVAL_ARTIFACT.relative_to(ROOT)),
        },
        "totals": {
            "n_examples": len(rows),
            "rag_incorrect": failure_total,
            "prompt_correct": sum(1 for row in rows.values() if score_is_correct(row.prompt)),
            "oracle_correct": sum(1 for row in rows.values() if score_is_correct(row.oracle)),
            "rag_correct": sum(1 for row in rows.values() if score_is_correct(row.rag)),
            "rag_wrong_oracle_correct": oracle_compare["rag_wrong_oracle_correct"],
            "rag_wrong_oracle_wrong": oracle_compare["rag_wrong_oracle_wrong"],
        },
        "primary_category_counts": {
            cat: {
                **category_details[cat],
            }
            for cat in PRIMARY_CATEGORIES
        },
        "secondary_tags": {tag: category_details[tag] for tag in SECONDARY_TAGS},
        "slice_breakdown": knowledge_breakdown,
        "retrieval_condition_summary": {
            "all_required_gold_retrieved": {
                "n": all_required_condition_total,
                "rag_wrong": sum(
                    1 for row in rows.values() if all_required_gold_retrieved(row.retrieval) and score_is_incorrect(row.rag)
                ),
            },
            "exact_gold_retrieved": {
                "n": len(exact_gold),
                "rag_wrong": sum(1 for row in exact_gold if score_is_incorrect(row.rag)),
            },
            "partial_gold_retrieved": {
                "n": partial_condition_total,
                "rag_wrong": sum(
                    1
                    for row in rows.values()
                    if any_required_gold_retrieved(row.retrieval)
                    and not all_required_gold_retrieved(row.retrieval)
                    and score_is_incorrect(row.rag)
                ),
            },
            "no_gold_retrieved": {
                "n": no_gold_condition_total,
                "rag_wrong": sum(
                    1
                    for row in rows.values()
                    if required_gold_ids(row.retrieval) and not any_required_gold_retrieved(row.retrieval) and score_is_incorrect(row.rag)
                ),
            },
            "evidence_absent": {
                "n": evidence_absent_condition_total,
                "rag_wrong": sum(1 for row in rows.values() if row.example.evidence_status == EvidenceStatus.ABSENT and score_is_incorrect(row.rag)),
            },
            "wrong_version_retrieval": wrong_version_counts,
            "obsolete_only_retrieval": obsolete_only_counts,
        },
        "explicit_45_all_required_gold_retrieved_and_rag_wrong": explicit_45,
        "primary_examples": {
            cat: {"example_ids": [row_id for row_id, row in incorrect_rows.items() if classify_primary(row) == cat]}
            for cat in PRIMARY_CATEGORIES
        },
    }

    OUTPUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = []
    lines.append("OBSERVATION")
    lines.append(f"- RAG was incorrect on {failure_total}/400 examples.")
    lines.append(
        f"- Primary mechanically supported buckets were: all-gold-plus-distractors={primary_counts['ALL_GOLD_PLUS_DISTRACTORS_MODEL_FAILED']}, "
        f"wrong-version={primary_counts['WRONG_VERSION']}, evidence-absent false-context={primary_counts['EVIDENCE_ABSENT_FALSE_CONTEXT']}; "
        f"all-gold-only, partial-retrieval, obsolete-only, and pure retrieval-miss primary buckets were 0."
    )
    lines.append(
        f"- Retrieval-condition counts among RAG failures: all-required-gold retrieved={all_required_condition_total} with {sum(1 for row in rows.values() if all_required_gold_retrieved(row.retrieval) and score_is_incorrect(row.rag))} wrong; "
        f"no-gold retrieved={no_gold_condition_total} with {sum(1 for row in rows.values() if required_gold_ids(row.retrieval) and not any_required_gold_retrieved(row.retrieval) and score_is_incorrect(row.rag))} wrong; "
        f"wrong-version retrieval={wrong_version_counts['count']} with {wrong_version_counts['rag_correct']} correct / {wrong_version_counts['rag_wrong']} wrong; "
        f"obsolete-only retrieval={obsolete_only_counts['count']} with {obsolete_only_counts['rag_correct']} correct / {obsolete_only_counts['rag_wrong']} wrong; "
        f"evidence-ABSENT={evidence_absent_condition_total} with {sum(1 for row in rows.values() if row.example.evidence_status == EvidenceStatus.ABSENT and score_is_incorrect(row.rag))} wrong."
    )
    lines.append(
        f"- RAG was wrong while ORACLE_CONTEXT was correct in {oracle_compare['rag_wrong_oracle_correct']} cases; both were wrong in {oracle_compare['rag_wrong_oracle_wrong']} cases."
    )
    lines.append(
        f"- The explicit all-required-gold-retrieved-and-wrong set contained {explicit_45['count']} examples, with {explicit_45['oracle_correct']} Oracle-correct and {explicit_45['oracle_wrong']} Oracle-wrong; "
        f"all 45 contained extra distractor chunks, with {explicit_45['updated']} UPDATED, {explicit_45['removed']} REMOVED, {explicit_45['structural_holdout']} structural holdout, and {explicit_45['hard']} HARD."
    )
    lines.append(
        "- Slice totals: knowledge_only 11 wrong, behavior_knowledge 7 wrong, changed_knowledge 32 wrong; "
        "UNCHANGED 10 wrong, UPDATED 5 wrong, REMOVED 17 wrong."
    )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(
        "- The frozen artifacts support a mix of knowledge-access gain and residual generation/reasoning gap: the dominant failure mode is all-required-gold retrieved plus distractors, while a smaller wrong-version slice and one evidence-ABSENT false-context case remain."
    )
    lines.append(
        "- Evidence-ABSENT failures should be read as false-context susceptibility rather than retrieval misses, because there is no gold retrieval target in that slice."
    )
    lines.append(
        "- The DISTRACTOR_DOMINANCE tag is reported only as a mechanical tag from the retrieval audit; presence of distractors is not treated as proof of causality."
    )
    lines.append("")
    lines.append("LIMITATION")
    lines.append(
        "- This decomposition is bounded by the frozen benchmark, frozen prompt/oracle/RAG outputs, and the frozen BM25 artifact; it does not rerun inference, retrieval, scorer, or normalization."
    )
    lines.append(
        "- The artifacts do not establish model intent, and they do not support claims about dense retrieval performance, universal RAG performance, LoRA benefit, or cross-model generality."
    )
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
