"""Milestone 4 Prompt/RAG/Oracle comparison with strict completeness gates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample

RAG_COMPARISON_SCHEMA_VERSION = "m4-rag-comparison-v1"
CONDITION_NAMES = ("PROMPT", "RAG", "ORACLE_CONTEXT")


class RAGComparisonBlockedError(ValueError):
    """Raised when canonical-condition artifacts are not analysis-ready."""


def _index_rows(rows: Iterable[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{condition} row missing example_id")
        if example_id in indexed:
            raise ValueError(f"{condition} contains duplicate example_id {example_id}")
        indexed[example_id] = row
    return indexed


def _require_complete_success(indexed: dict[str, dict[str, Any]], expected_ids: set[str], condition: str) -> None:
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - set(indexed))
        extra = sorted(set(indexed) - expected_ids)
        raise RAGComparisonBlockedError(
            f"{condition} is not the complete canonical 400-example set: missing={len(missing)} extra={len(extra)}"
        )
    failed = [eid for eid, row in indexed.items() if row.get("provider_error") is not None or row.get("score") is None]
    if failed:
        raise RAGComparisonBlockedError(
            f"{condition} has {len(failed)} unsuccessful model responses; performance analysis is blocked"
        )


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(float(r["score"]) for r in rows) / len(rows) if rows else 0.0


def _slice_specs(example: BenchmarkExample) -> tuple[tuple[str, str], ...]:
    return (
        ("overall", "overall"),
        ("task_family", example.task_family.value),
        ("difficulty", example.difficulty.value),
        ("split_type", example.split_type.value.upper()),
        ("knowledge_state", example.knowledge_state.value),
        ("evidence_status", example.evidence_status.value),
    )


def analyze_prompt_rag_oracle(*, examples: Iterable[BenchmarkExample], prompt_rows: Iterable[dict[str, Any]],
                              rag_rows: Iterable[dict[str, Any]], oracle_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare complete frozen canonical conditions; fail closed on any incomplete run."""
    examples = tuple(sorted(examples, key=lambda e: e.example_id))
    if len(examples) != 400:
        raise RAGComparisonBlockedError(f"canonical comparison requires 400 examples, found {len(examples)}")
    expected_ids = {e.example_id for e in examples}
    by_condition = {
        "PROMPT": _index_rows(prompt_rows, "PROMPT"),
        "RAG": _index_rows(rag_rows, "RAG"),
        "ORACLE_CONTEXT": _index_rows(oracle_rows, "ORACLE_CONTEXT"),
    }
    for name, rows in by_condition.items():
        _require_complete_success(rows, expected_ids, name)

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    behavior_disagreements: list[dict[str, Any]] = []
    for example in examples:
        for spec in _slice_specs(example):
            buckets[spec].append(example.example_id)
        if example.task_family.value == "behavior_only":
            p, r, o = (by_condition[name][example.example_id] for name in CONDITION_NAMES)
            hashes = {p.get("input_hash"), r.get("input_hash"), o.get("input_hash")}
            if len(hashes) != 1:
                raise RAGComparisonBlockedError(
                    f"behavior_only input mismatch for {example.example_id}; causal control failed"
                )
            outputs = {json.dumps(p.get("normalized_output"), sort_keys=True),
                       json.dumps(r.get("normalized_output"), sort_keys=True),
                       json.dumps(o.get("normalized_output"), sort_keys=True)}
            if len(outputs) != 1:
                behavior_disagreements.append({
                    "example_id": example.example_id,
                    "prompt": p.get("normalized_output"),
                    "rag": r.get("normalized_output"),
                    "oracle_context": o.get("normalized_output"),
                })

    slices: list[dict[str, Any]] = []
    for (dimension, value), ids in sorted(buckets.items()):
        condition_rows = {name: [by_condition[name][eid] for eid in ids] for name in CONDITION_NAMES}
        p = _accuracy(condition_rows["PROMPT"])
        r = _accuracy(condition_rows["RAG"])
        o = _accuracy(condition_rows["ORACLE_CONTEXT"])
        slices.append({
            "dimension": dimension,
            "value": value,
            "n": len(ids),
            "prompt_accuracy": p,
            "rag_accuracy": r,
            "gold_evidence_accuracy": o,
            "rag_minus_prompt": r - p,
            "gold_evidence_minus_rag": o - r,
        })

    report = {
        "schema_version": RAG_COMPARISON_SCHEMA_VERSION,
        "status": "COMPLETE",
        "terminology": {
            "rag_minus_prompt": "knowledge-access gain",
            "gold_evidence_minus_rag": "remaining retrieval gap / remaining generation-reasoning gap; decomposition requires retrieval-conditioned analysis",
        },
        "slices": slices,
        "behavior_only": {
            "byte_identical_inputs_verified": True,
            "output_disagreement_count": len(behavior_disagreements),
            "output_disagreements": behavior_disagreements,
        },
        "sections": {
            "OBSERVATION": "Frozen-condition accuracies and paired differences are reported without retuning.",
            "INTERPRETATION": "RAG-PROMPT is described as knowledge-access gain; Oracle-RAG is a remaining gap, not automatically a pure retrieval effect.",
            "LIMITATION": "This comparison alone cannot uniquely separate retrieval from generation/reasoning failures and makes no LoRA claim.",
        },
    }
    report["artifact_hash"] = sha256_bytes(canonical_json_bytes(report))
    return report


def write_blocked_comparison_artifact(*, output_dir: Path, reasons: list[str], rag_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": RAG_COMPARISON_SCHEMA_VERSION,
        "status": "BLOCKED",
        "reasons": list(reasons),
        "rag_execution_integrity": rag_summary,
        "sections": {
            "OBSERVATION": "Canonical three-condition performance analysis was not run because required canonical model outputs are incomplete.",
            "INTERPRETATION": "No RAG accuracy or gain can be inferred from provider-failure rows.",
            "LIMITATION": "The missing successful qwen3:8b responses prevent the requested Prompt/RAG/Oracle comparison.",
        },
    }
    report["artifact_hash"] = sha256_bytes(canonical_json_bytes(report))
    write_json(output_dir / "analysis_blocked.json", report)
    text = "\n".join([
        "OBSERVATION", report["sections"]["OBSERVATION"], "",
        "INTERPRETATION", report["sections"]["INTERPRETATION"], "",
        "LIMITATION", report["sections"]["LIMITATION"], "",
        "Reasons:", *[f"- {r}" for r in reasons],
    ]) + "\n"
    (output_dir / "analysis_blocked.txt").write_text(text, encoding="utf-8")
    return report
