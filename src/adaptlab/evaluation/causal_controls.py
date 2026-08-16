"""Mechanical causal-control validation for PROMPT, ORACLE_CONTEXT, and RAG."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import yaml

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.evaluation.inputs import (
    EVIDENCE_FORMAT_VERSION,
    construct_model_input,
    construct_rag_model_input,
    render_selected_evidence,
)
from adaptlab.evaluation.prompts import PromptContract
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.retrieval.frozen_artifact import FrozenRetrievalArtifact

CAUSAL_CONTROL_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class CausalControlReport:
    schema_version: str
    passed: bool
    example_count: int
    condition_checks: dict[str, bool]
    per_example_checks: tuple[dict[str, Any], ...]
    shared_condition_values: dict[str, Any]
    retrieval_run_id: str
    retrieval_artifact_hash: str
    evidence_renderer_version: str = EVIDENCE_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["per_example_checks"] = list(self.per_example_checks)
        return data

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def artifact_hash(self) -> str:
        return sha256_bytes(self.to_json_bytes())


class CausalControlValidationError(ValueError):
    """Raised when a causal control fails and performance analysis must stop."""


def load_condition_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evaluation condition config must be a mapping")
    return raw


def derive_rag_control_condition(prompt_condition: Mapping[str, Any]) -> dict[str, Any]:
    """Build the pre-freeze RAG control signature from the frozen PROMPT condition.

    Prompt 20 later freezes the canonical RAG condition. Prompt 17 only needs to
    establish that the intended RAG condition shares all non-retrieval causal
    controls, so this derivation changes adaptation identity and nothing else.
    """
    rag = deepcopy(dict(prompt_condition))
    rag["condition_id"] = "milestone4_rag_control_v1"
    rag["canonical_milestone"] = "4"
    rag["adaptation_method"] = "RAG"
    return rag


def _condition_signature(condition: Mapping[str, Any]) -> dict[str, Any]:
    provider = condition["provider"]
    request = condition["request"]
    scoring = condition["scoring"]
    benchmark = condition["benchmark"]
    prompt = condition["prompt"]
    return {
        "model": provider["model_tag"],
        "model_runtime_policy": {
            "provider": provider["name"],
            "base_url_policy": provider.get("base_url_policy"),
            "ollama_version": provider.get("ollama_version"),
            "ollama_model_digest": provider.get("ollama_model_digest"),
            "retry_policy": condition.get("retry_policy"),
        },
        "prompt_version": prompt["prompt_version"],
        "prompt_hash": prompt["prompt_hash"],
        "temperature": request["temperature"],
        "seed": request.get("seed"),
        "context_length": request.get("context_length"),
        "max_tokens": request["max_tokens"],
        "think": request.get("think"),
        "stream": request.get("stream"),
        "scorer_version": scoring["scorer_version"],
        "normalizer_version": scoring["normalizer_version"],
        "benchmark_manifest_hash": benchmark["benchmark_manifest_hash"],
    }


def validate_causal_controls(
    *,
    examples: Iterable[BenchmarkExample],
    chunks: Iterable[DocumentChunk] | Mapping[str, DocumentChunk],
    prompt_contract: PromptContract,
    retrieval_artifact: FrozenRetrievalArtifact,
    prompt_condition: Mapping[str, Any],
    oracle_condition: Mapping[str, Any],
    rag_condition: Mapping[str, Any],
) -> CausalControlReport:
    """Validate all Milestone 4 Prompt/Oracle/RAG causal controls mechanically."""
    signatures = {
        "PROMPT": _condition_signature(prompt_condition),
        "ORACLE_CONTEXT": _condition_signature(oracle_condition),
        "RAG": _condition_signature(rag_condition),
    }
    baseline = signatures["PROMPT"]
    condition_checks = {
        field: all(signature[field] == baseline[field] for signature in signatures.values())
        for field in baseline
    }
    condition_checks["adaptation_methods"] = (
        prompt_condition.get("adaptation_method") == "PROMPT"
        and oracle_condition.get("adaptation_method") == "ORACLE_CONTEXT"
        and rag_condition.get("adaptation_method") == "RAG"
    )
    condition_checks["prompt_contract_hash_matches"] = baseline["prompt_hash"] == prompt_contract.prompt_hash

    chunk_seq = tuple(chunks.values()) if isinstance(chunks, Mapping) else tuple(chunks)
    frozen_entries = {entry.example_id: entry for entry in retrieval_artifact.entries}
    rows: list[dict[str, Any]] = []
    for example in sorted(tuple(examples), key=lambda item: item.example_id):
        prompt_input = construct_model_input(
            example=example, method=AdaptationMethod.PROMPT, prompt_contract=prompt_contract
        )
        oracle_input = construct_model_input(
            example=example,
            method=AdaptationMethod.ORACLE_CONTEXT,
            prompt_contract=prompt_contract,
            chunks=chunk_seq,
        )
        rag_input = construct_rag_model_input(
            example=example,
            prompt_contract=prompt_contract,
            chunks=chunk_seq,
            retrieval_artifact=retrieval_artifact,
        )
        entry = frozen_entries.get(example.example_id)
        rag_matches_frozen = entry is not None and rag_input.evidence_chunk_ids == entry.chunk_ids
        oracle_matches_gold = (
            example.evidence_status is not EvidenceStatus.PRESENT
            or oracle_input.evidence_chunk_ids == tuple(sorted(example.gold_chunk_ids))
        )
        behavior_byte_equal = True
        if example.task_family is TaskFamily.behavior_only:
            behavior_byte_equal = (
                prompt_input.model_input_bytes()
                == oracle_input.model_input_bytes()
                == rag_input.model_input_bytes()
            )

        oracle_renderer_exact = True
        if oracle_input.evidence_chunk_ids:
            expected = f"{render_selected_evidence(oracle_input.evidence_chunk_ids, chunk_seq)}\n\n{example.question}"
            oracle_renderer_exact = oracle_input.model_input.user == expected
        else:
            oracle_renderer_exact = oracle_input.model_input.user == example.question

        rag_renderer_exact = True
        if rag_input.evidence_chunk_ids:
            expected = f"{render_selected_evidence(rag_input.evidence_chunk_ids, chunk_seq)}\n\n{example.question}"
            rag_renderer_exact = rag_input.model_input.user == expected
        else:
            rag_renderer_exact = rag_input.model_input.user == example.question

        shared_renderer = (
            oracle_input.evidence_format_version
            == rag_input.evidence_format_version
            == EVIDENCE_FORMAT_VERSION
            and oracle_renderer_exact
            and rag_renderer_exact
            and oracle_input.model_input.system == rag_input.model_input.system == prompt_contract.system_prompt
        )
        row_passed = all((behavior_byte_equal, rag_matches_frozen, oracle_matches_gold, shared_renderer))
        rows.append(
            {
                "example_id": example.example_id,
                "task_family": example.task_family.value,
                "evidence_status": example.evidence_status.value,
                "behavior_only_byte_identical": behavior_byte_equal,
                "rag_chunks_match_frozen_artifact": rag_matches_frozen,
                "oracle_chunks_match_permitted_gold": oracle_matches_gold,
                "oracle_rag_shared_renderer": shared_renderer,
                "passed": row_passed,
            }
        )

    passed = all(condition_checks.values()) and all(row["passed"] for row in rows)
    return CausalControlReport(
        schema_version=CAUSAL_CONTROL_SCHEMA_VERSION,
        passed=passed,
        example_count=len(rows),
        condition_checks=condition_checks,
        per_example_checks=tuple(rows),
        shared_condition_values=baseline,
        retrieval_run_id=retrieval_artifact.retrieval_run_id,
        retrieval_artifact_hash=retrieval_artifact.retrieval_artifact_hash,
    )


def require_causal_controls(report: CausalControlReport) -> None:
    if not report.passed:
        failed_conditions = sorted(key for key, value in report.condition_checks.items() if not value)
        failed_examples = [row["example_id"] for row in report.per_example_checks if not row["passed"]]
        raise CausalControlValidationError(
            "causal-control validation failed; stop before performance analysis: "
            f"conditions={failed_conditions}, examples={failed_examples[:10]}"
        )


def write_causal_control_artifact(report: CausalControlReport, path: str | Path) -> None:
    require_causal_controls(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(report.to_json_bytes())
