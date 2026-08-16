import hashlib
import json
from pathlib import Path

import pytest

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.evaluation.inputs import (
    canonical_model_input_bytes,
    construct_model_input,
    construct_rag_model_input,
    render_selected_evidence,
)
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.retrieval.frozen_artifact import (
    FrozenRetrievalArtifact,
    load_and_verify_frozen_retrieval_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data/generated/v0.0"
PROMPT_PATH = ROOT / "configs/prompts/prompt_v1.yaml"
FROZEN_RETRIEVAL = (
    ROOT
    / "artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json"
)


def _examples() -> list[BenchmarkExample]:
    return [
        BenchmarkExample.from_dict(item)
        for item in json.loads((BENCHMARK_DIR / "test.json").read_text(encoding="utf-8"))
    ]


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk.from_dict(item)
        for item in json.loads((BENCHMARK_DIR / "chunks.json").read_text(encoding="utf-8"))
    ]


def _artifact():
    return load_and_verify_frozen_retrieval_artifact(FROZEN_RETRIEVAL)


def _example_where(predicate):
    return next(example for example in _examples() if predicate(example))


def test_rag_eligible_input_uses_exact_frozen_chunk_sequence_and_shared_renderer() -> None:
    artifact = _artifact()
    entries = {entry.example_id: entry for entry in artifact.entries}
    example = _example_where(
        lambda e: e.task_family is TaskFamily.knowledge_only
        and entries[e.example_id].retrieval_eligible
        and bool(entries[e.example_id].chunk_ids)
    )
    entry = entries[example.example_id]
    chunks = _chunks()
    prompt = load_prompt_contract(PROMPT_PATH)

    result = construct_rag_model_input(
        example=example,
        prompt_contract=prompt,
        chunks=list(reversed(chunks)),
        retrieval_artifact=artifact,
    )

    expected_context = render_selected_evidence(entry.chunk_ids, chunks)
    assert result.model_input.system == prompt.system_prompt
    assert result.model_input.user == f"{expected_context}\n\n{example.question}"
    assert result.evidence_chunk_ids == entry.chunk_ids
    assert result.retrieval_run_id == artifact.retrieval_run_id
    assert result.retrieval_artifact_hash == artifact.retrieval_artifact_hash
    assert result.retrieved_context_hash == hashlib.sha256(expected_context.encode("utf-8")).hexdigest()
    assert result.input_hash == hashlib.sha256(canonical_model_input_bytes(result.model_input)).hexdigest()


def test_behavior_only_rag_is_byte_identical_to_prompt_and_oracle() -> None:
    artifact = _artifact()
    example = _example_where(lambda e: e.task_family is TaskFamily.behavior_only)
    prompt_contract = load_prompt_contract(PROMPT_PATH)
    chunks = _chunks()

    prompt = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=prompt_contract,
    )
    oracle = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=prompt_contract,
        chunks=chunks,
    )
    rag = construct_rag_model_input(
        example=example,
        prompt_contract=prompt_contract,
        chunks=chunks,
        retrieval_artifact=artifact,
    )

    assert rag.model_input == prompt.model_input == oracle.model_input
    assert rag.model_input_bytes() == prompt.model_input_bytes() == oracle.model_input_bytes()
    assert rag.input_hash == prompt.input_hash == oracle.input_hash
    assert rag.evidence_chunk_ids == ()
    assert rag.retrieved_context_hash == hashlib.sha256(b"").hexdigest()
    entry = next(entry for entry in artifact.entries if entry.example_id == example.example_id)
    assert entry.retrieval_eligible is False
    assert entry.chunk_ids == ()


def test_evidence_absent_rag_injects_frozen_bm25_context_without_hidden_absent_label() -> None:
    artifact = _artifact()
    entries = {entry.example_id: entry for entry in artifact.entries}
    example = _example_where(
        lambda e: e.evidence_status is EvidenceStatus.ABSENT
        and entries[e.example_id].retrieval_eligible
        and bool(entries[e.example_id].chunk_ids)
    )
    entry = entries[example.example_id]
    chunks = _chunks()
    prompt = load_prompt_contract(PROMPT_PATH)

    result = construct_rag_model_input(
        example=example,
        prompt_contract=prompt,
        chunks=chunks,
        retrieval_artifact=artifact,
    )

    assert result.evidence_chunk_ids == entry.chunk_ids
    assert result.model_input.user.startswith("--- BEGIN CONTEXT ---\n")
    # No special label is injected to tell the model that benchmark gold is absent.
    assert "evidence_status" not in result.model_input.user
    assert "EVIDENCE ABSENT" not in result.model_input.user
    assert result.model_input.user.endswith("\n\n" + example.question)


def test_rag_rejects_artifact_eligibility_mismatch() -> None:
    artifact = _artifact()
    example = _example_where(lambda e: e.task_family is TaskFamily.behavior_only)
    data = artifact.to_dict()
    entry = next(item for item in data["entries"] if item["example_id"] == example.example_id)
    entry["retrieval_eligible"] = True
    # Recompute the artifact hash so this is a structurally valid but semantically wrong frozen artifact.
    data.pop("retrieval_artifact_hash")
    from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
    data["retrieval_artifact_hash"] = sha256_bytes(canonical_json_bytes(data))
    bad = FrozenRetrievalArtifact.from_dict(data)

    with pytest.raises(ValueError, match="eligibility does not match"):
        construct_rag_model_input(
            example=example,
            prompt_contract=load_prompt_contract(PROMPT_PATH),
            chunks=_chunks(),
            retrieval_artifact=bad,
        )


def test_rag_requires_frozen_entry_for_example() -> None:
    artifact = _artifact()
    example = _examples()[0]
    data = artifact.to_dict()
    data["entries"] = [item for item in data["entries"] if item["example_id"] != example.example_id]
    data.pop("retrieval_artifact_hash")
    from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
    data["retrieval_artifact_hash"] = sha256_bytes(canonical_json_bytes(data))
    shortened = FrozenRetrievalArtifact.from_dict(data)

    with pytest.raises(ValueError, match="has no entry"):
        construct_rag_model_input(
            example=example,
            prompt_contract=load_prompt_contract(PROMPT_PATH),
            chunks=_chunks(),
            retrieval_artifact=shortened,
        )
