from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.evaluation.inputs import (
    EVIDENCE_FORMAT_VERSION,
    canonical_model_input_bytes,
    construct_model_input,
    _oracle_evidence_spans,
    _validate_oracle_evidence_spans,
)
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.schemas import AdaptationMethod

PROMPT_PATH = Path("configs/prompts/prompt_v1.yaml")
BENCHMARK_DIR = Path("data/generated/v0.0")


def _examples() -> list[BenchmarkExample]:
    raw = json.loads((BENCHMARK_DIR / "test.json").read_text(encoding="utf-8"))
    return [BenchmarkExample.from_dict(item) for item in raw]


def _chunks() -> list[DocumentChunk]:
    raw = json.loads((BENCHMARK_DIR / "chunks.json").read_text(encoding="utf-8"))
    return [DocumentChunk.from_dict(item) for item in raw]


def _example(*, evidence_status: EvidenceStatus, behavior_only: bool = False) -> BenchmarkExample:
    for example in _examples():
        if example.evidence_status is not evidence_status:
            continue
        if behavior_only and example.task_family is not TaskFamily.behavior_only:
            continue
        return example
    raise AssertionError("matching generated example not found")


def test_prompt_is_frozen_system_plus_question_only() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)

    result = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=contract,
    )

    assert result.model_input.system == contract.system_prompt
    assert result.model_input.user == example.question
    assert result.evidence_chunk_ids == ()
    assert result.evidence_format_version == EVIDENCE_FORMAT_VERSION


def test_oracle_present_injects_only_sorted_chunk_contents_and_question() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)
    chunk_index = {chunk.chunk_id: chunk for chunk in _chunks()}

    result = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=list(reversed(_chunks())),
    )

    expected_ids = tuple(sorted(example.gold_chunk_ids))
    assert result.evidence_chunk_ids == expected_ids
    assert result.model_input.system == contract.system_prompt
    assert result.model_input.user.endswith("\n\n" + example.question)
    positions = [result.model_input.user.index(chunk_index[cid].content) for cid in expected_ids]
    assert positions == sorted(positions)

    visible = result.model_input.user.lower()
    forbidden = (
        "gold",
        "correct answer",
        "authoritative answer",
        "expected_output",
        "knowledge_state",
        "task_family",
        "difficulty",
        "scoring_rule",
    )
    for token in forbidden:
        assert token not in visible
    for chunk_id in expected_ids:
        assert chunk_id not in result.model_input.user


def test_oracle_present_accepts_source_chunk_text_even_when_it_contains_the_correct_value() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)
    result = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=_chunks(),
    )

    assert example.expected_output in result.model_input.user
    assert result.evidence_chunk_ids == tuple(sorted(example.gold_chunk_ids))
    assert len(result.evidence_chunk_hashes) == len(result.evidence_chunk_ids)
    assert result.evidence_chunk_hashes == tuple(
        hashlib.sha256(next(chunk for chunk in _chunks() if chunk.chunk_id == chunk_id).content.encode("utf-8")).hexdigest()
        for chunk_id in result.evidence_chunk_ids
    )
    assert result.input_hash == hashlib.sha256(canonical_model_input_bytes(result.model_input)).hexdigest()


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda text, expected: f"{text}\nexpected_output: {expected}", "byte-for-byte"),
        (lambda text, expected: f"{text}\nanswer summary: {expected}", "byte-for-byte"),
        (lambda text, expected: f"{text}\nscoring_rule: CLASSIFICATION", "byte-for-byte"),
    ],
)
def test_oracle_evidence_provenance_rejects_answer_leakage_helpers(mutator, match) -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    chunks = _chunks()
    spans = _oracle_evidence_spans(example, chunks)
    bad_spans = tuple(
        (chunk_id, mutator(text, str(example.expected_output)), span_hash)
        for (chunk_id, text, span_hash) in spans
    )

    with pytest.raises(ValueError, match=match):
        _validate_oracle_evidence_spans(example, bad_spans, chunks)


def test_oracle_evidence_spans_are_traced_to_chunk_hashes() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    chunks = _chunks()
    spans = _oracle_evidence_spans(example, chunks)
    ordered_chunks = _validate_oracle_evidence_spans(example, spans, chunks)

    assert tuple(chunk.chunk_id for chunk in ordered_chunks) == tuple(sorted(example.gold_chunk_ids))
    assert all(
        span_hash
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
        for (_, text, span_hash) in spans
    )


def test_oracle_evidence_order_is_deterministic_under_chunk_input_permutation() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)
    chunks = _chunks()

    forward = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=chunks,
    )
    reverse = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=list(reversed(chunks)),
    )

    assert forward == reverse
    assert forward.model_input_bytes() == reverse.model_input_bytes()


@pytest.mark.parametrize("behavior_only", [False, True])
def test_no_evidence_oracle_is_byte_identical_to_prompt(behavior_only: bool) -> None:
    evidence_status = (
        EvidenceStatus.NOT_APPLICABLE if behavior_only else EvidenceStatus.ABSENT
    )
    example = _example(evidence_status=evidence_status, behavior_only=behavior_only)
    contract = load_prompt_contract(PROMPT_PATH)

    prompt = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=contract,
    )
    oracle = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=_chunks(),
    )

    assert oracle.model_input == prompt.model_input
    assert oracle.model_input_bytes() == prompt.model_input_bytes()
    assert oracle.input_hash == prompt.input_hash
    assert oracle.model_input.user == example.question
    assert "CONTEXT" not in oracle.model_input.user
    assert oracle.evidence_chunk_ids == ()


def test_input_hash_is_hash_of_exact_canonical_system_user_bytes() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)
    result = construct_model_input(
        example=example,
        method=AdaptationMethod.ORACLE_CONTEXT,
        prompt_contract=contract,
        chunks=_chunks(),
    )

    assert result.input_hash == hashlib.sha256(
        canonical_model_input_bytes(result.model_input)
    ).hexdigest()


def test_oracle_present_requires_referenced_chunks() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)

    with pytest.raises(ValueError, match="missing evidence chunks"):
        construct_model_input(
            example=example,
            method=AdaptationMethod.ORACLE_CONTEXT,
            prompt_contract=contract,
            chunks=(),
        )


def test_unsupported_methods_are_not_constructed() -> None:
    example = _example(evidence_status=EvidenceStatus.PRESENT)
    contract = load_prompt_contract(PROMPT_PATH)

    with pytest.raises(ValueError, match="only supports PROMPT and ORACLE_CONTEXT"):
        construct_model_input(
            example=example,
            method=AdaptationMethod.RAG,
            prompt_contract=contract,
        )


def test_shared_renderer_is_identical_for_oracle_and_rag_selected_chunks() -> None:
    """Identical selected chunks must render byte-identically for Oracle and RAG."""
    from adaptlab.evaluation.inputs import render_evidence, render_selected_evidence

    example = _example(evidence_status=EvidenceStatus.PRESENT)
    chunks = _chunks()
    chunk_index = {chunk.chunk_id: chunk for chunk in chunks}
    selected_ids = tuple(sorted(example.gold_chunk_ids))

    oracle_selected = tuple(chunk_index[chunk_id] for chunk_id in selected_ids)
    oracle_rendered = render_evidence(oracle_selected)
    rag_rendered = render_selected_evidence(selected_ids, list(reversed(chunks)))

    assert oracle_rendered == rag_rendered
    assert oracle_rendered.encode("utf-8") == rag_rendered.encode("utf-8")


def test_shared_renderer_preserves_supplied_selection_order() -> None:
    from adaptlab.evaluation.inputs import render_selected_evidence

    chunks = _chunks()
    selected = tuple(chunk.chunk_id for chunk in chunks[:2])
    forward = render_selected_evidence(selected, chunks)
    reverse = render_selected_evidence(tuple(reversed(selected)), chunks)

    assert forward != reverse
    assert forward.index(next(c.content for c in chunks if c.chunk_id == selected[0])) < forward.index(
        next(c.content for c in chunks if c.chunk_id == selected[1])
    )


def test_shared_renderer_emits_no_chunk_metadata() -> None:
    from adaptlab.evaluation.inputs import render_selected_evidence

    chunk = _chunks()[0]
    rendered = render_selected_evidence((chunk.chunk_id,), (chunk,))

    # Renderer output is exactly the raw frozen chunk text plus neutral delimiters.
    # If an identifier genuinely occurs inside chunk.content, preserving it is correct;
    # the renderer must not append any metadata beyond that source text.
    assert rendered == f"--- BEGIN CONTEXT ---\n{chunk.content}\n--- END CONTEXT ---"
    assert rendered.removeprefix("--- BEGIN CONTEXT ---\n").removesuffix("\n--- END CONTEXT ---") == chunk.content
