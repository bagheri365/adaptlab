"""Deterministic construction of model-visible evaluation inputs.

Only the frozen system prompt, benchmark question, and (for ORACLE_CONTEXT with
PRESENT evidence) benchmark-defined chunk *content* enter the model-visible
input. Benchmark labels and evidence provenance remain evaluation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus
from adaptlab.evaluation.prompts import PromptContract
from adaptlab.evaluation.schemas import AdaptationMethod, ModelInput

EVIDENCE_FORMAT_VERSION = "1"
_CONTEXT_BEGIN = "--- BEGIN CONTEXT ---"
_CONTEXT_SEPARATOR = "---"
_CONTEXT_END = "--- END CONTEXT ---"


@dataclass(frozen=True)
class ConstructedModelInput:
    """Canonical model input plus non-model-visible evidence provenance."""

    model_input: ModelInput
    input_hash: str
    evidence_chunk_ids: tuple[str, ...]
    evidence_chunk_hashes: tuple[str, ...] = ()
    evidence_format_version: str = EVIDENCE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if len(self.input_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.input_hash
        ):
            raise ValueError("input_hash must be a lowercase SHA-256 hex digest")
        if len(self.evidence_chunk_ids) != len(set(self.evidence_chunk_ids)):
            raise ValueError("evidence_chunk_ids must not contain duplicates")
        if self.evidence_chunk_hashes and len(self.evidence_chunk_hashes) != len(self.evidence_chunk_ids):
            raise ValueError("evidence_chunk_hashes must align with evidence_chunk_ids")
        if any(
            len(chunk_hash) != 64 or any(char not in "0123456789abcdef" for char in chunk_hash)
            for chunk_hash in self.evidence_chunk_hashes
        ):
            raise ValueError("evidence_chunk_hashes must be lowercase SHA-256 hex digests")
        if self.evidence_format_version != EVIDENCE_FORMAT_VERSION:
            raise ValueError(
                f"evidence_format_version must be {EVIDENCE_FORMAT_VERSION!r}"
            )

    def model_input_bytes(self) -> bytes:
        """Return the exact canonical bytes used to identify system/user input."""

        return canonical_model_input_bytes(self.model_input)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_model_input_bytes(model_input: ModelInput) -> bytes:
    """Canonical bytes for exact input equality and hashing."""

    return canonical_json_bytes(model_input.to_dict())


def _chunk_index(
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> dict[str, DocumentChunk]:
    if isinstance(chunks, Mapping):
        index = dict(chunks)
    else:
        index = {chunk.chunk_id: chunk for chunk in chunks}
    if len(index) != (len(chunks) if hasattr(chunks, "__len__") else len(index)):
        # Iterable inputs without __len__ cannot be checked after consumption; the
        # benchmark validator already enforces unique chunk IDs. Mapping inputs are
        # intrinsically unique by key.
        raise ValueError("chunks must have unique chunk_id values")
    for chunk_id, chunk in index.items():
        if chunk_id != chunk.chunk_id:
            raise ValueError("chunk mapping keys must equal chunk.chunk_id")
    return index


def _ordered_gold_chunks(
    example: BenchmarkExample,
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> tuple[DocumentChunk, ...]:
    index = _chunk_index(chunks)
    missing = sorted(set(example.gold_chunk_ids) - set(index))
    if missing:
        raise ValueError(
            f"example {example.example_id} references missing evidence chunks: {missing}"
        )
    return tuple(index[chunk_id] for chunk_id in sorted(example.gold_chunk_ids))


def _format_context(chunks: tuple[DocumentChunk, ...]) -> str:
    if not chunks:
        raise ValueError("context formatting requires at least one evidence chunk")
    contents = [chunk.content for chunk in chunks]
    if any(not content.strip() for content in contents):
        raise ValueError("evidence chunk content must be non-empty")
    body = f"\n\n{_CONTEXT_SEPARATOR}\n\n".join(contents)
    return f"{_CONTEXT_BEGIN}\n{body}\n{_CONTEXT_END}"


def _oracle_evidence_spans(
    example: BenchmarkExample,
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> tuple[tuple[str, str, str], ...]:
    """Return provenance-traced evidence spans for an evidence-present example.

    Each span is a byte-for-byte frozen chunk payload identified by chunk ID and
    content hash. The wrapper is intentionally neutral and does not annotate any
    answer, score, or label metadata.
    """

    ordered_chunks = _ordered_gold_chunks(example, chunks)
    return tuple((chunk.chunk_id, chunk.content, _sha256_text(chunk.content)) for chunk in ordered_chunks)


def _validate_oracle_evidence_spans(
    example: BenchmarkExample,
    spans: tuple[tuple[str, str, str], ...],
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> tuple[DocumentChunk, ...]:
    """Validate that every injected evidence span matches a frozen chunk exactly."""

    ordered_chunks = _ordered_gold_chunks(example, chunks)
    if len(spans) != len(ordered_chunks):
        raise ValueError(
            f"example {example.example_id} evidence span count {len(spans)} != gold chunk count {len(ordered_chunks)}"
        )
    for (span_chunk_id, span_text, span_hash), chunk in zip(spans, ordered_chunks):
        if span_chunk_id != chunk.chunk_id:
            raise ValueError(
                f"example {example.example_id} evidence span chunk_id {span_chunk_id!r} != permitted gold chunk {chunk.chunk_id!r}"
            )
        if span_text != chunk.content:
            raise ValueError(
                f"example {example.example_id} evidence span text must match frozen chunk {chunk.chunk_id} byte-for-byte"
            )
        if span_hash != _sha256_text(chunk.content):
            raise ValueError(
                f"example {example.example_id} evidence span hash must match frozen chunk {chunk.chunk_id}"
            )
    return ordered_chunks


def construct_model_input(
    *,
    example: BenchmarkExample,
    method: AdaptationMethod,
    prompt_contract: PromptContract,
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk] = (),
) -> ConstructedModelInput:
    """Build one canonical PROMPT or ORACLE_CONTEXT model input.

    PROMPT always contains only the frozen system prompt and benchmark question.
    ORACLE_CONTEXT differs only when ``evidence_status=PRESENT``: benchmark-defined
    gold chunk contents are sorted by chunk ID and prepended in a neutral context
    block. ABSENT and NOT_APPLICABLE examples take the exact PROMPT path, including
    byte-identical canonical input bytes and hash.
    """

    if method not in {AdaptationMethod.PROMPT, AdaptationMethod.ORACLE_CONTEXT}:
        raise ValueError("input construction only supports PROMPT and ORACLE_CONTEXT")

    user_prompt = example.question
    evidence_chunk_ids: tuple[str, ...] = ()
    evidence_chunk_hashes: tuple[str, ...] = ()

    if method is AdaptationMethod.ORACLE_CONTEXT and example.evidence_status is EvidenceStatus.PRESENT:
        spans = _oracle_evidence_spans(example, chunks)
        ordered_chunks = _validate_oracle_evidence_spans(example, spans, chunks)
        user_prompt = f"{_format_context(ordered_chunks)}\n\n{example.question}"
        evidence_chunk_ids = tuple(chunk.chunk_id for chunk in ordered_chunks)
        evidence_chunk_hashes = tuple(_sha256_text(chunk.content) for chunk in ordered_chunks)

    model_input = ModelInput(system=prompt_contract.system_prompt, user=user_prompt)
    input_bytes = canonical_model_input_bytes(model_input)
    return ConstructedModelInput(
        model_input=model_input,
        input_hash=sha256_bytes(input_bytes),
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_chunk_hashes=evidence_chunk_hashes,
    )
