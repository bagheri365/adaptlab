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
from adaptlab.retrieval.frozen_artifact import FrozenRetrievalArtifact
from adaptlab.retrieval.query_policy import is_retrieval_eligible

EVIDENCE_FORMAT_VERSION = "1"
EVIDENCE_RENDERER_VERSION = "evidence-renderer-v1"
_CONTEXT_BEGIN = "--- BEGIN CONTEXT ---"
_CONTEXT_SEPARATOR = "---"
_CONTEXT_END = "--- END CONTEXT ---"


def evidence_renderer_contract() -> dict[str, object]:
    """Return the frozen, model-visible evidence rendering contract."""
    return {
        "version": EVIDENCE_RENDERER_VERSION,
        "begin": _CONTEXT_BEGIN,
        "separator": _CONTEXT_SEPARATOR,
        "end": _CONTEXT_END,
        "content": "raw_frozen_chunk_text_only",
        "ordering": "preserve_supplied_chunk_sequence",
        "metadata": "none",
    }


def evidence_renderer_hash() -> str:
    """Hash the exact frozen evidence rendering policy."""
    return sha256_bytes(canonical_json_bytes(evidence_renderer_contract()))


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


@dataclass(frozen=True)
class ConstructedRAGInput:
    """Canonical RAG input plus frozen-retrieval provenance.

    Retrieval selection is supplied exclusively by a verified
    :class:`FrozenRetrievalArtifact`; this type contains no retrieval execution
    path. ``retrieved_context_hash`` is SHA-256 over the exact rendered context
    bytes, or SHA-256 of empty bytes for an explicit retrieval bypass / empty
    retrieval result.
    """

    model_input: ModelInput
    input_hash: str
    evidence_chunk_ids: tuple[str, ...]
    evidence_chunk_hashes: tuple[str, ...]
    retrieval_run_id: str
    retrieval_artifact_hash: str
    retriever_config_hash: str
    retrieved_context_hash: str
    evidence_format_version: str = EVIDENCE_FORMAT_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "input_hash",
            "retrieval_artifact_hash",
            "retriever_config_hash",
            "retrieved_context_hash",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if not self.retrieval_run_id.strip():
            raise ValueError("retrieval_run_id must be non-empty")
        if len(self.evidence_chunk_ids) != len(set(self.evidence_chunk_ids)):
            raise ValueError("evidence_chunk_ids must not contain duplicates")
        if len(self.evidence_chunk_hashes) != len(self.evidence_chunk_ids):
            raise ValueError("evidence_chunk_hashes must align with evidence_chunk_ids")
        if any(
            len(chunk_hash) != 64
            or any(char not in "0123456789abcdef" for char in chunk_hash)
            for chunk_hash in self.evidence_chunk_hashes
        ):
            raise ValueError("evidence_chunk_hashes must be lowercase SHA-256 hex digests")
        if self.evidence_format_version != EVIDENCE_FORMAT_VERSION:
            raise ValueError(
                f"evidence_format_version must be {EVIDENCE_FORMAT_VERSION!r}"
            )

    def model_input_bytes(self) -> bytes:
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


def render_evidence(chunks: Iterable[DocumentChunk]) -> str:
    """Render selected frozen chunks into the one canonical evidence block.

    The caller owns chunk selection and ordering.  This renderer deliberately
    ignores all benchmark/corpus metadata and emits only the raw frozen chunk
    text with the Milestone 3 delimiters.  Oracle and RAG must both call this
    function so chunk selection is their only intended evidence difference.
    """

    selected = tuple(chunks)
    if not selected:
        raise ValueError("evidence rendering requires at least one evidence chunk")
    contents = [chunk.content for chunk in selected]
    if any(not content.strip() for content in contents):
        raise ValueError("evidence chunk content must be non-empty")
    body = f"\n\n{_CONTEXT_SEPARATOR}\n\n".join(contents)
    return f"{_CONTEXT_BEGIN}\n{body}\n{_CONTEXT_END}"


def chunks_for_selected_ids(
    chunk_ids: Iterable[str],
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> tuple[DocumentChunk, ...]:
    """Load frozen chunks in exactly the supplied selected-ID sequence."""

    selected_ids = tuple(chunk_ids)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected chunk_ids must not contain duplicates")
    index = _chunk_index(chunks)
    missing = [chunk_id for chunk_id in selected_ids if chunk_id not in index]
    if missing:
        raise ValueError(f"selected evidence references missing chunks: {sorted(missing)}")
    return tuple(index[chunk_id] for chunk_id in selected_ids)


def render_selected_evidence(
    chunk_ids: Iterable[str],
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
) -> str:
    """Resolve a selected chunk sequence and render it canonically."""

    return render_evidence(chunks_for_selected_ids(chunk_ids, chunks))


# Backward-compatible private name used by existing Milestone 3 tests/code.
def _format_context(chunks: tuple[DocumentChunk, ...]) -> str:
    return render_evidence(chunks)


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
        user_prompt = f"{render_evidence(ordered_chunks)}\n\n{example.question}"
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


def construct_rag_model_input(
    *,
    example: BenchmarkExample,
    prompt_contract: PromptContract,
    chunks: Mapping[str, DocumentChunk] | Iterable[DocumentChunk],
    retrieval_artifact: FrozenRetrievalArtifact,
) -> ConstructedRAGInput:
    """Build one RAG input solely from a frozen canonical retrieval artifact.

    This function never invokes a retriever. Retrieval-eligible examples consume
    the exact selected chunk sequence frozen in ``retrieval_artifact`` and render
    it with the same ``render_evidence`` function used by ORACLE_CONTEXT.
    ``behavior_only`` examples require an explicit frozen bypass and take the
    exact PROMPT path, yielding byte-identical model input.
    """

    entries = {entry.example_id: entry for entry in retrieval_artifact.entries}
    if len(entries) != len(retrieval_artifact.entries):
        # FrozenRetrievalArtifact already enforces this, but keep the lookup
        # invariant local to input construction.
        raise ValueError("frozen retrieval artifact contains duplicate example IDs")
    try:
        entry = entries[example.example_id]
    except KeyError as exc:
        raise ValueError(
            f"frozen retrieval artifact has no entry for example {example.example_id}"
        ) from exc

    expected_eligible = is_retrieval_eligible(example)
    if entry.retrieval_eligible is not expected_eligible:
        raise ValueError(
            f"example {example.example_id} retrieval eligibility does not match frozen artifact"
        )

    evidence_chunk_ids: tuple[str, ...] = ()
    evidence_chunk_hashes: tuple[str, ...] = ()
    rendered_context = ""

    if expected_eligible and entry.chunk_ids:
        selected_chunks = chunks_for_selected_ids(entry.chunk_ids, chunks)
        rendered_context = render_evidence(selected_chunks)
        user_prompt = f"{rendered_context}\n\n{example.question}"
        evidence_chunk_ids = entry.chunk_ids
        evidence_chunk_hashes = tuple(_sha256_text(chunk.content) for chunk in selected_chunks)
    else:
        # Includes behavior_only's mandatory explicit bypass. Eligible empty
        # retrievals also remain question-only rather than inventing context.
        user_prompt = example.question

    model_input = ModelInput(system=prompt_contract.system_prompt, user=user_prompt)
    return ConstructedRAGInput(
        model_input=model_input,
        input_hash=sha256_bytes(canonical_model_input_bytes(model_input)),
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_chunk_hashes=evidence_chunk_hashes,
        retrieval_run_id=retrieval_artifact.retrieval_run_id,
        retrieval_artifact_hash=retrieval_artifact.retrieval_artifact_hash,
        retriever_config_hash=retrieval_artifact.retriever_config_hash,
        retrieved_context_hash=sha256_bytes(rendered_context.encode("utf-8")),
    )
