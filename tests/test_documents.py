"""Tests for deterministic Nimbus documentation generation."""

from __future__ import annotations

import json

import pytest

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.generate_docs import generate_documents
from adaptlab.benchmark.generate_world import generate_world
from adaptlab.domain.enums import DocumentStyle


def _serialized(value) -> str:
    return json.dumps([item.to_dict() for item in value], sort_keys=True, separators=(",", ":"))


def test_document_schema_validation() -> None:
    doc = Document(
        document_id="DOC_X",
        title="Example",
        version="v2",
        component_family="authentication",
        document_style=DocumentStyle.reference_documentation,
        content="Content",
        record_ids=("R1",),
        logical_fact_ids=("L1",),
    )
    assert doc.document_style is DocumentStyle.reference_documentation

    with pytest.raises(ValueError, match="document_id"):
        Document("", "Title", "v2", "authentication", DocumentStyle.reference_documentation, "x", (), ())

    with pytest.raises(ValueError, match="duplicates"):
        Document("D", "Title", "v2", "authentication", DocumentStyle.reference_documentation, "x", ("R1", "R1"), ("L1",))


def test_chunk_schema_validation() -> None:
    chunk = DocumentChunk(
        chunk_id="C1",
        document_id="D1",
        version="v2",
        component_family="projects",
        document_style="configuration_guide",
        content="x",
        record_ids=(),
        logical_fact_ids=(),
        is_authoritative=False,
        is_obsolete=False,
    )
    assert chunk.document_style is DocumentStyle.configuration_guide

    with pytest.raises(ValueError, match="invalid document_style"):
        DocumentChunk("C", "D", "v2", "projects", "invalid", "x", (), (), False, False)


def test_generation_has_unique_ids_and_all_styles() -> None:
    documents, chunks = generate_documents(generate_world(1729))

    assert len({doc.document_id for doc in documents}) == len(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {doc.document_style for doc in documents} == set(DocumentStyle)
    assert [doc.document_id for doc in documents] == sorted(doc.document_id for doc in documents)
    assert [chunk.chunk_id for chunk in chunks] == sorted(chunk.chunk_id for chunk in chunks)


def test_authoritative_fact_references_resolve_to_world() -> None:
    world = generate_world(1729)
    documents, chunks = generate_documents(world)
    record_ids = {fact.record_id for fact in world.facts}
    logical_ids = {fact.logical_fact_id for fact in world.facts}

    for doc in documents:
        assert set(doc.record_ids) <= record_ids
        assert set(doc.logical_fact_ids) <= logical_ids

    for chunk in chunks:
        if chunk.is_authoritative:
            assert chunk.record_ids
            assert set(chunk.record_ids) <= record_ids
            assert set(chunk.logical_fact_ids) <= logical_ids


def test_generation_contains_obsolete_competing_and_distractor_chunks() -> None:
    _, chunks = generate_documents(generate_world(1729))

    assert any(chunk.is_obsolete for chunk in chunks)
    assert any(chunk.chunk_id == "CHK_COMPETING_AUTH_V2_REFERENCE" for chunk in chunks)
    distractor = next(chunk for chunk in chunks if chunk.chunk_id == "CHK_DISTRACTOR_PROJECTS_UI")
    assert not distractor.record_ids
    assert not distractor.logical_fact_ids
    assert not distractor.is_authoritative


def test_generation_is_deterministic() -> None:
    docs_a, chunks_a = generate_documents(generate_world(1729))
    docs_b, chunks_b = generate_documents(generate_world(1729))

    assert _serialized(docs_a) == _serialized(docs_b)
    assert _serialized(chunks_a) == _serialized(chunks_b)


def test_document_and_chunk_serialization_round_trip() -> None:
    documents, chunks = generate_documents(generate_world(1729))
    document = documents[0]
    chunk = chunks[0]

    restored_document = Document.from_dict(json.loads(json.dumps(document.to_dict())))
    restored_chunk = DocumentChunk.from_dict(json.loads(json.dumps(chunk.to_dict())))

    assert restored_document == document
    assert restored_chunk == chunk
