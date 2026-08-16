from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.domain.enums import DocumentStyle
from adaptlab.retrieval.policies import (
    INDEXING_POLICY_VERSION,
    TOKENIZATION_POLICY_VERSION,
    TIE_BREAK_POLICY,
    index_text,
    indexing_policy_hash,
    ranking_sort_key,
    sort_scored_chunk_ids,
    tokenize,
    tokenization_policy_hash,
    verify_frozen_indexing_policy,
    verify_frozen_tokenization_policy,
)


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-002",
        document_id="doc-secret",
        version="v2",
        component_family="SECRET_COMPONENT",
        document_style=DocumentStyle.reference_documentation,
        content="Nimbus_Mode-2 uses API.v3; Café COST is 42.50%.",
        record_ids=("SECRET_RECORD",),
        logical_fact_ids=("SECRET_LOGICAL_FACT",),
        is_authoritative=True,
        is_obsolete=False,
    )


def test_frozen_policy_files_match_executable_contracts() -> None:
    assert verify_frozen_indexing_policy(Path("config/retrieval/indexing_policy_v1.json")) == indexing_policy_hash()
    assert verify_frozen_tokenization_policy(Path("config/retrieval/tokenization_policy_v1.json")) == tokenization_policy_hash()
    assert INDEXING_POLICY_VERSION == "retrieval-indexing-v1"
    assert TOKENIZATION_POLICY_VERSION == "retrieval-tokenization-v1"
    assert len(indexing_policy_hash()) == 64
    assert len(tokenization_policy_hash()) == 64


def test_index_representation_is_chunk_content_only() -> None:
    chunk = _chunk()
    assert index_text(chunk) == chunk.content
    assert "SECRET_RECORD" not in index_text(chunk)
    assert "SECRET_LOGICAL_FACT" not in index_text(chunk)
    assert "SECRET_COMPONENT" not in index_text(chunk)


def test_hidden_corpus_metadata_cannot_change_indexed_text_or_tokens() -> None:
    chunk = _chunk()
    changed = replace(
        chunk,
        chunk_id="chunk-999",
        document_id="other-doc",
        version="obsolete-secret",
        component_family="OTHER_SECRET_COMPONENT",
        record_ids=("OTHER_SECRET_RECORD",),
        logical_fact_ids=("OTHER_SECRET_FACT",),
        is_authoritative=False,
        is_obsolete=True,
    )
    assert index_text(changed) == index_text(chunk)
    assert tokenize(index_text(changed)) == tokenize(index_text(chunk))


def test_tokenization_contract_is_deterministic_and_explicit() -> None:
    text = "Nimbus_Mode-2 uses API.v3; Café COST is 42.50%."
    expected = ("nimbus", "mode", "2", "uses", "api", "v3", "café", "cost", "is", "42", "50")
    assert tokenize(text) == expected
    assert tokenize(text) == tokenize(text)
    assert tokenize("THE the And") == ("the", "the", "and")  # no stopwords


def test_identifier_and_punctuation_handling_use_boundaries() -> None:
    assert tokenize("ABC_123/x.y-z") == ("abc", "123", "x", "y", "z")
    assert tokenize("") == ()
    assert tokenize("---___...") == ()


def test_no_stemming_or_query_expansion_is_implicit() -> None:
    assert tokenize("runs running runner") == ("runs", "running", "runner")


def test_equal_scores_tie_break_by_chunk_id_ascending() -> None:
    scored = (("chunk-z", 1.25), ("chunk-a", 1.25), ("chunk-m", 2.0))
    assert sort_scored_chunk_ids(scored) == (
        ("chunk-m", 2.0),
        ("chunk-a", 1.25),
        ("chunk-z", 1.25),
    )
    assert ranking_sort_key(1.0, "a") < ranking_sort_key(1.0, "b")
    assert TIE_BREAK_POLICY == "equal_bm25_score_then_chunk_id_ascending"
