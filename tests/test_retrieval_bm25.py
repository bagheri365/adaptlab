from dataclasses import replace

import pytest

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.domain.enums import DocumentStyle
from adaptlab.retrieval.bm25 import BM25Retriever, bm25_config_hash


def chunk(chunk_id: str, content: str, **kw) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, document_id=f"doc-{chunk_id}", version="v2",
        component_family=kw.get("component_family", "routing"),
        document_style=DocumentStyle.reference_documentation, content=content,
        record_ids=tuple(kw.get("record_ids", (f"rec-{chunk_id}",))),
        logical_fact_ids=tuple(kw.get("logical_fact_ids", (f"fact-{chunk_id}",))),
        is_authoritative=kw.get("is_authoritative", True),
        is_obsolete=kw.get("is_obsolete", False),
    )


def test_known_lexical_match_ranks_first():
    r = BM25Retriever([chunk("b", "alpha beta"), chunk("a", "zebra routing zebra")])
    hits = r.retrieve("zebra", top_k=2)
    assert hits[0].chunk_id == "a"
    assert hits[0].score > hits[1].score
    assert [h.rank for h in hits] == [1, 2]


def test_deterministic_across_input_order():
    chunks = [chunk("c", "alpha"), chunk("a", "alpha beta"), chunk("b", "beta")]
    assert BM25Retriever(chunks).retrieve("alpha beta", top_k=3) == BM25Retriever(list(reversed(chunks))).retrieve("alpha beta", top_k=3)


def test_equal_scores_tie_break_by_chunk_id():
    r = BM25Retriever([chunk("z", "same text"), chunk("a", "same text")])
    assert [h.chunk_id for h in r.retrieve("same", top_k=2)] == ["a", "z"]


def test_empty_query_returns_no_hits():
    r = BM25Retriever([chunk("a", "alpha")])
    assert r.retrieve("...___---", top_k=5) == ()


def test_top_k_limits_and_caps_at_corpus_size():
    r = BM25Retriever([chunk("a", "alpha"), chunk("b", "beta"), chunk("c", "gamma")])
    assert len(r.retrieve("alpha", top_k=1)) == 1
    assert len(r.retrieve("alpha", top_k=10)) == 3
    with pytest.raises(ValueError):
        r.retrieve("alpha", top_k=0)


def test_hidden_metadata_cannot_change_ranking_or_corpus_index_terms():
    base = [chunk("a", "visible alpha"), chunk("b", "visible beta")]
    changed = [
        replace(base[0], component_family="other", record_ids=("secret-zebra",), logical_fact_ids=("gold-zebra",), is_obsolete=True),
        base[1],
    ]
    before = BM25Retriever(base).retrieve("zebra", top_k=2)
    after = BM25Retriever(changed).retrieve("zebra", top_k=2)
    assert [(x.chunk_id, x.score) for x in before] == [(x.chunk_id, x.score) for x in after]


def test_config_and_corpus_hashes_are_stable():
    chunks = [chunk("b", "beta"), chunk("a", "alpha")]
    one = BM25Retriever(chunks)
    two = BM25Retriever(list(reversed(chunks)))
    assert one.retriever_config_hash == two.retriever_config_hash == bm25_config_hash()
    assert one.corpus_hash == two.corpus_hash
    assert len(one.retriever_config_hash) == 64
    assert len(one.corpus_hash) == 64
