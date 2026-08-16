"""Deterministic local BM25 retrieval over frozen Nimbus chunks."""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Protocol, Sequence

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, serialize_chunks, sha256_bytes
from adaptlab.retrieval.policies import (
    INDEXING_POLICY_VERSION,
    TOKENIZATION_POLICY_VERSION,
    TIE_BREAK_POLICY,
    index_text,
    indexing_policy_hash,
    ranking_sort_key,
    tokenization_policy_hash,
    tokenize,
)

BM25_RETRIEVER_NAME = "BM25"
BM25_RETRIEVER_VERSION = "bm25-v1"
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    rank: int


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievalHit, ...]: ...


def bm25_config_payload(*, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> dict[str, object]:
    return {
        "b": float(b),
        "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
        "indexing_policy_hash": indexing_policy_hash(),
        "indexing_policy_version": INDEXING_POLICY_VERSION,
        "k1": float(k1),
        "retriever_name": BM25_RETRIEVER_NAME,
        "retriever_version": BM25_RETRIEVER_VERSION,
        "tie_break_policy": TIE_BREAK_POLICY,
        "tokenization_policy_hash": tokenization_policy_hash(),
        "tokenization_policy_version": TOKENIZATION_POLICY_VERSION,
    }


def bm25_config_hash(*, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> str:
    return sha256_bytes(canonical_json_bytes(bm25_config_payload(k1=k1, b=b)))


def frozen_corpus_hash(chunks: Sequence[DocumentChunk]) -> str:
    """Hash the complete frozen chunk artifact representation, deterministically."""
    return sha256_bytes(canonical_json_bytes(serialize_chunks(chunks)))


class BM25Retriever:
    """Small deterministic BM25 implementation with no external dependencies."""

    def __init__(self, chunks: Sequence[DocumentChunk], *, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        if not chunks:
            raise ValueError("chunks must be non-empty")
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        ordered = tuple(sorted(chunks, key=lambda c: c.chunk_id))
        ids = [c.chunk_id for c in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk_id values must be unique")

        self._chunks = ordered
        self.k1 = float(k1)
        self.b = float(b)
        self.retriever_name = BM25_RETRIEVER_NAME
        self.retriever_version = BM25_RETRIEVER_VERSION
        self.retriever_config_hash = bm25_config_hash(k1=self.k1, b=self.b)
        self.corpus_hash = frozen_corpus_hash(ordered)

        self._tokens = tuple(tokenize(index_text(c)) for c in ordered)
        self._lengths = tuple(len(tokens) for tokens in self._tokens)
        self._avgdl = sum(self._lengths) / len(self._lengths)
        self._term_freqs = tuple(self._counts(tokens) for tokens in self._tokens)
        df: dict[str, int] = {}
        for counts in self._term_freqs:
            for term in counts:
                df[term] = df.get(term, 0) + 1
        self._df = df

    @staticmethod
    def _counts(tokens: tuple[str, ...]) -> dict[str, int]:
        result: dict[str, int] = {}
        for token in tokens:
            result[token] = result.get(token, 0) + 1
        return result

    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievalHit, ...]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query_terms = tuple(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return ()

        n_docs = len(self._chunks)
        scored: list[tuple[str, float]] = []
        for chunk, counts, dl in zip(self._chunks, self._term_freqs, self._lengths, strict=True):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = self._df[term]
                idf = log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * (tf * (self.k1 + 1.0) / norm)
            scored.append((chunk.chunk_id, score))

        scored.sort(key=lambda item: ranking_sort_key(item[1], item[0]))
        limit = min(top_k, len(scored))
        return tuple(
            RetrievalHit(chunk_id=chunk_id, score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(scored[:limit], start=1)
        )
