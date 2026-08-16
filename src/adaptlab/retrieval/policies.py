"""Frozen indexing and tokenization contracts for Milestone 4 BM25.

These policies are intentionally minimal: only model-visible frozen chunk content
is indexable. Benchmark annotations and corpus provenance metadata are never part
of the retrieval document representation.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes

INDEXING_POLICY_VERSION = "retrieval-indexing-v1"
TOKENIZATION_POLICY_VERSION = "retrieval-tokenization-v1"
INDEXING_POLICY_FILENAME = "indexing_policy_v1.json"
TOKENIZATION_POLICY_FILENAME = "tokenization_policy_v1.json"
TIE_BREAK_POLICY = "equal_bm25_score_then_chunk_id_ascending"

# Maximal Unicode alphanumeric runs. Underscores and punctuation are boundaries,
# so identifiers such as "Nimbus_Mode-2" become ("nimbus", "mode", "2").
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def indexing_policy_payload() -> dict[str, object]:
    """Canonical semantic payload for the frozen chunk-index representation."""
    return {
        "document_representation": "chunk_content_only",
        "excluded_metadata": [
            "component_family",
            "difficulty",
            "expected_output",
            "gold_annotations",
            "knowledge_state",
            "logical_fact_id",
            "record_id",
            "task_family",
        ],
        "included_fields": ["content"],
        "title_or_heading_policy": "not_indexed",
        "version": INDEXING_POLICY_VERSION,
    }


def tokenization_policy_payload() -> dict[str, object]:
    """Canonical semantic payload for deterministic BM25 tokenization/ranking."""
    return {
        "case_normalization": "unicode_casefold",
        "identifier_handling": (
            "split_on_non_alphanumeric_boundaries; preserve contiguous unicode alphanumeric runs"
        ),
        "punctuation_handling": "treat_as_token_boundary",
        "stemming": "none",
        "stopword_policy": "none",
        "tie_break_policy": TIE_BREAK_POLICY,
        "token_splitting": "maximal_unicode_alphanumeric_runs",
        "version": TOKENIZATION_POLICY_VERSION,
    }


def indexing_policy_hash() -> str:
    return sha256_bytes(canonical_json_bytes(indexing_policy_payload()))


def tokenization_policy_hash() -> str:
    return sha256_bytes(canonical_json_bytes(tokenization_policy_payload()))


def index_text(chunk: DocumentChunk) -> str:
    """Return exactly the frozen, model-visible chunk content to index."""
    return chunk.content


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize deterministically according to retrieval-tokenization-v1."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def ranking_sort_key(score: float, chunk_id: str) -> tuple[float, str]:
    """Canonical rank key: higher score first, then chunk_id ascending."""
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError("score must be numeric")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")
    return (-float(score), chunk_id)


def sort_scored_chunk_ids(items: Iterable[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    """Sort scored chunk IDs using the precommitted deterministic tie-break."""
    return tuple(sorted(items, key=lambda item: ranking_sort_key(item[1], item[0])))


def _verify_frozen_policy(path: Path, expected: dict[str, object], label: str) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw != expected:
        raise ValueError(f"frozen {label} policy does not match executable policy")
    return sha256_bytes(canonical_json_bytes(expected))


def verify_frozen_indexing_policy(path: Path) -> str:
    return _verify_frozen_policy(path, indexing_policy_payload(), "indexing")


def verify_frozen_tokenization_policy(path: Path) -> str:
    return _verify_frozen_policy(path, tokenization_policy_payload(), "tokenization")
