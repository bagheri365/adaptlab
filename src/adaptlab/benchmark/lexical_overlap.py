"""Deterministic lexical-overlap diagnostics for the full Nimbus benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from statistics import mean
from typing import Any, Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Difficulty, EvidenceStatus, TaskFamily

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*")
_IDENTIFIER_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*|[A-Z][A-Z0-9]*_[A-Z0-9_-]+)\b")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _identifiers(text: str) -> set[str]:
    return {match.group(0) for match in _IDENTIFIER_RE.finditer(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True, slots=True)
class ChunkOverlap:
    chunk_id: str
    token_overlap_count: int
    jaccard: float
    identifier_overlap: tuple[str, ...]
    rare_token_overlap: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "token_overlap_count": self.token_overlap_count,
            "jaccard": round(self.jaccard, 8),
            "identifier_overlap": list(self.identifier_overlap),
            "rare_token_overlap": list(self.rare_token_overlap),
        }


@dataclass(frozen=True, slots=True)
class ExampleOverlap:
    example_id: str
    difficulty: Difficulty
    gold: tuple[ChunkOverlap, ...]
    distractors: tuple[ChunkOverlap, ...]
    best_gold_jaccard: float
    best_distractor_jaccard: float
    identifier_shortcut: bool
    shortcut_identifiers: tuple[str, ...]
    suspicious_hard: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "difficulty": self.difficulty.value,
            "gold": [item.to_dict() for item in self.gold],
            "distractors": [item.to_dict() for item in self.distractors],
            "best_gold_jaccard": round(self.best_gold_jaccard, 8),
            "best_distractor_jaccard": round(self.best_distractor_jaccard, 8),
            "identifier_shortcut": self.identifier_shortcut,
            "shortcut_identifiers": list(self.shortcut_identifiers),
            "suspicious_hard": self.suspicious_hard,
        }


@dataclass(frozen=True, slots=True)
class LexicalOverlapAudit:
    examples: tuple[ExampleOverlap, ...]
    distributions_by_difficulty: dict[str, dict[str, float | int]]
    identifier_shortcut_count: int
    highest_overlap_examples: tuple[str, ...]
    suspicious_hard_cases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": [item.to_dict() for item in self.examples],
            "distributions_by_difficulty": self.distributions_by_difficulty,
            "identifier_shortcut_count": self.identifier_shortcut_count,
            "highest_overlap_examples": list(self.highest_overlap_examples),
            "suspicious_hard_cases": list(self.suspicious_hard_cases),
        }

    def human_summary(self) -> str:
        lines = [
            "Lexical-overlap audit",
            f"Knowledge-bearing examples: {len(self.examples)}",
            f"Identifier shortcuts: {self.identifier_shortcut_count}",
            f"Suspicious HARD cases: {len(self.suspicious_hard_cases)}",
            "Overlap by difficulty:",
        ]
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            stats = self.distributions_by_difficulty.get(difficulty.value, {})
            lines.append(
                f"  {difficulty.value}: count={stats.get('count', 0)}, "
                f"mean_best_gold_jaccard={stats.get('mean_best_gold_jaccard', 0.0):.4f}, "
                f"mean_best_distractor_jaccard={stats.get('mean_best_distractor_jaccard', 0.0):.4f}"
            )
        if self.highest_overlap_examples:
            lines.append("Highest-overlap examples: " + ", ".join(self.highest_overlap_examples))
        if self.suspicious_hard_cases:
            lines.append("Suspicious HARD cases: " + ", ".join(self.suspicious_hard_cases))
        return "\n".join(lines)


def _chunk_overlap(
    question_tokens: set[str],
    question_identifiers: set[str],
    chunk: DocumentChunk,
    rare_tokens: set[str],
) -> ChunkOverlap:
    chunk_tokens = _token_set(chunk.content)
    common = question_tokens & chunk_tokens
    return ChunkOverlap(
        chunk_id=chunk.chunk_id,
        token_overlap_count=len(common),
        jaccard=_jaccard(question_tokens, chunk_tokens),
        identifier_overlap=tuple(sorted(question_identifiers & _identifiers(chunk.content))),
        rare_token_overlap=tuple(sorted(common & rare_tokens)),
    )


def run_lexical_overlap_audit(
    examples: Iterable[BenchmarkExample],
    chunks: Iterable[DocumentChunk],
    *,
    top_n: int = 10,
) -> LexicalOverlapAudit:
    """Audit lexical shortcuts without embeddings or model-based similarity."""

    all_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in all_chunks}
    chunk_token_sets = {chunk.chunk_id: _token_set(chunk.content) for chunk in all_chunks}

    document_frequency: Counter[str] = Counter()
    for token_set in chunk_token_sets.values():
        document_frequency.update(token_set)
    rare_limit = max(1, int(len(all_chunks) * 0.02))
    rare_tokens = {token for token, count in document_frequency.items() if count <= rare_limit}

    identifier_chunk_counts: Counter[str] = Counter()
    for chunk in all_chunks:
        identifier_chunk_counts.update(_identifiers(chunk.content))

    knowledge_bearing = sorted(
        (
            example
            for example in examples
            if example.task_family is not TaskFamily.behavior_only
            and example.evidence_status in (EvidenceStatus.PRESENT, EvidenceStatus.ABSENT)
        ),
        key=lambda example: example.example_id,
    )

    results: list[ExampleOverlap] = []
    for example in knowledge_bearing:
        qtokens = _token_set(example.question)
        qids = _identifiers(example.question)
        gold_ids = set(example.gold_chunk_ids)
        gold_chunks = [chunk_by_id[chunk_id] for chunk_id in sorted(gold_ids) if chunk_id in chunk_by_id]
        # Distractor candidates deliberately include all non-gold corpus chunks: obsolete,
        # competing, domain distractors, and other authoritative chunks are all plausible
        # retrieval alternatives from the model's perspective.
        distractor_chunks = [chunk for chunk in all_chunks if chunk.chunk_id not in gold_ids]

        gold = tuple(_chunk_overlap(qtokens, qids, chunk, rare_tokens) for chunk in gold_chunks)
        distractors_all = [
            _chunk_overlap(qtokens, qids, chunk, rare_tokens) for chunk in distractor_chunks
        ]
        # Keep the machine-readable report compact but deterministic: top 5 distractors
        # by Jaccard, then overlap count, then chunk id.
        distractors = tuple(
            sorted(
                distractors_all,
                key=lambda item: (-item.jaccard, -item.token_overlap_count, item.chunk_id),
            )[:5]
        )
        best_gold = max((item.jaccard for item in gold), default=0.0)
        best_distractor = max((item.jaccard for item in distractors_all), default=0.0)

        shortcut_ids = tuple(
            sorted(
                identifier
                for identifier in qids
                if identifier_chunk_counts[identifier] == 1
                and any(identifier in item.identifier_overlap for item in gold)
            )
        )
        identifier_shortcut = bool(shortcut_ids)

        # HARD should not be dominated by a trivial exact lexical match. This is
        # diagnostic-only: flag high gold overlap or a unique-identifier shortcut.
        suspicious_hard = (
            example.difficulty is Difficulty.HARD
            and (best_gold >= 0.50 or identifier_shortcut)
        )
        results.append(
            ExampleOverlap(
                example_id=example.example_id,
                difficulty=example.difficulty,
                gold=gold,
                distractors=distractors,
                best_gold_jaccard=best_gold,
                best_distractor_jaccard=best_distractor,
                identifier_shortcut=identifier_shortcut,
                shortcut_identifiers=shortcut_ids,
                suspicious_hard=suspicious_hard,
            )
        )

    by_difficulty: dict[Difficulty, list[ExampleOverlap]] = defaultdict(list)
    for result in results:
        by_difficulty[result.difficulty].append(result)
    distributions: dict[str, dict[str, float | int]] = {}
    for difficulty in Difficulty:
        rows = by_difficulty.get(difficulty, [])
        distributions[difficulty.value] = {
            "count": len(rows),
            "mean_best_gold_jaccard": round(mean([r.best_gold_jaccard for r in rows]), 8) if rows else 0.0,
            "mean_best_distractor_jaccard": round(mean([r.best_distractor_jaccard for r in rows]), 8) if rows else 0.0,
            "max_best_gold_jaccard": round(max((r.best_gold_jaccard for r in rows), default=0.0), 8),
        }

    highest = tuple(
        item.example_id
        for item in sorted(results, key=lambda item: (-item.best_gold_jaccard, item.example_id))[:top_n]
    )
    suspicious = tuple(item.example_id for item in results if item.suspicious_hard)
    return LexicalOverlapAudit(
        examples=tuple(results),
        distributions_by_difficulty=distributions,
        identifier_shortcut_count=sum(item.identifier_shortcut for item in results),
        highest_overlap_examples=highest,
        suspicious_hard_cases=suspicious,
    )
