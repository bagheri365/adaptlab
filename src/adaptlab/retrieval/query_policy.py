"""Frozen Milestone 4 retrieval eligibility and query-construction policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar
import json

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import TaskFamily

QUERY_POLICY_VERSION = "retrieval-query-v1"
QUERY_POLICY_FILENAME = "query_policy_v1.json"
_ELIGIBLE = frozenset({
    TaskFamily.knowledge_only,
    TaskFamily.behavior_knowledge,
    TaskFamily.changed_knowledge,
})
_T = TypeVar("_T")


@dataclass(frozen=True)
class RetrievalQuery:
    """Policy decision for one example; ineligible examples carry an empty query."""

    example_id: str
    retrieval_eligible: bool
    query_text: str
    query_hash: str


def is_retrieval_eligible(example: BenchmarkExample) -> bool:
    """Return whether the example is permitted to invoke retrieval."""
    return example.task_family in _ELIGIBLE


def construct_retrieval_query(example: BenchmarkExample) -> RetrievalQuery:
    """Construct exactly the benchmark question, or an explicit bypass record."""
    if not is_retrieval_eligible(example):
        text = ""
        return RetrievalQuery(example.example_id, False, text, sha256_bytes(text.encode("utf-8")))
    text = example.question
    return RetrievalQuery(example.example_id, True, text, sha256_bytes(text.encode("utf-8")))


def retrieve_if_eligible(
    example: BenchmarkExample,
    retrieve: Callable[[str], _T],
) -> tuple[RetrievalQuery, _T | None]:
    """Apply the frozen policy and guarantee behavior_only never invokes retrieval."""
    query = construct_retrieval_query(example)
    if not query.retrieval_eligible:
        return query, None
    return query, retrieve(query.query_text)


def query_policy_payload() -> dict[str, object]:
    """Canonical semantic policy payload used for hashing and freeze verification."""
    return {
        "eligible_task_families": [family.value for family in sorted(_ELIGIBLE, key=lambda x: x.value)],
        "ineligible_task_families": [TaskFamily.behavior_only.value],
        "query_source": "question_only",
        "version": QUERY_POLICY_VERSION,
    }


def query_policy_hash() -> str:
    """Stable hash of the executable query-policy contract."""
    return sha256_bytes(canonical_json_bytes(query_policy_payload()))


def verify_frozen_query_policy(path: Path) -> str:
    """Verify a checked-in policy file matches the executable policy."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw != query_policy_payload():
        raise ValueError("frozen query policy does not match executable query policy")
    return query_policy_hash()
