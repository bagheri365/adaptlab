from __future__ import annotations

from dataclasses import replace

import pytest

from adaptlab.evaluation.rag_completeness import (
    CANONICAL_RAG_EXPECTED_COUNT,
    RAGExampleCompletion,
    RAGRunIdentity,
    canonical_rag_completeness,
    require_canonical_rag_complete,
    resume_identity_matches,
)

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64


def _success(i: int) -> RAGExampleCompletion:
    return RAGExampleCompletion(
        example_id=f"test-{i:04d}",
        retrieval_succeeded=True,
        model_response_succeeded=True,
    )


def test_complete_400_of_400_rag_run_is_valid() -> None:
    rows = [_success(i) for i in range(CANONICAL_RAG_EXPECTED_COUNT)]
    record = canonical_rag_completeness(rows)
    assert record.expected_count == 400
    assert record.represented_count == 400
    assert record.completed_successful_model_responses == 400
    assert record.unresolved_provider_failures == 0
    assert record.valid is True
    require_canonical_rag_complete(record)


def test_retrieval_miss_is_not_an_infrastructure_failure() -> None:
    # Retrieval relevance is intentionally absent from this schema. A frozen
    # retrieval result that missed gold still counts as a valid consumed result.
    row = RAGExampleCompletion(
        example_id="test-miss",
        retrieval_succeeded=True,
        model_response_succeeded=True,
    )
    record = canonical_rag_completeness([row], expected_count=1)
    assert record.valid is True
    assert record.unresolved_provider_failures == 0


def test_provider_failure_makes_run_incomplete_but_keeps_example_represented() -> None:
    rows = [_success(i) for i in range(399)] + [
        RAGExampleCompletion(
            example_id="test-failure",
            retrieval_succeeded=True,
            model_response_succeeded=False,
            provider_error="TransientProviderError: connection reset",
        )
    ]
    record = canonical_rag_completeness(rows)
    assert record.represented_count == 400
    assert record.completed_successful_model_responses == 399
    assert record.unresolved_provider_failures == 1
    assert record.valid is False
    with pytest.raises(ValueError, match="unresolved_provider_failures=1"):
        require_canonical_rag_complete(record)


def test_resume_accepts_same_frozen_retrieval_identity() -> None:
    identity = RAGRunIdentity(
        canonical_rag_config_hash=H1,
        retrieval_artifact_hash=H2,
        benchmark_manifest_hash=H3,
    )
    assert resume_identity_matches(identity, identity)


def test_changed_retrieval_artifact_invalidates_run_identity_and_resume() -> None:
    prior = RAGRunIdentity(
        canonical_rag_config_hash=H1,
        retrieval_artifact_hash=H2,
        benchmark_manifest_hash=H3,
    )
    changed = replace(prior, retrieval_artifact_hash=H4)
    assert prior.run_identity_hash != changed.run_identity_hash
    assert resume_identity_matches(prior, changed) is False


def test_missing_frozen_retrieval_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="frozen retrieval/bypass"):
        RAGExampleCompletion(
            example_id="test-no-retrieval",
            retrieval_succeeded=False,
            model_response_succeeded=True,
        )


def test_duplicate_examples_cannot_be_silently_counted() -> None:
    row = _success(0)
    with pytest.raises(ValueError, match="duplicate"):
        canonical_rag_completeness([row, row], expected_count=2)
