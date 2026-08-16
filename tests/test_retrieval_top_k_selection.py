import json

import pytest

from adaptlab.retrieval.top_k_selection import (
    ALL_REQUIRED_TOLERANCE,
    TOP_K_CANDIDATES,
    select_top_k,
    top_k_selection_policy_hash,
    top_k_selection_policy_payload,
    verify_frozen_top_k_selection_policy,
)


def test_frozen_policy_matches_executable_contract():
    verify_frozen_top_k_selection_policy()
    payload = top_k_selection_policy_payload()
    assert payload["candidate_top_k_values"] == [1, 3, 5, 10]
    assert payload["primary_metric"] == "ALL_REQUIRED_GOLD@k"
    assert payload["absolute_tolerance"] == 0.02
    assert payload["generation_metrics_allowed"] is False
    assert payload["primary_test_metrics_allowed"] is False


def test_policy_hash_is_deterministic_sha256():
    digest = top_k_selection_policy_hash()
    assert digest == top_k_selection_policy_hash()
    assert len(digest) == 64
    int(digest, 16)


def test_selects_smallest_k_within_two_points_of_best():
    decision = select_top_k({1: 0.70, 3: 0.81, 5: 0.82, 10: 0.825})
    assert decision.selected_top_k == 3
    assert decision.best_all_required_gold == 0.825
    assert decision.threshold == pytest.approx(0.805)


def test_context_budget_prefers_smallest_eligible_k():
    decision = select_top_k({1: 0.90, 3: 0.91, 5: 0.91, 10: 0.91})
    assert decision.selected_top_k == 1


def test_requires_exact_precommitted_candidates():
    with pytest.raises(ValueError, match="exactly candidate"):
        select_top_k({1: 0.5, 3: 0.6, 5: 0.7})
    with pytest.raises(ValueError, match="exactly candidate"):
        select_top_k({1: 0.5, 3: 0.6, 5: 0.7, 10: 0.8, 20: 0.9})


def test_rejects_invalid_metric_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_top_k({1: 0.5, 3: 0.6, 5: 0.7, 10: 1.1})


def test_decision_serialization_records_policy_and_all_candidates():
    decision = select_top_k({1: 0.5, 3: 0.7, 5: 0.71, 10: 0.72})
    payload = decision.to_dict()
    assert payload["policy_hash"] == top_k_selection_policy_hash()
    assert [row["top_k"] for row in payload["candidate_metrics"]] == list(TOP_K_CANDIDATES)
    json.dumps(payload, sort_keys=True)
    assert ALL_REQUIRED_TOLERANCE == 0.02
