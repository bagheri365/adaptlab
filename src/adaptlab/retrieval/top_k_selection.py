"""Precommitted top-k selection contract for Milestone 4 retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

TOP_K_SELECTION_VERSION = "top-k-selection-v1"
TOP_K_CANDIDATES = (1, 3, 5, 10)
ALL_REQUIRED_TOLERANCE = 0.02
PRIMARY_METRIC = "ALL_REQUIRED_GOLD@k"
SECONDARY_TIE_BREAK = "smallest_k_context_budget"
FROZEN_TOP_K_SELECTION_CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "retrieval" / "top_k_selection_v1.json"
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def top_k_selection_policy_payload() -> dict[str, object]:
    """Return the executable precommitted selection policy."""
    return {
        "candidate_top_k_values": list(TOP_K_CANDIDATES),
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": (
            "choose_smallest_k_within_absolute_tolerance_of_best_validation_"
            "all_required_gold"
        ),
        "absolute_tolerance": ALL_REQUIRED_TOLERANCE,
        "secondary_tie_break": SECONDARY_TIE_BREAK,
        "generation_metrics_allowed": False,
        "primary_test_metrics_allowed": False,
        "version": TOP_K_SELECTION_VERSION,
    }


def top_k_selection_policy_hash() -> str:
    return hashlib.sha256(
        _canonical_json(top_k_selection_policy_payload()).encode("utf-8")
    ).hexdigest()


def verify_frozen_top_k_selection_policy(
    path: Path = FROZEN_TOP_K_SELECTION_CONFIG,
) -> None:
    """Verify the checked-in decision artifact matches executable policy semantics."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != top_k_selection_policy_payload():
        raise ValueError("frozen top_k selection policy does not match executable policy")


@dataclass(frozen=True)
class TopKSelectionDecision:
    """Deterministic selection output to be persisted after validation retrieval."""

    selected_top_k: int
    best_all_required_gold: float
    threshold: float
    candidate_metrics: tuple[tuple[int, float], ...]
    policy_version: str = TOP_K_SELECTION_VERSION
    policy_hash: str = ""

    def __post_init__(self) -> None:
        if not self.policy_hash:
            object.__setattr__(self, "policy_hash", top_k_selection_policy_hash())

    def to_dict(self) -> dict[str, object]:
        return {
            "best_all_required_gold": self.best_all_required_gold,
            "candidate_metrics": [
                {"top_k": k, "all_required_gold": score}
                for k, score in self.candidate_metrics
            ],
            "policy_hash": self.policy_hash,
            "policy_version": self.policy_version,
            "selected_top_k": self.selected_top_k,
            "threshold": self.threshold,
        }


def select_top_k(validation_all_required_gold: Mapping[int, float]) -> TopKSelectionDecision:
    """Apply the precommitted rule using retrieval metrics only.

    The caller must provide validation ALL_REQUIRED_GOLD values for exactly the
    predeclared candidates. Generation accuracy and primary-test metrics are not
    accepted by this API.
    """
    keys = tuple(sorted(validation_all_required_gold))
    if keys != TOP_K_CANDIDATES:
        raise ValueError(
            f"validation metrics must contain exactly candidate k values {TOP_K_CANDIDATES}"
        )

    normalized: list[tuple[int, float]] = []
    for k in TOP_K_CANDIDATES:
        score = float(validation_all_required_gold[k])
        if not 0.0 <= score <= 1.0:
            raise ValueError("ALL_REQUIRED_GOLD validation metrics must be in [0, 1]")
        normalized.append((k, score))

    best = max(score for _, score in normalized)
    threshold = best - ALL_REQUIRED_TOLERANCE
    selected = min(k for k, score in normalized if score >= threshold)
    return TopKSelectionDecision(
        selected_top_k=selected,
        best_all_required_gold=best,
        threshold=threshold,
        candidate_metrics=tuple(normalized),
    )
