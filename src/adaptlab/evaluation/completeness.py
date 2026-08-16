"""Canonical run completeness checks shared across evaluation paths."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompletenessRecord:
    expected_count: int
    completed_successful_responses: int
    valid: bool


def completeness_record(*, expected_count: int, completed_successful_responses: int) -> CompletenessRecord:
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 0:
        raise ValueError("expected_count must be a non-negative integer")
    if (
        not isinstance(completed_successful_responses, int)
        or isinstance(completed_successful_responses, bool)
        or completed_successful_responses < 0
    ):
        raise ValueError("completed_successful_responses must be a non-negative integer")
    return CompletenessRecord(
        expected_count=expected_count,
        completed_successful_responses=completed_successful_responses,
        valid=expected_count == completed_successful_responses,
    )


def require_no_silent_drop(*, expected_count: int, actual_count: int) -> None:
    if expected_count != actual_count:
        raise ValueError(
            f"accidental dropped example: expected {expected_count} results but got {actual_count}"
        )
