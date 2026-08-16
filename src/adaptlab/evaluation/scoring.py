"""Deterministic normalization and scoring for benchmark model outputs.

The evaluator does not reinterpret benchmark truth.  ``BenchmarkExample`` objects
already carry expected outputs that are mechanically validated against the
benchmark's typed scoring contracts; this module only normalizes a model's text
conservatively and compares it to that validated expected output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import ScoringRule

SCORER_VERSION = "1"
NORMALIZER_VERSION = "1"

SUPPORTED_SCORING_RULES = frozenset({
    ScoringRule.FACT_VALUE,
    ScoringRule.RETIRED_STATUS,
    ScoringRule.STRUCTURED_EXTRACTION,
    ScoringRule.CONDITIONAL_RULE,
    ScoringRule.CLASSIFICATION,
    ScoringRule.ABSTENTION,
})


@dataclass(frozen=True, slots=True)
class ScoredOutput:
    """Normalized output plus a deterministic exact-match score."""

    raw_output: str
    normalized_output: Any
    score: float
    valid: bool
    error: str | None = None
    scorer_version: str = SCORER_VERSION
    normalizer_version: str = NORMALIZER_VERSION


def _parse_json_scalar(text: str, expected: Any) -> tuple[bool, Any]:
    """Parse a scalar only when JSON preserves the expected primitive type."""

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None

    # bool is a subclass of int in Python, so test it first.
    if isinstance(expected, bool):
        return (isinstance(value, bool), value)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return (isinstance(value, int) and not isinstance(value, bool), value)
    if isinstance(expected, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok and not math.isfinite(float(value)):
            ok = False
        return ok, float(value) if ok else None
    return False, None


def normalize_output(raw_output: str, *, expected_output: Any, scoring_rule: ScoringRule) -> tuple[bool, Any, str | None]:
    """Normalize model text conservatively according to benchmark output shape.

    Allowed normalization is deliberately narrow:
    * trim surrounding whitespace for every output;
    * parse JSON when the benchmark requires a structured object/list;
    * parse canonical JSON primitives for numeric/boolean expected values.

    Free-form prose extraction, case folding, label guessing, markdown-fence
    stripping, and semantic equivalence are intentionally not performed.
    """

    if scoring_rule not in SUPPORTED_SCORING_RULES:
        raise ValueError(f"unsupported scoring rule: {scoring_rule!r}")
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")

    text = raw_output.strip()

    if isinstance(expected_output, (dict, list)):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False, None, "malformed JSON"
        if type(parsed) is not type(expected_output):
            return False, parsed, "structured output has wrong JSON type"
        return True, parsed, None

    if isinstance(expected_output, (bool, int, float)) and not isinstance(expected_output, str):
        valid, parsed = _parse_json_scalar(text, expected_output)
        if not valid:
            return False, None, "output is not the required JSON primitive type"
        return True, parsed, None

    if isinstance(expected_output, str):
        return True, text, None

    # Current benchmark outputs are JSON-compatible primitives/objects. Fail
    # closed if a future benchmark introduces another output type.
    return False, None, f"unsupported expected output type: {type(expected_output).__name__}"


def score_output(example: BenchmarkExample, raw_output: str) -> ScoredOutput:
    """Score a raw model output against a benchmark-validated expected output."""

    if example.scoring_rule is None:
        raise ValueError(f"example {example.example_id} has no scoring_rule")
    if example.scoring_rule not in SUPPORTED_SCORING_RULES:
        raise ValueError(f"unsupported scoring rule: {example.scoring_rule.value}")

    valid, normalized, error = normalize_output(
        raw_output,
        expected_output=example.expected_output,
        scoring_rule=example.scoring_rule,
    )
    score = 1.0 if valid and normalized == example.expected_output else 0.0
    return ScoredOutput(
        raw_output=raw_output,
        normalized_output=normalized,
        score=score,
        valid=valid,
        error=error,
    )
