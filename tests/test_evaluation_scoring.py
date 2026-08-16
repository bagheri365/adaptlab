from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    EvidenceStatus,
    KnowledgeState,
    ScoringRule,
    Split,
    SplitType,
    TaskFamily,
)
from adaptlab.evaluation.scoring import (
    NORMALIZER_VERSION,
    SCORER_VERSION,
    SUPPORTED_SCORING_RULES,
    normalize_output,
    score_output,
)


def make_example(rule: ScoringRule, expected_output):
    return BenchmarkExample(
        example_id=f"score-{rule.value.lower()}",
        benchmark_version="0.0.0",
        task_family=TaskFamily.behavior_only,
        behavior_type=BehaviorType.SCHEMA_ADHERENCE,
        difficulty=Difficulty.EASY,
        split=Split.validation,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version=None,
        knowledge_state=KnowledgeState.NOT_APPLICABLE,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        question="Return the requested output.",
        expected_output=expected_output,
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
        generation_seed=1,
        scoring_rule=rule,
        scoring_parameters={"prompt_only": True},
    )


@pytest.mark.parametrize("rule", sorted(SUPPORTED_SCORING_RULES, key=lambda x: x.value))
def test_every_benchmark_scoring_rule_supported(rule: ScoringRule) -> None:
    assert rule in SUPPORTED_SCORING_RULES


def test_versions_are_explicit() -> None:
    assert SCORER_VERSION == "1"
    assert NORMALIZER_VERSION == "1"


def test_correct_output_scores_one_and_preserves_raw() -> None:
    example = make_example(ScoringRule.FACT_VALUE, "queue")
    result = score_output(example, "  queue\n")
    assert result.raw_output == "  queue\n"
    assert result.normalized_output == "queue"
    assert result.score == 1.0
    assert result.valid


def test_wrong_output_scores_zero() -> None:
    result = score_output(make_example(ScoringRule.CLASSIFICATION, "HIGH"), "LOW")
    assert result.score == 0.0
    assert result.valid


def test_extra_prose_is_not_silently_extracted() -> None:
    result = score_output(make_example(ScoringRule.CONDITIONAL_RULE, "PASS"), "The answer is PASS")
    assert result.normalized_output == "The answer is PASS"
    assert result.score == 0.0


def test_malformed_json_fails_structured_output() -> None:
    result = score_output(make_example(ScoringRule.STRUCTURED_EXTRACTION, {"value": 7}), '{"value": 7')
    assert result.score == 0.0
    assert not result.valid
    assert result.normalized_output is None
    assert result.error == "malformed JSON"


def test_schema_violation_wrong_json_type_fails() -> None:
    result = score_output(make_example(ScoringRule.STRUCTURED_EXTRACTION, {"value": 7}), '[7]')
    assert result.score == 0.0
    assert not result.valid
    assert result.normalized_output == [7]
    assert "wrong JSON type" in result.error


def test_structured_output_requires_exact_schema_and_values() -> None:
    example = make_example(ScoringRule.STRUCTURED_EXTRACTION, {"value": 7})
    assert score_output(example, '{"value":7}').score == 1.0
    assert score_output(example, '{"value":7,"extra":1}').score == 0.0
    assert score_output(example, '{"value":"7"}').score == 0.0


def test_allowed_normalization_is_surrounding_whitespace_only_for_strings() -> None:
    example = make_example(ScoringRule.FACT_VALUE, "NMB-AUTH-102-v2")
    assert score_output(example, "\n NMB-AUTH-102-v2 \t").score == 1.0
    assert score_output(example, '"NMB-AUTH-102-v2"').score == 0.0
    assert score_output(example, "nmb-auth-102-v2").score == 0.0


def test_json_primitives_are_typed_conservatively() -> None:
    assert score_output(make_example(ScoringRule.FACT_VALUE, 82), " 82\n").score == 1.0
    assert score_output(make_example(ScoringRule.FACT_VALUE, 82), '"82"').score == 0.0
    assert score_output(make_example(ScoringRule.FACT_VALUE, True), "true").score == 1.0
    assert score_output(make_example(ScoringRule.FACT_VALUE, True), "1").score == 0.0


def test_abstention_uses_benchmark_expected_contract_exactly() -> None:
    assert score_output(make_example(ScoringRule.ABSTENTION, "INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_EVIDENCE").score == 1.0
    assert score_output(make_example(ScoringRule.ABSTENTION, "INSUFFICIENT_INFORMATION"), "INSUFFICIENT_INFORMATION").score == 1.0
    assert score_output(make_example(ScoringRule.ABSTENTION, "INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_INFORMATION").score == 0.0


def test_retired_status_is_exact() -> None:
    example = make_example(ScoringRule.RETIRED_STATUS, "RETIRED")
    assert score_output(example, "RETIRED\n").score == 1.0
    assert score_output(example, "retired").score == 0.0


def test_normalizer_rejects_non_string_raw_output() -> None:
    with pytest.raises(TypeError, match="raw_output must be a string"):
        normalize_output(7, expected_output=7, scoring_rule=ScoringRule.FACT_VALUE)  # type: ignore[arg-type]


def test_generated_benchmark_scoring_rules_all_score_their_expected_output() -> None:
    root = Path(__file__).parents[1] / "data" / "generated" / "v0.0"
    examples = []
    for filename in ("train.json", "validation.json", "test.json"):
        payload = json.loads((root / filename).read_text())
        examples.extend(BenchmarkExample.from_dict(item) for item in payload)

    assert examples
    assert {e.scoring_rule for e in examples} == SUPPORTED_SCORING_RULES
    for example in examples:
        expected = example.expected_output
        if isinstance(expected, str):
            raw = expected
        else:
            raw = json.dumps(expected, separators=(",", ":"), sort_keys=True)
        result = score_output(example, raw)
        assert result.score == 1.0, example.example_id
        assert result.valid, example.example_id
