from __future__ import annotations

import json

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
from adaptlab.evaluation.metrics import aggregate_metrics
from adaptlab.evaluation.schemas import EvaluationResult, ModelInput

HASH = "a" * 64


def _result(
    example_id: str,
    *,
    score: float | None,
    task_family: TaskFamily = TaskFamily.behavior_only,
    difficulty: Difficulty = Difficulty.EASY,
    behavior_type: BehaviorType | None = BehaviorType.SCHEMA_ADHERENCE,
    knowledge_state: KnowledgeState = KnowledgeState.NOT_APPLICABLE,
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_APPLICABLE,
    split_type: SplitType = SplitType.iid,
    provider_error: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        example_id=example_id,
        split=Split.test,
        task_family=task_family,
        difficulty=difficulty,
        behavior_type=behavior_type,
        knowledge_state=knowledge_state,
        evidence_status=evidence_status,
        split_type=split_type,
        input_hash=HASH,
        model_input=ModelInput(system="s", user="u"),
        raw_output=None if score is None else "x",
        normalized_output=None if score is None else "x",
        expected_output="x",
        score=score,
        scoring_rule=ScoringRule.CLASSIFICATION,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        provider_error=provider_error,
        retry_count=0,
    )


def test_overall_and_group_accuracies_include_n():
    results = [
        _result("b", score=0.0, difficulty=Difficulty.HARD),
        _result("a", score=1.0),
        _result(
            "c",
            score=1.0,
            task_family=TaskFamily.knowledge_only,
            behavior_type=None,
            knowledge_state=KnowledgeState.UPDATED,
            evidence_status=EvidenceStatus.PRESENT,
            split_type=SplitType.structural_holdout,
        ),
    ]
    metrics = aggregate_metrics(results)
    data = metrics.to_dict()

    assert data["primary"]["overall_accuracy"] == {"n": 3, "accuracy": 2 / 3}
    assert data["confirmatory"]["task_family"]["behavior_only"] == {
        "n": 2,
        "accuracy": 0.5,
    }
    assert data["confirmatory"]["task_family"]["knowledge_only"] == {
        "n": 1,
        "accuracy": 1.0,
    }
    assert data["exploratory"]["knowledge_state"]["UPDATED"] == {
        "n": 1,
        "accuracy": 1.0,
    }
    assert data["exploratory"]["split_type"]["STRUCTURAL_HOLDOUT"] == {
        "n": 1,
        "accuracy": 1.0,
    }


def test_empty_groups_have_n_but_no_accuracy():
    data = aggregate_metrics([_result("a", score=1.0)]).to_dict()
    empty = data["confirmatory"]["task_family"]["changed_knowledge"]
    assert empty == {"n": 0}
    assert "accuracy" not in empty


def test_provider_errors_are_not_silently_counted_as_wrong():
    metrics = aggregate_metrics(
        [
            _result("ok", score=1.0),
            _result("err", score=None, provider_error="temporary failure"),
        ]
    )
    assert metrics.primary["overall_accuracy"].n == 1
    assert metrics.primary["overall_accuracy"].accuracy == 1.0


def test_json_is_deterministic_and_machine_readable():
    first = _result("a", score=1.0)
    second = _result("z", score=0.0, difficulty=Difficulty.HARD)
    a = aggregate_metrics([second, first]).to_json_bytes()
    b = aggregate_metrics([first, second]).to_json_bytes()
    assert a == b
    parsed = json.loads(a)
    assert parsed["schema_version"] == "1"
    assert parsed["primary"]["overall_accuracy"]["n"] == 2


def test_human_summary_is_compact_and_omits_percentage_for_empty_groups():
    text = aggregate_metrics([_result("a", score=1.0)]).human_summary()
    assert "overall: n=1 accuracy=1.000" in text
    assert "changed_knowledge n=0" in text
    assert "changed_knowledge n=0 accuracy=" not in text
    assert text.endswith("\n")


def test_all_required_group_dimensions_are_present():
    data = aggregate_metrics([]).to_dict()
    assert list(data["confirmatory"]) == ["task_family", "difficulty"]
    assert list(data["exploratory"]) == [
        "behavior_type",
        "knowledge_state",
        "evidence_status",
        "split_type",
    ]
    assert list(data["confirmatory"]["difficulty"]) == ["EASY", "MEDIUM", "HARD"]
    assert list(data["exploratory"]["evidence_status"]) == [
        "NOT_APPLICABLE",
        "PRESENT",
        "ABSENT",
    ]
    assert list(data["exploratory"]["split_type"]) == ["IID", "STRUCTURAL_HOLDOUT"]
