from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptlab.benchmark.sentinel import generate_generalization_sentinel
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split
from adaptlab.evaluation.completeness import completeness_record
from adaptlab.evaluation.errors import TransientProviderError
from adaptlab.evaluation.providers import ModelProvider, ModelRequest, ModelResponse
from adaptlab.evaluation.providers.fake import FakeModelProvider, FakePermanentFailure
from adaptlab.evaluation.runner import run_evaluation
from adaptlab.evaluation.schemas import AdaptationMethod, EvaluationRunStatus
from adaptlab.evaluation.scoring import normalize_output

BENCHMARK = Path("data/generated/v0.0")
PROMPT = Path("configs/prompts/prompt_v1.yaml")


def _expected_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _benchmark_examples(split: str) -> list[BenchmarkExample]:
    raw = json.loads((BENCHMARK / f"{split}.json").read_text(encoding="utf-8"))
    return sorted((BenchmarkExample.from_dict(item) for item in raw), key=lambda example: example.example_id)


def _test_responses() -> list[ModelResponse]:
    return [ModelResponse(text=_expected_text(item.expected_output)) for item in _benchmark_examples("test")]


def _validation_pair_responses() -> list[ModelResponse]:
    examples = _benchmark_examples("validation")[:2]
    return [ModelResponse(text=_expected_text(examples[0].expected_output)), FakePermanentFailure("model not found")]


class _AlwaysTransientProvider(ModelProvider):
    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise TransientProviderError("temporary Ollama server failure", provider=self.provider_name)


def test_complete_primary_run_records_400_of_400_and_emits_accuracy(tmp_path):
    provider = FakeModelProvider(_test_responses())
    run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=tmp_path / "primary-complete",
        split=Split.test,
        expected_count=400,
        max_retries=0,
        run_id="primary-complete",
    )

    manifest = json.loads((tmp_path / "primary-complete" / "run_manifest.json").read_text())
    results = json.loads((tmp_path / "primary-complete" / "results.json").read_text())

    assert run.status is EvaluationRunStatus.COMPLETED
    assert len(results) == 400
    assert manifest["expected_count"] == 400
    assert manifest["completed_successful_responses"] == 400
    assert manifest["canonical_accuracy_emitted"] is True
    assert manifest["completed_count"] == 400
    assert manifest["provider_error_count"] == 0
    assert json.loads((tmp_path / "primary-complete" / "metrics.json").read_text())["primary"]["overall_accuracy"] == {
        "n": 400,
        "accuracy": 1.0,
    }


def test_one_unresolved_provider_failure_is_recorded_not_dropped(tmp_path):
    provider = FakeModelProvider(_validation_pair_responses())
    run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=tmp_path / "one-failure",
        split=Split.validation,
        limit=2,
        expected_count=2,
        max_retries=0,
    )

    results = json.loads((tmp_path / "one-failure" / "results.json").read_text())
    manifest = json.loads((tmp_path / "one-failure" / "run_manifest.json").read_text())

    assert run.status is EvaluationRunStatus.INCOMPLETE
    assert len(results) == 2
    assert manifest["expected_count"] == 2
    assert manifest["completed_successful_responses"] == 1
    assert manifest["provider_error_count"] == 1
    assert manifest["canonical_accuracy_emitted"] is False
    assert any(row["provider_error"] is not None for row in results)
    assert any(row["score"] is None for row in results)
    assert json.loads((tmp_path / "one-failure" / "metrics.json").read_text())["primary"]["overall_accuracy"] == {"n": 1}


def test_resumed_incomplete_run_preserves_successful_responses(tmp_path):
    out = tmp_path / "resume"
    first_provider = FakeModelProvider(_validation_pair_responses())
    first_run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=first_provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=2,
        expected_count=2,
        max_retries=0,
    )
    assert first_run.status is EvaluationRunStatus.INCOMPLETE

    second_provider = FakeModelProvider([ModelResponse(text=_expected_text(_benchmark_examples("validation")[1].expected_output))])
    resumed = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=second_provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=2,
        expected_count=2,
        resume=True,
        max_retries=0,
    )

    resumed_results = json.loads((out / "results.json").read_text())
    assert resumed.status is EvaluationRunStatus.COMPLETED
    assert len(second_provider.requests) == 1
    assert len(resumed_results) == 2
    assert all(row["provider_error"] is None for row in resumed_results)


def test_retry_exhaustion_marks_run_incomplete(tmp_path):
    run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=_AlwaysTransientProvider(),
        prompt_config=PROMPT,
        output_dir=tmp_path / "retry-exhaust",
        split=Split.validation,
        limit=1,
        expected_count=1,
        max_retries=1,
    )

    result = json.loads((tmp_path / "retry-exhaust" / "results.json").read_text())[0]
    assert run.status is EvaluationRunStatus.INCOMPLETE
    assert result["provider_error"] is not None
    assert result["score"] is None
    assert result["retry_count"] == 1


def test_completed_sentinel_is_100_of_100_and_canonical_complete() -> None:
    examples = generate_generalization_sentinel(seed=1729)
    provider = FakeModelProvider(
        [ModelResponse(text=_expected_text(example.expected_output)) for example in examples]
    )
    successful = 0
    for example in examples:
        response = provider.generate(
            ModelRequest(
                system_prompt="Follow the requested format precisely.",
                user_prompt=example.question,
                temperature=0.0,
                max_tokens=256,
                seed=1729,
            )
        )
        valid, normalized, _error = normalize_output(
            response.text,
            expected_output=example.expected_output,
            scoring_rule=example.scoring_rule,
        )
        if valid and normalized == example.expected_output:
            successful += 1

    record = completeness_record(expected_count=100, completed_successful_responses=successful)
    assert len(examples) == 100
    assert successful == 100
    assert record.valid is True


def test_accidental_dropped_example_is_rejected_before_completion(tmp_path, monkeypatch):
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(runner, "load_benchmark_split", lambda _dir, _split: _benchmark_examples("validation")[:2])
    with pytest.raises(ValueError, match="expected_count 3"):
        run_evaluation(
            benchmark_dir=BENCHMARK,
            method=AdaptationMethod.PROMPT,
            model_id="qwen3:8b",
            provider=FakeModelProvider([]),
            prompt_config=PROMPT,
            output_dir=tmp_path / "dropped",
            split=Split.validation,
            expected_count=3,
        )
