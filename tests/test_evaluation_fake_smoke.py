from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split, TaskFamily
from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.providers.fake import FakeModelProvider, FakeTransientFailure
from adaptlab.evaluation.runner import run_evaluation
from adaptlab.evaluation.schemas import AdaptationMethod, EvaluationRunStatus

BENCHMARK = Path("data/generated/v0.0")
PROMPT = Path("configs/prompts/prompt_v1.yaml")
FAMILIES = (
    TaskFamily.behavior_only,
    TaskFamily.knowledge_only,
    TaskFamily.behavior_knowledge,
    TaskFamily.changed_knowledge,
)


def _validation_examples() -> list[BenchmarkExample]:
    raw = json.loads((BENCHMARK / "validation.json").read_text(encoding="utf-8"))
    return [BenchmarkExample.from_dict(item) for item in raw]


def _smoke_examples() -> list[BenchmarkExample]:
    """Choose 16 stable validation examples: four from every task family."""
    examples = _validation_examples()
    selected: list[BenchmarkExample] = []
    for family in FAMILIES:
        family_examples = [example for example in examples if example.task_family is family]
        selected.extend(family_examples[:4])
    return sorted(selected, key=lambda example: example.example_id)


def _expected_text(example: BenchmarkExample) -> str:
    value = example.expected_output
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_provider_end_to_end_smoke(monkeypatch, tmp_path):
    """Exercise the full Milestone-3 fake-provider path over 16 mixed examples."""
    selected = _smoke_examples()
    assert len(selected) == 16
    assert {example.task_family for example in selected} == set(FAMILIES)

    # Keep frozen benchmark verification real while selecting a compact,
    # stratified smoke slice after the verified split has been loaded.
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(runner, "load_benchmark_split", lambda _directory, _split: list(selected))

    # Deliberately include malformed output, wrong output, and a recovered
    # transient provider failure. Every other response is mechanically correct.
    prompt_steps = []
    for index, example in enumerate(selected):
        if index == 0:
            prompt_steps.append(ModelResponse(text="{malformed-json", model_revision="fake-snapshot-1"))
        elif index == 1:
            prompt_steps.extend([
                FakeTransientFailure("temporary fake outage"),
                ModelResponse(text=_expected_text(example), model_revision="fake-snapshot-1"),
            ])
        elif index == 2:
            prompt_steps.append(ModelResponse(text="DELIBERATELY_WRONG", model_revision="fake-snapshot-1"))
        else:
            prompt_steps.append(ModelResponse(text=_expected_text(example), model_revision="fake-snapshot-1"))

    cache_dir = tmp_path / "shared-cache"
    prompt_dir = tmp_path / "prompt-run"
    prompt_provider = FakeModelProvider(prompt_steps)
    prompt_run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-model",
        provider=prompt_provider,
        prompt_config=PROMPT,
        output_dir=prompt_dir,
        split=Split.validation,
        limit=16,
        max_retries=1,
        cache_dir=cache_dir,
        run_id="fake-smoke-prompt",
    )
    assert prompt_run.status is EvaluationRunStatus.COMPLETED
    assert len(prompt_provider.requests) == 17  # one transient retry

    prompt_results = json.loads((prompt_dir / "results.json").read_text(encoding="utf-8"))
    assert len(prompt_results) == 16
    assert [row["example_id"] for row in prompt_results] == sorted(row["example_id"] for row in prompt_results)
    assert sum(row["score"] for row in prompt_results) == 14.0
    assert sum(row["retry_count"] for row in prompt_results) == 1
    assert all(row["provider_error"] is None for row in prompt_results)
    assert prompt_results[0]["raw_output"] == "{malformed-json"
    assert prompt_results[0]["score"] == 0.0
    assert prompt_results[2]["raw_output"] == "DELIBERATELY_WRONG"
    assert prompt_results[2]["score"] == 0.0

    prompt_metrics = json.loads((prompt_dir / "metrics.json").read_text(encoding="utf-8"))
    assert prompt_metrics["primary"]["overall_accuracy"] == {"n": 16, "accuracy": 14 / 16}
    assert sum(
        group["n"] for group in prompt_metrics["confirmatory"]["task_family"].values()
    ) == 16

    prompt_manifest = json.loads((prompt_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert prompt_manifest["completed_count"] == 16
    assert prompt_manifest["provider_error_count"] == 0
    assert prompt_manifest["model_revision"] == "fake-snapshot-1"
    assert prompt_manifest["result_hashes"]["results.json"] == _sha256(prompt_dir / "results.json")
    assert prompt_manifest["metric_hashes"]["metrics.json"] == _sha256(prompt_dir / "metrics.json")
    assert prompt_manifest["metric_hashes"]["summary.txt"] == _sha256(prompt_dir / "summary.txt")

    # Resume must preserve completed raw responses and make zero provider calls.
    raw_before_resume = [row["raw_output"] for row in prompt_results]
    resume_provider = FakeModelProvider([])
    resumed = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-model",
        provider=resume_provider,
        prompt_config=PROMPT,
        output_dir=prompt_dir,
        split=Split.validation,
        limit=16,
        max_retries=1,
        cache_dir=cache_dir,
        run_id="fake-smoke-prompt",
        resume=True,
    )
    assert resumed.status is EvaluationRunStatus.COMPLETED
    assert resume_provider.requests == []
    resumed_results = json.loads((prompt_dir / "results.json").read_text(encoding="utf-8"))
    assert [row["raw_output"] for row in resumed_results] == raw_before_resume

    # Exact-request cache reuse must also work in a fresh output directory.
    cached_dir = tmp_path / "prompt-from-cache"
    cached_provider = FakeModelProvider([])
    cached = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-model",
        provider=cached_provider,
        prompt_config=PROMPT,
        output_dir=cached_dir,
        split=Split.validation,
        limit=16,
        max_retries=1,
        cache_dir=cache_dir,
        run_id="fake-smoke-prompt-cache",
    )
    assert cached.status is EvaluationRunStatus.COMPLETED
    assert cached_provider.requests == []
    cached_results = json.loads((cached_dir / "results.json").read_text(encoding="utf-8"))
    assert [row["raw_output"] for row in cached_results] == raw_before_resume

    # Run Oracle on the exact same examples. This exercises gold evidence
    # injection while keeping behavior-only controls byte-identical to Prompt.
    oracle_dir = tmp_path / "oracle-run"
    oracle_provider = FakeModelProvider([
        ModelResponse(text=_expected_text(example), model_revision="fake-snapshot-1")
        for example in selected
    ])
    oracle_run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.ORACLE_CONTEXT,
        model_id="fake-model",
        provider=oracle_provider,
        prompt_config=PROMPT,
        output_dir=oracle_dir,
        split=Split.validation,
        limit=16,
        run_id="fake-smoke-oracle",
    )
    assert oracle_run.status is EvaluationRunStatus.COMPLETED
    oracle_results = json.loads((oracle_dir / "results.json").read_text(encoding="utf-8"))
    assert sum(row["score"] for row in oracle_results) == 16.0

    prompt_by_id = {row["example_id"]: row for row in prompt_results}
    oracle_by_id = {row["example_id"]: row for row in oracle_results}
    saw_injected_evidence = False
    for example in selected:
        prompt_row = prompt_by_id[example.example_id]
        oracle_row = oracle_by_id[example.example_id]
        if example.task_family is TaskFamily.behavior_only:
            assert prompt_row["input_hash"] == oracle_row["input_hash"]
            assert prompt_row["model_input"] == oracle_row["model_input"]
        else:
            if example.evidence_status.value == "PRESENT":
                saw_injected_evidence = True
                assert prompt_row["input_hash"] != oracle_row["input_hash"]
                assert prompt_row["model_input"] != oracle_row["model_input"]
    assert saw_injected_evidence
