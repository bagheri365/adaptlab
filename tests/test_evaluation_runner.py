from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from adaptlab.domain.enums import Split
from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.providers.fake import FakeModelProvider, FakePermanentFailure
from adaptlab.evaluation.runner import run_evaluation, verify_frozen_benchmark
from adaptlab.evaluation.schemas import AdaptationMethod, EvaluationRunStatus

BENCHMARK = Path("data/generated/v0.0")
PROMPT = Path("configs/prompts/prompt_v1.yaml")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _examples(split: str = "validation"):
    return json.loads((BENCHMARK / f"{split}.json").read_text())


def test_verify_frozen_benchmark_returns_manifest_hash():
    manifest, digest = verify_frozen_benchmark(BENCHMARK)
    assert manifest["benchmark_version"] == "0.0.0"
    assert digest == hashlib.sha256((BENCHMARK / "manifest.json").read_bytes()).hexdigest()


def test_verify_frozen_benchmark_rejects_tampered_split(tmp_path):
    copy = tmp_path / "benchmark"
    shutil.copytree(BENCHMARK, copy)
    (copy / "test.json").write_bytes((copy / "test.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="test.json"):
        verify_frozen_benchmark(copy)


def test_runner_end_to_end_writes_canonical_outputs_without_mutating_benchmark(tmp_path):
    before = _tree_hashes(BENCHMARK)
    raw = _examples()[:3]
    provider = FakeModelProvider([
        ModelResponse(text=json.dumps(item["expected_output"]) if not isinstance(item["expected_output"], str) else item["expected_output"], input_tokens=10, output_tokens=2, latency_ms=1.5)
        for item in raw
    ])
    out = tmp_path / "run"
    run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-model",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=3,
        run_id="run-test",
    )
    assert run.status is EvaluationRunStatus.COMPLETED
    results = json.loads((out / "results.json").read_text())
    assert [r["example_id"] for r in results] == sorted(r["example_id"] for r in results)
    assert len(results) == 3 and all(r["score"] == 1.0 for r in results)
    assert (out / "metrics.json").exists()
    assert (out / "summary.txt").exists()
    assert json.loads((out / "run_manifest.json").read_text())["run_id"] == "run-test"
    assert _tree_hashes(BENCHMARK) == before


def test_provider_failure_is_recorded_not_omitted(tmp_path):
    provider = FakeModelProvider([FakePermanentFailure("nope")])
    out = tmp_path / "failed"
    run = run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-model",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=1,
    )
    result = json.loads((out / "results.json").read_text())[0]
    assert run.status is EvaluationRunStatus.INCOMPLETE
    assert result["score"] is None
    assert "PermanentProviderError" in result["provider_error"]


def test_resume_reuses_completed_result_without_provider_call(tmp_path):
    raw = _examples()[0]
    text = json.dumps(raw["expected_output"]) if not isinstance(raw["expected_output"], str) else raw["expected_output"]
    out = tmp_path / "resume"
    first = FakeModelProvider([ModelResponse(text=text)])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=first, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    second = FakeModelProvider([])
    rerun = run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=second, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1, resume=True,
    )
    assert rerun.status is EvaluationRunStatus.COMPLETED
    assert second.requests == []
