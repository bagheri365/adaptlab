from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptlab.domain.enums import Split
from adaptlab.evaluation.provenance import GitState, require_canonical_git_state
from adaptlab.evaluation.providers import FakeModelProvider, ModelResponse, OllamaModelProvider
from adaptlab.evaluation.providers.ollama import _HttpResult
from adaptlab.evaluation.runner import run_evaluation
from adaptlab.evaluation.runtime import OllamaRuntimeProvenance
from adaptlab.evaluation.schemas import AdaptationMethod

BENCHMARK = Path("data/generated/v0.0")
PROMPT = Path("configs/prompts/prompt_v1.yaml")


def _expected_text(item: dict) -> str:
    value = item["expected_output"]
    return json.dumps(value) if not isinstance(value, str) else value


def _ollama_body(text: str, *, model: str = "fake-model") -> bytes:
    payload = {
        "model": model,
        "message": {"role": "assistant", "content": text},
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 3,
    }
    return json.dumps(payload).encode("utf-8")


def test_canonical_git_requires_available_clean_state():
    with pytest.raises(ValueError, match="available Git provenance"):
        require_canonical_git_state(GitState(None, None, False, "archive"), allow_dirty_git=False)
    with pytest.raises(ValueError, match="clean Git"):
        require_canonical_git_state(GitState("a" * 40, True, True), allow_dirty_git=False)
    require_canonical_git_state(GitState("a" * 40, True, True), allow_dirty_git=True)
    require_canonical_git_state(GitState("a" * 40, False, True), allow_dirty_git=False)


def test_run_manifest_contains_canonical_provenance_and_hashes(tmp_path, monkeypatch):
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(
        runner,
        "capture_git_state",
        lambda _path: GitState("1" * 40, False, True),
    )
    monkeypatch.setattr(
        runner,
        "capture_ollama_runtime",
        lambda _provider: OllamaRuntimeProvenance(
            ollama_version="0.32.6",
            ollama_base_url_policy="http://localhost:11434",
            model_tag="fake-model",
            model_digest="2" * 64,
            context_length=4096,
            stream=False,
            think=False,
        ),
    )
    raw = json.loads((BENCHMARK / "validation.json").read_text())[:2]
    responses = iter(raw)

    def transport(_request, _timeout):
        item = next(responses)
        return _HttpResult(status=200, headers={}, body=_ollama_body(_expected_text(item), model="fake-model"))

    provider = OllamaModelProvider(
        model_id="fake-model",
        base_url="http://localhost:11434",
        context_length=4096,
        think=False,
        stream=False,
        transport=transport,
        max_retries=0,
    )
    out = tmp_path / "run"
    run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="fake-alias",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=2,
        seed=7,
        canonical=True,
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["git_commit_sha"] == "1" * 40
    assert manifest["git_dirty"] is False
    assert manifest["canonical"] is True
    assert manifest["ollama_version"] == "0.32.6"
    assert manifest["ollama_base_url_policy"] == "http://localhost:11434"
    assert manifest["model_tag"] == "fake-model"
    assert manifest["model_digest"] == "2" * 64
    assert manifest["context_length"] == 4096
    assert manifest["stream"] is False
    assert manifest["think"] is False
    assert manifest["model_revision"] is None
    assert "unresolved or mutable model alias" in manifest["model_revision_limitation"]
    assert manifest["seed_policy"] == "FIXED_REQUEST_SEED:7"
    assert manifest["example_count"] == 2
    assert manifest["completed_count"] == 2
    assert manifest["provider_error_count"] == 0
    assert manifest["inference_determinism_claimed"] is False
    assert manifest["result_hashes"]["results.json"] == hashlib.sha256((out / "results.json").read_bytes()).hexdigest()
    assert manifest["metric_hashes"]["metrics.json"] == hashlib.sha256((out / "metrics.json").read_bytes()).hexdigest()
    assert manifest["metric_hashes"]["summary.txt"] == hashlib.sha256((out / "summary.txt").read_bytes()).hexdigest()


def test_mutable_model_alias_limitation_is_explicit(tmp_path, monkeypatch):
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(runner, "capture_git_state", lambda _path: GitState("2" * 40, False, True))
    monkeypatch.setattr(
        runner,
        "capture_ollama_runtime",
        lambda _provider: OllamaRuntimeProvenance(
            ollama_version=None,
            ollama_base_url_policy="http://localhost:11434",
            model_tag="mutable-alias",
            model_digest=None,
            context_length=4096,
            stream=False,
            think=False,
        ),
    )
    raw = json.loads((BENCHMARK / "validation.json").read_text())[0]
    provider = OllamaModelProvider(
        model_id="mutable-alias",
        base_url="http://localhost:11434",
        context_length=4096,
        think=False,
        stream=False,
        transport=lambda _request, _timeout: _HttpResult(
            status=200,
            headers={},
            body=_ollama_body(_expected_text(raw), model="mutable-alias"),
        ),
        max_retries=0,
    )
    out = tmp_path / "run"
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="mutable-alias",
        provider=provider, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["model_revision"] is None
    assert "mutable model alias" in manifest["model_revision_limitation"]
    assert manifest["inference_determinism_claimed"] is False


def test_missing_ollama_runtime_metadata_is_recorded_as_null(tmp_path, monkeypatch):
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(
        runner,
        "capture_git_state",
        lambda _path: GitState("4" * 40, False, True),
    )
    monkeypatch.setattr(
        runner,
        "capture_ollama_runtime",
        lambda _provider: OllamaRuntimeProvenance(
            ollama_version=None,
            ollama_base_url_policy="http://localhost:11434",
            model_tag="qwen3:8b",
            model_digest=None,
            context_length=4096,
            stream=False,
            think=False,
        ),
    )
    raw = json.loads((BENCHMARK / "validation.json").read_text())[0]
    provider = OllamaModelProvider(
        model_id="qwen3:8b",
        base_url="http://localhost:11434",
        context_length=4096,
        think=False,
        stream=False,
        transport=lambda _request, _timeout: _HttpResult(
            status=200,
            headers={},
            body=_ollama_body(_expected_text(raw), model="qwen3:8b"),
        ),
        max_retries=0,
    )
    out = tmp_path / "run"
    run_evaluation(
        benchmark_dir=BENCHMARK,
        method=AdaptationMethod.PROMPT,
        model_id="qwen3:8b",
        provider=provider,
        prompt_config=PROMPT,
        output_dir=out,
        split=Split.validation,
        limit=1,
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["ollama_version"] is None
    assert manifest["model_digest"] is None
    assert manifest["ollama_base_url_policy"] == "http://localhost:11434"


def test_inconsistent_provider_revisions_are_rejected(tmp_path, monkeypatch):
    import adaptlab.evaluation.runner as runner

    monkeypatch.setattr(runner, "capture_git_state", lambda _path: GitState("3" * 40, False, True))
    raw = json.loads((BENCHMARK / "validation.json").read_text())[:2]
    provider = FakeModelProvider([
        ModelResponse(text=_expected_text(raw[0]), model_revision="rev-a"),
        ModelResponse(text=_expected_text(raw[1]), model_revision="rev-b"),
    ])
    with pytest.raises(ValueError, match="inconsistent model revisions"):
        run_evaluation(
            benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="alias",
            provider=provider, prompt_config=PROMPT, output_dir=tmp_path / "run",
            split=Split.validation, limit=2,
        )
