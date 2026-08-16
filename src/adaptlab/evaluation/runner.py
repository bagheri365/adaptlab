"""End-to-end evaluation runner for frozen AdaptLab benchmarks."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split
from adaptlab.evaluation.cache import ExactRequestCache, InferenceCacheKey, ResultArtifactIdentity
from adaptlab.evaluation.errors import ProviderError, TransientProviderError
from adaptlab.evaluation.completeness import completeness_record, require_no_silent_drop
from adaptlab.evaluation.inputs import construct_model_input
from adaptlab.evaluation.metrics import AccuracyMetric, aggregate_metrics
from adaptlab.evaluation.prompts import PromptContract, load_prompt_contract
from adaptlab.evaluation.runtime import capture_ollama_runtime
from adaptlab.evaluation.provenance import capture_git_state, require_canonical_git_state
from adaptlab.evaluation.providers import ModelProvider, ModelRequest
from adaptlab.evaluation.providers.ollama import OllamaModelProvider
from adaptlab.evaluation.schemas import (
    AdaptationMethod,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunStatus,
)
from adaptlab.evaluation.scoring import NORMALIZER_VERSION, SCORER_VERSION, score_output

RESULTS_FILENAME = "results.json"
METRICS_FILENAME = "metrics.json"
SUMMARY_FILENAME = "summary.txt"
RUN_MANIFEST_FILENAME = "run_manifest.json"
CACHE_DIRNAME = "cache"
RESUME_STATE_FILENAME = "resume_state.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_frozen_benchmark(directory: Path) -> tuple[dict, str]:
    """Verify freeze marker plus integrity hashes needed by evaluation."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    freeze_path = directory / "benchmark_freeze.json"
    manifest = _read_json(manifest_path)
    freeze = _read_json(freeze_path)
    if freeze.get("ready") is not True or freeze.get("decision") != "V0_0_BENCHMARK_READY":
        raise ValueError("benchmark is not frozen/ready")
    if freeze.get("intended_git_tag") != "v0.0-benchmark":
        raise ValueError("benchmark freeze artifact does not reference v0.0-benchmark")
    if freeze.get("benchmark_version") != manifest.get("benchmark_version"):
        raise ValueError("benchmark freeze/manifest version mismatch")
    checks = {
        "world.json": "world_hash",
        "train.json": "train_hash",
        "validation.json": "validation_hash",
        "test.json": "test_hash",
        "sentinel.json": "sentinel_hash",
    }
    for filename, key in checks.items():
        expected = manifest.get(key)
        if not isinstance(expected, str) or _sha256(directory / filename) != expected:
            raise ValueError(f"benchmark artifact hash mismatch: {filename}")
    return manifest, _sha256(manifest_path)


def load_benchmark_split(directory: Path, split: Split) -> list[BenchmarkExample]:
    raw = _read_json(Path(directory) / f"{split.value}.json")
    return [BenchmarkExample.from_dict(item) for item in raw]


def load_chunks(directory: Path) -> list[DocumentChunk]:
    return [DocumentChunk.from_dict(item) for item in _read_json(Path(directory) / "chunks.json")]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_from_failure(example: BenchmarkExample, constructed, exc: ProviderError, retry_count: int) -> EvaluationResult:
    return EvaluationResult(
        example_id=example.example_id,
        split=example.split,
        task_family=example.task_family,
        difficulty=example.difficulty,
        behavior_type=example.behavior_type,
        knowledge_state=example.knowledge_state,
        evidence_status=example.evidence_status,
        split_type=example.split_type,
        input_hash=constructed.input_hash,
        model_input=constructed.model_input,
        raw_output=None,
        normalized_output=None,
        expected_output=example.expected_output,
        score=None,
        scoring_rule=example.scoring_rule,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        provider_error=f"{type(exc).__name__}: {exc}",
        retry_count=retry_count,
    )


def _result_from_response(example: BenchmarkExample, constructed, response, retry_count: int) -> EvaluationResult:
    scored = score_output(example, response.text)
    return EvaluationResult(
        example_id=example.example_id, split=example.split, task_family=example.task_family,
        difficulty=example.difficulty, behavior_type=example.behavior_type, knowledge_state=example.knowledge_state,
        evidence_status=example.evidence_status, split_type=example.split_type, input_hash=constructed.input_hash,
        model_input=constructed.model_input, raw_output=response.text, normalized_output=scored.normalized_output,
        expected_output=example.expected_output, score=scored.score, scoring_rule=example.scoring_rule,
        latency_ms=response.latency_ms, input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        provider_error=None, retry_count=retry_count,
    )


def _evaluate_one(*, example: BenchmarkExample, constructed, provider: ModelProvider, temperature: float, max_tokens: int, seed: int | None, max_retries: int):
    request = ModelRequest(
        system_prompt=constructed.model_input.system,
        user_prompt=constructed.model_input.user,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    retry_count = 0
    while True:
        try:
            response = provider.generate(request)
            return _result_from_response(example, constructed, response, retry_count), response
        except TransientProviderError as exc:
            if retry_count >= max_retries:
                return _result_from_failure(example, constructed, exc, retry_count), None
            retry_count += 1
        except ProviderError as exc:
            return _result_from_failure(example, constructed, exc, retry_count), None


def _load_resume(path: Path) -> dict[str, EvaluationResult]:
    if not path.exists():
        return {}
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("resume results must be a JSON list")
    results = [EvaluationResult.from_dict(item) for item in data]
    if len({r.example_id for r in results}) != len(results):
        raise ValueError("resume results contain duplicate example IDs")
    return {r.example_id: r for r in results}


def _load_resume_state(path: Path) -> dict[str, ResultArtifactIdentity]:
    if not path.exists():
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("resume state must be a JSON object")
    try:
        return {example_id: ResultArtifactIdentity.from_dict(value) for example_id, value in data.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"corrupted resume state: {exc}") from exc


def _write_resume_state(path: Path, state: dict[str, ResultArtifactIdentity]) -> None:
    write_json(path, {key: state[key].to_dict() for key in sorted(state)})


def _rescore_preserved(example: BenchmarkExample, existing: EvaluationResult) -> EvaluationResult:
    if existing.raw_output is None or existing.provider_error is not None:
        raise ValueError("only successful preserved raw responses can be rescored")
    scored = score_output(example, existing.raw_output)
    return replace(existing, normalized_output=scored.normalized_output, score=scored.score)


def run_evaluation(
    *,
    benchmark_dir: Path,
    method: AdaptationMethod,
    model_id: str,
    provider: ModelProvider,
    model_revision: str | None = None,
    prompt_config: Path,
    output_dir: Path,
    split: Split = Split.test,
    limit: int | None = None,
    resume: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int | None = None,
    max_retries: int = 2,
    run_id: str | None = None,
    cache_dir: Path | None = None,
    expected_count: int | None = None,
    scorer_version: str = SCORER_VERSION,
    normalizer_version: str = NORMALIZER_VERSION,
    canonical: bool = False,
    allow_dirty_git: bool = False,
) -> EvaluationRun:
    """Execute one evaluation condition without mutating benchmark artifacts."""
    if method not in {AdaptationMethod.PROMPT, AdaptationMethod.ORACLE_CONTEXT}:
        raise ValueError("runner only supports PROMPT and ORACLE_CONTEXT")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    manifest, manifest_hash = verify_frozen_benchmark(benchmark_dir)
    git_state = capture_git_state(Path.cwd())
    if canonical:
        require_canonical_git_state(git_state, allow_dirty_git=allow_dirty_git)
    prompt = load_prompt_contract(prompt_config)
    ollama_runtime = (
        capture_ollama_runtime(provider)
        if isinstance(provider, OllamaModelProvider)
        else None
    )
    examples = sorted(load_benchmark_split(benchmark_dir, split), key=lambda e: e.example_id)
    if limit is not None:
        examples = examples[:limit]
    required_count = len(examples) if expected_count is None else expected_count
    if expected_count is not None and expected_count != len(examples):
        raise ValueError(
            f"expected_count {expected_count} does not match loaded example count {len(examples)}"
        )
    chunks = load_chunks(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or output_dir.name
    started_at = _now()
    prior = _load_resume(output_dir / RESULTS_FILENAME) if resume else {}
    prior_state = _load_resume_state(output_dir / RESUME_STATE_FILENAME) if resume else {}
    cache = ExactRequestCache(cache_dir or (output_dir / CACHE_DIRNAME))
    state: dict[str, ResultArtifactIdentity] = {}
    results: list[EvaluationResult] = []
    observed_model_revisions: set[str] = set()

    for example in examples:
        constructed = construct_model_input(example=example, method=method, prompt_contract=prompt, chunks=chunks)
        inference_key = InferenceCacheKey(
            benchmark_manifest_hash=manifest_hash,
            example_id=example.example_id,
            provider=provider.provider_name,
            ollama_base_url_policy=(
                ollama_runtime.ollama_base_url_policy if ollama_runtime is not None else None
            ),
            ollama_version=ollama_runtime.ollama_version if ollama_runtime is not None else None,
            model_id=model_id,
            model_tag=(ollama_runtime.model_tag if ollama_runtime is not None else None),
            model_digest=(ollama_runtime.model_digest if ollama_runtime is not None else None),
            model_revision=model_revision,
            prompt_hash=prompt.prompt_hash,
            method=method,
            temperature=temperature,
            context_length=(ollama_runtime.context_length if ollama_runtime is not None else None),
            max_tokens=max_tokens,
            seed=seed,
            stream=(ollama_runtime.stream if ollama_runtime is not None else None),
            think=(ollama_runtime.think if ollama_runtime is not None else None),
            input_hash=constructed.input_hash,
        )
        result_identity = ResultArtifactIdentity(
            inference=inference_key, scorer_version=scorer_version, normalizer_version=normalizer_version
        )
        existing = prior.get(example.example_id)
        existing_identity = prior_state.get(example.example_id)
        if existing is not None and existing_identity is not None and existing_identity.inference == inference_key and existing.raw_output is not None and existing.provider_error is None:
            if existing_identity == result_identity:
                result = existing
            else:
                # Scorer/normalizer changes invalidate the scored artifact, not the preserved raw response.
                result = _rescore_preserved(example, existing)
            results.append(result)
            state[example.example_id] = result_identity
            continue

        cached_response = cache.get(inference_key)
        if cached_response is not None:
            if cached_response.model_revision:
                observed_model_revisions.add(cached_response.model_revision)
            result = _result_from_response(example, constructed, cached_response, retry_count=0)
        else:
            result, response = _evaluate_one(
                example=example, constructed=constructed, provider=provider, temperature=temperature,
                max_tokens=max_tokens, seed=seed, max_retries=max_retries,
            )
            if response is not None:
                if response.model_revision:
                    observed_model_revisions.add(response.model_revision)
                cache.put(inference_key, response)
        results.append(result)
        state[example.example_id] = result_identity
        # Persist after every example so interrupted runs can resume safely.
        write_json(output_dir / RESULTS_FILENAME, [r.to_dict() for r in sorted(results, key=lambda r: r.example_id)])
        _write_resume_state(output_dir / RESUME_STATE_FILENAME, state)

    results.sort(key=lambda r: r.example_id)
    require_no_silent_drop(expected_count=required_count, actual_count=len(results))
    completeness = completeness_record(
        expected_count=required_count,
        completed_successful_responses=sum(r.provider_error is None and r.raw_output is not None for r in results),
    )
    metrics = aggregate_metrics(results)
    if expected_count is not None and not completeness.valid:
        metrics = replace(
            metrics,
            primary={
                **metrics.primary,
                "overall_accuracy": AccuracyMetric(
                    n=completeness.completed_successful_responses,
                    accuracy=None,
                ),
            },
        )
    write_json(output_dir / RESULTS_FILENAME, [r.to_dict() for r in results])
    _write_resume_state(output_dir / RESUME_STATE_FILENAME, state)
    write_json(output_dir / METRICS_FILENAME, metrics.to_dict())
    (output_dir / SUMMARY_FILENAME).write_text(metrics.human_summary(), encoding="utf-8")
    status = (
        EvaluationRunStatus.COMPLETED
        if all(r.provider_error is None for r in results) and completeness.valid
        else EvaluationRunStatus.INCOMPLETE
    )
    resolved_model_revision = model_revision
    if resolved_model_revision is None and len(observed_model_revisions) == 1:
        resolved_model_revision = next(iter(observed_model_revisions))
    if len(observed_model_revisions) > 1:
        raise ValueError("provider returned inconsistent model revisions within one run")
    run = EvaluationRun(
        run_id=run_id,
        benchmark_version=str(manifest["benchmark_version"]),
        benchmark_manifest_hash=manifest_hash,
        model_id=model_id,
        model_tag=(ollama_runtime.model_tag if ollama_runtime is not None else model_id),
        model_digest=(ollama_runtime.model_digest if ollama_runtime is not None else None),
        model_revision=resolved_model_revision,
        provider=provider.provider_name,
        ollama_version=(ollama_runtime.ollama_version if ollama_runtime is not None else None),
        ollama_base_url_policy=(ollama_runtime.ollama_base_url_policy if ollama_runtime is not None else None),
        adaptation_method=method,
        prompt_version=prompt.prompt_version,
        prompt_hash=prompt.prompt_hash,
        scorer_version=scorer_version,
        normalizer_version=normalizer_version,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        context_length=(ollama_runtime.context_length if ollama_runtime is not None else None),
        stream=(ollama_runtime.stream if ollama_runtime is not None else None),
        think=(ollama_runtime.think if ollama_runtime is not None else None),
        started_at=started_at,
        completed_at=_now(),
        status=status,
        expected_count=completeness.expected_count,
        completed_successful_responses=completeness.completed_successful_responses,
        completeness_valid=completeness.valid,
    )
    result_hashes = {RESULTS_FILENAME: _sha256(output_dir / RESULTS_FILENAME)}
    metric_hashes = {
        METRICS_FILENAME: _sha256(output_dir / METRICS_FILENAME),
        SUMMARY_FILENAME: _sha256(output_dir / SUMMARY_FILENAME),
    }
    provider_error_count = sum(r.provider_error is not None for r in results)
    completed_count = completeness.completed_successful_responses
    run_manifest = run.to_dict()
    run_manifest.update({
        "timestamp": run.completed_at,
        "git_commit_sha": git_state.commit_sha,
        "git_dirty": git_state.dirty,
        "git_state_available": git_state.available,
        "git_provenance_limitation": git_state.limitation,
        "canonical": canonical,
        "dirty_git_override": bool(allow_dirty_git),
        "seed_policy": "UNSPECIFIED" if seed is None else f"FIXED_REQUEST_SEED:{seed}",
        "example_count": len(examples),
        "completed_count": completed_count,
        "completed_successful_responses": completed_count,
        "expected_count": completeness.expected_count,
        "completeness_valid": completeness.valid,
        "canonical_accuracy_emitted": expected_count is not None and completeness.valid,
        "provider_error_count": provider_error_count,
        "result_hashes": result_hashes,
        "metric_hashes": metric_hashes,
        "model_revision_limitation": (
            None if resolved_model_revision is not None
            else "Provider/model exposed only an unresolved or mutable model alias; immutable revision unavailable."
        ),
        "ollama_model_digest_limitation": (
            None if (ollama_runtime is None or ollama_runtime.model_digest is not None)
            else "Local Ollama registry did not expose a digest for the requested model tag."
        ),
        "inference_determinism_claimed": False,
    })
    write_json(output_dir / RUN_MANIFEST_FILENAME, run_manifest)
    return run
