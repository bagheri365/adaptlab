from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptlab.benchmark.io import canonical_json_bytes
from adaptlab.domain.enums import Split
from adaptlab.evaluation.cache import ExactRequestCache, InferenceCacheKey, ResultArtifactIdentity
from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.providers.fake import FakeModelProvider
from adaptlab.evaluation.runner import run_evaluation
from adaptlab.evaluation.schemas import AdaptationMethod

BENCHMARK = Path("data/generated/v0.0")
PROMPT = Path("configs/prompts/prompt_v1.yaml")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def make_key(**overrides) -> InferenceCacheKey:
    values = dict(
        benchmark_manifest_hash=HASH_A,
        example_id="validation-0001",
        provider="fake",
        ollama_base_url_policy=None,
        ollama_version=None,
        model_id="model-a",
        model_tag="model-a",
        model_digest=None,
        model_revision="snapshot-1",
        prompt_hash=HASH_B,
        method=AdaptationMethod.PROMPT,
        temperature=0.0,
        context_length=None,
        max_tokens=256,
        seed=7,
        stream=None,
        think=None,
        input_hash=HASH_C,
    )
    values.update(overrides)
    return InferenceCacheKey(**values)


def first_expected_text() -> str:
    item = json.loads((BENCHMARK / "validation.json").read_text())[0]
    value = item["expected_output"]
    return value if isinstance(value, str) else json.dumps(value)


def test_cache_key_is_deterministic_and_exact_request_sensitive():
    key = make_key()
    assert key.request_hash == hashlib.sha256(key.to_json_bytes()).hexdigest()
    assert key.request_hash == make_key().request_hash
    variants = [
        make_key(provider="other"),
        make_key(ollama_base_url_policy="http://localhost:11435"),
        make_key(ollama_version="0.33.0"),
        make_key(model_id="model-b"),
        make_key(model_tag="model-b"),
        make_key(model_digest="d" * 64),
        make_key(model_revision="snapshot-2"),
        make_key(prompt_hash="d" * 64),
        make_key(method=AdaptationMethod.ORACLE_CONTEXT),
        make_key(temperature=0.2),
        make_key(context_length=8192),
        make_key(max_tokens=128),
        make_key(seed=8),
        make_key(stream=True),
        make_key(think=True),
        make_key(input_hash="e" * 64),
        make_key(example_id="validation-0002"),
        make_key(benchmark_manifest_hash="f" * 64),
    ]
    assert all(v.request_hash != key.request_hash for v in variants)


def test_result_identity_includes_scorer_and_normalizer_versions():
    key = make_key()
    a = ResultArtifactIdentity(key, "scorer-1", "normalizer-1")
    b = ResultArtifactIdentity(key, "scorer-2", "normalizer-1")
    c = ResultArtifactIdentity(key, "scorer-1", "normalizer-2")
    assert a != b and a != c


def test_cache_round_trip_and_corruption_fails_closed(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    key = make_key()
    response = ModelResponse(text="ALLOW", input_tokens=10, output_tokens=1, latency_ms=2.5, model_revision="snapshot-1", provider_metadata={"x": 1})
    assert cache.get(key) is None
    cache.put(key, response)
    assert cache.get(key) == response
    path = next((tmp_path / "cache").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["payload"]["response"]["text"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupted cache entry"):
        cache.get(key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_length", 8192),
        ("think", True),
        ("stream", True),
        ("seed", 99),
        ("temperature", 0.1),
        ("model_tag", "model-b"),
        ("model_digest", "d" * 64),
        ("ollama_version", "0.33.0"),
        ("ollama_base_url_policy", "http://localhost:11435"),
        ("input_hash", "e" * 64),
        ("prompt_hash", "f" * 64),
    ],
)
def test_cache_misses_when_request_affecting_fields_change(field, value):
    base = make_key()
    assert make_key(**{field: value}).request_hash != base.request_hash


def test_identical_request_hits_cache(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    key = make_key()
    response = ModelResponse(text="ALLOW")
    cache.put(key, response)
    assert cache.get(make_key()) == response


def test_legacy_cache_schema_version_is_rejected(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    key = make_key()
    response = ModelResponse(text="ALLOW")
    cache.put(key, response)
    path = next((tmp_path / "cache").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["payload"]["schema_version"] = "1"
    payload["payload_sha256"] = hashlib.sha256(canonical_json_bytes(payload["payload"])).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported cache schema version"):
        cache.get(key)


def test_runner_reuses_exact_cache_without_resume_flag(tmp_path):
    out = tmp_path / "run"
    first = FakeModelProvider([ModelResponse(text=first_expected_text())])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=first, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    second = FakeModelProvider([])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=second, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    assert second.requests == []


def test_changed_model_invalidates_cache_and_requires_provider(tmp_path):
    out = tmp_path / "run"
    first = FakeModelProvider([ModelResponse(text=first_expected_text())])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="model-a",
        provider=first, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    second = FakeModelProvider([ModelResponse(text=first_expected_text())])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="model-b",
        provider=second, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    assert len(second.requests) == 1


def test_corrected_oracle_input_cannot_reuse_invalid_oracle_cache_entry(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    invalid_oracle = make_key(
        method=AdaptationMethod.ORACLE_CONTEXT,
        input_hash="a" * 64,
    )
    corrected_oracle = make_key(
        method=AdaptationMethod.ORACLE_CONTEXT,
        input_hash="b" * 64,
    )
    cache.put(invalid_oracle, ModelResponse(text="OLD"))
    assert cache.get(corrected_oracle) is None


def test_resume_rescores_preserved_raw_response_when_scorer_version_changes(tmp_path):
    out = tmp_path / "run"
    first = FakeModelProvider([ModelResponse(text=first_expected_text())])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=first, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
        scorer_version="scorer-1",
    )
    before = json.loads((out / "results.json").read_text())[0]["raw_output"]
    second = FakeModelProvider([])
    run = run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=second, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
        resume=True, scorer_version="scorer-2",
    )
    after = json.loads((out / "results.json").read_text())[0]["raw_output"]
    assert second.requests == []
    assert before == after
    assert run.scorer_version == "scorer-2"
    state = json.loads((out / "resume_state.json").read_text())
    assert next(iter(state.values()))["scorer_version"] == "scorer-2"


def test_corrupted_cache_does_not_fall_back_to_provider(tmp_path):
    out = tmp_path / "run"
    first = FakeModelProvider([ModelResponse(text=first_expected_text())])
    run_evaluation(
        benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
        provider=first, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
    )
    cache_file = next((out / "cache").glob("*.json"))
    cache_file.write_text("{bad json")
    second = FakeModelProvider([ModelResponse(text="SHOULD_NOT_BE_USED")])
    with pytest.raises(ValueError, match="corrupted cache entry"):
        run_evaluation(
            benchmark_dir=BENCHMARK, method=AdaptationMethod.PROMPT, model_id="fake-model",
            provider=second, prompt_config=PROMPT, output_dir=out, split=Split.validation, limit=1,
        )
    assert second.requests == []


def make_rag_key(**overrides) -> InferenceCacheKey:
    values = dict(
        method=AdaptationMethod.RAG,
        retrieval_run_id="m4-primary-test-bm25-test",
        retrieval_artifact_hash="1" * 64,
        retriever_config_hash="2" * 64,
        retrieved_context_hash="3" * 64,
    )
    values.update(overrides)
    return make_key(**values)


def test_rag_cache_key_requires_retrieval_provenance():
    with pytest.raises(ValueError, match="requires retrieval"):
        make_key(method=AdaptationMethod.RAG)


def test_same_rag_input_and_same_retrieval_artifact_hits_cache(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    key = make_rag_key()
    response = ModelResponse(text="ALLOW")
    cache.put(key, response)
    assert cache.get(make_rag_key()) == response


def test_changed_rag_retrieved_chunk_set_causes_cache_miss():
    base = make_rag_key()
    changed = make_rag_key(input_hash="4" * 64, retrieved_context_hash="5" * 64)
    assert changed.request_hash != base.request_hash


def test_changed_rag_retrieval_artifact_hash_causes_cache_miss():
    base = make_rag_key()
    changed = make_rag_key(retrieval_artifact_hash="6" * 64)
    assert changed.request_hash != base.request_hash


def test_corrected_rag_input_cannot_reuse_stale_rag_response(tmp_path):
    cache = ExactRequestCache(tmp_path / "cache")
    stale = make_rag_key(input_hash="7" * 64, retrieved_context_hash="8" * 64)
    corrected = make_rag_key(input_hash="9" * 64, retrieved_context_hash="a" * 64)
    cache.put(stale, ModelResponse(text="STALE"))
    assert cache.get(corrected) is None


def test_rag_scorer_only_change_preserves_raw_inference_identity():
    key = make_rag_key()
    first = ResultArtifactIdentity(key, "scorer-1", "normalizer-1")
    rescored = ResultArtifactIdentity(key, "scorer-2", "normalizer-1")
    assert first.inference == rescored.inference
    assert first != rescored
