from pathlib import Path
import json

from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.providers.fake import FakeModelProvider
from adaptlab.evaluation.rag_smoke import freeze_validation_candidate, run_validation_rag_smoke, select_validation_smoke_examples
from adaptlab.evaluation.runner import load_benchmark_split
from adaptlab.domain.enums import Split

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/generated/v0.0"
RESULTS = ROOT / "artifacts/retrieval/m4/validation_bm25_candidates_v1/results_k10.json"
MANIFEST = ROOT / "artifacts/retrieval/m4/validation_bm25_candidates_v1/run_manifest.json"
PROMPT = ROOT / "configs/prompts/prompt_v1.yaml"


def test_freeze_validation_candidate_matches_persisted_k10():
    artifact = freeze_validation_candidate(results_path=RESULTS, manifest_path=MANIFEST)
    assert len(artifact.entries) == 150
    assert artifact.retrieval_run_id.startswith("m4-validation-bm25-")


def test_smoke_selection_is_deterministic_and_balanced():
    examples = load_benchmark_split(BENCH, Split.validation)
    a = select_validation_smoke_examples(examples, count=24)
    b = select_validation_smoke_examples(reversed(examples), count=24)
    assert [x.example_id for x in a] == [x.example_id for x in b]
    assert len({x.task_family for x in a}) == 4


def test_fake_smoke_consumes_frozen_retrieval_scores_and_cache(tmp_path):
    responses = [ModelResponse(text="INSUFFICIENT_EVIDENCE") for _ in range(24)]
    p1 = FakeModelProvider(responses)
    out = tmp_path / "smoke"
    s1 = run_validation_rag_smoke(
        benchmark_dir=BENCH, prompt_config=PROMPT, retrieval_results_path=RESULTS,
        retrieval_manifest_path=MANIFEST, provider=p1, model_id="qwen3:8b", output_dir=out,
    )
    assert s1.selected_count == s1.successful_count == 24
    assert s1.provider_failure_count == 0
    assert len(p1.requests) == 24
    rows = json.loads((out / "results.json").read_text())
    assert all(row["retrieval_artifact_hash"] == s1.retrieval_artifact_hash for row in rows)
    assert all(row["retrieval_run_id"] == s1.retrieval_run_id for row in rows)
    assert all(row["provider_error"] is None for row in rows)

    p2 = FakeModelProvider([])
    s2 = run_validation_rag_smoke(
        benchmark_dir=BENCH, prompt_config=PROMPT, retrieval_results_path=RESULTS,
        retrieval_manifest_path=MANIFEST, provider=p2, model_id="qwen3:8b", output_dir=out,
    )
    assert s2.cache_hit_count == 24
    assert not p2.requests
