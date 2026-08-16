from pathlib import Path
import json

from adaptlab.retrieval.validation_run import run_validation_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_validation_candidate_run_is_validation_only_and_deterministic(tmp_path: Path):
    kwargs = dict(
        validation_path=ROOT / "data/generated/v0.0/validation.json",
        chunks_path=ROOT / "data/generated/v0.0/chunks.json",
        benchmark_manifest_path=ROOT / "data/generated/v0.0/manifest.json",
    )
    first = run_validation_candidates(output_dir=tmp_path / "a", **kwargs)
    second = run_validation_candidates(output_dir=tmp_path / "b", **kwargs)
    assert first == second
    assert first["split"] == "validation"
    assert first["example_count"] == 150
    assert first["eligible_count"] == 112
    assert set(first["candidate_all_required_gold_at_k"]) == {"1", "3", "5", "10"}
    assert first["selected_top_k"] in {1, 3, 5, 10}
    assert (tmp_path / "a/run_manifest.json").read_bytes() == (tmp_path / "b/run_manifest.json").read_bytes()


def test_selection_decision_records_validation_retrieval_only(tmp_path: Path):
    run_validation_candidates(
        validation_path=ROOT / "data/generated/v0.0/validation.json",
        chunks_path=ROOT / "data/generated/v0.0/chunks.json",
        benchmark_manifest_path=ROOT / "data/generated/v0.0/manifest.json",
        output_dir=tmp_path,
    )
    decision = json.loads((tmp_path / "top_k_selection_decision.json").read_text())
    assert decision["selection_input"] == "validation_retrieval_metrics_only"
    assert [x["top_k"] for x in decision["candidate_metrics"]] == [1, 3, 5, 10]
