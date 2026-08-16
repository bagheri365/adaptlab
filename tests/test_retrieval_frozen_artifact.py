import json
from pathlib import Path

import pytest

from adaptlab.retrieval.frozen_artifact import (
    FrozenRetrievalArtifact,
    build_frozen_retrieval_artifact,
    freeze_canonical_retrieval_results,
    load_and_verify_frozen_retrieval_artifact,
)
from adaptlab.retrieval.schemas import RetrievalResult, RetrievalRunManifest

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "artifacts/retrieval/m4/primary_test_bm25_v1"


def _load_inputs():
    results = tuple(RetrievalResult.from_dict(x) for x in json.loads((PRIMARY / "results.json").read_text()))
    manifest = RetrievalRunManifest.from_dict(json.loads((PRIMARY / "run_manifest.json").read_text()))
    return results, manifest


def test_build_freezes_all_400_results_and_behavior_bypass():
    results, manifest = _load_inputs()
    artifact = build_frozen_retrieval_artifact(results=results, manifest=manifest)
    assert len(artifact.entries) == 400
    by_id = {x.example_id: x for x in artifact.entries}
    for result in results:
        entry = by_id[result.example_id]
        if result.retrieval_eligible:
            assert entry.chunk_ids == result.candidate_chunk_ids
            assert entry.ranks == result.candidate_ranks
            assert entry.scores == result.candidate_scores
        else:
            assert entry.chunk_ids == ()
            assert entry.ranks == ()
            assert entry.scores == ()


def test_artifact_is_deterministic_and_bound_to_source_results():
    results, manifest = _load_inputs()
    a = build_frozen_retrieval_artifact(results=results, manifest=manifest)
    b = build_frozen_retrieval_artifact(results=reversed(results), manifest=manifest)
    assert a.to_json_bytes() == b.to_json_bytes()
    assert a.source_results_hash == manifest.result_hashes["canonical"]


def test_round_trip_verifies_hash(tmp_path):
    out = tmp_path / "canonical_retrieval.json"
    artifact = freeze_canonical_retrieval_results(
        results_path=PRIMARY / "results.json",
        manifest_path=PRIMARY / "run_manifest.json",
        output_path=out,
    )
    loaded = load_and_verify_frozen_retrieval_artifact(out)
    assert loaded == artifact


def test_tampering_changes_identity_and_is_rejected():
    results, manifest = _load_inputs()
    artifact = build_frozen_retrieval_artifact(results=results, manifest=manifest)
    data = artifact.to_dict()
    eligible = next(x for x in data["entries"] if x["retrieval_eligible"])
    eligible["chunk_ids"] = ("tampered-chunk",) + tuple(eligible["chunk_ids"][1:])
    with pytest.raises(ValueError, match="retrieval_artifact_hash"):
        FrozenRetrievalArtifact.from_dict(data)


def test_manifest_mismatch_is_rejected():
    results, manifest = _load_inputs()
    bad = manifest.to_dict()
    bad["result_hashes"] = dict(bad["result_hashes"])
    bad["result_hashes"]["canonical"] = "0" * 64
    with pytest.raises(ValueError, match="canonical result hash"):
        build_frozen_retrieval_artifact(results=results, manifest=RetrievalRunManifest.from_dict(bad))
