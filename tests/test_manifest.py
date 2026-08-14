from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptlab.benchmark.build import build_full_benchmark
from adaptlab.benchmark.manifest import FinalManifestError, generate_final_benchmark_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark_v0.0.yaml"


def _prepare_candidate(tmp_path: Path) -> Path:
    out = tmp_path / "candidate"
    build_full_benchmark(CONFIG, out)
    # Exercise final-manifest mechanics with synthetic passing gates; the
    # canonical repository state remains blocked and pending human review.
    prelim_path = out / "preliminary_manifest.json"
    prelim = json.loads(prelim_path.read_text(encoding="utf-8"))
    prelim["candidate_status"] = "VALID_CANDIDATE"
    prelim["blocking_issue_count"] = 0
    prelim["blockers"] = []
    prelim_path.write_text(json.dumps(prelim, sort_keys=True), encoding="utf-8")
    target = out / "audits" / "human_audit.json"
    target.write_text(json.dumps({
        "audit_version": "2",
        "sample_size": 50,
        "summary": {"passed": 50, "failed": 0, "correction_required": 0, "pending_human_review": 0},
        "reviews": [],
    }, sort_keys=True), encoding="utf-8")
    return out


def test_final_manifest_contains_required_provenance_and_no_self_hash(tmp_path: Path) -> None:
    out = _prepare_candidate(tmp_path)
    manifest = generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)

    assert manifest["train_count"] == 300
    assert manifest["validation_count"] == 150
    assert manifest["test_count"] == 400
    assert manifest["generalization_sentinel_count"] == 100
    assert manifest["corpus_chunk_count"] == 180
    assert set(manifest["training_subset_hashes"]) == {
        "train_050", "train_100", "train_200", "train_300"
    }
    assert "human_audit.json" in manifest["audit_artifact_hashes"]
    assert manifest["human_audit_artifact_hash"] == manifest["audit_artifact_hashes"]["human_audit.json"]
    assert manifest["split_policy_version"]
    assert "manifest_hash" not in manifest
    assert "self_hash" not in manifest
    assert "final_manifest_hash" not in manifest

    persisted = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_final_manifest_is_deterministic(tmp_path: Path) -> None:
    out = _prepare_candidate(tmp_path)
    first = generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)
    first_bytes = (out / "manifest.json").read_bytes()
    second = generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)
    assert second == first
    assert (out / "manifest.json").read_bytes() == first_bytes


def test_final_manifest_rejects_blocked_candidate(tmp_path: Path) -> None:
    out = _prepare_candidate(tmp_path)
    path = out / "preliminary_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["candidate_status"] = "BLOCKED"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FinalManifestError, match="not VALID_CANDIDATE"):
        generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)


def test_final_manifest_rejects_failed_human_audit(tmp_path: Path) -> None:
    out = _prepare_candidate(tmp_path)
    path = out / "audits" / "human_audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["summary"]["failed"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FinalManifestError, match="human audit"):
        generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)


def test_final_manifest_rejects_pending_human_audit(tmp_path: Path) -> None:
    out = _prepare_candidate(tmp_path)
    path = out / "audits" / "human_audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["summary"]["passed"] = 0
    data["summary"]["pending_human_review"] = 50
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FinalManifestError, match="pending human review"):
        generate_final_benchmark_manifest(repo_root=ROOT, output_dir=out)
