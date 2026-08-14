from __future__ import annotations

import json
from pathlib import Path

from adaptlab.benchmark.freeze import (
    NOT_READY,
    READY,
    derive_freeze_artifact,
    validate_freeze_artifact_sync,
    write_freeze_artifact,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_freeze_artifact_is_not_ready_when_blockers_exist(tmp_path: Path) -> None:
    _write_json(tmp_path / "preliminary_manifest.json", {
        "benchmark_name": "AdaptLab Nimbus Benchmark",
        "benchmark_version": "0.0.0",
        "candidate_version": "0.0.0-candidate.1",
        "generation_seed": 1729,
        "candidate_status": "BLOCKED",
        "blockers": ["semantic leakage"],
    })
    _write_json(tmp_path / "audits" / "human_audit.json", {
        "sample_size": 50,
        "summary": {"passed": 50, "failed": 0, "correction_required": 0, "pending_human_review": 0},
    })
    artifact = derive_freeze_artifact(tmp_path)
    assert artifact["decision"] == NOT_READY
    assert artifact["ready"] is False
    assert artifact["blocking_issue_count"] == 1
    assert artifact["pending_human_review"] == 0
    assert artifact["intended_git_tag"] is None


def test_freeze_artifact_is_not_ready_while_human_review_pending(tmp_path: Path) -> None:
    _write_json(tmp_path / "preliminary_manifest.json", {
        "benchmark_name": "AdaptLab Nimbus Benchmark",
        "benchmark_version": "0.0.0",
        "candidate_version": "0.0.0-candidate.1",
        "generation_seed": 1729,
        "candidate_status": "VALID_CANDIDATE",
        "blockers": [],
    })
    _write_json(tmp_path / "audits" / "human_audit.json", {
        "sample_size": 50,
        "summary": {"passed": 0, "failed": 0, "correction_required": 0, "pending_human_review": 50},
    })
    artifact = derive_freeze_artifact(tmp_path)
    assert artifact["decision"] == NOT_READY
    assert artifact["pending_human_review"] == 50
    assert artifact["human_audit"]["complete"] is False


def test_freeze_artifact_ready_only_when_all_current_gates_pass(tmp_path: Path) -> None:
    _write_json(tmp_path / "preliminary_manifest.json", {
        "benchmark_name": "AdaptLab Nimbus Benchmark",
        "benchmark_version": "0.0.0",
        "candidate_version": "0.0.0-candidate.1",
        "generation_seed": 1729,
        "candidate_status": "VALID_CANDIDATE",
        "blockers": [],
    })
    _write_json(tmp_path / "audits" / "human_audit.json", {
        "sample_size": 50,
        "summary": {"passed": 50, "failed": 0, "correction_required": 0, "pending_human_review": 0},
    })
    artifact = derive_freeze_artifact(tmp_path)
    assert artifact["decision"] == READY
    assert artifact["ready"] is True
    assert artifact["pending_human_review"] == 0
    assert artifact["intended_git_tag"] == "v0.0-benchmark"


def test_freeze_artifact_sync_rejects_stale_blocker_state(tmp_path: Path) -> None:
    _write_json(tmp_path / "preliminary_manifest.json", {
        "benchmark_name": "AdaptLab Nimbus Benchmark",
        "benchmark_version": "0.0.0",
        "candidate_version": "0.0.0-candidate.1",
        "generation_seed": 1729,
        "candidate_status": "VALID_CANDIDATE",
        "blockers": [],
    })
    _write_json(tmp_path / "audits" / "human_audit.json", {
        "sample_size": 50,
        "summary": {"passed": 0, "failed": 0, "correction_required": 0, "pending_human_review": 50},
    })
    write_freeze_artifact(tmp_path)
    freeze_path = tmp_path / "benchmark_freeze.json"
    stale = json.loads(freeze_path.read_text(encoding="utf-8"))
    stale["candidate_status"] = "BLOCKED"
    stale["blocking_issue_count"] = 31
    del stale["pending_human_review"]
    stale["reasons"] = ["stale semantic leakage"]
    freeze_path.write_text(json.dumps(stale, sort_keys=True), encoding="utf-8")

    errors = validate_freeze_artifact_sync(tmp_path)
    assert any("candidate_status" in error for error in errors)
    assert any("blocking_issue_count" in error for error in errors)
    assert any("pending_human_review" in error for error in errors)


def test_canonical_freeze_artifact_matches_current_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "generated" / "v0.0"
    freeze = json.loads((output_dir / "benchmark_freeze.json").read_text(encoding="utf-8"))
    assert freeze["pending_human_review"] == 0
    assert validate_freeze_artifact_sync(output_dir) == []
