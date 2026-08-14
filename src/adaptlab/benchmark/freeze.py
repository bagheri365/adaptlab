"""Derive benchmark freeze readiness from current machine-readable gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FREEZE_ARTIFACT_VERSION = "2"
READY = "V0_0_BENCHMARK_READY"
NOT_READY = "V0_0_BENCHMARK_NOT_READY"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pending_human_review_count(human: dict[str, Any]) -> int:
    summary = human.get("summary", {})
    pending = summary.get("pending_human_review")
    if pending is not None:
        return int(pending)

    reviews = human.get("reviews", [])
    if not isinstance(reviews, list):
        return 0
    return sum(
        1
        for review in reviews
        if isinstance(review, dict) and review.get("review_status", "PENDING_HUMAN_REVIEW") == "PENDING_HUMAN_REVIEW"
    )


def derive_freeze_artifact(output_dir: Path) -> dict[str, Any]:
    """Build a truthful freeze decision from the current candidate and human-audit state."""
    output_dir = Path(output_dir)
    preliminary_path = output_dir / "preliminary_manifest.json"
    human_path = output_dir / "audits" / "human_audit.json"
    if not preliminary_path.is_file():
        raise FileNotFoundError(preliminary_path)
    if not human_path.is_file():
        raise FileNotFoundError(human_path)

    preliminary = _read_json(preliminary_path)
    human = _read_json(human_path)
    summary = human.get("summary", {})
    pending = _pending_human_review_count(human)
    failed = int(summary.get("failed", 0))
    corrections = int(summary.get("correction_required", 0))
    blockers = list(preliminary.get("blockers", []))
    candidate_valid = preliminary.get("candidate_status") == "VALID_CANDIDATE" and not blockers
    human_complete = pending == 0 and failed == 0 and corrections == 0
    ready = candidate_valid and human_complete

    reasons: list[str] = []
    if blockers:
        reasons.append(f"{len(blockers)} blocking benchmark issue(s) remain")
    if not candidate_valid and not blockers:
        reasons.append("candidate status is not VALID_CANDIDATE")
    if pending:
        reasons.append(f"{pending} human-audit records remain pending")
    if failed:
        reasons.append(f"{failed} human-audit record(s) failed")
    if corrections:
        reasons.append(f"{corrections} human-audit correction(s) are required")

    return {
        "artifact_type": "benchmark_freeze_decision",
        "artifact_version": FREEZE_ARTIFACT_VERSION,
        "benchmark_name": preliminary.get("benchmark_name"),
        "benchmark_version": preliminary.get("benchmark_version"),
        "candidate_version": preliminary.get("candidate_version"),
        "generation_seed": preliminary.get("generation_seed"),
        "pending_human_review": pending,
        "decision": READY if ready else NOT_READY,
        "ready": ready,
        "candidate_status": preliminary.get("candidate_status"),
        "blocking_issue_count": len(blockers),
        "human_audit": {
            "sample_size": human.get("sample_size"),
            "passed": int(summary.get("passed", 0)),
            "failed": failed,
            "correction_required": corrections,
            "pending_human_review": pending,
            "complete": human_complete,
        },
        "reasons": reasons,
        "preliminary_manifest_path": "data/generated/v0.0/preliminary_manifest.json",
        "preliminary_manifest_sha256": _sha256(preliminary_path),
        "human_audit_path": "data/generated/v0.0/audits/human_audit.json",
        "human_audit_sha256": _sha256(human_path),
        "intended_git_tag": "v0.0-benchmark" if ready else None,
    }



def validate_freeze_artifact_sync(output_dir: Path) -> list[str]:
    """Return contradictions between the persisted freeze artifact and current gates."""
    output_dir = Path(output_dir)
    persisted_path = output_dir / "benchmark_freeze.json"
    if not persisted_path.is_file():
        return ["benchmark_freeze.json is missing"]
    persisted = _read_json(persisted_path)
    expected = derive_freeze_artifact(output_dir)
    fields = (
        "decision",
        "ready",
        "candidate_status",
        "blocking_issue_count",
        "pending_human_review",
        "intended_git_tag",
        "preliminary_manifest_sha256",
        "human_audit_sha256",
    )
    errors: list[str] = []
    for field in fields:
        if persisted.get(field) != expected.get(field):
            errors.append(
                f"benchmark_freeze.json field {field!r} is stale: "
                f"persisted={persisted.get(field)!r}, expected={expected.get(field)!r}"
            )
    if persisted.get("human_audit") != expected.get("human_audit"):
        errors.append("benchmark_freeze.json human_audit summary is stale")
    return errors

def write_freeze_artifact(output_dir: Path) -> dict[str, Any]:
    from adaptlab.benchmark.io import write_json

    artifact = derive_freeze_artifact(output_dir)
    write_json(Path(output_dir) / "benchmark_freeze.json", artifact)
    return artifact
