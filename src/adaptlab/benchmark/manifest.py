"""Final machine-readable manifest generation for the frozen-candidate benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json

MANIFEST_SCHEMA_VERSION = "1"
GENERATOR_VERSIONS = {
    "world_generator": "full-v0.0-world-v1",
    "corpus_generator": "full-v0.0-corpus-v1",
    "task_generator": "full-v0.0-tasks-v1",
    "sentinel_generator": "v0.0.0",
    "training_subset_generator": "1",
    "manifest_generator": MANIFEST_SCHEMA_VERSION,
}


class FinalManifestError(ValueError):
    """Raised when final-manifest preconditions are not satisfied."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalManifestError(f"required manifest input is missing: {path}") from exc


def _sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except FileNotFoundError as exc:
        raise FinalManifestError(f"required manifest input is missing: {path}") from exc


def _hash_named_payload(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def generate_final_benchmark_manifest(
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate the final v0.0 provenance manifest after all pre-freeze gates pass.

    The manifest deliberately never includes a hash of itself.  It is derived only
    from deterministic benchmark/config/audit artifacts already written to disk.
    """

    repo_root = Path(repo_root)
    output_dir = Path(output_dir)

    preliminary_path = output_dir / "preliminary_manifest.json"
    preliminary = _read_json(preliminary_path)
    if preliminary.get("candidate_status") != "VALID_CANDIDATE":
        raise FinalManifestError("candidate is not VALID_CANDIDATE")
    if preliminary.get("blocking_issue_count") != 0 or preliminary.get("blockers"):
        raise FinalManifestError("candidate still contains blocking issues")

    human_path = output_dir / "audits" / "human_audit.json"
    human = _read_json(human_path)
    human_summary = human.get("summary", {})
    if human_summary.get("failed", 0) != 0 or human_summary.get("correction_required", 0) != 0:
        raise FinalManifestError("human audit still contains failed or correction-required cases")
    if human_summary.get("pending_human_review", 0) != 0:
        raise FinalManifestError("human audit is incomplete: pending human review remains")

    required_artifacts = {
        "world": output_dir / "world.json",
        "documents": output_dir / "documents.json",
        "chunks": output_dir / "chunks.json",
        "train": output_dir / "train.json",
        "validation": output_dir / "validation.json",
        "test": output_dir / "test.json",
        "sentinel": output_dir / "sentinel.json",
        "training_subsets": output_dir / "training_subsets.json",
        "holdout_policy": output_dir / "holdout_policy.json",
    }
    artifact_hashes = {name: _sha256_file(path) for name, path in required_artifacts.items()}

    # Corpus provenance is the deterministic composition of its two serialized
    # source artifacts, avoiding dependence on filesystem concatenation details.
    corpus_hash = _hash_named_payload(
        {
            "documents_hash": artifact_hashes["documents"],
            "chunks_hash": artifact_hashes["chunks"],
        }
    )

    subsets = _read_json(required_artifacts["training_subsets"])
    subset_payloads = subsets.get("subsets", {})
    expected_subset_names = ("train_050", "train_100", "train_200", "train_300")
    if tuple(sorted(subset_payloads)) != tuple(sorted(expected_subset_names)):
        raise FinalManifestError("training subset artifact does not contain the expected four subsets")
    training_subset_hashes = {
        name: _hash_named_payload(subset_payloads[name]) for name in expected_subset_names
    }

    audit_dir = output_dir / "audits"
    required_audits = (
        "anti_confounding.json",
        "corpus_composition.json",
        "human_audit.json",
        "leakage.json",
        "lexical_overlap.json",
    )
    audit_artifact_hashes = {
        filename: _sha256_file(audit_dir / filename) for filename in required_audits
    }

    benchmark_config = repo_root / "configs" / "benchmark_v0.0.yaml"
    evaluation_policy = repo_root / "configs" / "evaluation_policy_v0.0.yaml"
    external_validity = repo_root / "configs" / "external_validity_v0.0.yaml"
    policy_hashes = {
        "evaluation_policy_v0.0.yaml": _sha256_file(evaluation_policy),
        "external_validity_v0.0.yaml": _sha256_file(external_validity),
    }

    holdout_policy = _read_json(required_artifacts["holdout_policy"])
    split_policy_version = holdout_policy.get("policy_version")
    if not split_policy_version:
        raise FinalManifestError("holdout policy is missing policy_version")

    counts = preliminary.get("counts", {})
    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_name": preliminary["benchmark_name"],
        "benchmark_version": preliminary["benchmark_version"],
        "candidate_version": preliminary.get("candidate_version"),
        "generation_seed": preliminary["generation_seed"],
        "world_schema_version": preliminary["world_schema_version"],
        "train_count": counts.get("train"),
        "validation_count": counts.get("validation"),
        "test_count": counts.get("test"),
        "generalization_sentinel_count": counts.get("sentinel"),
        "corpus_chunk_count": counts.get("chunks"),
        "config_hash": _sha256_file(benchmark_config),
        "world_hash": artifact_hashes["world"],
        "corpus_hash": corpus_hash,
        "train_hash": artifact_hashes["train"],
        "validation_hash": artifact_hashes["validation"],
        "test_hash": artifact_hashes["test"],
        "sentinel_hash": artifact_hashes["sentinel"],
        "training_subset_hashes": training_subset_hashes,
        "generator_versions": dict(GENERATOR_VERSIONS),
        "split_policy_version": split_policy_version,
        "audit_artifact_hashes": dict(sorted(audit_artifact_hashes.items())),
        "human_audit_artifact_hash": audit_artifact_hashes["human_audit.json"],
        "policy_hashes": dict(sorted(policy_hashes.items())),
        "preliminary_manifest_hash": _sha256_file(preliminary_path),
        "provenance": {
            "evaluation_policy": "configs/evaluation_policy_v0.0.yaml",
            "external_validity_policy": "configs/external_validity_v0.0.yaml",
            "holdout_policy": "data/generated/v0.0/holdout_policy.json",
            "human_audit": "data/generated/v0.0/audits/human_audit.json",
        },
    }

    # Explicitly guard against recursive self-hashing by contract.
    forbidden = {"manifest_hash", "self_hash", "final_manifest_hash"}
    if forbidden.intersection(manifest):
        raise AssertionError("final manifest must not contain its own hash")

    write_json(output_dir / "manifest.json", manifest)
    return manifest
