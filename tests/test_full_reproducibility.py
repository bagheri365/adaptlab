"""Full v0.0 benchmark reproducibility integration tests.

These tests are intentionally heavier than the normal unit suite because a single
assertion may build the complete 850-example candidate benchmark more than once.
Run them explicitly with ``pytest -m full_reproducibility``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptlab.benchmark.build import build_full_benchmark


CONFIG = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


@pytest.fixture(scope="module")
def canonical_build_pair(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("full-repro")
    directory_a = root / "a"
    directory_b = root / "b"
    result_a = build_full_benchmark(CONFIG, directory_a)
    result_b = build_full_benchmark(CONFIG, directory_b)
    return result_a, result_b, directory_a, directory_b


@pytest.mark.full_reproducibility
def test_same_config_and_seed_reproduce_every_full_artifact_byte_for_byte(
    canonical_build_pair,
) -> None:
    result_a, result_b, directory_a, directory_b = canonical_build_pair

    files_a = _relative_files(directory_a)
    files_b = _relative_files(directory_b)
    assert files_a == files_b

    # This covers world, corpus, all three primary splits, sentinel, nested
    # training subsets, holdout artifacts, every deterministic audit artifact,
    # and the preliminary manifest.
    for relative in files_a:
        path_a = directory_a / relative
        path_b = directory_b / relative
        assert path_a.read_bytes() == path_b.read_bytes(), f"bytes differ for {relative}"
        assert _sha256(path_a) == _sha256(path_b), f"hash differs for {relative}"

    assert result_a.manifest == result_b.manifest
    assert result_a.manifest["artifact_hashes"] == result_b.manifest["artifact_hashes"]


@pytest.mark.full_reproducibility
def test_manifest_hashes_match_exact_full_artifact_bytes(canonical_build_pair) -> None:
    result_a, _, directory_a, _ = canonical_build_pair

    for relative_name, expected_hash in result_a.manifest["artifact_hashes"].items():
        assert _sha256(directory_a / relative_name) == expected_hash

    # The manifest remains outside its own artifact hash map.
    assert "preliminary_manifest.json" not in result_a.manifest["artifact_hashes"]


@pytest.mark.full_reproducibility
def test_full_serialization_has_explicit_canonical_collection_ordering(
    canonical_build_pair,
) -> None:
    _, _, directory_a, _ = canonical_build_pair

    world = json.loads((directory_a / "world.json").read_text(encoding="utf-8"))
    assert [item["record_id"] for item in world["facts"]] == sorted(
        item["record_id"] for item in world["facts"]
    )

    documents = json.loads((directory_a / "documents.json").read_text(encoding="utf-8"))
    assert [item["document_id"] for item in documents] == sorted(
        item["document_id"] for item in documents
    )

    chunks = json.loads((directory_a / "chunks.json").read_text(encoding="utf-8"))
    assert [item["chunk_id"] for item in chunks] == sorted(item["chunk_id"] for item in chunks)

    for filename in ("train.json", "validation.json", "test.json", "sentinel.json"):
        examples = json.loads((directory_a / filename).read_text(encoding="utf-8"))
        assert [item["example_id"] for item in examples] == sorted(
            item["example_id"] for item in examples
        )

    subsets = json.loads((directory_a / "training_subsets.json").read_text(encoding="utf-8"))
    for subset in subsets["subsets"].values():
        assert [item["example_id"] for item in subset] == sorted(
            item["example_id"] for item in subset
        )


@pytest.mark.full_reproducibility
def test_different_seed_preserves_non_leakage_integrity_invariants(tmp_path: Path) -> None:
    config_text = CONFIG.read_text(encoding="utf-8")
    different_config = tmp_path / "benchmark_seed_1730.yaml"
    different_config.write_text(
        config_text.replace("generation_seed: 1729", "generation_seed: 1730", 1),
        encoding="utf-8",
    )

    result = build_full_benchmark(different_config, tmp_path / "different-seed")

    assert result.world.generation_seed == 1730
    assert result.answer_validation.passed
    assert result.holdout_validation.passed
    assert result.sentinel_validation.passed
    assert result.manifest["counts"]["train"] == 300
    assert result.manifest["counts"]["validation"] == 150
    assert result.manifest["counts"]["test"] == 400
    assert result.manifest["counts"]["sentinel"] == 100


@pytest.mark.full_reproducibility
def test_different_seed_produces_a_freeze_valid_full_benchmark(tmp_path: Path) -> None:
    config_text = CONFIG.read_text(encoding="utf-8")
    different_config = tmp_path / "benchmark_seed_1731.yaml"
    different_config.write_text(
        config_text.replace("generation_seed: 1729", "generation_seed: 1731", 1),
        encoding="utf-8",
    )

    result = build_full_benchmark(different_config, tmp_path / "different-seed-validity")
    assert result.passed
