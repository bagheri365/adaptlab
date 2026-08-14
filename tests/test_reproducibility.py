"""Milestone-blocking byte-for-byte reproducibility tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from adaptlab.benchmark.build import build_prototype_fixture
from adaptlab.benchmark.io import ARTIFACT_FILENAMES, write_benchmark_artifacts
from adaptlab.domain.world import NimbusWorld

ALL_FIXTURE_FILES = (*ARTIFACT_FILENAMES, "manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_1729_reproduces_every_artifact_byte_for_byte_and_hash_for_hash(
    tmp_path: Path,
) -> None:
    """The canonical seed must reproduce the actual serialized fixture bytes."""

    directory_a = tmp_path / "a"
    directory_b = tmp_path / "b"

    result_a = build_prototype_fixture(1729, directory_a)
    result_b = build_prototype_fixture(1729, directory_b)

    for filename in ALL_FIXTURE_FILES:
        bytes_a = (directory_a / filename).read_bytes()
        bytes_b = (directory_b / filename).read_bytes()
        assert bytes_a == bytes_b, f"serialized bytes differ for {filename}"
        assert _sha256(directory_a / filename) == _sha256(directory_b / filename)

    # The manifest's artifact hashes must themselves be identical and must match
    # the exact bytes written for every reproducibility-hashed artifact.
    assert result_a.manifest["hashes"] == result_b.manifest["hashes"]
    for filename in ARTIFACT_FILENAMES:
        expected_hash = _sha256(directory_a / filename)
        assert result_a.manifest["hashes"][filename] == expected_hash
        assert result_b.manifest["hashes"][filename] == expected_hash


def test_different_seed_still_builds_a_valid_fixture(tmp_path: Path) -> None:
    result = build_prototype_fixture(1730, tmp_path / "different-seed")

    assert result.validation.passed
    assert result.world.generation_seed == 1730
    assert result.manifest["generation_seed"] == 1730
    for filename in ALL_FIXTURE_FILES:
        assert (tmp_path / "different-seed" / filename).is_file()


def test_artifact_writer_enforces_canonical_order_independent_of_input_order(
    tmp_path: Path,
) -> None:
    """Reversing every collection must not change serialized artifact bytes."""

    canonical_dir = tmp_path / "canonical"
    reordered_dir = tmp_path / "reordered"
    result = build_prototype_fixture(1729, canonical_dir)

    reversed_world = NimbusWorld(
        world_schema_version=result.world.world_schema_version,
        generation_seed=result.world.generation_seed,
        facts=tuple(reversed(result.world.facts)),
    )
    reordered_hashes = write_benchmark_artifacts(
        reordered_dir,
        reversed_world,
        reversed(result.documents),
        reversed(result.chunks),
        reversed(result.examples),
    )

    for filename in ARTIFACT_FILENAMES:
        assert (reordered_dir / filename).read_bytes() == (canonical_dir / filename).read_bytes()
        assert reordered_hashes[filename] == result.manifest["hashes"][filename]
