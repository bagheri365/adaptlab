import hashlib
import json
from pathlib import Path

from adaptlab.benchmark.build import build_prototype_fixture
from adaptlab.benchmark.io import ARTIFACT_FILENAMES, canonical_json_bytes


def test_build_pipeline_writes_all_artifacts_and_valid_manifest(tmp_path: Path) -> None:
    result = build_prototype_fixture(1729, tmp_path)

    expected_files = {*ARTIFACT_FILENAMES, "manifest.json"}
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    assert result.validation.passed
    assert result.manifest["generation_seed"] == 1729
    assert result.manifest["fact_count"] == len(result.world.facts)
    assert result.manifest["document_count"] == len(result.documents)
    assert result.manifest["chunk_count"] == len(result.chunks)
    assert result.manifest["example_count"] == len(result.examples)
    assert set(result.manifest["hashes"]) == set(ARTIFACT_FILENAMES)
    assert "manifest.json" not in result.manifest["hashes"]

    for filename in ARTIFACT_FILENAMES:
        actual = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert result.manifest["hashes"][filename] == actual


def test_build_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_prototype_fixture(1729, a)
    build_prototype_fixture(1729, b)

    for filename in (*ARTIFACT_FILENAMES, "manifest.json"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()


def test_serialized_collections_are_canonically_sorted(tmp_path: Path) -> None:
    build_prototype_fixture(1729, tmp_path)

    world = json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))
    documents = json.loads((tmp_path / "documents.json").read_text(encoding="utf-8"))
    chunks = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    examples = json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))

    assert [item["record_id"] for item in world["facts"]] == sorted(
        item["record_id"] for item in world["facts"]
    )
    assert [item["document_id"] for item in documents] == sorted(
        item["document_id"] for item in documents
    )
    assert [item["chunk_id"] for item in chunks] == sorted(
        item["chunk_id"] for item in chunks
    )
    assert [item["example_id"] for item in examples] == sorted(
        item["example_id"] for item in examples
    )


def test_canonical_json_has_stable_key_order_and_utf8() -> None:
    data = canonical_json_bytes({"z": "Nimbus ☁", "a": 1})
    assert data.decode("utf-8") == '{\n  "a": 1,\n  "z": "Nimbus ☁"\n}\n'


def test_different_seed_builds_valid_fixture(tmp_path: Path) -> None:
    result = build_prototype_fixture(1730, tmp_path)
    assert result.validation.passed
    assert result.world.generation_seed == 1730
