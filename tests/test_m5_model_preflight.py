from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.m5.model_preflight import (
    CANONICAL_MLX_BASE_IDENTITY_HASH,
    CANONICAL_SOURCE_MANIFEST_HASH,
    CANONICAL_SOURCE_REPOSITORY,
    CANONICAL_SOURCE_REVISION,
    MLX_BASE_ENV_VAR,
    M5LocalModelPreflightResult,
    M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH,
    M5_PREFLIGHT_MLX_BASE_MISSING,
    M5_PREFLIGHT_NO_METAL_DEVICE,
    M5_PREFLIGHT_READY,
    M5_PREFLIGHT_SOURCE_HASH_MISMATCH,
    M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING,
    M5RuntimePaths,
    SOURCE_SNAPSHOT_ENV_VAR,
    format_m5_local_model_preflight_failure,
    preflight_m5_local_model,
    resolve_m5_runtime_paths,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROVENANCE_V1 = ROOT / "artifacts/evaluation/m5/local_lora_runtime_provenance_v1.json"
TRAINING_CONFIG_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_training_config_v1.json"
TRAINING_FORMATTER_PATH = ROOT / "artifacts/evaluation/m5/m5_training_formatter_v1.json"
LORA_POLICY_PATH = ROOT / "artifacts/evaluation/m5/m5_lora_trainable_policy_v1.json"
MODULE_INSPECTION_PATH = ROOT / "artifacts/evaluation/m5/m5_model_module_inspection_v1.json"
SELECTION_POLICY_PATH = ROOT / "artifacts/evaluation/m5/m5_validation_selection_policy_v1.json"
BENCHMARK_DIR = ROOT / "data/generated/v0.0"
PROMPT_PATH = ROOT / "configs/prompts/prompt_v1.yaml"
RETRIEVAL_ARTIFACT_PATH = ROOT / "artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json"


def _write_manifest_dir(base: Path, *, files: dict[str, str]) -> tuple[Path, list[dict[str, object]]]:
    base.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for name, content in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        manifest.append({"file": name, "size_bytes": path.stat().st_size, "sha256": sha256_bytes(path.read_bytes())})
    manifest.sort(key=lambda item: item["file"])
    return base, manifest


def _runtime_provenance(
    tmp_path: Path,
    *,
    source_dir: Path,
    source_manifest: list[dict[str, object]],
    mlx_dir: Path,
    mlx_manifest: list[dict[str, object]],
) -> Path:
    payload = json.loads(RUNTIME_PROVENANCE_V1.read_text(encoding="utf-8"))
    payload["source_lineage"]["local_snapshot_path"] = str(source_dir)
    payload["source_lineage"]["file_hashes"] = source_manifest
    payload["mlx_conversion"]["output_path"] = str(mlx_dir)
    payload["mlx_conversion"]["output_file_hashes"] = mlx_manifest
    path = tmp_path / "runtime_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _ready_preflight(tmp_path: Path, *, source_name: str = "source", mlx_name: str = "mlx") -> tuple[Path, Path, Path]:
    source_dir, source_manifest = _write_manifest_dir(
        tmp_path / source_name,
        files={
            "config.json": "{\"model\":\"qwen3\"}\n",
            "tokenizer.json": "{\"tokenizer\":\"qwen3\"}\n",
        },
    )
    mlx_dir, mlx_manifest = _write_manifest_dir(
        tmp_path / mlx_name,
        files={
            "config.json": "{\"model_type\":\"qwen3\"}\n",
            "model.safetensors": "weights\n",
        },
    )
    provenance = _runtime_provenance(
        tmp_path,
        source_dir=source_dir,
        source_manifest=source_manifest,
        mlx_dir=mlx_dir,
        mlx_manifest=mlx_manifest,
    )
    return provenance, source_dir, mlx_dir


def test_preflight_reports_no_metal_device(tmp_path: Path) -> None:
    provenance, source_dir, mlx_dir = _ready_preflight(tmp_path)
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: False,
    )
    assert result.status == M5_PREFLIGHT_NO_METAL_DEVICE
    assert M5_PREFLIGHT_NO_METAL_DEVICE in result.status_codes
    assert result.ready is False


def test_preflight_reports_missing_source_snapshot(tmp_path: Path) -> None:
    provenance, _source_dir, mlx_dir = _ready_preflight(tmp_path)
    missing_source = tmp_path / "missing_source"
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=missing_source,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: True,
    )
    assert result.status == M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING
    assert M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING in result.status_codes


def test_preflight_reports_missing_mlx_base(tmp_path: Path) -> None:
    provenance, source_dir, _mlx_dir = _ready_preflight(tmp_path)
    missing_mlx = tmp_path / "missing_mlx"
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=missing_mlx,
        metal_probe=lambda: True,
    )
    assert result.status == M5_PREFLIGHT_MLX_BASE_MISSING
    assert M5_PREFLIGHT_MLX_BASE_MISSING in result.status_codes


def test_preflight_reports_source_hash_mismatch(tmp_path: Path) -> None:
    provenance, source_dir, mlx_dir = _ready_preflight(tmp_path)
    (source_dir / "config.json").write_text("corrupted\n", encoding="utf-8")
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: True,
    )
    assert result.status == M5_PREFLIGHT_SOURCE_HASH_MISMATCH
    assert M5_PREFLIGHT_SOURCE_HASH_MISMATCH in result.status_codes


def test_preflight_reports_mlx_identity_mismatch(tmp_path: Path) -> None:
    provenance, source_dir, mlx_dir = _ready_preflight(tmp_path)
    (mlx_dir / "model.safetensors").write_text("different\n", encoding="utf-8")
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: True,
    )
    assert result.status == M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH
    assert M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH in result.status_codes


def test_explicit_runtime_paths_override_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default_source, default_source_manifest = _write_manifest_dir(
        tmp_path / "default_source",
        files={"config.json": "{\"default\":true}\n"},
    )
    default_mlx, default_mlx_manifest = _write_manifest_dir(
        tmp_path / "default_mlx",
        files={"config.json": "{\"default\":true}\n"},
    )
    explicit_source, explicit_source_manifest = _write_manifest_dir(
        tmp_path / "explicit_source",
        files={"config.json": "{\"explicit\":true}\n"},
    )
    explicit_mlx, explicit_mlx_manifest = _write_manifest_dir(
        tmp_path / "explicit_mlx",
        files={"config.json": "{\"explicit\":true}\n"},
    )
    provenance = _runtime_provenance(
        tmp_path,
        source_dir=default_source,
        source_manifest=default_source_manifest,
        mlx_dir=default_mlx,
        mlx_manifest=default_mlx_manifest,
    )
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=explicit_source,
        mlx_base_path=explicit_mlx,
        metal_probe=lambda: True,
    )
    assert result.runtime_paths.source_snapshot_path == explicit_source
    assert result.runtime_paths.mlx_base_path == explicit_mlx
    assert result.runtime_paths.source_snapshot_source.startswith("explicit:")
    assert result.runtime_paths.mlx_base_source.startswith("explicit:")


def test_environment_variable_paths_override_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default_source, default_source_manifest = _write_manifest_dir(
        tmp_path / "default_source",
        files={"config.json": "{\"default\":true}\n"},
    )
    default_mlx, default_mlx_manifest = _write_manifest_dir(
        tmp_path / "default_mlx",
        files={"config.json": "{\"default\":true}\n"},
    )
    env_source, env_source_manifest = _write_manifest_dir(
        tmp_path / "env_source",
        files={"config.json": "{\"env\":true}\n"},
    )
    env_mlx, env_mlx_manifest = _write_manifest_dir(
        tmp_path / "env_mlx",
        files={"config.json": "{\"env\":true}\n"},
    )
    provenance = _runtime_provenance(
        tmp_path,
        source_dir=default_source,
        source_manifest=default_source_manifest,
        mlx_dir=default_mlx,
        mlx_manifest=default_mlx_manifest,
    )
    monkeypatch.setenv(SOURCE_SNAPSHOT_ENV_VAR, str(env_source))
    monkeypatch.setenv(MLX_BASE_ENV_VAR, str(env_mlx))
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        metal_probe=lambda: True,
    )
    assert result.runtime_paths.source_snapshot_path == env_source
    assert result.runtime_paths.mlx_base_path == env_mlx
    assert result.runtime_paths.source_snapshot_source.startswith("env:")
    assert result.runtime_paths.mlx_base_source.startswith("env:")


def test_canonical_hashes_remain_unchanged_regardless_of_path(tmp_path: Path) -> None:
    provenance, source_dir, mlx_dir = _ready_preflight(tmp_path)
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: True,
    )
    assert result.canonical_source_repository == CANONICAL_SOURCE_REPOSITORY
    assert result.canonical_source_revision == CANONICAL_SOURCE_REVISION
    assert result.canonical_source_manifest_hash == CANONICAL_SOURCE_MANIFEST_HASH
    assert result.canonical_mlx_base_identity_hash == CANONICAL_MLX_BASE_IDENTITY_HASH


def test_preflight_happens_before_model_loading() -> None:
    source = (ROOT / "src/adaptlab/m5/smoke_training.py").read_text(encoding="utf-8")
    body = source[source.index("def run_m5_lora_smoke("):]
    assert body.index("preflight_m5_local_model(") < body.index("_load_model_and_tokenizer(")


def test_preflight_result_is_mechanically_serializable(tmp_path: Path) -> None:
    provenance, source_dir, mlx_dir = _ready_preflight(tmp_path)
    result = preflight_m5_local_model(
        runtime_provenance_path=provenance,
        source_snapshot_path=source_dir,
        mlx_base_path=mlx_dir,
        metal_probe=lambda: True,
    )
    payload = result.to_dict()
    assert payload["status"] == M5_PREFLIGHT_READY
    assert payload["runtime_paths"]["source_snapshot_path"] == str(source_dir)
    assert payload["runtime_paths"]["mlx_base_path"] == str(mlx_dir)
    assert sha256_bytes(canonical_json_bytes(payload))
