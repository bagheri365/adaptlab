"""Milestone 5 local-model path resolution and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import struct
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes

SOURCE_SNAPSHOT_ENV_VAR = "ADAPTLAB_M5_SOURCE_SNAPSHOT"
MLX_BASE_ENV_VAR = "ADAPTLAB_M5_MLX_BASE"

CANONICAL_RUNTIME_VERSIONS = {
    "python_version": "3.12.13",
    "macos_version": "26.5.2",
    "machine_architecture": "arm64",
    "mlx_version": "0.32.0",
    "mlx_lm_version": "0.31.3",
}
CANONICAL_SOURCE_REPOSITORY = "Qwen/Qwen3-8B-Base"
CANONICAL_SOURCE_REVISION = "7b8a267e13df1a9427e7dfa2691f69a417c58d94"
CANONICAL_SOURCE_MANIFEST_HASH = "507f79d4086e495f0852327e79ea6a4daa53afe2beb591a0fd8489dc16fe8397"
CANONICAL_MLX_BASE_IDENTITY_HASH = "d07ae738ad42baadb62b16115f6b2d90c32fbaa859acc81a4a0a95195e833c80"
CANONICAL_MLX_BASE_CONTAINER_HASH = "12292b1c28ee6beca7888d596f8b6129c966256c43c9500dc576902830173714"
CANONICAL_MLX_BASE_SEMANTIC_TENSOR_HASH = "3732cb9f906cb18f8c9d2844191270fb12d95c6e89ea9295f540b3a8126cac27"

M5_PREFLIGHT_READY = "READY"
M5_PREFLIGHT_NO_METAL_DEVICE = "NO_METAL_DEVICE"
M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING = "SOURCE_SNAPSHOT_MISSING"
M5_PREFLIGHT_MLX_BASE_MISSING = "MLX_BASE_MISSING"
M5_PREFLIGHT_SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
M5_PREFLIGHT_MLX_SEMANTIC_HASH_MISMATCH = "MLX_SEMANTIC_HASH_MISMATCH"
M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH = "MLX_BASE_HASH_MISMATCH"
M5_PREFLIGHT_RUNTIME_VERSION_MISMATCH = "RUNTIME_VERSION_MISMATCH"

_PREFLIGHT_FAILURE_ORDER = (
    M5_PREFLIGHT_NO_METAL_DEVICE,
    M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING,
    M5_PREFLIGHT_MLX_BASE_MISSING,
    M5_PREFLIGHT_SOURCE_HASH_MISMATCH,
    M5_PREFLIGHT_MLX_SEMANTIC_HASH_MISMATCH,
    M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH,
    M5_PREFLIGHT_RUNTIME_VERSION_MISMATCH,
)


@dataclass(frozen=True, slots=True)
class M5RuntimePaths:
    """Resolved runtime paths for the canonical Milestone 5 model assets."""

    source_snapshot_path: Path
    mlx_base_path: Path
    source_snapshot_source: str
    mlx_base_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_snapshot_path": str(self.source_snapshot_path),
            "mlx_base_path": str(self.mlx_base_path),
            "source_snapshot_source": self.source_snapshot_source,
            "mlx_base_source": self.mlx_base_source,
        }


@dataclass(frozen=True, slots=True)
class M5LocalModelPreflightResult:
    """Mechanical status summary for the local canonical M5 model assets."""

    status: str
    status_codes: tuple[str, ...]
    metal_available: bool
    source_snapshot_exists: bool
    mlx_base_exists: bool
    source_manifest_matches: bool
    mlx_base_identity_matches: bool
    runtime_version_matches: bool
    runtime_paths: M5RuntimePaths
    canonical_source_snapshot_path: Path
    canonical_mlx_base_path: Path
    canonical_source_repository: str
    canonical_source_revision: str
    canonical_source_manifest_hash: str
    canonical_mlx_base_identity_hash: str
    source_manifest_hash: str
    mlx_base_manifest_hash: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_codes": list(self.status_codes),
            "metal_available": self.metal_available,
            "source_snapshot_exists": self.source_snapshot_exists,
            "mlx_base_exists": self.mlx_base_exists,
            "source_manifest_matches": self.source_manifest_matches,
            "mlx_base_identity_matches": self.mlx_base_identity_matches,
            "runtime_version_matches": self.runtime_version_matches,
            "runtime_paths": self.runtime_paths.to_dict(),
            "canonical_source_snapshot_path": str(self.canonical_source_snapshot_path),
            "canonical_mlx_base_path": str(self.canonical_mlx_base_path),
            "canonical_source_repository": self.canonical_source_repository,
            "canonical_source_revision": self.canonical_source_revision,
            "canonical_source_manifest_hash": self.canonical_source_manifest_hash,
            "canonical_mlx_base_identity_hash": self.canonical_mlx_base_identity_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "mlx_base_manifest_hash": self.mlx_base_manifest_hash,
            "details": self.details,
        }

    @property
    def ready(self) -> bool:
        return self.status == M5_PREFLIGHT_READY


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(
    *,
    explicit_path: str | Path | None,
    env_var: str,
    default_path: str | Path,
    source_label: str,
) -> tuple[Path, str]:
    if explicit_path is not None:
        return Path(explicit_path), f"explicit:{source_label}"
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value), f"env:{env_var}"
    return Path(default_path), "provenance"


def resolve_m5_runtime_paths(
    *,
    runtime_provenance_path: str | Path,
    source_snapshot_path: str | Path | None = None,
    mlx_base_path: str | Path | None = None,
) -> M5RuntimePaths:
    """Resolve runtime model paths without changing frozen identity checks."""

    provenance = _load_json(runtime_provenance_path)
    if "source_lineage" in provenance:
        canonical_source_snapshot = provenance["source_lineage"]["local_snapshot_path"]
    else:
        canonical_source_snapshot = provenance["runtime_locations"]["resolved_source_snapshot_path"]
    if "mlx_conversion" in provenance:
        canonical_mlx_base = provenance["mlx_conversion"]["output_path"]
    else:
        canonical_mlx_base = provenance["runtime_locations"]["resolved_mlx_base_path"]
    resolved_source_snapshot, source_source = _resolve_path(
        explicit_path=source_snapshot_path,
        env_var=SOURCE_SNAPSHOT_ENV_VAR,
        default_path=canonical_source_snapshot,
        source_label="source_snapshot_path",
    )
    resolved_mlx_base, mlx_source = _resolve_path(
        explicit_path=mlx_base_path,
        env_var=MLX_BASE_ENV_VAR,
        default_path=canonical_mlx_base,
        source_label="mlx_base_path",
    )
    return M5RuntimePaths(
        source_snapshot_path=resolved_source_snapshot,
        mlx_base_path=resolved_mlx_base,
        source_snapshot_source=source_source,
        mlx_base_source=mlx_source,
    )


def _manifest_hash(file_hashes: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (
            {
                "file": str(item["file"]),
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
            for item in file_hashes
        ),
        key=lambda item: item["file"],
    )
    return sha256_bytes(canonical_json_bytes(ordered))


def _check_manifest(root: Path, file_hashes: Iterable[Mapping[str, Any]]) -> bool:
    for item in file_hashes:
        path = root / str(item["file"])
        if not path.exists():
            return False
        if sha256_bytes(path.read_bytes()) != str(item["sha256"]):
            return False
    return True


def _read_safetensors_tensor_identity(path: str | Path) -> dict[str, Any]:
    """Compute a deterministic tensor-semantic manifest from a Safetensors file."""

    blob = Path(path).read_bytes()
    if len(blob) < 8:
        raise ValueError(f"{path} is too small to be a Safetensors file")
    header_len = struct.unpack("<Q", blob[:8])[0]
    header_end = 8 + header_len
    raw_header = blob[8:header_end]
    header = json.loads(raw_header.decode("utf-8"))
    metadata = header.get("__metadata__") or header.get("metadata") or {}
    tensors: list[dict[str, Any]] = []
    for name, spec in header.items():
        if name.startswith("__"):
            continue
        start, end = spec["data_offsets"]
        raw = blob[header_end + int(start): header_end + int(end)]
        tensors.append(
            {
                "name": name,
                "dtype": spec["dtype"],
                "shape": list(spec["shape"]),
                "sha256": sha256_bytes(raw),
            }
        )
    tensors.sort(key=lambda item: item["name"])
    aggregate_semantic_tensor_hash = sha256_bytes(
        canonical_json_bytes(
            [{"name": item["name"], "dtype": item["dtype"], "shape": item["shape"], "sha256": item["sha256"]} for item in tensors]
        )
    )
    return {
        "tensor_count": len(tensors),
        "file_metadata": metadata,
        "tensors": tensors,
        "aggregate_semantic_tensor_hash": aggregate_semantic_tensor_hash,
    }


def _runtime_version_matches(runtime: Mapping[str, Any]) -> bool:
    runtime_environment = runtime["runtime_environment"]
    if "installed_packages" in runtime_environment:
        mlx_version = runtime_environment["installed_packages"]["mlx"]
        mlx_lm_version = runtime_environment["installed_packages"]["mlx-lm"]
    elif "mlx_version" in runtime_environment and "mlx_lm_version" in runtime_environment:
        mlx_version = runtime_environment["mlx_version"]
        mlx_lm_version = runtime_environment["mlx_lm_version"]
    else:
        mlx_version = runtime_environment["mlx"]
        mlx_lm_version = runtime_environment["mlx_lm"]
    observed = {
        "python_version": runtime_environment["python_version"],
        "macos_version": runtime_environment["macos_version"],
        "machine_architecture": runtime_environment["machine_architecture"],
        "mlx_version": mlx_version,
        "mlx_lm_version": mlx_lm_version,
    }
    return observed == CANONICAL_RUNTIME_VERSIONS


def _metal_available(probe: Callable[[], bool] | None = None) -> bool:
    if probe is not None:
        return bool(probe())
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


def preflight_m5_local_model(
    *,
    runtime_provenance_path: str | Path,
    source_snapshot_path: str | Path | None = None,
    mlx_base_path: str | Path | None = None,
    metal_probe: Callable[[], bool] | None = None,
) -> M5LocalModelPreflightResult:
    """Run the mechanical local-model preflight before any MLX model load."""

    provenance = _load_json(runtime_provenance_path)
    if "source_lineage" in provenance:
        provenance_source_lineage = provenance["source_lineage"]
        canonical_source_repository = str(provenance_source_lineage["repository"])
        canonical_source_revision = str(provenance_source_lineage["revision"])
        canonical_source_manifest_hash = str(
            provenance_source_lineage.get("source_manifest_hash", CANONICAL_SOURCE_MANIFEST_HASH)
        )
        source_file_hashes = provenance_source_lineage["file_hashes"]
        if "mlx_conversion" in provenance:
            mlx_file_hashes = provenance["mlx_conversion"]["output_file_hashes"]
        elif "rebuilt_mlx_manifest" in provenance:
            mlx_file_hashes = provenance["rebuilt_mlx_manifest"]["output_file_hashes"]
        elif "rebased_identity" in provenance:
            mlx_file_hashes = provenance["rebased_identity"]["container_file_hashes"]
        else:
            raise KeyError("runtime provenance is missing MLX output file hashes")
    else:
        provenance_identity = provenance["canonical_identity"]
        canonical_source_repository = str(provenance_identity["source_repository"])
        canonical_source_revision = str(provenance_identity["source_revision"])
        canonical_source_manifest_hash = str(provenance_identity["source_manifest_hash"])
        source_file_hashes = _load_json(Path(provenance["supersedes"]["artifact_path"]))["source_lineage"]["file_hashes"]
        mlx_file_hashes = _load_json(Path(provenance["supersedes"]["artifact_path"]))["mlx_conversion"]["output_file_hashes"]
    runtime_paths = resolve_m5_runtime_paths(
        runtime_provenance_path=runtime_provenance_path,
        source_snapshot_path=source_snapshot_path,
        mlx_base_path=mlx_base_path,
    )

    source_manifest_hash = _manifest_hash(source_file_hashes)
    mlx_manifest_hash = _manifest_hash(mlx_file_hashes)

    source_snapshot_exists = runtime_paths.source_snapshot_path.exists()
    mlx_base_exists = runtime_paths.mlx_base_path.exists()
    metal_available = _metal_available(metal_probe)
    source_manifest_matches = source_snapshot_exists and _check_manifest(runtime_paths.source_snapshot_path, source_file_hashes)
    runtime_version_matches = _runtime_version_matches(provenance)

    rebased_identity = provenance.get("rebased_identity")
    if rebased_identity is not None:
        expected_semantic_hash = str(
            rebased_identity.get("new_canonical_semantic_tensor_hash", CANONICAL_MLX_BASE_SEMANTIC_TENSOR_HASH)
        )
        expected_container_hash = str(
            rebased_identity.get("new_canonical_container_hash", CANONICAL_MLX_BASE_CONTAINER_HASH)
        )
        semantic_manifest_path = rebased_identity.get("semantic_tensor_identity_path")
        semantic_identity_matches = False
        observed_semantic_hash = None
        semantic_identity_detail: dict[str, Any] | None = None
        if mlx_base_exists:
            model_path = runtime_paths.mlx_base_path / "model.safetensors"
            if model_path.exists():
                semantic_identity_detail = _read_safetensors_tensor_identity(model_path)
                observed_semantic_hash = semantic_identity_detail["aggregate_semantic_tensor_hash"]
                semantic_identity_matches = observed_semantic_hash == expected_semantic_hash
        semantic_manifest_matches = False
        if semantic_manifest_path is not None and Path(semantic_manifest_path).exists():
            semantic_manifest = _load_json(semantic_manifest_path)
            semantic_manifest_matches = (
                str(semantic_manifest.get("aggregate_semantic_tensor_hash")) == expected_semantic_hash
            )
        mlx_base_identity_matches = mlx_base_exists and _check_manifest(
            runtime_paths.mlx_base_path,
            rebased_identity.get("container_file_hashes", mlx_file_hashes),
        )
    else:
        expected_semantic_hash = CANONICAL_MLX_BASE_IDENTITY_HASH
        expected_container_hash = None
        semantic_identity_matches = None
        observed_semantic_hash = None
        semantic_identity_detail = None
        semantic_manifest_matches = None
        mlx_base_identity_matches = mlx_base_exists and _check_manifest(runtime_paths.mlx_base_path, mlx_file_hashes)

    if rebased_identity is not None and rebased_identity.get("container_file_hashes") is not None:
        mlx_file_hashes = rebased_identity["container_file_hashes"]
        mlx_manifest_hash = _manifest_hash(mlx_file_hashes)

    status_codes: list[str] = []
    if not metal_available:
        status_codes.append(M5_PREFLIGHT_NO_METAL_DEVICE)
    if not source_snapshot_exists:
        status_codes.append(M5_PREFLIGHT_SOURCE_SNAPSHOT_MISSING)
    if not mlx_base_exists:
        status_codes.append(M5_PREFLIGHT_MLX_BASE_MISSING)
    if source_snapshot_exists and not source_manifest_matches:
        status_codes.append(M5_PREFLIGHT_SOURCE_HASH_MISMATCH)
    if rebased_identity is not None:
        if not semantic_identity_matches or not semantic_manifest_matches:
            status_codes.append(M5_PREFLIGHT_MLX_SEMANTIC_HASH_MISMATCH)
    if mlx_base_exists and not mlx_base_identity_matches:
        status_codes.append(M5_PREFLIGHT_MLX_BASE_HASH_MISMATCH)
    if not runtime_version_matches:
        status_codes.append(M5_PREFLIGHT_RUNTIME_VERSION_MISMATCH)

    status = M5_PREFLIGHT_READY if not status_codes else status_codes[0]
    details = {
        "resolved_paths_source": runtime_paths.source_snapshot_source,
        "resolved_paths_mlx": runtime_paths.mlx_base_source,
    }
    return M5LocalModelPreflightResult(
        status=status,
        status_codes=tuple(status_codes),
        metal_available=metal_available,
        source_snapshot_exists=source_snapshot_exists,
        mlx_base_exists=mlx_base_exists,
        source_manifest_matches=source_manifest_matches,
        mlx_base_identity_matches=mlx_base_identity_matches,
        runtime_version_matches=runtime_version_matches,
        runtime_paths=runtime_paths,
        canonical_source_snapshot_path=Path(
            provenance["source_lineage"]["local_snapshot_path"]
            if "source_lineage" in provenance
            else provenance["runtime_locations"]["resolved_source_snapshot_path"]
        ),
        canonical_mlx_base_path=Path(
            provenance["mlx_conversion"]["output_path"]
            if "mlx_conversion" in provenance
            else provenance["runtime_locations"]["resolved_mlx_base_path"]
        ),
        canonical_source_repository=canonical_source_repository,
        canonical_source_revision=canonical_source_revision,
        canonical_source_manifest_hash=CANONICAL_SOURCE_MANIFEST_HASH,
        canonical_mlx_base_identity_hash=(
            expected_semantic_hash if rebased_identity is not None else CANONICAL_MLX_BASE_IDENTITY_HASH
        ),
        source_manifest_hash=source_manifest_hash,
        mlx_base_manifest_hash=mlx_manifest_hash,
        details={
            **details,
            "expected_container_hash": expected_container_hash,
            "expected_semantic_tensor_hash": expected_semantic_hash,
            "observed_semantic_tensor_hash": observed_semantic_hash,
            "semantic_identity_matches": semantic_identity_matches,
            "semantic_manifest_matches": semantic_manifest_matches,
            "semantic_tensor_identity": semantic_identity_detail,
        },
    )


def format_m5_local_model_preflight_failure(result: M5LocalModelPreflightResult) -> str:
    """Format a concise, human-readable preflight failure message."""

    if result.ready:
        return "M5 local model preflight passed."
    return (
        "M5 local model preflight failed: "
        f"{', '.join(result.status_codes)}; "
        f"resolved_source_snapshot_path={result.runtime_paths.source_snapshot_path}; "
        f"resolved_mlx_base_path={result.runtime_paths.mlx_base_path}"
    )
