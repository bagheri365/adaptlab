"""Versioned, frozen prompt contracts for AdaptLab evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from adaptlab.benchmark.io import sha256_bytes

PROMPT_CONFIG_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class PromptContract:
    """A single generic system prompt plus immutable artifact provenance."""

    prompt_version: str
    system_prompt: str
    prompt_hash: str
    frozen: bool
    schema_version: str = PROMPT_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("prompt_version", "system_prompt", "prompt_hash", "schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if len(self.prompt_hash) != 64 or any(c not in "0123456789abcdef" for c in self.prompt_hash):
            raise ValueError("prompt_hash must be a lowercase SHA-256 hex digest")
        if not self.frozen:
            raise ValueError("evaluation prompt contract must be frozen")

    def provenance(self) -> dict[str, str]:
        """Fields copied into every evaluation run manifest."""

        return {
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
        }


def _require_exact_keys(raw: dict[str, Any]) -> None:
    expected = {"schema_version", "prompt_version", "frozen", "system_prompt"}
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise ValueError("invalid prompt config fields: " + ", ".join(detail))


def load_prompt_contract(path: str | Path) -> PromptContract:
    """Load a frozen prompt artifact and hash the exact bytes consumed."""

    config_path = Path(path)
    artifact_bytes = config_path.read_bytes()
    raw = yaml.safe_load(artifact_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("prompt config must be a mapping")
    _require_exact_keys(raw)

    if raw["schema_version"] != PROMPT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prompt config schema_version: {raw['schema_version']!r}"
        )
    if raw["frozen"] is not True:
        raise ValueError("prompt config must declare frozen: true")

    return PromptContract(
        schema_version=raw["schema_version"],
        prompt_version=raw["prompt_version"],
        system_prompt=raw["system_prompt"],
        prompt_hash=sha256_bytes(artifact_bytes),
        frozen=raw["frozen"],
    )
