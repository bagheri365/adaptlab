"""Deterministic JSON serialization for AdaptLab benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.world import NimbusWorld

ARTIFACT_FILENAMES = ("world.json", "documents.json", "chunks.json", "examples.json")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically as UTF-8 with a trailing newline."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    )
    return (text + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> bytes:
    """Write deterministic JSON and return the exact bytes written."""

    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_world(world: NimbusWorld) -> dict[str, Any]:
    data = world.to_dict()
    data["facts"] = [fact.to_dict() for fact in sorted(world.facts, key=lambda f: f.record_id)]
    return data


def serialize_documents(documents: Iterable[Document]) -> list[dict[str, Any]]:
    return [doc.to_dict() for doc in sorted(documents, key=lambda d: d.document_id)]


def serialize_chunks(chunks: Iterable[DocumentChunk]) -> list[dict[str, Any]]:
    return [chunk.to_dict() for chunk in sorted(chunks, key=lambda c: c.chunk_id)]


def serialize_examples(examples: Iterable[BenchmarkExample]) -> list[dict[str, Any]]:
    return [example.to_dict() for example in sorted(examples, key=lambda e: e.example_id)]


def write_benchmark_artifacts(
    output_dir: Path,
    world: NimbusWorld,
    documents: Iterable[Document],
    chunks: Iterable[DocumentChunk],
    examples: Iterable[BenchmarkExample],
) -> dict[str, str]:
    """Write canonical benchmark artifacts and return SHA-256 hashes by filename."""

    payloads = {
        "world.json": serialize_world(world),
        "documents.json": serialize_documents(documents),
        "chunks.json": serialize_chunks(chunks),
        "examples.json": serialize_examples(examples),
    }
    hashes: dict[str, str] = {}
    for filename in ARTIFACT_FILENAMES:
        data = write_json(output_dir / filename, payloads[filename])
        hashes[filename] = sha256_bytes(data)
    return hashes
