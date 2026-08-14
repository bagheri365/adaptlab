"""End-to-end deterministic build pipeline for the prototype Nimbus fixture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptlab.benchmark.audit import AuditResult, audit_benchmark
from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.generate_docs import generate_documents
from adaptlab.benchmark.generate_tasks import BENCHMARK_VERSION, generate_tasks
from adaptlab.benchmark.generate_world import generate_world
from adaptlab.benchmark.io import write_benchmark_artifacts, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.benchmark.validate import (
    ValidationResult,
    apply_structural_holdout_rules,
    validate_fixture,
)
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import NimbusFact, NimbusWorld

BENCHMARK_NAME = "AdaptLab Nimbus"


@dataclass(frozen=True, slots=True)
class BuildResult:
    world: NimbusWorld
    documents: tuple[Document, ...]
    chunks: tuple[DocumentChunk, ...]
    examples: tuple[BenchmarkExample, ...]
    validation: ValidationResult
    audit: AuditResult
    manifest: dict[str, Any]


def _classify_world_lifecycle(world: NimbusWorld) -> dict[str, str]:
    """Classify every logical fact that has a v1 record.

    This makes lifecycle classification an explicit pipeline stage while keeping
    the structured world authoritative. The task generator independently uses
    the same deterministic lifecycle function for changed-knowledge examples.
    """

    versions: dict[str, dict[str, NimbusFact]] = {}
    for fact in world.facts:
        versions.setdefault(fact.logical_fact_id, {})[fact.version] = fact

    states: dict[str, str] = {}
    for logical_fact_id in sorted(versions):
        by_version = versions[logical_fact_id]
        v1 = by_version.get("v1")
        if v1 is None:
            continue
        states[logical_fact_id] = classify_knowledge_state(v1, by_version.get("v2")).value
    return states


def build_prototype_fixture(seed: int, output_dir: Path) -> BuildResult:
    """Build, validate, audit, serialize, and manifest the prototype fixture."""

    output_dir = Path(output_dir)

    world = generate_world(seed)
    _classify_world_lifecycle(world)
    documents, chunks = generate_documents(world)
    examples = generate_tasks(world, documents, chunks)
    examples = apply_structural_holdout_rules(world, examples)

    validation = validate_fixture(
        world,
        documents,
        chunks,
        examples,
        expected_generation_seed=seed,
    )
    if not validation.passed:
        details = "\n".join(f"- {error}" for error in validation.errors)
        raise ValueError(f"benchmark validation failed; fixture was not serialized:\n{details}")

    audit = audit_benchmark(world, documents, examples)
    hashes = write_benchmark_artifacts(output_dir, world, documents, chunks, examples)

    manifest: dict[str, Any] = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "generation_seed": seed,
        "world_schema_version": world.world_schema_version,
        "fact_count": len(world.facts),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "example_count": len(examples),
        "hashes": hashes,
    }
    write_json(output_dir / "manifest.json", manifest)

    return BuildResult(
        world=world,
        documents=tuple(documents),
        chunks=tuple(chunks),
        examples=tuple(examples),
        validation=validation,
        audit=audit,
        manifest=manifest,
    )
