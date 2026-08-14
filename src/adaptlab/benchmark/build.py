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


# --- Full v0.0 benchmark build -------------------------------------------------

@dataclass(frozen=True, slots=True)
class FullBuildResult:
    """Result of a full candidate benchmark build.

    A candidate may be fully generated while still carrying freeze blockers from
    deterministic audits.  ``passed`` means all blocking validations/audits passed;
    it does not mean the benchmark has been frozen.
    """

    config: Any
    world: NimbusWorld
    holdout_policy: Any
    documents: tuple[Document, ...]
    chunks: tuple[DocumentChunk, ...]
    examples: tuple[BenchmarkExample, ...]
    training_subsets: Any
    sentinel: tuple[Any, ...]
    answer_validation: ValidationResult
    holdout_validation: Any
    sentinel_validation: Any
    leakage_audit: Any
    lexical_overlap_audit: Any
    corpus_report: Any
    audit: AuditResult
    manifest: dict[str, Any]
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def build_full_benchmark(config_path: Path, output_dir: Path) -> FullBuildResult:
    """Build the complete deterministic full-v0.0 candidate benchmark.

    The prototype pipeline above remains intentionally separate.  This full path
    consumes the declarative benchmark configuration, freezes holdouts before task
    generation, runs all currently implemented blocking validation/audits, writes
    deterministic candidate artifacts, and emits a *preliminary* manifest whose
    ``frozen`` field is always false at this milestone stage.
    """

    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import generate_full_world, summarize_world
    from adaptlab.benchmark.holdout import (
        build_full_holdout_policy,
        render_holdout_report,
        validate_full_holdout_examples,
    )
    from adaptlab.benchmark.generate_docs import generate_full_documents, summarize_corpus
    from adaptlab.benchmark.generate_tasks import generate_full_tasks
    from adaptlab.benchmark.training_subsets import generate_training_subsets
    from adaptlab.benchmark.sentinel import (
        generate_generalization_sentinel,
        validate_generalization_sentinel,
    )
    from adaptlab.benchmark.validate import validate_answer_integrity
    from adaptlab.benchmark.leakage import run_leakage_audit
    from adaptlab.benchmark.lexical_overlap import run_lexical_overlap_audit
    from adaptlab.benchmark.audit import audit_benchmark
    from adaptlab.benchmark.io import sha256_bytes

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_benchmark_config(config_path)

    world = generate_full_world(config)
    holdout_policy = build_full_holdout_policy(config, world)
    documents, chunks = generate_full_documents(world, config)
    examples = generate_full_tasks(world, documents, chunks, config, holdout_policy)
    training_subsets = generate_training_subsets(examples, config)
    sentinel = generate_generalization_sentinel(
        seed=config.generation_seed,
        count=config.generalization_sentinel.count,
    )

    answer_validation = validate_answer_integrity(world, documents, chunks, examples)
    holdout_validation = validate_full_holdout_examples(world, examples, holdout_policy)
    sentinel_validation = validate_generalization_sentinel(
        sentinel,
        expected_count=config.generalization_sentinel.count,
        expected_seed=config.generation_seed,
    )
    leakage_audit = run_leakage_audit(examples, world=world, holdout_policy=holdout_policy)
    lexical_overlap_audit = run_lexical_overlap_audit(examples, chunks)
    corpus_report = summarize_corpus(documents, chunks)
    audit = audit_benchmark(world, documents, examples, full_scale=True)

    blockers: list[str] = []
    if not answer_validation.passed:
        blockers.extend(f"answer integrity: {error}" for error in answer_validation.errors)
    if not holdout_validation.passed:
        blockers.extend(f"structural holdout: {error}" for error in holdout_validation.errors)
    if not sentinel_validation.passed:
        blockers.extend(f"sentinel: {error}" for error in sentinel_validation.errors)
    blockers.extend(f"leakage: {blocker}" for blocker in leakage_audit.blockers)

    # Deterministic serialization for all currently available full-build artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {
        "world.json": world.to_dict(),
        "documents.json": [item.to_dict() for item in sorted(documents, key=lambda x: x.document_id)],
        "chunks.json": [item.to_dict() for item in sorted(chunks, key=lambda x: x.chunk_id)],
        "train.json": [item.to_dict() for item in examples if item.split.value == "train"],
        "validation.json": [item.to_dict() for item in examples if item.split.value == "validation"],
        "test.json": [item.to_dict() for item in examples if item.split.value == "test"],
        "training_subsets.json": training_subsets.to_dict(),
        "sentinel.json": [item.to_dict() for item in sentinel],
        "holdout_policy.json": holdout_policy.to_dict(),
        "audits/corpus_composition.json": corpus_report.to_dict(),
        "audits/leakage.json": leakage_audit.to_dict(),
        "audits/lexical_overlap.json": lexical_overlap_audit.to_dict(),
        "audits/anti_confounding.json": audit.to_dict(),
    }
    hashes: dict[str, str] = {}
    for relative_name in sorted(payloads):
        data = write_json(output_dir / relative_name, payloads[relative_name])
        hashes[relative_name] = sha256_bytes(data)

    write_json(output_dir / "holdout_report.json", {
        "policy_version": holdout_policy.policy_version,
        "report": render_holdout_report(holdout_policy),
    })

    config_hash = sha256_bytes(config_path.read_bytes())
    manifest: dict[str, Any] = {
        "benchmark_name": config.benchmark_name,
        "benchmark_version": config.benchmark_version,
        "candidate_version": f"{config.benchmark_version}-candidate.1",
        "generation_seed": config.generation_seed,
        "world_schema_version": config.world_schema_version,
        "frozen": False,
        "candidate_status": "BLOCKED" if blockers else "VALID_CANDIDATE",
        "counts": {
            "world_logical_facts": sum(summarize_world(world).logical_facts_per_component.values()),
            "world_records": len(world.facts),
            "documents": len(documents),
            "chunks": len(chunks),
            "train": sum(item.split.value == "train" for item in examples),
            "validation": sum(item.split.value == "validation" for item in examples),
            "test": sum(item.split.value == "test" for item in examples),
            "sentinel": len(sentinel),
        },
        "config_hash": config_hash,
        "artifact_hashes": dict(sorted(hashes.items())),
        "blocking_issue_count": len(blockers),
        "blockers": list(blockers),
    }
    write_json(output_dir / "preliminary_manifest.json", manifest)

    return FullBuildResult(
        config=config,
        world=world,
        holdout_policy=holdout_policy,
        documents=tuple(documents),
        chunks=tuple(chunks),
        examples=tuple(examples),
        training_subsets=training_subsets,
        sentinel=tuple(sentinel),
        answer_validation=answer_validation,
        holdout_validation=holdout_validation,
        sentinel_validation=sentinel_validation,
        leakage_audit=leakage_audit,
        lexical_overlap_audit=lexical_overlap_audit,
        corpus_report=corpus_report,
        audit=audit,
        manifest=manifest,
        blockers=tuple(blockers),
    )
