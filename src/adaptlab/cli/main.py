"""Minimal command-line interface for AdaptLab benchmark utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adaptlab.benchmark.audit import audit_benchmark
from adaptlab.benchmark.build import build_prototype_fixture
from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.benchmark.validate import validate_fixture
from adaptlab.domain.world import NimbusWorld

DEFAULT_FIXTURE_DIR = Path("data/fixtures/prototype")


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_fixture(directory: Path):
    directory = Path(directory)
    world = NimbusWorld.from_dict(_read_json(directory / "world.json"))
    documents = [Document.from_dict(item) for item in _read_json(directory / "documents.json")]
    chunks = [DocumentChunk.from_dict(item) for item in _read_json(directory / "chunks.json")]
    examples = [BenchmarkExample.from_dict(item) for item in _read_json(directory / "examples.json")]
    return world, documents, chunks, examples


def _build_fixture(args: argparse.Namespace) -> int:
    result = build_prototype_fixture(args.seed, args.output_dir)
    print(
        f"Built fixture at {args.output_dir} with seed {args.seed}: "
        f"{len(result.world.facts)} records, {len(result.documents)} documents, "
        f"{len(result.chunks)} chunks, {len(result.examples)} examples."
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        world, documents, chunks, examples = _load_fixture(args.fixture_dir)
        result = validate_fixture(
            world,
            documents,
            chunks,
            examples,
            expected_generation_seed=world.generation_seed,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Validation failed: {exc}")
        return 1

    if not result.passed:
        print(f"Validation failed for {args.fixture_dir}:")
        for error in result.errors:
            print(f"- {error}")
        return 1

    print(
        f"Validation passed for {args.fixture_dir}: "
        f"{result.statistics['example_count']} examples, "
        f"{result.statistics['document_count']} documents, "
        f"{result.statistics['chunk_count']} chunks."
    )
    return 0


def _audit(args: argparse.Namespace) -> int:
    try:
        world, documents, _chunks, examples = _load_fixture(args.fixture_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Audit failed: {exc}")
        return 1

    result = audit_benchmark(world, documents, examples)
    print(f"Audit completed for {args.fixture_dir}.")
    print(json.dumps(result.summary, sort_keys=True))
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Nimbus benchmark utilities")
    benchmark_subparsers = benchmark.add_subparsers(dest="benchmark_command", required=True)

    build = benchmark_subparsers.add_parser("build-fixture", help="build the prototype fixture")
    build.add_argument("--seed", type=int, default=1729)
    build.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help=f"fixture output directory (default: {DEFAULT_FIXTURE_DIR})",
    )
    build.set_defaults(handler=_build_fixture)

    validate = benchmark_subparsers.add_parser("validate", help="validate a fixture")
    validate.add_argument("fixture_dir", type=Path, nargs="?", default=DEFAULT_FIXTURE_DIR)
    validate.set_defaults(handler=_validate)

    audit = benchmark_subparsers.add_parser("audit", help="audit a fixture")
    audit.add_argument("fixture_dir", type=Path, nargs="?", default=DEFAULT_FIXTURE_DIR)
    audit.set_defaults(handler=_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
