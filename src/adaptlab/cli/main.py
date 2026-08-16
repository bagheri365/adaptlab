"""Minimal command-line interface for AdaptLab benchmark utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from adaptlab.benchmark.audit import audit_benchmark
from adaptlab.benchmark.build import build_full_benchmark, build_prototype_fixture
from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.human_audit import load_human_review_queue, update_human_review_record
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.benchmark.validate import validate_fixture
from adaptlab.domain.world import NimbusWorld
from adaptlab.domain.enums import Split
from adaptlab.evaluation.providers import FakeModelProvider, OllamaModelProvider
from adaptlab.evaluation.runner import run_evaluation
from adaptlab.evaluation.schemas import AdaptationMethod

DEFAULT_FIXTURE_DIR = Path("data/fixtures/prototype")
DEFAULT_FULL_CONFIG = Path("configs/benchmark_v0.0.yaml")
DEFAULT_FULL_OUTPUT = Path("data/generated/v0.0")


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



def _build_full(args: argparse.Namespace) -> int:
    try:
        result = build_full_benchmark(args.config, args.output)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Full benchmark build failed: {exc}")
        return 1

    print(
        f"Built full benchmark candidate at {args.output}: "
        f"{result.manifest['counts']['train']} train, "
        f"{result.manifest['counts']['validation']} validation, "
        f"{result.manifest['counts']['test']} test, "
        f"{result.manifest['counts']['sentinel']} sentinel."
    )
    if result.blockers:
        print("Build completed with blocking validation findings; benchmark is not frozen:")
        for blocker in result.blockers:
            print(f"- {blocker}")
        return 1
    print("Full benchmark candidate passed blocking validation. Benchmark remains unfrozen.")
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



def _find_chunk_texts(audit_path: Path) -> dict[str, str]:
    """Resolve chunk text from the benchmark output adjacent to an audit file."""
    audit_path = Path(audit_path)
    candidates = [audit_path.parent.parent / "chunks.json", audit_path.parent / "chunks.json"]
    for chunks_path in candidates:
        if chunks_path.exists():
            chunks = _read_json(chunks_path)
            return {str(item["chunk_id"]): str(item["content"]) for item in chunks}
    return {}


def review_human_audit(
    audit_path: Path,
    *,
    blind: bool = False,
    input_fn=input,
    output_fn=print,
) -> int:
    """Interactively review pending records; each completed decision is saved immediately."""
    audit_path = Path(audit_path)
    data = load_human_review_queue(audit_path)
    chunk_text = _find_chunk_texts(audit_path)
    pending = [
        review for review in data["reviews"]
        if review.get("review_status", "PENDING_HUMAN_REVIEW") == "PENDING_HUMAN_REVIEW"
    ]
    if not pending:
        output_fn("No pending human-review records.")
        return 0

    for index, review in enumerate(pending, start=1):
        output_fn(f"\n[{index}/{len(pending)}] {review['example_id']}")
        for key in ("task_family", "behavior_type", "difficulty", "knowledge_state", "evidence_status"):
            output_fn(f"{key}: {review.get(key)}")
        output_fn(f"question: {review.get('question', '')}")
        output_fn(f"required records: {', '.join(review.get('required_records', [])) or '(none)'}")
        output_fn("structured truth:")
        truth_items = list(review.get("structured_truth", []))
        if truth_items:
            for item in truth_items:
                output_fn(f"- {json.dumps(item, sort_keys=True, ensure_ascii=False)}")
        else:
            output_fn("- (prompt/scoring contract only; no world record required)")
        gold_ids = list(review.get("gold_chunks", []))
        embedded_text = list(review.get("gold_evidence_text", []))
        output_fn("gold evidence:")
        if gold_ids:
            for idx, chunk_id in enumerate(gold_ids):
                text = embedded_text[idx] if idx < len(embedded_text) and embedded_text[idx] else chunk_text.get(chunk_id, "[chunk text unavailable]")
                output_fn(f"- {chunk_id}: {text}")
        else:
            output_fn("- (none)")

        if blind:
            reveal = input_fn("Reveal expected output? [y/N]: ").strip().lower()
            if reveal == "y":
                output_fn(f"expected_output: {review.get('expected_output', '')}")
        else:
            output_fn(f"expected_output: {review.get('expected_output', '')}")

        while True:
            raw = input_fn("Decision [PASS/FAIL/CORRECTION_REQUIRED/SKIP/QUIT]: ").strip().upper()
            if raw in {"PASS", "FAIL", "CORRECTION_REQUIRED", "SKIP", "QUIT"}:
                break
            output_fn("Invalid decision.")
        if raw == "QUIT":
            output_fn("Review stopped; completed decisions are already saved.")
            return 0
        if raw == "SKIP":
            continue
        notes = input_fn("Optional notes: ").strip()
        update_human_review_record(audit_path, str(review["example_id"]), raw, notes)
        output_fn(f"Saved {raw} for {review['example_id']}.")

    final = load_human_review_queue(audit_path)
    output_fn(f"Review summary: {json.dumps(final['summary'], sort_keys=True)}")
    return 0


def _review_human_audit(args: argparse.Namespace) -> int:
    try:
        return review_human_audit(args.audit_path, blind=args.blind)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Human audit review failed: {exc}")
        return 1

def _evaluate_run(args: argparse.Namespace) -> int:
    method_name = args.method.upper().replace("-", "_")
    try:
        method = AdaptationMethod(method_name)
        if args.provider == "fake":
            provider = FakeModelProvider([], provider_name="fake")
        elif args.provider == "ollama":
            provider = OllamaModelProvider(
                model_id=args.model,
                base_url=args.base_url,
                context_length=args.context_length,
                think=args.think,
                stream=False,
            )
        else:
            raise ValueError(f"unsupported provider: {args.provider}")
        runner_max_retries = 0 if args.provider == "ollama" else 2
        run = run_evaluation(
            benchmark_dir=args.benchmark,
            method=method,
            model_id=args.model,
            provider=provider,
            prompt_config=args.prompt_config,
            output_dir=args.output,
            split=Split(args.split),
            limit=args.limit,
            resume=args.resume,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            canonical=args.canonical,
            allow_dirty_git=args.allow_dirty_git,
            max_retries=runner_max_retries,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    print(f"Evaluation {run.status.value.lower()}: {run.run_id} -> {args.output}")
    return 0 if run.status.value == "COMPLETED" else 1


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

    full_build = benchmark_subparsers.add_parser("build", help="build the full benchmark candidate")
    full_build.add_argument(
        "--config", type=Path, default=DEFAULT_FULL_CONFIG,
        help=f"benchmark config (default: {DEFAULT_FULL_CONFIG})",
    )
    full_build.add_argument(
        "--output", type=Path, default=DEFAULT_FULL_OUTPUT,
        help=f"candidate output directory (default: {DEFAULT_FULL_OUTPUT})",
    )
    full_build.set_defaults(handler=_build_full)

    validate = benchmark_subparsers.add_parser("validate", help="validate a fixture")
    validate.add_argument("fixture_dir", type=Path, nargs="?", default=DEFAULT_FIXTURE_DIR)
    validate.set_defaults(handler=_validate)

    audit = benchmark_subparsers.add_parser("audit", help="audit a fixture")
    audit.add_argument("fixture_dir", type=Path, nargs="?", default=DEFAULT_FIXTURE_DIR)
    audit.set_defaults(handler=_audit)

    review = benchmark_subparsers.add_parser(
        "review-human-audit", help="interactively review pending human-audit records"
    )
    review.add_argument("audit_path", type=Path)
    review.add_argument(
        "--blind", action="store_true",
        help="hide expected output until the reviewer explicitly reveals it",
    )
    review.set_defaults(handler=_review_human_audit)

    evaluate = subparsers.add_parser("evaluate", help="evaluation harness utilities")
    evaluate_subparsers = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_run = evaluate_subparsers.add_parser("run", help="run a frozen benchmark evaluation")
    evaluate_run.add_argument("--benchmark", type=Path, required=True)
    evaluate_run.add_argument("--method", choices=("prompt", "oracle_context", "oracle-context"), required=True)
    evaluate_run.add_argument("--model", required=True)
    evaluate_run.add_argument("--provider", choices=("fake", "ollama"), default="fake")
    evaluate_run.add_argument("--prompt-config", type=Path, required=True)
    evaluate_run.add_argument("--output", type=Path, required=True)
    evaluate_run.add_argument("--split", choices=tuple(item.value for item in Split), default=Split.test.value)
    evaluate_run.add_argument("--limit", type=int)
    evaluate_run.add_argument("--resume", action="store_true")
    evaluate_run.add_argument("--temperature", type=float, default=0.0)
    evaluate_run.add_argument("--max-tokens", type=int, default=256)
    evaluate_run.add_argument("--seed", type=int)
    evaluate_run.add_argument("--base-url", type=str, default="http://localhost:11434")
    evaluate_run.add_argument("--context-length", type=int, default=40960)
    evaluate_run.add_argument("--think", action="store_true", help="send think=true when using Ollama")
    evaluate_run.add_argument("--canonical", action="store_true", help="enforce canonical clean-Git provenance")
    evaluate_run.add_argument("--allow-dirty-git", action="store_true", help="explicitly override dirty-tree refusal for a canonical run")
    evaluate_run.set_defaults(handler=_evaluate_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
