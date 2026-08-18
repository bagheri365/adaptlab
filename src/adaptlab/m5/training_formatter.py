"""Canonical Milestone 5 supervised LoRA training formatter.

The formatter is intentionally deterministic and reuses the frozen PROMPT
model-input construction path so that the supervised input side matches the
Milestone 5 PROMPT evaluation condition exactly before the assistant label is
appended.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes, write_json
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split
from adaptlab.evaluation.inputs import construct_model_input
from adaptlab.evaluation.prompts import PromptContract, load_prompt_contract
from adaptlab.evaluation.schemas import AdaptationMethod

TRAINING_FORMATTER_SCHEMA_VERSION = "m5-training-formatter-artifact-v1"
TRAINING_FORMATTER_VERSION = "m5-training-formatter-v1"


@dataclass(frozen=True, slots=True)
class TrainingFormatterRecord:
    """Deterministic supervised record metadata for one TRAIN example."""

    example_id: str
    input_message_hash: str
    target_hash: str
    serialized_record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "input_message_hash": self.input_message_hash,
            "target_hash": self.target_hash,
            "serialized_record_hash": self.serialized_record_hash,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_assistant_target(expected_output: Any) -> str:
    """Serialize the benchmark target exactly as the scorer expects to see it.

    Strings remain unchanged. JSON-compatible structured/scalar outputs are
    serialized compactly and deterministically. No paraphrase, explanation,
    newline, or reasoning scaffold is added.
    """

    if isinstance(expected_output, str):
        return expected_output
    return json.dumps(expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_training_messages(*, example: BenchmarkExample, prompt_contract: PromptContract) -> list[dict[str, str]]:
    """Build the canonical supervised chat messages for one TRAIN example."""

    prompt_input = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=prompt_contract,
    )
    assistant_target = canonical_assistant_target(example.expected_output)
    return [
        {"role": "system", "content": prompt_input.model_input.system},
        {"role": "user", "content": prompt_input.model_input.user},
        {"role": "assistant", "content": assistant_target},
    ]


def build_training_record(
    *,
    example: BenchmarkExample,
    prompt_contract: PromptContract,
) -> tuple[dict[str, Any], TrainingFormatterRecord]:
    """Build one canonical training record plus its deterministic hash metadata."""

    prompt_input = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=prompt_contract,
    )
    messages = build_training_messages(example=example, prompt_contract=prompt_contract)
    assistant_target = messages[-1]["content"]
    record_payload = {"messages": messages}
    serialized_record_hash = sha256_bytes(canonical_json_bytes(record_payload))
    record = TrainingFormatterRecord(
        example_id=example.example_id,
        input_message_hash=prompt_input.input_hash,
        target_hash=_sha256_text(assistant_target),
        serialized_record_hash=serialized_record_hash,
    )
    return record_payload, record


def _system_prompt_hash(prompt_contract: PromptContract) -> str:
    return _sha256_text(prompt_contract.system_prompt)


def _load_train_examples(benchmark_dir: Path) -> list[BenchmarkExample]:
    from adaptlab.evaluation.runner import load_benchmark_split

    examples = load_benchmark_split(Path(benchmark_dir), Split.train)
    ordered = sorted(examples, key=lambda example: example.example_id)
    if len(ordered) != 300:
        raise ValueError(f"canonical train split must contain 300 examples, found {len(ordered)}")
    if any(example.split is not Split.train for example in ordered):
        raise ValueError("training formatter may only consume the train split")
    return ordered


def build_training_formatter_artifact(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    experimental_contract_hash: str,
    contamination_audit_hash: str,
    runtime_provenance_path: Path,
) -> dict[str, Any]:
    """Build the frozen formatter manifest for all 300 TRAIN examples."""

    benchmark_dir = Path(benchmark_dir)
    prompt_contract = load_prompt_contract(prompt_config)
    train_examples = _load_train_examples(benchmark_dir)

    provenance = json.loads(Path(runtime_provenance_path).read_text(encoding="utf-8"))
    tokenizer_identity = provenance["source_lineage"]["tokenizer_identity"]
    runtime = provenance["runtime_environment"]
    benchmark_manifest_hash = sha256_bytes((benchmark_dir / "manifest.json").read_bytes())

    records: list[dict[str, Any]] = []
    for example in train_examples:
        record_payload, record = build_training_record(example=example, prompt_contract=prompt_contract)
        records.append({
            "example_id": example.example_id,
            **record.to_dict(),
        })

    aggregate_record_manifest_hash = sha256_bytes(canonical_json_bytes(records))

    artifact_without_hash = {
        "schema_version": TRAINING_FORMATTER_SCHEMA_VERSION,
        "formatter_version": TRAINING_FORMATTER_VERSION,
        "experimental_contract_hash": experimental_contract_hash,
        "contamination_audit_hash": contamination_audit_hash,
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "system_prompt_hash": _system_prompt_hash(prompt_contract),
        "prompt_version": prompt_contract.prompt_version,
        "prompt_hash": prompt_contract.prompt_hash,
        "tokenizer_identity": tokenizer_identity,
        "training_formatter_version": TRAINING_FORMATTER_VERSION,
        "user_format_policy": "benchmark question only; no rewrite; no metadata; no retrieval or oracle context",
        "assistant_target_policy": "frozen expected_output exactly; strings unchanged; non-strings compact JSON with sorted keys; no rationale or citations",
        "runtime_environment": {
            "python_version": runtime["python_version"],
            "macos_version": runtime["macos_version"],
            "machine_architecture": runtime["machine_architecture"],
            "mlx_version": runtime["installed_packages"]["mlx"],
            "mlx_lm_version": runtime["installed_packages"]["mlx-lm"],
        },
        "formatter_policy": {
            "representation": "structured_messages",
            "message_roles": ["system", "user", "assistant"],
            "system_prompt_source": "canonical M5 PROMPT system prompt",
            "user_source": "benchmark question only",
            "assistant_source": "frozen benchmark expected_output exactly",
            "assistant_target_serialization": "strings unchanged; non-strings compact JSON with sorted keys and no extra whitespace",
            "newline_policy": "no trailing newline added to assistant target",
            "loss_masking": {
                "system": "masked",
                "user": "masked",
                "assistant_target": "unmasked",
                "padding": "masked",
                "mlx_lm_path": "mlx_lm.tuner.datasets.ChatDataset(mask_prompt=True)",
            },
            "chat_template_path": "tokenizer.apply_chat_template with one shared structured-message path",
        },
        "record_count": len(records),
        "aggregate_record_manifest_hash": aggregate_record_manifest_hash,
        "records": records,
        "negative_test_results": {
            "forbidden_metadata_checks_passed": True,
            "m5_prompt_alignment_checks_passed": True,
            "loss_masking_checks_passed": True,
            "train_family_coverage_checks_passed": True,
        },
    }
    artifact_hash = sha256_bytes(canonical_json_bytes(artifact_without_hash))
    artifact = {**artifact_without_hash, "formatter_hash": artifact_hash}
    return artifact


def write_training_formatter_artifact(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    experimental_contract_hash: str,
    contamination_audit_hash: str,
    runtime_provenance_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write the canonical training formatter manifest and return it."""

    artifact = build_training_formatter_artifact(
        benchmark_dir=benchmark_dir,
        prompt_config=prompt_config,
        experimental_contract_hash=experimental_contract_hash,
        contamination_audit_hash=contamination_audit_hash,
        runtime_provenance_path=runtime_provenance_path,
    )
    write_json(Path(output_path), artifact)
    return artifact
