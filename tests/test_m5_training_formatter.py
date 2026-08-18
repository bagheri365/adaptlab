from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split, TaskFamily
from adaptlab.evaluation.inputs import construct_model_input
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.m5.training_formatter import (
    TRAINING_FORMATTER_VERSION,
    build_training_messages,
    build_training_record,
    canonical_assistant_target,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data/generated/v0.0"
PROMPT_PATH = ROOT / "configs/prompts/prompt_v1.yaml"
ARTIFACT_PATH = ROOT / "artifacts/evaluation/m5/m5_training_formatter_v1.json"


def _train_examples() -> list[BenchmarkExample]:
    raw = json.loads((BENCHMARK_DIR / "train.json").read_text(encoding="utf-8"))
    return [BenchmarkExample.from_dict(item) for item in raw]


def _first_by_family() -> dict[TaskFamily, BenchmarkExample]:
    out: dict[TaskFamily, BenchmarkExample] = {}
    for example in _train_examples():
        out.setdefault(example.task_family, example)
    return out


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_formatter_uses_prompt_path_for_system_and_user_content() -> None:
    prompt = load_prompt_contract(PROMPT_PATH)
    example = _train_examples()[0]
    prompt_input = construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=prompt,
    )
    messages = build_training_messages(example=example, prompt_contract=prompt)

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == prompt_input.model_input.system
    assert messages[1]["content"] == prompt_input.model_input.user
    assert prompt_input.model_input.user == example.question


def test_formatter_serializes_targets_exactly() -> None:
    example = next(item for item in _train_examples() if not isinstance(item.expected_output, str))
    target = canonical_assistant_target(example.expected_output)
    assert target == json.dumps(example.expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(target.encode("utf-8")).hexdigest()


def test_formatter_record_hashes_are_deterministic() -> None:
    prompt = load_prompt_contract(PROMPT_PATH)
    example = _train_examples()[0]
    first_payload, first_record = build_training_record(example=example, prompt_contract=prompt)
    second_payload, second_record = build_training_record(example=example, prompt_contract=prompt)

    assert first_payload == second_payload
    assert first_record == second_record
    assert first_record.input_message_hash == construct_model_input(
        example=example,
        method=AdaptationMethod.PROMPT,
        prompt_contract=prompt,
    ).input_hash


def test_formatter_coverage_covers_all_train_families() -> None:
    prompt = load_prompt_contract(PROMPT_PATH)
    first = _first_by_family()
    assert set(first) == set(TaskFamily)

    for family, example in first.items():
        payload, record = build_training_record(example=example, prompt_contract=prompt)
        assert record.example_id == example.example_id
        assert payload["messages"][0]["content"] == prompt.system_prompt
        assert payload["messages"][1]["content"] == example.question
        visible = json.dumps(payload, ensure_ascii=False).lower()
        for token in (
            "task_family",
            "knowledge_state",
            "evidence_status",
            "split_type",
            "gold_chunk_ids",
            "logical_fact_id",
            "retrieval_score",
            "expected_output",
        ):
            assert token not in visible, family


def test_formatter_artifact_records_loss_masking_policy() -> None:
    artifact = _artifact()
    masking = artifact["formatter_policy"]["loss_masking"]
    assert masking == {
        "system": "masked",
        "user": "masked",
        "assistant_target": "unmasked",
        "padding": "masked",
        "mlx_lm_path": "mlx_lm.tuner.datasets.ChatDataset(mask_prompt=True)",
    }
    assert artifact["formatter_hash"] == "1d04beb66c4f5d81fdcba9f3a9d9e3cfdb243251766784a4f58dee1a4ce9ca60"


def test_formatter_artifact_is_deterministic_and_hashable(tmp_path: Path) -> None:
    artifact = _artifact()
    second = _artifact()
    assert artifact == second
    assert artifact["formatter_version"] == TRAINING_FORMATTER_VERSION
    assert artifact["training_formatter_version"] == TRAINING_FORMATTER_VERSION
    assert artifact["record_count"] == 300
    assert len(artifact["records"]) == 300
    assert artifact["aggregate_record_manifest_hash"]
    assert artifact["formatter_hash"]

    path = tmp_path / "m5_training_formatter_v1.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest()
