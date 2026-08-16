from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from adaptlab.evaluation.prompts import load_prompt_contract

PROMPT_PATH = Path("configs/prompts/prompt_v1.yaml")


def test_prompt_v1_is_frozen_and_exact_artifact_hash_is_recorded() -> None:
    contract = load_prompt_contract(PROMPT_PATH)

    assert contract.prompt_version == "prompt_v1"
    assert contract.frozen is True
    assert contract.schema_version == "1"
    assert contract.prompt_hash == hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    assert contract.provenance() == {
        "prompt_version": "prompt_v1",
        "prompt_hash": contract.prompt_hash,
    }


def test_prompt_contract_is_one_generic_system_prompt() -> None:
    contract = load_prompt_contract(PROMPT_PATH)
    prompt = contract.system_prompt

    assert "Follow the requested format precisely." in prompt
    assert "Return only the requested output" in prompt
    assert "Do not invent unsupported information." in prompt
    assert "INSUFFICIENT_EVIDENCE" in prompt


def test_prompt_artifact_contains_no_benchmark_metadata_or_nimbus_facts() -> None:
    # Guard both the exact artifact bytes and the loaded system prompt. The frozen
    # contract must not smuggle benchmark labels or answer/evidence metadata.
    artifact = PROMPT_PATH.read_text(encoding="utf-8").lower()
    prompt = load_prompt_contract(PROMPT_PATH).system_prompt.lower()

    forbidden = (
        "nimbus",
        "expected_output",
        "expected answer",
        "task_family",
        "difficulty",
        "knowledge_state",
        "gold",
        "gold evidence",
        "split_type",
        "split type",
        "scoring_rule",
        "scoring rule",
    )
    for token in forbidden:
        assert token not in artifact
        assert token not in prompt


def test_prompt_artifact_does_not_contain_example_answers_or_gold_ids() -> None:
    prompt_bytes = PROMPT_PATH.read_bytes()
    # The prompt is intentionally tiny and generic; benchmark artifacts are not
    # inputs to its loader. This test also prevents common identifier namespaces.
    text = prompt_bytes.decode("utf-8").lower()
    for identifier_prefix in ("example_", "chunk_", "doc_", "fact_"):
        assert identifier_prefix not in text


def test_loader_rejects_unfrozen_prompt(tmp_path: Path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_text(
        'schema_version: "1"\nprompt_version: "test"\nfrozen: false\nsystem_prompt: "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen"):
        load_prompt_contract(path)


def test_loader_rejects_unversioned_or_extended_config(tmp_path: Path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_text(
        'schema_version: "1"\nprompt_version: "test"\nfrozen: true\nsystem_prompt: "x"\n'
        'task_family: "behavior_only"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid prompt config fields"):
        load_prompt_contract(path)
