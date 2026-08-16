from __future__ import annotations

from pathlib import Path

import yaml


PROMPT_CFG = Path("configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml")
ORACLE_CFG = Path("configs/evaluation_conditions/milestone3_ollama_oracle_context_v1.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_milestone3_canonical_configs_share_the_required_common_settings() -> None:
    prompt = _load(PROMPT_CFG)
    oracle = _load(ORACLE_CFG)

    shared_paths = (
        ("schema_version",),
        ("condition_version",),
        ("canonical_milestone",),
        ("benchmark", "benchmark_version"),
        ("benchmark", "benchmark_manifest_hash"),
        ("prompt", "prompt_version"),
        ("prompt", "prompt_hash"),
        ("provider", "name"),
        ("provider", "base_url_policy"),
        ("provider", "model_tag"),
        ("provider", "ollama_version"),
        ("provider", "ollama_model_digest"),
        ("provider", "ollama_metadata_policy"),
        ("request", "temperature"),
        ("request", "seed"),
        ("request", "context_length"),
        ("request", "max_tokens"),
        ("request", "stream"),
        ("request", "think"),
        ("retry_policy", "runner_max_retries"),
        ("retry_policy", "provider_max_retries"),
        ("scoring", "scorer_version"),
        ("scoring", "normalizer_version"),
        ("scoring", "evidence_format_version"),
        ("cache_policy", "type"),
        ("cache_policy", "storage"),
        ("cache_policy", "read_mode"),
        ("selection_controls", "validation_split_only"),
        ("selection_controls", "test_set_access_during_selection"),
        ("selection_controls", "sentinel_access_during_selection"),
        ("selection_controls", "repeated_prompt_search"),
    )

    for path in shared_paths:
        left = prompt
        right = oracle
        for key in path:
            left = left[key]
            right = right[key]
        assert left == right


def test_oracle_config_freezes_the_only_intended_difference() -> None:
    prompt = _load(PROMPT_CFG)
    oracle = _load(ORACLE_CFG)

    assert prompt["adaptation_method"] == "PROMPT"
    assert oracle["adaptation_method"] == "ORACLE_CONTEXT"
    assert prompt["input_construction"]["prompt"] == "benchmark question only"
    assert oracle["input_construction"]["prompt"] == "benchmark question only"
    assert oracle["input_construction"]["oracle_context"]["enabled_when_evidence_status"] == "PRESENT"
    assert oracle["input_construction"]["oracle_context"]["absent_evidence_behavior"] == (
        "byte-identical to PROMPT; do not add an empty evidence section"
    )

