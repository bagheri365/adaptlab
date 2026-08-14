from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from adaptlab.benchmark.config import BenchmarkConfigError, benchmark_config_from_mapping, load_benchmark_config
from adaptlab.domain.enums import BehaviorType, Difficulty, KnowledgeState, TaskFamily


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"


def _raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_full_v0_config() -> None:
    config = load_benchmark_config(CONFIG_PATH)

    assert config.benchmark_name == "AdaptLab Nimbus Benchmark"
    assert config.benchmark_version == "0.0.0"
    assert config.generation_seed == 1729
    assert config.world_schema_version == "1"
    assert config.splits.train == 300
    assert config.splits.validation == 150
    assert config.splits.test == 400
    assert config.generalization_sentinel.count == 100
    assert config.corpus.total_chunks == 180

    assert config.test_task_families.by_family()[TaskFamily.behavior_only] == 100
    assert config.test_difficulty.by_difficulty()[Difficulty.HARD] == 120
    assert config.changed_knowledge.by_state()[KnowledgeState.UPDATED] == 40
    assert config.behavior_only_test_behavior_types.by_behavior_type()[BehaviorType.SCHEMA_ADHERENCE] == 20


def test_rejects_negative_counts() -> None:
    raw = _raw_config()
    raw["splits"]["train"] = -1
    with pytest.raises(BenchmarkConfigError, match="non-negative"):
        benchmark_config_from_mapping(raw)


def test_rejects_task_family_total_mismatch() -> None:
    raw = _raw_config()
    raw["test_task_families"]["knowledge_only"] = 99
    with pytest.raises(BenchmarkConfigError, match="test_task_families total"):
        benchmark_config_from_mapping(raw)


def test_rejects_difficulty_total_mismatch() -> None:
    raw = _raw_config()
    raw["test_difficulty"]["HARD"] = 119
    with pytest.raises(BenchmarkConfigError, match="test_difficulty total"):
        benchmark_config_from_mapping(raw)


def test_rejects_changed_knowledge_total_mismatch() -> None:
    raw = _raw_config()
    raw["changed_knowledge"]["REMOVED"] = 29
    with pytest.raises(BenchmarkConfigError, match="changed_knowledge total"):
        benchmark_config_from_mapping(raw)


def test_rejects_evidence_absent_subtotal_mismatch() -> None:
    raw = _raw_config()
    raw["evidence_absent"]["changed_knowledge"] = 9
    with pytest.raises(BenchmarkConfigError, match="evidence_absent subtotals"):
        benchmark_config_from_mapping(raw)


def test_rejects_behavior_target_total_mismatch() -> None:
    raw = _raw_config()
    raw["behavior_only_test_behavior_types"]["ABSTENTION_BEHAVIOR"] = 19
    with pytest.raises(BenchmarkConfigError, match="behavior-only behavior-type targets"):
        benchmark_config_from_mapping(raw)


def test_rejects_corpus_composition_mismatch() -> None:
    raw = _raw_config()
    raw["corpus"]["domain_distractor"] = 29
    with pytest.raises(BenchmarkConfigError, match="corpus category counts"):
        benchmark_config_from_mapping(raw)


def test_rejects_unknown_fields() -> None:
    raw = _raw_config()
    raw["splits"]["future"] = 1
    with pytest.raises(BenchmarkConfigError, match="unknown fields"):
        benchmark_config_from_mapping(raw)
