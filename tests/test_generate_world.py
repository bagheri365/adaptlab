"""Tests for deterministic prototype Nimbus world generation."""

from __future__ import annotations

import json

from adaptlab.benchmark.generate_world import generate_world
from adaptlab.domain.enums import KnowledgeState
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import NimbusWorld


def _serialized(world: NimbusWorld) -> str:
    return json.dumps(world.to_dict(), sort_keys=True, separators=(",", ":"))


def _by_logical_id(world: NimbusWorld):
    grouped: dict[str, dict[str, object]] = {}
    for fact in world.facts:
        grouped.setdefault(fact.logical_fact_id, {})[fact.version] = fact
    return grouped


def test_same_seed_produces_identical_serialized_world() -> None:
    first = generate_world(1729)
    second = generate_world(1729)

    assert _serialized(first) == _serialized(second)
    assert first.to_dict() == second.to_dict()


def test_different_seed_produces_valid_world() -> None:
    first = generate_world(1729)
    second = generate_world(1730)

    assert isinstance(second, NimbusWorld)
    assert second.generation_seed == 1730
    assert 8 <= len({fact.logical_fact_id for fact in second.facts}) <= 12
    assert {fact.component_family for fact in second.facts} == {
        "authentication",
        "projects",
        "deployments",
    }
    assert _serialized(first) != _serialized(second)


def test_world_contains_all_required_lifecycle_examples() -> None:
    world = generate_world(1729)
    grouped = _by_logical_id(world)

    states = {
        classify_knowledge_state(versions["v1"], versions.get("v2"))
        for versions in grouped.values()
    }

    assert KnowledgeState.UNCHANGED in states
    assert KnowledgeState.UPDATED in states
    assert KnowledgeState.REMOVED in states


def test_facts_are_canonically_sorted_by_record_id() -> None:
    world = generate_world(1729)
    record_ids = [fact.record_id for fact in world.facts]

    assert record_ids == sorted(record_ids)


def test_logical_fact_pairing_uses_logical_fact_id() -> None:
    world = generate_world(1729)
    grouped = _by_logical_id(world)

    paired = grouped["AUTH_TOKEN_TTL"]
    assert paired["v1"].logical_fact_id == paired["v2"].logical_fact_id
    assert paired["v1"].record_id != paired["v2"].record_id


def test_full_world_is_deterministic_and_uses_configured_seed() -> None:
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import generate_full_world

    config = load_benchmark_config("configs/benchmark_v0.0.yaml")
    first = generate_full_world(config)
    second = generate_full_world(config)

    assert _serialized(first) == _serialized(second)
    assert first.generation_seed == config.generation_seed
    assert first.world_schema_version == config.world_schema_version


def test_full_world_has_minimum_structural_coverage() -> None:
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import (
        FULL_COMPONENT_FAMILIES,
        FULL_FACT_FAMILIES,
        generate_full_world,
        summarize_world,
    )

    config = load_benchmark_config("configs/benchmark_v0.0.yaml")
    world = generate_full_world(config)
    summary = summarize_world(world)

    assert set(summary.logical_facts_per_component) == set(FULL_COMPONENT_FAMILIES)
    assert all(count >= 8 for count in summary.logical_facts_per_component.values())
    assert set(summary.facts_per_fact_family) == set(FULL_FACT_FAMILIES)
    assert all(count >= 8 for count in summary.facts_per_fact_family.values())
    assert sum(summary.logical_facts_per_component.values()) >= 64
    assert summary.records_per_version["v1"] >= 64
    assert summary.records_per_version["v2"] >= 48
    assert summary.knowledge_lifecycle_counts["UNCHANGED"] >= 20
    assert summary.knowledge_lifecycle_counts["UPDATED"] >= 20
    assert summary.knowledge_lifecycle_counts["REMOVED"] >= 10


def test_full_world_contains_both_removal_mechanisms() -> None:
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import generate_full_world
    from adaptlab.domain.world import FactStatus

    world = generate_full_world(load_benchmark_config("configs/benchmark_v0.0.yaml"))
    grouped = _by_logical_id(world)

    explicit_retirements = [
        versions
        for versions in grouped.values()
        if versions.get("v2") is not None and versions["v2"].status is FactStatus.RETIRED
    ]
    missing_v2 = [versions for versions in grouped.values() if "v2" not in versions]

    assert explicit_retirements
    assert missing_v2


def test_full_world_summary_is_deterministic() -> None:
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_world import generate_full_world, summarize_world

    config = load_benchmark_config("configs/benchmark_v0.0.yaml")
    first = summarize_world(generate_full_world(config)).to_dict()
    second = summarize_world(generate_full_world(config)).to_dict()

    assert first == second
