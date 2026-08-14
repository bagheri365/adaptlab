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
