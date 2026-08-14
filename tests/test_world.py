import json

import pytest

from adaptlab.domain.enums import KnowledgeState
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld


def make_fact(
    *,
    logical_fact_id: str = "ERR_E17",
    record_id: str = "ERR_E17_V1",
    version: str = "v1",
    value: object = "retry",
    status: FactStatus = FactStatus.ACTIVE,
) -> NimbusFact:
    return NimbusFact(
        logical_fact_id=logical_fact_id,
        record_id=record_id,
        component_family="deployments",
        fact_type="error_policy",
        entity_id="E17",
        version=version,
        value=value,
        status=status,
        severity="warning",
        description="Fictional Nimbus deployment behavior.",
    )


def test_valid_world_creation() -> None:
    world = NimbusWorld(
        world_schema_version="1.0",
        generation_seed=1729,
        facts=(make_fact(),),
    )

    assert world.facts[0].record_id == "ERR_E17_V1"


def test_duplicate_record_id_rejected() -> None:
    first = make_fact()
    duplicate = make_fact(logical_fact_id="ERR_E18", version="v2")

    with pytest.raises(ValueError, match="duplicate record_id"):
        NimbusWorld("1.0", 1729, (first, duplicate))


def test_duplicate_logical_fact_version_rejected() -> None:
    first = make_fact()
    duplicate = make_fact(record_id="OTHER_RECORD")

    with pytest.raises(ValueError, match=r"duplicate \(logical_fact_id, version\)"):
        NimbusWorld("1.0", 1729, (first, duplicate))


def test_unchanged_lifecycle() -> None:
    v1 = make_fact()
    v2 = make_fact(record_id="ERR_E17_V2", version="v2")

    assert classify_knowledge_state(v1, v2) is KnowledgeState.UNCHANGED


def test_updated_lifecycle() -> None:
    v1 = make_fact()
    v2 = make_fact(record_id="ERR_E17_V2", version="v2", value="abort")

    assert classify_knowledge_state(v1, v2) is KnowledgeState.UPDATED


def test_removed_lifecycle_through_missing_v2() -> None:
    assert classify_knowledge_state(make_fact(), None) is KnowledgeState.REMOVED


def test_removed_lifecycle_through_explicit_retirement() -> None:
    v2 = make_fact(
        record_id="ERR_E17_V2",
        version="v2",
        value=None,
        status=FactStatus.RETIRED,
    )

    assert classify_knowledge_state(make_fact(), v2) is KnowledgeState.REMOVED


def test_lifecycle_rejects_mismatched_logical_fact_ids() -> None:
    v2 = make_fact(
        logical_fact_id="ERR_E18",
        record_id="ERR_E18_V2",
        version="v2",
    )

    with pytest.raises(ValueError, match="different logical_fact_id"):
        classify_knowledge_state(make_fact(), v2)


def test_serialization_deserialization_round_trip() -> None:
    world = NimbusWorld(
        world_schema_version="1.0",
        generation_seed=1729,
        facts=(make_fact(),),
    )

    serialized = json.dumps(world.to_dict(), sort_keys=True)
    restored = NimbusWorld.from_dict(json.loads(serialized))

    assert restored == world
    assert restored.facts[0].status is FactStatus.ACTIVE
