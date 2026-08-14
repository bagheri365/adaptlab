"""Deterministic generators for fictional Nimbus benchmark worlds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING

from adaptlab.domain.enums import KnowledgeState
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld

if TYPE_CHECKING:
    from adaptlab.benchmark.config import BenchmarkConfig

WORLD_SCHEMA_VERSION = "1.0"

FULL_COMPONENT_FAMILIES = (
    "authentication",
    "projects",
    "deployments",
    "storage",
    "billing",
    "observability",
    "permissions",
    "configuration",
)

FULL_FACT_FAMILIES = (
    "error_code",
    "limit",
    "permission",
    "endpoint_behavior",
    "configuration_value",
    "retention_setting",
    "feature_availability",
    "policy",
)


@dataclass(frozen=True, slots=True)
class WorldSummary:
    """Deterministic summary statistics for a generated Nimbus world."""

    logical_facts_per_component: dict[str, int]
    records_per_version: dict[str, int]
    facts_per_fact_family: dict[str, int]
    knowledge_lifecycle_counts: dict[str, int]

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "logical_facts_per_component": dict(self.logical_facts_per_component),
            "records_per_version": dict(self.records_per_version),
            "facts_per_fact_family": dict(self.facts_per_fact_family),
            "knowledge_lifecycle_counts": dict(self.knowledge_lifecycle_counts),
        }


def _record(
    logical_fact_id: str,
    component_family: str,
    fact_type: str,
    entity_id: str,
    version: str,
    value: object,
    *,
    status: FactStatus = FactStatus.ACTIVE,
    severity: str | None = None,
    description: str | None = None,
) -> NimbusFact:
    return NimbusFact(
        logical_fact_id=logical_fact_id,
        record_id=f"{logical_fact_id}_{version.upper()}",
        component_family=component_family,
        fact_type=fact_type,
        entity_id=entity_id,
        version=version,
        value=value,
        status=status,
        severity=severity,
        description=description,
    )


def generate_world(seed: int) -> NimbusWorld:
    """Generate the small deterministic prototype Nimbus world for ``seed``."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")

    rng = random.Random(seed)

    token_v1 = rng.choice([30, 45, 60])
    token_v2 = token_v1 + rng.choice([15, 30])
    project_limit = rng.choice([40, 50, 60])
    deploy_window_v1 = rng.choice([10, 15, 20])
    deploy_window_v2 = deploy_window_v1 + 10
    region = rng.choice(["nimbus-east", "nimbus-central", "nimbus-west"])
    retry_limit = rng.choice([2, 3, 4])

    facts = [
        _record("AUTH_TOKEN_TTL", "authentication", "duration_minutes", "access_token", "v1", token_v1),
        _record("AUTH_TOKEN_TTL", "authentication", "duration_minutes", "access_token", "v2", token_v2),
        _record("AUTH_MFA_METHOD", "authentication", "policy", "mfa", "v1", "totp"),
        _record("AUTH_MFA_METHOD", "authentication", "policy", "mfa", "v2", "totp"),
        _record("AUTH_LEGACY_KEY", "authentication", "credential_policy", "legacy_key", "v1", "enabled"),
        _record(
            "AUTH_LEGACY_KEY",
            "authentication",
            "credential_policy",
            "legacy_key",
            "v2",
            "retired",
            status=FactStatus.RETIRED,
            description="Legacy Nimbus keys are retired in v2.",
        ),
        _record("PROJ_MEMBER_LIMIT", "projects", "limit", "project_members", "v1", project_limit),
        _record("PROJ_MEMBER_LIMIT", "projects", "limit", "project_members", "v2", project_limit),
        _record("PROJ_DEFAULT_REGION", "projects", "default", "project_region", "v1", region),
        _record("PROJ_DEFAULT_REGION", "projects", "default", "project_region", "v2", region),
        _record("PROJ_ARCHIVE_DELAY", "projects", "duration_days", "archive", "v1", 14),
        _record("PROJ_ARCHIVE_DELAY", "projects", "duration_days", "archive", "v2", 7),
        _record("DEPLOY_ROLLBACK_WINDOW", "deployments", "duration_minutes", "rollback", "v1", deploy_window_v1),
        _record("DEPLOY_ROLLBACK_WINDOW", "deployments", "duration_minutes", "rollback", "v2", deploy_window_v2),
        _record("DEPLOY_RETRY_LIMIT", "deployments", "limit", "retry", "v1", retry_limit),
        _record("DEPLOY_RETRY_LIMIT", "deployments", "limit", "retry", "v2", retry_limit),
        _record("DEPLOY_CLASSIC_MODE", "deployments", "feature_flag", "classic_mode", "v1", True),
        _record("DEPLOY_HEALTH_PATH", "deployments", "path", "health_check", "v1", "/nimbus/health"),
        _record("DEPLOY_HEALTH_PATH", "deployments", "path", "health_check", "v2", "/nimbus/health"),
    ]

    return NimbusWorld(
        world_schema_version=WORLD_SCHEMA_VERSION,
        generation_seed=seed,
        facts=tuple(sorted(facts, key=lambda fact: fact.record_id)),
    )


def _full_value(
    rng: random.Random,
    component: str,
    fact_family: str,
    ordinal: int,
) -> object:
    """Create one fictional structured value without external or model input."""

    short = component[:4]
    if fact_family == "error_code":
        return f"NMB-{short.upper()}-{100 + ordinal + rng.randint(0, 19)}"
    if fact_family == "limit":
        return rng.choice([25, 40, 50, 75, 100, 125]) + ordinal
    if fact_family == "permission":
        return rng.choice(["viewer", "operator", "maintainer", "owner"])
    if fact_family == "endpoint_behavior":
        return rng.choice(["reject", "queue", "retry", "accept_with_warning"])
    if fact_family == "configuration_value":
        return f"{short}-{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}-{ordinal}"
    if fact_family == "retention_setting":
        return rng.choice([7, 14, 21, 30, 45, 60])
    if fact_family == "feature_availability":
        return rng.choice([True, False])
    if fact_family == "policy":
        return rng.choice(["strict", "standard", "permissive", "delegated"])
    raise ValueError(f"unsupported fact family: {fact_family}")


def _updated_value(value: object, fact_family: str) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 5
    if fact_family == "permission":
        order = ["viewer", "operator", "maintainer", "owner"]
        return order[(order.index(str(value)) + 1) % len(order)]
    if fact_family == "endpoint_behavior":
        order = ["reject", "queue", "retry", "accept_with_warning"]
        return order[(order.index(str(value)) + 1) % len(order)]
    if fact_family == "policy":
        order = ["strict", "standard", "permissive", "delegated"]
        return order[(order.index(str(value)) + 1) % len(order)]
    return f"{value}-v2"


def generate_full_world(config: "BenchmarkConfig") -> NimbusWorld:
    """Generate the expanded full-v0.0 Nimbus world from typed configuration.

    The world intentionally has 64 logical facts (eight per component), enough
    to support varied task composition without creating hundreds of one-off
    facts. Lifecycle assignment is frozen from stable component/family order,
    while seed-controlled values provide reproducible variation.
    """

    seed = config.generation_seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("config.generation_seed must be an int")

    rng = random.Random(seed)
    facts: list[NimbusFact] = []
    logical_ordinal = 0

    for component_index, component in enumerate(FULL_COMPONENT_FAMILIES):
        prefix = component[:4].upper()
        for family_index, fact_family in enumerate(FULL_FACT_FAMILIES):
            logical_ordinal += 1
            logical_fact_id = f"{prefix}_{fact_family.upper()}_{family_index + 1:02d}"
            entity_id = f"{component}_{fact_family}_{family_index + 1:02d}"
            v1_value = _full_value(rng, component, fact_family, logical_ordinal)

            # A stable lifecycle pattern produces substantial representation of
            # every state. Every fifth fact is removed; removal mechanism
            # alternates between explicit v2 tombstone and missing-v2 record.
            lifecycle_slot = (component_index * len(FULL_FACT_FAMILIES) + family_index) % 5
            if lifecycle_slot in (0, 1):
                lifecycle = KnowledgeState.UPDATED
            elif lifecycle_slot in (2, 3):
                lifecycle = KnowledgeState.UNCHANGED
            else:
                lifecycle = KnowledgeState.REMOVED

            facts.append(
                _record(
                    logical_fact_id,
                    component,
                    fact_family,
                    entity_id,
                    "v1",
                    v1_value,
                    description=f"Fictional Nimbus {component} {fact_family} fact.",
                )
            )

            if lifecycle is KnowledgeState.UPDATED:
                facts.append(
                    _record(
                        logical_fact_id,
                        component,
                        fact_family,
                        entity_id,
                        "v2",
                        _updated_value(v1_value, fact_family),
                        description=f"Fictional Nimbus {component} {fact_family} fact.",
                    )
                )
            elif lifecycle is KnowledgeState.UNCHANGED:
                facts.append(
                    _record(
                        logical_fact_id,
                        component,
                        fact_family,
                        entity_id,
                        "v2",
                        v1_value,
                        description=f"Fictional Nimbus {component} {fact_family} fact.",
                    )
                )
            elif logical_ordinal % 2 == 0:
                facts.append(
                    _record(
                        logical_fact_id,
                        component,
                        fact_family,
                        entity_id,
                        "v2",
                        "retired",
                        status=FactStatus.RETIRED,
                        description=f"Fictional Nimbus {component} {fact_family} fact retired in v2.",
                    )
                )
            # Odd removed ordinals intentionally omit v2 to model missing-v2 removal.

    return NimbusWorld(
        world_schema_version=config.world_schema_version,
        generation_seed=seed,
        facts=tuple(sorted(facts, key=lambda fact: fact.record_id)),
    )


def summarize_world(world: NimbusWorld) -> WorldSummary:
    """Return stable summary statistics for a Nimbus world."""

    grouped: dict[str, dict[str, NimbusFact]] = {}
    for fact in world.facts:
        grouped.setdefault(fact.logical_fact_id, {})[fact.version] = fact

    logical_per_component = Counter(
        versions["v1"].component_family for versions in grouped.values()
    )
    records_per_version = Counter(fact.version for fact in world.facts)
    facts_per_family = Counter(versions["v1"].fact_type for versions in grouped.values())
    lifecycle_counts = Counter(
        classify_knowledge_state(versions["v1"], versions.get("v2")).value
        for versions in grouped.values()
    )

    return WorldSummary(
        logical_facts_per_component=dict(sorted(logical_per_component.items())),
        records_per_version=dict(sorted(records_per_version.items())),
        facts_per_fact_family=dict(sorted(facts_per_family.items())),
        knowledge_lifecycle_counts=dict(sorted(lifecycle_counts.items())),
    )
