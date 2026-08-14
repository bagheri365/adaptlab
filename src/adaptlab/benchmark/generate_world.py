"""Deterministic generator for the prototype fictional Nimbus world."""

from __future__ import annotations

import random

from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld

WORLD_SCHEMA_VERSION = "1.0"


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
    """Generate a small deterministic Nimbus world for ``seed``.

    The generator is deliberately template-driven: benchmark truth is created
    here as structured data, without model calls or generated documentation.
    Facts are explicitly sorted by ``record_id`` before constructing the world.
    """

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
        # authentication — unchanged, updated, and explicitly retired examples
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
        # projects — stable and changed configuration facts
        _record("PROJ_MEMBER_LIMIT", "projects", "limit", "project_members", "v1", project_limit),
        _record("PROJ_MEMBER_LIMIT", "projects", "limit", "project_members", "v2", project_limit),
        _record("PROJ_DEFAULT_REGION", "projects", "default", "project_region", "v1", region),
        _record("PROJ_DEFAULT_REGION", "projects", "default", "project_region", "v2", region),
        _record("PROJ_ARCHIVE_DELAY", "projects", "duration_days", "archive", "v1", 14),
        _record("PROJ_ARCHIVE_DELAY", "projects", "duration_days", "archive", "v2", 7),
        # deployments — includes missing-v2 removal
        _record("DEPLOY_ROLLBACK_WINDOW", "deployments", "duration_minutes", "rollback", "v1", deploy_window_v1),
        _record("DEPLOY_ROLLBACK_WINDOW", "deployments", "duration_minutes", "rollback", "v2", deploy_window_v2),
        _record("DEPLOY_RETRY_LIMIT", "deployments", "limit", "retry", "v1", retry_limit),
        _record("DEPLOY_RETRY_LIMIT", "deployments", "limit", "retry", "v2", retry_limit),
        _record("DEPLOY_CLASSIC_MODE", "deployments", "feature_flag", "classic_mode", "v1", True),
        # v2 intentionally absent => REMOVED
        _record("DEPLOY_HEALTH_PATH", "deployments", "path", "health_check", "v1", "/nimbus/health"),
        _record("DEPLOY_HEALTH_PATH", "deployments", "path", "health_check", "v2", "/nimbus/health"),
    ]

    return NimbusWorld(
        world_schema_version=WORLD_SCHEMA_VERSION,
        generation_seed=seed,
        facts=tuple(sorted(facts, key=lambda fact: fact.record_id)),
    )
