"""Deterministic lifecycle classification for Nimbus logical facts."""

from __future__ import annotations

from adaptlab.domain.enums import KnowledgeState
from adaptlab.domain.world import FactStatus, NimbusFact


def _authoritative_meaning(fact: NimbusFact) -> tuple[object, ...]:
    """Fields that define fact meaning independently of record/version identity."""

    return (
        fact.component_family,
        fact.fact_type,
        fact.entity_id,
        fact.value,
        fact.severity,
        fact.description,
    )


def classify_knowledge_state(
    v1_fact: NimbusFact,
    v2_fact: NimbusFact | None,
) -> KnowledgeState:
    """Classify how one logical fact changes from v1 to v2.

    Missing v2 records and explicit retirement/tombstones are REMOVED. Active
    records with unchanged authoritative meaning are UNCHANGED; otherwise they
    are UPDATED. No text similarity or model inference is involved.
    """

    if v2_fact is None:
        return KnowledgeState.REMOVED

    if v1_fact.logical_fact_id != v2_fact.logical_fact_id:
        raise ValueError(
            "cannot classify lifecycle for different logical_fact_id values: "
            f"{v1_fact.logical_fact_id!r} != {v2_fact.logical_fact_id!r}"
        )

    if v2_fact.status is FactStatus.RETIRED:
        return KnowledgeState.REMOVED

    if _authoritative_meaning(v1_fact) == _authoritative_meaning(v2_fact):
        return KnowledgeState.UNCHANGED

    return KnowledgeState.UPDATED
