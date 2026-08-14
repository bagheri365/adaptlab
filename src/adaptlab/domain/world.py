"""Authoritative typed schema for the fictional Nimbus benchmark world."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class FactStatus(str, Enum):
    """Lifecycle status of a versioned fact record."""

    ACTIVE = "active"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class NimbusFact:
    """A versioned record for one logical Nimbus fact."""

    logical_fact_id: str
    record_id: str
    component_family: str
    fact_type: str
    entity_id: str
    version: str
    value: Any
    status: FactStatus = FactStatus.ACTIVE
    severity: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "logical_fact_id",
            "record_id",
            "component_family",
            "fact_type",
            "entity_id",
            "version",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")

        if not isinstance(self.status, FactStatus):
            try:
                object.__setattr__(self, "status", FactStatus(self.status))
            except ValueError as exc:
                raise ValueError(f"invalid fact status: {self.status!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NimbusFact":
        """Construct a fact from its serialized representation."""

        return cls(**data)


@dataclass(frozen=True, slots=True)
class NimbusWorld:
    """Structured world truth from which documentation and tasks are derived."""

    world_schema_version: str
    generation_seed: int
    facts: tuple[NimbusFact, ...]

    def __post_init__(self) -> None:
        if not self.world_schema_version:
            raise ValueError("world_schema_version must be non-empty")
        if isinstance(self.generation_seed, bool) or not isinstance(self.generation_seed, int):
            raise TypeError("generation_seed must be an int")

        normalized_facts = tuple(
            fact if isinstance(fact, NimbusFact) else NimbusFact.from_dict(fact)
            for fact in self.facts
        )
        object.__setattr__(self, "facts", normalized_facts)

        record_ids: set[str] = set()
        logical_versions: set[tuple[str, str]] = set()
        for fact in self.facts:
            if fact.record_id in record_ids:
                raise ValueError(f"duplicate record_id: {fact.record_id}")
            record_ids.add(fact.record_id)

            pair = (fact.logical_fact_id, fact.version)
            if pair in logical_versions:
                raise ValueError(
                    "duplicate (logical_fact_id, version): "
                    f"({fact.logical_fact_id}, {fact.version})"
                )
            logical_versions.add(pair)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of authoritative world truth."""

        return {
            "world_schema_version": self.world_schema_version,
            "generation_seed": self.generation_seed,
            "facts": [fact.to_dict() for fact in self.facts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NimbusWorld":
        """Reconstruct a world from serialized data."""

        return cls(
            world_schema_version=data["world_schema_version"],
            generation_seed=data["generation_seed"],
            facts=tuple(NimbusFact.from_dict(fact) for fact in data["facts"]),
        )
