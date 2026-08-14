"""Typed document schemas for deterministic Nimbus benchmark documentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from adaptlab.domain.enums import DocumentStyle


def _normalize_style(value: DocumentStyle | str) -> DocumentStyle:
    if isinstance(value, DocumentStyle):
        return value
    try:
        return DocumentStyle(value)
    except ValueError as exc:
        raise ValueError(f"invalid document_style: {value!r}") from exc


def _normalize_ids(values: tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not value for value in result):
        raise ValueError(f"{field_name} entries must be non-empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    title: str
    version: str
    component_family: str
    document_style: DocumentStyle
    content: str
    record_ids: tuple[str, ...]
    logical_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("document_id", "title", "version", "component_family", "content"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        object.__setattr__(self, "document_style", _normalize_style(self.document_style))
        object.__setattr__(self, "record_ids", _normalize_ids(self.record_ids, "record_ids"))
        object.__setattr__(
            self,
            "logical_fact_ids",
            _normalize_ids(self.logical_fact_ids, "logical_fact_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document_style"] = self.document_style.value
        data["record_ids"] = list(self.record_ids)
        data["logical_fact_ids"] = list(self.logical_fact_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    version: str
    component_family: str
    document_style: DocumentStyle
    content: str
    record_ids: tuple[str, ...]
    logical_fact_ids: tuple[str, ...]
    is_authoritative: bool
    is_obsolete: bool

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "document_id", "version", "component_family", "content"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        object.__setattr__(self, "document_style", _normalize_style(self.document_style))
        object.__setattr__(self, "record_ids", _normalize_ids(self.record_ids, "record_ids"))
        object.__setattr__(
            self,
            "logical_fact_ids",
            _normalize_ids(self.logical_fact_ids, "logical_fact_ids"),
        )
        if not isinstance(self.is_authoritative, bool):
            raise TypeError("is_authoritative must be a bool")
        if not isinstance(self.is_obsolete, bool):
            raise TypeError("is_obsolete must be a bool")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document_style"] = self.document_style.value
        data["record_ids"] = list(self.record_ids)
        data["logical_fact_ids"] = list(self.logical_fact_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)
