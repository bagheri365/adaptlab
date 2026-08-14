"""Deterministic documentation generation from authoritative Nimbus world truth."""

from __future__ import annotations

from collections import defaultdict

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.domain.enums import DocumentStyle
from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld


def _value_text(fact: NimbusFact) -> str:
    if fact.status is FactStatus.RETIRED:
        return f"{fact.entity_id} is retired"
    return f"{fact.entity_id} = {fact.value}"


def _render_content(style: DocumentStyle, component: str, version: str, facts: list[NimbusFact]) -> str:
    lines = [f"Nimbus {component} {style.value.replace('_', ' ')} ({version})."]
    if style is DocumentStyle.reference_documentation:
        lines.append("Authoritative settings:")
    elif style is DocumentStyle.troubleshooting_guide:
        lines.append("Use these authoritative Nimbus facts when diagnosing behavior:")
    elif style is DocumentStyle.release_note:
        lines.append("Version facts recorded for this Nimbus release:")
    else:
        lines.append("Configuration values:")
    lines.extend(f"- {fact.logical_fact_id}: {_value_text(fact)}." for fact in facts)
    return "\n".join(lines)


def _make_document(style: DocumentStyle, component: str, version: str, facts: list[NimbusFact]) -> Document:
    suffix = style.value.upper()
    document_id = f"DOC_{component.upper()}_{version.upper()}_{suffix}"
    return Document(
        document_id=document_id,
        title=f"Nimbus {component.title()} {style.value.replace('_', ' ').title()} {version}",
        version=version,
        component_family=component,
        document_style=style,
        content=_render_content(style, component, version, facts),
        record_ids=tuple(fact.record_id for fact in facts),
        logical_fact_ids=tuple(dict.fromkeys(fact.logical_fact_id for fact in facts)),
    )


def _chunk_for_document(document: Document, *, authoritative: bool, obsolete: bool, suffix: str = "001") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"CHK_{document.document_id}_{suffix}",
        document_id=document.document_id,
        version=document.version,
        component_family=document.component_family,
        document_style=document.document_style,
        content=document.content,
        record_ids=document.record_ids,
        logical_fact_ids=document.logical_fact_ids,
        is_authoritative=authoritative,
        is_obsolete=obsolete,
    )


def generate_documents(world: NimbusWorld) -> tuple[list[Document], list[DocumentChunk]]:
    """Generate deterministic documents and chunks from structured world truth.

    Current v2 documentation is authoritative. A compact set of v1 documents is
    retained as obsolete material, while competing and distractor chunks are
    explicitly non-authoritative and never introduce benchmark truth.
    """

    by_component_version: dict[tuple[str, str], list[NimbusFact]] = defaultdict(list)
    for fact in world.facts:
        by_component_version[(fact.component_family, fact.version)].append(fact)

    for facts in by_component_version.values():
        facts.sort(key=lambda fact: fact.record_id)

    documents: list[Document] = []
    chunks: list[DocumentChunk] = []

    # All four required styles, across all component families, for current v2 truth.
    components = sorted({fact.component_family for fact in world.facts})
    for component in components:
        v2_facts = by_component_version.get((component, "v2"), [])
        for style in DocumentStyle:
            document = _make_document(style, component, "v2", v2_facts)
            documents.append(document)
            chunks.append(_chunk_for_document(document, authoritative=True, obsolete=False))

    # Retain one obsolete v1 reference document derived solely from v1 truth.
    obsolete_facts = by_component_version[("authentication", "v1")]
    obsolete_doc = _make_document(
        DocumentStyle.reference_documentation,
        "authentication",
        "v1",
        obsolete_facts,
    )
    documents.append(obsolete_doc)
    chunks.append(_chunk_for_document(obsolete_doc, authoritative=False, obsolete=True))

    # A semantically competing near-duplicate: same structured references and truth,
    # deliberately different phrasing. It is non-authoritative, not a new fact source.
    current_auth = _make_document(
        DocumentStyle.reference_documentation,
        "authentication",
        "v2",
        by_component_version[("authentication", "v2")],
    )
    chunks.append(
        DocumentChunk(
            chunk_id="CHK_COMPETING_AUTH_V2_REFERENCE",
            document_id=current_auth.document_id,
            version="v2",
            component_family="authentication",
            document_style=DocumentStyle.reference_documentation,
            content="Nimbus authentication summary.\n" + "\n".join(
                f"- {_value_text(fact)}." for fact in by_component_version[("authentication", "v2")]
            ),
            record_ids=current_auth.record_ids,
            logical_fact_ids=current_auth.logical_fact_ids,
            is_authoritative=False,
            is_obsolete=False,
        )
    )

    # Nimbus-plausible distractor with no authoritative fact references.
    distractor_parent = next(
        doc
        for doc in documents
        if doc.component_family == "projects"
        and doc.version == "v2"
        and doc.document_style is DocumentStyle.troubleshooting_guide
    )
    chunks.append(
        DocumentChunk(
            chunk_id="CHK_DISTRACTOR_PROJECTS_UI",
            document_id=distractor_parent.document_id,
            version="v2",
            component_family="projects",
            document_style=DocumentStyle.troubleshooting_guide,
            content=(
                "Nimbus console tip: collapsing the project navigation panel can make "
                "dense troubleshooting sessions easier to scan."
            ),
            record_ids=(),
            logical_fact_ids=(),
            is_authoritative=False,
            is_obsolete=False,
        )
    )

    documents.sort(key=lambda doc: doc.document_id)
    chunks.sort(key=lambda chunk: chunk.chunk_id)
    return documents, chunks
