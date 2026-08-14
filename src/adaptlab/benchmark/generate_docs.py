"""Deterministic documentation generation from authoritative Nimbus world truth."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.domain.enums import DocumentStyle
from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld

if TYPE_CHECKING:
    from adaptlab.benchmark.config import BenchmarkConfig


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


@dataclass(frozen=True, slots=True)
class CorpusCompositionReport:
    """Deterministic composition summary for the full retrieval corpus."""

    total_chunks: int
    current_authoritative: int
    obsolete_versioned: int
    competing_near_duplicate: int
    domain_distractor: int
    document_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_chunks": self.total_chunks,
            "current_authoritative": self.current_authoritative,
            "obsolete_versioned": self.obsolete_versioned,
            "competing_near_duplicate": self.competing_near_duplicate,
            "domain_distractor": self.domain_distractor,
            "document_count": self.document_count,
        }


def _full_doc(
    *,
    document_id: str,
    title: str,
    version: str,
    component: str,
    style: DocumentStyle,
    content: str,
    facts: list[NimbusFact],
) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        version=version,
        component_family=component,
        document_style=style,
        content=content,
        record_ids=tuple(f.record_id for f in facts),
        logical_fact_ids=tuple(dict.fromkeys(f.logical_fact_id for f in facts)),
    )


def _fact_sentence(fact: NimbusFact) -> str:
    if fact.status is FactStatus.RETIRED:
        return f"{fact.entity_id} is retired in Nimbus {fact.version}."
    return f"For {fact.entity_id}, the Nimbus {fact.version} value is {fact.value}."


def _category_from_chunk_id(chunk_id: str) -> str:
    prefixes = {
        "CHK_CUR_": "current_authoritative",
        "CHK_OBS_": "obsolete_versioned",
        "CHK_CMP_": "competing_near_duplicate",
        "CHK_DST_": "domain_distractor",
    }
    for prefix, category in prefixes.items():
        if chunk_id.startswith(prefix):
            return category
    raise ValueError(f"unrecognized full-corpus chunk category: {chunk_id}")


def generate_full_documents(
    world: NimbusWorld,
    config: "BenchmarkConfig",
) -> tuple[list[Document], list[DocumentChunk]]:
    """Generate the deterministic full-v0.0 retrieval corpus.

    The four corpus categories are represented by stable chunk-ID prefixes. This
    keeps the existing Document/DocumentChunk contract intact while allowing
    exact composition auditing. Authoritative and obsolete content is rendered
    only from structured world records; competing chunks paraphrase the same
    referenced truth; distractors carry no fact provenance.
    """

    if world.generation_seed != config.generation_seed:
        raise ValueError("world generation_seed must match benchmark config")
    if world.world_schema_version != config.world_schema_version:
        raise ValueError("world schema version must match benchmark config")

    by_component_version: dict[tuple[str, str], list[NimbusFact]] = defaultdict(list)
    by_record = {fact.record_id: fact for fact in world.facts}
    for fact in world.facts:
        by_component_version[(fact.component_family, fact.version)].append(fact)
    for facts in by_component_version.values():
        facts.sort(key=lambda f: f.record_id)

    current_records = sorted(
        (f for f in world.facts if f.version == "v2"), key=lambda f: f.record_id
    )
    obsolete_records = sorted(
        (f for f in world.facts if f.version == "v1"), key=lambda f: f.record_id
    )
    if not current_records or not obsolete_records:
        raise ValueError("full corpus generation requires both v1 and v2 world records")

    documents: list[Document] = []
    chunks: list[DocumentChunk] = []
    document_ids: set[str] = set()

    # Build one current document per component/style. Content is derived solely
    # from current structured truth for that component.
    current_docs: dict[tuple[str, DocumentStyle], Document] = {}
    components = sorted({fact.component_family for fact in world.facts})
    for component in components:
        facts = by_component_version.get((component, "v2"), [])
        for style in DocumentStyle:
            doc = _make_document(style, component, "v2", facts)
            current_docs[(component, style)] = doc
            documents.append(doc)
            document_ids.add(doc.document_id)

    # 90 current authoritative chunks. Records are cycled deterministically so
    # some facts appear in multiple styles/contexts instead of mapping 1:1 to a
    # single obvious chunk.
    styles = tuple(DocumentStyle)
    for index in range(config.corpus.current_authoritative):
        fact = current_records[index % len(current_records)]
        style = styles[(index // len(current_records) + index) % len(styles)]
        parent = current_docs[(fact.component_family, style)]
        # Every fourth chunk includes a neighboring current fact from the same
        # component when possible, creating controlled multi-fact context.
        refs = [fact]
        if index % 4 == 3:
            same_component = by_component_version[(fact.component_family, "v2")]
            if len(same_component) > 1:
                pos = same_component.index(fact)
                neighbor = same_component[(pos + 1) % len(same_component)]
                if neighbor.record_id != fact.record_id:
                    refs.append(neighbor)
        content = (
            f"Nimbus {fact.component_family} {style.value.replace('_', ' ')} excerpt.\n"
            + " ".join(_fact_sentence(ref) for ref in refs)
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"CHK_CUR_{index + 1:03d}_{fact.record_id}",
                document_id=parent.document_id,
                version="v2",
                component_family=fact.component_family,
                document_style=style,
                content=content,
                record_ids=tuple(ref.record_id for ref in refs),
                logical_fact_ids=tuple(dict.fromkeys(ref.logical_fact_id for ref in refs)),
                is_authoritative=True,
                is_obsolete=False,
            )
        )

    # 30 explicitly versioned obsolete chunks. Create compact v1 documents as
    # parents, with metadata making their historical status unambiguous.
    obsolete_doc_cache: dict[tuple[str, DocumentStyle], Document] = {}
    for index in range(config.corpus.obsolete_versioned):
        fact = obsolete_records[index % len(obsolete_records)]
        style = styles[(index + 1) % len(styles)]
        key = (fact.component_family, style)
        if key not in obsolete_doc_cache:
            component_facts = by_component_version[(fact.component_family, "v1")]
            base_id = f"DOC_{fact.component_family.upper()}_V1_{style.value.upper()}_OBSOLETE"
            doc = _full_doc(
                document_id=base_id,
                title=(
                    f"Nimbus {fact.component_family.title()} {style.value.replace('_', ' ').title()} "
                    "v1 (Obsolete)"
                ),
                version="v1",
                component=fact.component_family,
                style=style,
                content=(
                    "OBSOLETE NIMBUS v1 MATERIAL. Historical reference only.\n"
                    + _render_content(style, fact.component_family, "v1", component_facts)
                ),
                facts=component_facts,
            )
            obsolete_doc_cache[key] = doc
            documents.append(doc)
            document_ids.add(doc.document_id)
        parent = obsolete_doc_cache[key]
        chunks.append(
            DocumentChunk(
                chunk_id=f"CHK_OBS_{index + 1:03d}_{fact.record_id}",
                document_id=parent.document_id,
                version="v1",
                component_family=fact.component_family,
                document_style=style,
                content=(
                    "OBSOLETE Nimbus v1 excerpt. "
                    + _fact_sentence(fact)
                    + " Do not treat this historical text as current v2 authority."
                ),
                record_ids=(fact.record_id,),
                logical_fact_ids=(fact.logical_fact_id,),
                is_authoritative=False,
                is_obsolete=True,
            )
        )

    # 30 near-duplicate/competing chunks. They retain exact v2 provenance and
    # correct truth, but wording and identifiers overlap enough to challenge
    # retrieval. They are deliberately non-authoritative.
    for index in range(config.corpus.competing_near_duplicate):
        fact = current_records[(index * 7 + 3) % len(current_records)]
        style = styles[(index + 2) % len(styles)]
        parent = current_docs[(fact.component_family, style)]
        value_text = "retired" if fact.status is FactStatus.RETIRED else str(fact.value)
        chunks.append(
            DocumentChunk(
                chunk_id=f"CHK_CMP_{index + 1:03d}_{fact.logical_fact_id}",
                document_id=parent.document_id,
                version="v2",
                component_family=fact.component_family,
                document_style=style,
                content=(
                    f"Nimbus field note for {fact.logical_fact_id} / {fact.entity_id}: "
                    f"current setting is {value_text}. Nearby similarly named settings may differ."
                ),
                record_ids=(fact.record_id,),
                logical_fact_ids=(fact.logical_fact_id,),
                is_authoritative=False,
                is_obsolete=False,
            )
        )

    # 30 domain-plausible distractors. They are Nimbus-flavored operational text
    # but intentionally have no record/logical-fact provenance and answer no
    # structured-world question.
    for index in range(config.corpus.domain_distractor):
        component = components[index % len(components)]
        style = styles[(index + 3) % len(styles)]
        parent = current_docs[(component, style)]
        chunks.append(
            DocumentChunk(
                chunk_id=f"CHK_DST_{index + 1:03d}_{component.upper()}",
                document_id=parent.document_id,
                version="v2",
                component_family=component,
                document_style=style,
                content=(
                    f"Nimbus {component} operator note {index + 1}: the console groups related "
                    "controls together for easier navigation during routine maintenance."
                ),
                record_ids=(),
                logical_fact_ids=(),
                is_authoritative=False,
                is_obsolete=False,
            )
        )

    documents.sort(key=lambda doc: doc.document_id)
    chunks.sort(key=lambda chunk: chunk.chunk_id)

    # Defensive exact-count check keeps the generator tied to declarative config.
    report = summarize_corpus(documents, chunks)
    expected = config.corpus
    actual = report.to_dict()
    targets = {
        "total_chunks": expected.total_chunks,
        "current_authoritative": expected.current_authoritative,
        "obsolete_versioned": expected.obsolete_versioned,
        "competing_near_duplicate": expected.competing_near_duplicate,
        "domain_distractor": expected.domain_distractor,
    }
    for key, target in targets.items():
        if actual[key] != target:
            raise ValueError(f"full corpus {key} mismatch: {actual[key]} != {target}")

    # Ensure every referenced record still resolves exactly to structured truth.
    for chunk in chunks:
        for record_id in chunk.record_ids:
            if record_id not in by_record:
                raise ValueError(f"chunk {chunk.chunk_id} references unknown record {record_id}")

    return documents, chunks


def summarize_corpus(
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> CorpusCompositionReport:
    """Return deterministic full-corpus category counts."""

    counts = {
        "current_authoritative": 0,
        "obsolete_versioned": 0,
        "competing_near_duplicate": 0,
        "domain_distractor": 0,
    }
    for chunk in chunks:
        counts[_category_from_chunk_id(chunk.chunk_id)] += 1
    return CorpusCompositionReport(
        total_chunks=len(chunks),
        current_authoritative=counts["current_authoritative"],
        obsolete_versioned=counts["obsolete_versioned"],
        competing_near_duplicate=counts["competing_near_duplicate"],
        domain_distractor=counts["domain_distractor"],
        document_count=len(documents),
    )
