"""Deterministic prototype benchmark task generation for Nimbus."""

from __future__ import annotations

from collections.abc import Iterable

from adaptlab.benchmark.documents import Document, DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import (
    BehaviorType,
    Difficulty,
    DocumentStyle,
    EvidenceStatus,
    KnowledgeState,
    Split,
    SplitType,
    TaskFamily,
    ScoringRule,
)
from adaptlab.domain.lifecycle import classify_knowledge_state
from adaptlab.domain.world import FactStatus, NimbusFact, NimbusWorld

BENCHMARK_VERSION = "0.1-prototype"


def _facts_by_logical_id(world: NimbusWorld) -> dict[str, dict[str, NimbusFact]]:
    grouped: dict[str, dict[str, NimbusFact]] = {}
    for fact in world.facts:
        grouped.setdefault(fact.logical_fact_id, {})[fact.version] = fact
    return grouped


def _reference_evidence(
    fact: NimbusFact,
    documents: Iterable[Document],
    chunks: Iterable[DocumentChunk],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return deterministic authoritative reference-document evidence for ``fact``."""

    matching_documents = sorted(
        (
            document
            for document in documents
            if document.version == fact.version
            and document.component_family == fact.component_family
            and document.document_style is DocumentStyle.reference_documentation
            and fact.record_id in document.record_ids
        ),
        key=lambda document: document.document_id,
    )
    if not matching_documents:
        raise ValueError(f"no reference document found for record_id={fact.record_id}")

    document = matching_documents[0]
    matching_chunks = sorted(
        (
            chunk
            for chunk in chunks
            if chunk.document_id == document.document_id
            and chunk.is_authoritative
            and not chunk.is_obsolete
            and fact.record_id in chunk.record_ids
        ),
        key=lambda chunk: chunk.chunk_id,
    )
    if not matching_chunks:
        raise ValueError(f"no authoritative chunk found for record_id={fact.record_id}")

    return (document.document_id,), (matching_chunks[0].chunk_id,)


def _present_example(
    *,
    example_id: str,
    task_family: TaskFamily,
    behavior_type: BehaviorType | None,
    difficulty: Difficulty,
    fact: NimbusFact,
    question: str,
    expected_output: object,
    seed: int,
    documents: list[Document],
    chunks: list[DocumentChunk],
    knowledge_state: KnowledgeState = KnowledgeState.NOT_APPLICABLE,
    scoring_rule: ScoringRule = ScoringRule.FACT_VALUE,
    scoring_parameters: dict[str, object] | None = None,
) -> BenchmarkExample:
    gold_document_ids, gold_chunk_ids = _reference_evidence(fact, documents, chunks)
    return BenchmarkExample(
        example_id=example_id,
        benchmark_version=BENCHMARK_VERSION,
        task_family=task_family,
        behavior_type=behavior_type,
        difficulty=difficulty,
        split=Split.train,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version=fact.version,
        knowledge_state=knowledge_state,
        evidence_status=EvidenceStatus.PRESENT,
        question=question,
        expected_output=expected_output,
        required_record_ids=(fact.record_id,),
        required_logical_fact_ids=(fact.logical_fact_id,),
        gold_document_ids=gold_document_ids,
        gold_chunk_ids=gold_chunk_ids,
        generation_seed=seed,
        scoring_rule=scoring_rule,
        scoring_parameters=scoring_parameters,
        lifecycle_logical_fact_id=(fact.logical_fact_id if task_family is TaskFamily.changed_knowledge else None),
    )


def _behavior_only_examples(seed: int) -> list[BenchmarkExample]:
    specs = [
        (
            "BHV_001_SCHEMA",
            BehaviorType.SCHEMA_ADHERENCE,
            Difficulty.EASY,
            "The Nimbus incident has code E17 and severity high. Return exactly a JSON object with keys code and severity.",
            {"code": "E17", "severity": "high"},
            ScoringRule.STRUCTURED_EXTRACTION,
        ),
        (
            "BHV_002_DECISION",
            BehaviorType.CONDITIONAL_DECISION_RULE,
            Difficulty.MEDIUM,
            "Policy: approve a deployment only when tests_passed=true and rollback_ready=true. Candidate: tests_passed=true, rollback_ready=false. Answer APPROVE or DENY.",
            "DENY",
            ScoringRule.CONDITIONAL_RULE,
        ),
        (
            "BHV_003_EXTRACT",
            BehaviorType.TRANSFORMATION_EXTRACTION,
            Difficulty.EASY,
            "Extract the Nimbus project IDs from this text in appearance order: 'Reviewed PRJ-104, ignored a note, then opened PRJ-208.' Return only a list of IDs.",
            ["PRJ-104", "PRJ-208"],
            ScoringRule.STRUCTURED_EXTRACTION,
        ),
        (
            "BHV_004_CLASSIFY",
            BehaviorType.CLASSIFICATION_POLICY,
            Difficulty.MEDIUM,
            "Classification rule: URGENT if severity=critical or customer_blocked=true; otherwise ROUTINE. Case: severity=medium, customer_blocked=true. Answer only the class.",
            "URGENT",
            ScoringRule.CLASSIFICATION,
        ),
        (
            "BHV_005_ABSTAIN",
            BehaviorType.ABSTENTION_BEHAVIOR,
            Difficulty.HARD,
            "Use only the supplied facts. Facts: component=projects; owner=team-iris. Question: What is the project's retention period? If the answer is not supplied, answer INSUFFICIENT_INFORMATION.",
            "INSUFFICIENT_INFORMATION",
            ScoringRule.ABSTENTION,
        ),
    ]

    return [
        BenchmarkExample(
            example_id=example_id,
            benchmark_version=BENCHMARK_VERSION,
            task_family=TaskFamily.behavior_only,
            behavior_type=behavior_type,
            difficulty=difficulty,
            split=Split.train,
            split_type=SplitType.iid,
            holdout_dimension=None,
            holdout_group=None,
            knowledge_version=None,
            knowledge_state=KnowledgeState.NOT_APPLICABLE,
            evidence_status=EvidenceStatus.NOT_APPLICABLE,
            question=question,
            expected_output=expected_output,
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_document_ids=(),
            gold_chunk_ids=(),
            generation_seed=seed,
            scoring_rule=scoring_rule,
        )
        for example_id, behavior_type, difficulty, question, expected_output, scoring_rule in specs
    ]


def _knowledge_only_examples(
    world: NimbusWorld,
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> list[BenchmarkExample]:
    facts = _facts_by_logical_id(world)
    seed = world.generation_seed
    specs = [
        ("KNW_001_TOKEN_TTL", Difficulty.EASY, facts["AUTH_TOKEN_TTL"]["v2"], "What is the Nimbus v2 access-token TTL?"),
        ("KNW_002_MEMBER_LIMIT", Difficulty.EASY, facts["PROJ_MEMBER_LIMIT"]["v2"], "What is the Nimbus v2 project member limit?"),
        ("KNW_003_REGION", Difficulty.MEDIUM, facts["PROJ_DEFAULT_REGION"]["v2"], "Which region is the Nimbus v2 default for new projects?"),
        ("KNW_004_RETRY", Difficulty.MEDIUM, facts["DEPLOY_RETRY_LIMIT"]["v2"], "What retry limit does Nimbus v2 use for deployments?"),
    ]
    examples = [
        _present_example(
            example_id=example_id,
            task_family=TaskFamily.knowledge_only,
            behavior_type=None,
            difficulty=difficulty,
            fact=fact,
            question=question,
            expected_output=fact.value,
            seed=seed,
            documents=documents,
            chunks=chunks,
        )
        for example_id, difficulty, fact, question in specs
    ]
    examples.append(
        BenchmarkExample(
            example_id="KNW_005_ABSENT",
            benchmark_version=BENCHMARK_VERSION,
            task_family=TaskFamily.knowledge_only,
            behavior_type=None,
            difficulty=Difficulty.HARD,
            split=Split.train,
            split_type=SplitType.iid,
            holdout_dimension=None,
            holdout_group=None,
            knowledge_version="v2",
            knowledge_state=KnowledgeState.NOT_APPLICABLE,
            evidence_status=EvidenceStatus.ABSENT,
            question="What is the Nimbus v2 billing export retention period?",
            expected_output="INSUFFICIENT_EVIDENCE",
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_document_ids=(),
            gold_chunk_ids=(),
            generation_seed=seed,
            scoring_rule=ScoringRule.ABSTENTION,
        )
    )
    return examples


def _behavior_knowledge_examples(
    world: NimbusWorld,
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> list[BenchmarkExample]:
    facts = _facts_by_logical_id(world)
    seed = world.generation_seed
    specs = [
        (
            "BKN_001_SCHEMA",
            BehaviorType.SCHEMA_ADHERENCE,
            Difficulty.MEDIUM,
            facts["AUTH_TOKEN_TTL"]["v2"],
            "Using Nimbus v2 documentation, return the access-token TTL as exactly {\"ttl_minutes\": <integer>}.",
            lambda fact: {"ttl_minutes": fact.value},
        ),
        (
            "BKN_002_DECISION",
            BehaviorType.CONDITIONAL_DECISION_RULE,
            Difficulty.HARD,
            facts["PROJ_MEMBER_LIMIT"]["v2"],
            "A project requests 55 members. Using the documented Nimbus v2 member limit, answer ALLOW if 55 is at or below the limit; otherwise DENY.",
            lambda fact: "ALLOW" if 55 <= int(fact.value) else "DENY",
        ),
        (
            "BKN_003_EXTRACT",
            BehaviorType.TRANSFORMATION_EXTRACTION,
            Difficulty.MEDIUM,
            facts["PROJ_DEFAULT_REGION"]["v2"],
            "From Nimbus v2 evidence, extract the default project region and return only the region string.",
            lambda fact: fact.value,
        ),
        (
            "BKN_004_CLASSIFY",
            BehaviorType.CLASSIFICATION_POLICY,
            Difficulty.HARD,
            facts["DEPLOY_ROLLBACK_WINDOW"]["v2"],
            "Classify the Nimbus v2 rollback window as SHORT when <=20 minutes and EXTENDED when >20 minutes. Use external Nimbus evidence and answer only the class.",
            lambda fact: "SHORT" if int(fact.value) <= 20 else "EXTENDED",
        ),
        (
            "BKN_005_ABSTAIN",
            BehaviorType.ABSTENTION_BEHAVIOR,
            Difficulty.HARD,
            facts["AUTH_LEGACY_KEY"]["v2"],
            "Using Nimbus v2 evidence, state ACTIVE if legacy keys remain active or RETIRED if the authoritative record explicitly retires them. Abstain only if the evidence cannot decide.",
            lambda fact: "RETIRED" if fact.status is FactStatus.RETIRED else "ACTIVE",
        ),
    ]
    return [
        _present_example(
            example_id=example_id,
            task_family=TaskFamily.behavior_knowledge,
            behavior_type=behavior_type,
            difficulty=difficulty,
            fact=fact,
            question=question,
            expected_output=make_output(fact),
            seed=seed,
            documents=documents,
            chunks=chunks,
            scoring_rule={
                "BKN_001_SCHEMA": ScoringRule.STRUCTURED_EXTRACTION,
                "BKN_002_DECISION": ScoringRule.CONDITIONAL_RULE,
                "BKN_003_EXTRACT": ScoringRule.STRUCTURED_EXTRACTION,
                "BKN_004_CLASSIFY": ScoringRule.CLASSIFICATION,
                "BKN_005_ABSTAIN": ScoringRule.RETIRED_STATUS,
            }[example_id],
            scoring_parameters={
                "BKN_001_SCHEMA": {"output_key": "ttl_minutes", "coerce": "int"},
                "BKN_002_DECISION": {"candidate": 55, "operator": "lte", "true_output": "ALLOW", "false_output": "DENY"},
                "BKN_003_EXTRACT": {"mode": "scalar"},
                "BKN_004_CLASSIFY": {"threshold": 20, "operator": "gt", "true_output": "EXTENDED", "false_output": "SHORT"},
                "BKN_005_ABSTAIN": None,
            }[example_id],
        )
        for example_id, behavior_type, difficulty, fact, question, make_output in specs
    ]


def _changed_knowledge_examples(
    world: NimbusWorld,
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> list[BenchmarkExample]:
    grouped = _facts_by_logical_id(world)
    seed = world.generation_seed
    logical_ids = [
        "AUTH_MFA_METHOD",       # unchanged
        "AUTH_TOKEN_TTL",        # updated
        "AUTH_LEGACY_KEY",       # removed via retirement
        "DEPLOY_RETRY_LIMIT",    # unchanged
    ]
    difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.HARD]
    examples: list[BenchmarkExample] = []

    current_questions = {
        "AUTH_MFA_METHOD": "What is the current Nimbus MFA method?",
        "AUTH_TOKEN_TTL": "What is the current Nimbus access-token TTL?",
        "AUTH_LEGACY_KEY": "What is the current Nimbus status of legacy authentication keys?",
        "DEPLOY_RETRY_LIMIT": "What is the current Nimbus deployment retry limit?",
    }

    for index, (logical_id, difficulty) in enumerate(zip(logical_ids, difficulties, strict=True), start=1):
        versions = grouped[logical_id]
        v1 = versions["v1"]
        v2 = versions.get("v2")
        state = classify_knowledge_state(v1, v2)
        if v2 is None:
            raise ValueError(
                f"present changed-knowledge example requires a current v2 record: {logical_id}"
            )

        expected_output = "RETIRED" if state is KnowledgeState.REMOVED else v2.value
        examples.append(
            _present_example(
                example_id=f"CHG_{index:03d}_{state.value}",
                task_family=TaskFamily.changed_knowledge,
                behavior_type=None,
                difficulty=difficulty,
                fact=v2,
                question=current_questions[logical_id],
                expected_output=expected_output,
                seed=seed,
                documents=documents,
                chunks=chunks,
                knowledge_state=state,
                scoring_rule=(ScoringRule.RETIRED_STATUS if state is KnowledgeState.REMOVED else ScoringRule.FACT_VALUE),
            )
        )

    # Explicit evidence-absent changed-knowledge case. The lifecycle state remains
    # benchmark metadata, while the task asks an ordinary current-knowledge question.
    versions = grouped["DEPLOY_CLASSIC_MODE"]
    state = classify_knowledge_state(versions["v1"], versions.get("v2"))
    examples.append(
        BenchmarkExample(
            example_id="CHG_005_ABSENT",
            benchmark_version=BENCHMARK_VERSION,
            task_family=TaskFamily.changed_knowledge,
            behavior_type=None,
            difficulty=Difficulty.HARD,
            split=Split.train,
            split_type=SplitType.iid,
            holdout_dimension=None,
            holdout_group=None,
            knowledge_version="v2",
            knowledge_state=state,
            evidence_status=EvidenceStatus.ABSENT,
            question="Is Nimbus classic deployment mode currently available?",
            expected_output="INSUFFICIENT_EVIDENCE",
            required_record_ids=(),
            required_logical_fact_ids=(),
            gold_document_ids=(),
            gold_chunk_ids=(),
            generation_seed=seed,
            scoring_rule=ScoringRule.ABSTENTION,
            lifecycle_logical_fact_id="DEPLOY_CLASSIC_MODE",
        )
    )
    return examples


def generate_tasks(
    world: NimbusWorld,
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> list[BenchmarkExample]:
    """Generate the deterministic prototype benchmark examples.

    The prototype intentionally optimizes for invariant and coverage checks rather
    than statistical balance. Output is explicitly canonicalized by ``example_id``.
    """

    examples = [
        *_behavior_only_examples(world.generation_seed),
        *_knowledge_only_examples(world, documents, chunks),
        *_behavior_knowledge_examples(world, documents, chunks),
        *_changed_knowledge_examples(world, documents, chunks),
    ]
    examples.sort(key=lambda example: example.example_id)
    return examples
