"""Deterministic prototype benchmark task generation for Nimbus."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from adaptlab.benchmark.config import BenchmarkConfig
    from adaptlab.benchmark.holdout import FullHoldoutPolicy

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
            {"literal_value": {"code": "E17", "severity": "high"}, "mode": "literal_scalar"},
        ),
        (
            "BHV_002_DECISION",
            BehaviorType.CONDITIONAL_DECISION_RULE,
            Difficulty.MEDIUM,
            "Policy: approve a deployment only when tests_passed=true and rollback_ready=true. Candidate: tests_passed=true, rollback_ready=false. Answer APPROVE or DENY.",
            "DENY",
            ScoringRule.CONDITIONAL_RULE,
            {"candidate": 0, "threshold": 1, "operator": "gte", "true_output": "APPROVE", "false_output": "DENY"},
        ),
        (
            "BHV_003_EXTRACT",
            BehaviorType.TRANSFORMATION_EXTRACTION,
            Difficulty.EASY,
            "Extract the Nimbus project IDs from this text in appearance order: 'Reviewed PRJ-104, ignored a note, then opened PRJ-208.' Return only a list of IDs.",
            ["PRJ-104", "PRJ-208"],
            ScoringRule.STRUCTURED_EXTRACTION,
            {"literal_value": ["PRJ-104", "PRJ-208"], "mode": "literal_scalar"},
        ),
        (
            "BHV_004_CLASSIFY",
            BehaviorType.CLASSIFICATION_POLICY,
            Difficulty.MEDIUM,
            "Classification rule: URGENT if severity=critical or customer_blocked=true; otherwise ROUTINE. Case: severity=medium, customer_blocked=true. Answer only the class.",
            "URGENT",
            ScoringRule.CLASSIFICATION,
            {"value": 1, "threshold": 1, "operator": "gte", "true_output": "URGENT", "false_output": "ROUTINE"},
        ),
        (
            "BHV_005_ABSTAIN",
            BehaviorType.ABSTENTION_BEHAVIOR,
            Difficulty.HARD,
            "Use only the supplied facts. Facts: component=projects; owner=team-iris. Question: What is the project's retention period? If the answer is not supplied, answer INSUFFICIENT_INFORMATION.",
            "INSUFFICIENT_INFORMATION",
            ScoringRule.ABSTENTION,
            {"abstention_output": "INSUFFICIENT_INFORMATION", "prompt_only": True},
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
            scoring_parameters=scoring_parameters,
        )
        for example_id, behavior_type, difficulty, question, expected_output, scoring_rule, scoring_parameters in specs
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

# ---------------------------------------------------------------------------
# Full v0.0 benchmark task generation
# ---------------------------------------------------------------------------


def _full_current_evidence(
    fact: NimbusFact,
    documents: list[Document],
    chunks: list[DocumentChunk],
    *,
    cardinality: int = 1,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return deterministic current authoritative gold evidence for a fact."""

    if cardinality < 1:
        raise ValueError("evidence cardinality must be positive")

    matches = sorted(
        (
            chunk
            for chunk in chunks
            if chunk.version == "v2"
            and chunk.is_authoritative
            and not chunk.is_obsolete
            and fact.record_id in chunk.record_ids
        ),
        key=lambda chunk: chunk.chunk_id,
    )
    if len(matches) < cardinality:
        raise ValueError(
            f"only {len(matches)} current authoritative chunks exist for {fact.record_id}; "
            f"requested {cardinality}"
        )
    selected = matches[:cardinality]
    document_ids = tuple(sorted({chunk.document_id for chunk in selected}))
    chunk_ids = tuple(chunk.chunk_id for chunk in selected)
    return document_ids, chunk_ids


_KNOWLEDGE_INTENTS = (
    "current_value_lookup",
    "version_specific_lookup",
    "fact_family_lookup",
    "component_scoped_lookup",
    "component_fact_family_lookup",
)


def _required_source_scope(cardinality: int) -> str:
    if cardinality < 1:
        raise ValueError("required evidence cardinality must be positive for evidence-present tasks")
    if cardinality == 1:
        return "one current authoritative Nimbus source"
    return f"exactly {cardinality} current authoritative Nimbus sources"


def _knowledge_question(fact: NimbusFact, intent: str, evidence_cardinality: int = 1) -> str:
    scope = _required_source_scope(evidence_cardinality)
    if intent == "current_value_lookup":
        return f"Using {scope}, what is the current Nimbus value for {fact.entity_id}?"
    if intent == "version_specific_lookup":
        return f"Using {scope}, what current authoritative value does Nimbus v2 record for {fact.entity_id}?"
    if intent == "fact_family_lookup":
        family = fact.fact_type.replace("_", " ")
        return f"Using {scope}, what current Nimbus {family} value applies to {fact.entity_id}?"
    if intent == "component_scoped_lookup":
        return (
            f"Using {scope} within Nimbus {fact.component_family}, what current value is assigned to "
            f"{fact.entity_id}?"
        )
    if intent == "component_fact_family_lookup":
        family = fact.fact_type.replace("_", " ")
        return (
            f"Using {scope}, within Nimbus {fact.component_family}, what current {family} value "
            f"applies to {fact.entity_id}?"
        )
    raise ValueError(f"unsupported knowledge-only intent: {intent}")


def _full_knowledge_task_inventory(
    facts: list[NimbusFact],
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> list[tuple[NimbusFact, str, int]]:
    """Enumerate genuinely distinct, evidence-supported knowledge-only tasks.

    Distinctness comes from structured task semantics: logical fact, a supported
    lookup intent, and required evidence cardinality. No split labels, nonces, or
    wording-only variants are used.
    """

    inventory: list[tuple[NimbusFact, str, int]] = []
    for fact in sorted(facts, key=lambda item: item.record_id):
        available = sum(
            1
            for chunk in chunks
            if chunk.version == "v2"
            and chunk.is_authoritative
            and not chunk.is_obsolete
            and fact.record_id in chunk.record_ids
        )
        for intent in _KNOWLEDGE_INTENTS:
            for cardinality in range(1, min(available, 3) + 1):
                inventory.append((fact, intent, cardinality))
    return sorted(inventory, key=lambda item: (item[2], item[0].record_id, item[1]))


_CHANGED_INTENTS = (
    "current_value_lookup",
    "version_specific_lookup",
    "fact_family_lookup",
    "component_scoped_lookup",
    "status_aware_current_lookup",
)


def _changed_question(fact: NimbusFact, intent: str, evidence_cardinality: int = 1) -> str:
    """Render a current-knowledge question where version change is task-relevant.

    Changed-knowledge questions deliberately name the v1-to-v2 transition so they
    remain distinct from ordinary knowledge-only lookups without adding cosmetic
    nonces or split-specific prose. The answer is still the current v2 authority.
    """

    evidence_scope = (
        "the current authoritative Nimbus source"
        if evidence_cardinality == 1
        else f"{evidence_cardinality} current authoritative Nimbus sources"
    )
    if intent == "current_value_lookup":
        return (
            f"Using {evidence_scope}, after any Nimbus v1-to-v2 change, what is the "
            f"current value for {fact.entity_id}?"
        )
    if intent == "version_specific_lookup":
        return (
            f"Using {evidence_scope}, after accounting for Nimbus v1 history, what "
            f"authoritative value does Nimbus v2 record for {fact.entity_id}?"
        )
    if intent == "fact_family_lookup":
        family = fact.fact_type.replace("_", " ")
        return (
            f"Using {evidence_scope}, for the current Nimbus v2 state after any "
            f"prior-version change, what {family} value applies to {fact.entity_id}?"
        )
    if intent == "component_scoped_lookup":
        return (
            f"Using {evidence_scope} for Nimbus {fact.component_family}, after accounting "
            f"for v1-to-v2 changes, what current value is assigned to {fact.entity_id}?"
        )
    if intent == "status_aware_current_lookup":
        return (
            f"Using {evidence_scope}, inspect the current Nimbus v2 record for {fact.entity_id}. "
            "Return RETIRED if the current record is retired; otherwise return its current value."
        )
    raise ValueError(f"unsupported changed-knowledge intent: {intent}")


def _changed_present_inventory(pairs, documents, chunks):
    inventory = []
    for v1, v2 in sorted(pairs, key=lambda pair: pair[0].logical_fact_id):
        if v2 is None:
            continue
        state = classify_knowledge_state(v1, v2)
        available = sum(1 for chunk in chunks if chunk.version == "v2" and chunk.is_authoritative and not chunk.is_obsolete and v2.record_id in chunk.record_ids)
        for intent in _CHANGED_INTENTS:
            for cardinality in range(1, min(available, 3) + 1):
                inventory.append((v2, state, intent, cardinality))
    return sorted(inventory, key=lambda item: (item[3], item[0].record_id, item[2]))


def _allocate_changed_tasks(world, holdout_policy, documents, chunks, config):
    grouped = _facts_by_logical_id(world)
    used = set()
    allocations = {}
    for split, need in ((Split.train, 75), (Split.validation, 37)):
        pairs = []
        for fact in _eligible_current_facts(world, holdout_policy, split):
            versions = grouped[fact.logical_fact_id]
            pairs.append((versions["v1"], versions.get("v2")))
        chosen = []
        for item in _changed_present_inventory(pairs, documents, chunks):
            fact, state, intent, cardinality = item
            key = (fact.logical_fact_id, state.value, intent, cardinality)
            if key in used:
                continue
            chosen.append(item); used.add(key)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            raise ValueError(f"only {len(chosen)} independent changed-knowledge tasks are available for {split.value}; need {need}")
        allocations[split] = chosen

    lifecycle = _full_facts_by_lifecycle(world)
    test_chosen = []
    for state in (KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED):
        target = config.changed_knowledge.by_state()[state]
        absent = config.evidence_absent.changed_knowledge if state is KnowledgeState.REMOVED else 0
        present_need = target - absent
        pairs = lifecycle[state]
        if state is KnowledgeState.REMOVED:
            pairs = [pair for pair in pairs if pair[1] is not None and pair[1].status is FactStatus.RETIRED]
        state_items = []
        for item in _changed_present_inventory(pairs, documents, chunks):
            fact, item_state, intent, cardinality = item
            key = (fact.logical_fact_id, item_state.value, intent, cardinality)
            if key in used:
                continue
            state_items.append(item); used.add(key)
            if len(state_items) == present_need:
                break
        if len(state_items) != present_need:
            raise ValueError(f"only {len(state_items)} independent test changed-knowledge {state.value} tasks are available; need {present_need}")
        test_chosen.extend(state_items)
    allocations[Split.test] = test_chosen

    absent_candidates = []
    for v1, v2 in lifecycle[KnowledgeState.REMOVED]:
        if v2 is None:
            absent_candidates.extend(((v1, "current_value_lookup"), (v1, "current_status_lookup")))
    absent_need = config.evidence_absent.changed_knowledge
    if len(absent_candidates) < absent_need:
        raise ValueError(f"only {len(absent_candidates)} independent missing-v2 changed tasks are available; need {absent_need}")
    return allocations, absent_candidates[:absent_need]


def _full_facts_by_lifecycle(world: NimbusWorld) -> dict[KnowledgeState, list[tuple[NimbusFact, NimbusFact | None]]]:
    grouped = _facts_by_logical_id(world)
    result: dict[KnowledgeState, list[tuple[NimbusFact, NimbusFact | None]]] = {
        KnowledgeState.UNCHANGED: [],
        KnowledgeState.UPDATED: [],
        KnowledgeState.REMOVED: [],
    }
    for logical_id in sorted(grouped):
        versions = grouped[logical_id]
        v1 = versions.get("v1")
        if v1 is None:
            continue
        v2 = versions.get("v2")
        state = classify_knowledge_state(v1, v2)
        result[state].append((v1, v2))
    return result


def _difficulty_schedule(config: "BenchmarkConfig") -> list[Difficulty]:
    """Freeze exact test difficulty assignments before task construction."""

    import random

    schedule = (
        [Difficulty.EASY] * config.test_difficulty.EASY
        + [Difficulty.MEDIUM] * config.test_difficulty.MEDIUM
        + [Difficulty.HARD] * config.test_difficulty.HARD
    )
    random.Random(f"{config.generation_seed}:full-test-difficulty").shuffle(schedule)
    return schedule


def _difficulty_params(
    difficulty: Difficulty,
    ordinal: int,
    *,
    required_evidence_cardinality: int | None = None,
    retrieval_applicable: bool = True,
) -> dict[str, object]:
    """Return unambiguous difficulty metadata for one generated task.

    ``required_evidence_cardinality`` is the number of gold chunks the task
    explicitly requires. ``retrieval_candidate_count`` inside the difficulty
    metadata is a separate construction property describing the broader set of
    relevant/competing retrieval candidates.
    """

    from dataclasses import replace
    from adaptlab.benchmark.difficulty import build_difficulty_plan, validate_difficulty_plan

    variant = 1 if difficulty is Difficulty.HARD else ordinal
    plan = build_difficulty_plan(difficulty, variant)
    if not retrieval_applicable:
        plan = replace(
            plan,
            required_evidence_cardinality=0,
            retrieval_candidate_count=0,
            retrieval_applicable=False,
        )
    elif required_evidence_cardinality is not None:
        plan = replace(plan, required_evidence_cardinality=required_evidence_cardinality)
    errors = validate_difficulty_plan(plan)
    if errors:
        raise ValueError("invalid generated difficulty plan: " + "; ".join(errors))
    return {"difficulty": plan.to_dict()}


def _take_compatible_test_difficulty(
    schedule: list[Difficulty], cursor: int, required_evidence_cardinality: int
) -> Difficulty:
    """Take the next configured test difficulty compatible with evidence needs.

    This preserves the exact configured difficulty totals while preventing an
    EASY task from requiring multiple gold chunks or a MEDIUM task from requiring
    three. Selection is deterministic and happens during construction.
    """

    def compatible(value: Difficulty) -> bool:
        if required_evidence_cardinality <= 1:
            return True
        if required_evidence_cardinality == 2:
            return value in (Difficulty.MEDIUM, Difficulty.HARD)
        return value is Difficulty.HARD

    for index in range(cursor, len(schedule)):
        if compatible(schedule[index]):
            schedule[cursor], schedule[index] = schedule[index], schedule[cursor]
            return schedule[cursor]
    raise ValueError(
        f"no remaining configured difficulty can support required evidence cardinality "
        f"{required_evidence_cardinality}"
    )


def _compatible_split_difficulty(base: Difficulty, required_evidence_cardinality: int) -> Difficulty:
    if required_evidence_cardinality >= 3:
        return Difficulty.HARD
    if required_evidence_cardinality == 2 and base is Difficulty.EASY:
        return Difficulty.MEDIUM
    return base


def _eligible_current_facts(
    world: NimbusWorld,
    policy: "FullHoldoutPolicy",
    split: Split,
) -> list[NimbusFact]:
    from adaptlab.benchmark.holdout import error_family_for_fact

    current = [fact for fact in world.facts if fact.version == "v2" and fact.status is FactStatus.ACTIVE]
    selected: list[NimbusFact] = []
    for fact in current:
        component_role = policy.component_family.role_for(fact.component_family)
        error_group = error_family_for_fact(fact)
        error_role = policy.error_family.role_for(error_group) if error_group else None
        if split is Split.train:
            if component_role == "train" and error_role != "structural_test":
                selected.append(fact)
        elif split is Split.validation:
            if component_role == "validation" and error_role != "structural_test":
                selected.append(fact)
        else:
            if component_role in ("iid_test", "structural_test") or error_role == "structural_test":
                selected.append(fact)
    return sorted(selected, key=lambda fact: fact.record_id)


def _behavior_instance_value(seed: int, instance_index: int) -> int:
    """Return a deterministic, globally unique concrete value for a behavior instance.

    ``instance_index`` is allocated once across all full-benchmark behavior-only
    examples. The arithmetic permutation keeps visible values from encoding split
    identity while ensuring that concrete task parameters do not repeat across
    train, validation, and test.
    """

    if instance_index < 1:
        raise ValueError("behavior instance_index must be positive")
    return 10_000 + ((seed + instance_index * 7_919) % 900_000)


def _full_behavior_only_example(
    *,
    example_id: str,
    behavior_type: BehaviorType,
    difficulty: Difficulty,
    split: Split,
    seed: int,
    ordinal: int,
    benchmark_version: str,
) -> BenchmarkExample:
    base = _behavior_instance_value(seed, ordinal)
    token = f"NMB-{base}"
    if behavior_type is BehaviorType.SCHEMA_ADHERENCE:
        question = f"Value: {base}. Return exactly a JSON object with key value and that integer."
        expected: object = {"value": base}
        rule = ScoringRule.STRUCTURED_EXTRACTION
        scoring_params = {"source_value": base, "output_key": "value"}
    elif behavior_type is BehaviorType.CONDITIONAL_DECISION_RULE:
        threshold = base + (3 if ordinal % 2 else -3)
        candidate = base
        question = (
            f"Rule: PASS when candidate <= {threshold}, otherwise FAIL. "
            f"Candidate={candidate}. Return only PASS or FAIL."
        )
        expected = "PASS" if candidate <= threshold else "FAIL"
        rule = ScoringRule.CONDITIONAL_RULE
        scoring_params = {
            "candidate": candidate,
            "threshold": threshold,
            "operator": "lte",
            "true_output": "PASS",
            "false_output": "FAIL",
        }
    elif behavior_type is BehaviorType.TRANSFORMATION_EXTRACTION:
        question = f"Extract the identifier from 'ticket {token} accepted'. Return only the identifier."
        expected = token
        rule = ScoringRule.STRUCTURED_EXTRACTION
        scoring_params = {"literal_value": token, "mode": "literal_scalar"}
    elif behavior_type is BehaviorType.CLASSIFICATION_POLICY:
        severity = base
        threshold = base + (2 if ordinal % 2 else -2)
        question = (
            f"Classify HIGH when severity >= {threshold}, otherwise LOW. "
            f"severity={severity}. Return only the class."
        )
        expected = "HIGH" if severity >= threshold else "LOW"
        rule = ScoringRule.CLASSIFICATION
        scoring_params = {
            "value": severity,
            "threshold": threshold,
            "operator": "gte",
            "true_output": "HIGH",
            "false_output": "LOW",
        }
    else:
        question = (
            f"Use only these supplied facts: identifier={token}. What is its owner? "
            "If the owner is not supplied, answer INSUFFICIENT_INFORMATION."
        )
        expected = "INSUFFICIENT_INFORMATION"
        rule = ScoringRule.ABSTENTION
        scoring_params = {
            "identifier": token,
            "abstention_output": "INSUFFICIENT_INFORMATION",
            "prompt_only": True,
        }


    return BenchmarkExample(
        example_id=example_id,
        benchmark_version=benchmark_version,
        task_family=TaskFamily.behavior_only,
        behavior_type=behavior_type,
        difficulty=difficulty,
        split=split,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version=None,
        knowledge_state=KnowledgeState.NOT_APPLICABLE,
        evidence_status=EvidenceStatus.NOT_APPLICABLE,
        question=question,
        expected_output=expected,
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
        generation_seed=seed,
        scoring_rule=rule,
        scoring_parameters={**scoring_params, **_difficulty_params(difficulty, ordinal, retrieval_applicable=False)},
    )


def _full_present_fact_example(
    *,
    example_id: str,
    task_family: TaskFamily,
    behavior_type: BehaviorType | None,
    difficulty: Difficulty,
    split: Split,
    fact: NimbusFact,
    seed: int,
    benchmark_version: str,
    documents: list[Document],
    chunks: list[DocumentChunk],
    ordinal: int,
    knowledge_state: KnowledgeState = KnowledgeState.NOT_APPLICABLE,
    question_intent: str | None = None,
    evidence_cardinality: int = 1,
) -> BenchmarkExample:
    gold_documents, gold_chunks = _full_current_evidence(
        fact, documents, chunks, cardinality=evidence_cardinality
    )
    if task_family is TaskFamily.knowledge_only:
        intent = question_intent or "current_value_lookup"
        question = _knowledge_question(fact, intent, evidence_cardinality)
        expected: object = fact.value
        scoring_rule = ScoringRule.FACT_VALUE
        params = {
            "question_intent": intent,
            "required_evidence_cardinality": evidence_cardinality,
            **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
        }
    elif task_family is TaskFamily.behavior_knowledge:
        if behavior_type is None:
            raise ValueError("behavior_knowledge requires behavior_type")
        evidence_scope = _required_source_scope(evidence_cardinality)
        if behavior_type is BehaviorType.SCHEMA_ADHERENCE:
            question = (
                f"Using {evidence_scope}, return the value for {fact.entity_id} "
                'as exactly {"value": <value>} with no extra keys.'
            )
            expected = {"value": fact.value}
            scoring_rule = ScoringRule.STRUCTURED_EXTRACTION
            params = {
                "output_key": "value",
                "question_intent": "schema_current_value",
                "required_evidence_cardinality": evidence_cardinality,
                **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
            }
        elif behavior_type is BehaviorType.CONDITIONAL_DECISION_RULE:
            question = (
                f"Using {evidence_scope}, inspect the value for {fact.entity_id}. "
                "Return MATCH only if the documented value is exactly 'enabled'; otherwise return OTHER."
            )
            expected = "MATCH" if str(fact.value) == "enabled" else "OTHER"
            scoring_rule = ScoringRule.CONDITIONAL_RULE
            params = {
                "operator": "eq",
                "threshold": "enabled",
                "true_output": "MATCH",
                "false_output": "OTHER",
                "question_intent": "conditional_current_value",
                "required_evidence_cardinality": evidence_cardinality,
                **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
            }
        elif behavior_type is BehaviorType.TRANSFORMATION_EXTRACTION:
            question = (
                f"From {evidence_scope}, extract the value for {fact.entity_id} "
                "and return only that value, with no label or explanation."
            )
            expected = fact.value
            scoring_rule = ScoringRule.STRUCTURED_EXTRACTION
            params = {
                "mode": "scalar",
                "question_intent": "extract_current_value",
                "required_evidence_cardinality": evidence_cardinality,
                **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
            }
        elif behavior_type is BehaviorType.CLASSIFICATION_POLICY:
            question = (
                f"Using {evidence_scope}, classify the value for {fact.entity_id} "
                "as NUMERIC if it is a number and TEXT otherwise. Return only the class."
            )
            try:
                float(fact.value)
                expected = "NUMERIC"
            except (TypeError, ValueError):
                expected = "TEXT"
            scoring_rule = ScoringRule.CLASSIFICATION
            params = {
                "operator": "is_numeric",
                "true_output": "NUMERIC",
                "false_output": "TEXT",
                "question_intent": "classify_current_value",
                "required_evidence_cardinality": evidence_cardinality,
                **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
            }
        else:
            question = (
                f"Consult {evidence_scope} for {fact.entity_id}. "
                "If a current value is explicitly supported, return only that value; "
                "otherwise return INSUFFICIENT_EVIDENCE."
            )
            expected = fact.value
            scoring_rule = ScoringRule.FACT_VALUE
            params = {
                "evidence_gate": True,
                "question_intent": "supported_current_value",
                "required_evidence_cardinality": evidence_cardinality,
                **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
            }
    else:
        intent = question_intent or "current_value_lookup"
        question = _changed_question(fact, intent, evidence_cardinality)
        expected = "RETIRED" if fact.status is FactStatus.RETIRED else fact.value
        scoring_rule = ScoringRule.RETIRED_STATUS if fact.status is FactStatus.RETIRED else ScoringRule.FACT_VALUE
        params = {
            "question_intent": intent,
            "required_evidence_cardinality": evidence_cardinality,
            **_difficulty_params(difficulty, ordinal, required_evidence_cardinality=evidence_cardinality),
        }

    return BenchmarkExample(
        example_id=example_id,
        benchmark_version=benchmark_version,
        task_family=task_family,
        behavior_type=behavior_type,
        difficulty=difficulty,
        split=split,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version="v2",
        knowledge_state=knowledge_state,
        evidence_status=EvidenceStatus.PRESENT,
        question=question,
        expected_output=expected,
        required_record_ids=(fact.record_id,),
        required_logical_fact_ids=(fact.logical_fact_id,),
        gold_document_ids=gold_documents,
        gold_chunk_ids=gold_chunks,
        generation_seed=seed,
        scoring_rule=scoring_rule,
        scoring_parameters=params,
        lifecycle_logical_fact_id=(fact.logical_fact_id if task_family is TaskFamily.changed_knowledge else None),
    )


def _full_absent_example(
    *,
    example_id: str,
    task_family: TaskFamily,
    behavior_type: BehaviorType | None,
    difficulty: Difficulty,
    split: Split,
    seed: int,
    benchmark_version: str,
    ordinal: int,
    lifecycle_logical_fact_id: str | None = None,
    knowledge_state: KnowledgeState = KnowledgeState.NOT_APPLICABLE,
    entity_label: str | None = None,
    question_intent: str = "current_value_lookup",
) -> BenchmarkExample:
    label = entity_label or f"unpublished_control_{ordinal:03d}"
    question = f"What is the current Nimbus value for {label}?"
    if task_family is TaskFamily.changed_knowledge and question_intent == "current_status_lookup":
        question = f"What is the current Nimbus status for {label}?"
    if task_family is TaskFamily.behavior_knowledge:
        question = (
            f"Consult current Nimbus documentation for {label}. If a current value is explicitly supported, "
            "return only that value; otherwise return INSUFFICIENT_EVIDENCE."
        )
    return BenchmarkExample(
        example_id=example_id,
        benchmark_version=benchmark_version,
        task_family=task_family,
        behavior_type=behavior_type,
        difficulty=difficulty,
        split=split,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        knowledge_version="v2",
        knowledge_state=knowledge_state,
        evidence_status=EvidenceStatus.ABSENT,
        question=question,
        expected_output="INSUFFICIENT_EVIDENCE",
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
        generation_seed=seed,
        scoring_rule=ScoringRule.ABSTENTION,
        scoring_parameters={
            "semantic_target": label,
            "question_intent": question_intent,
            "required_evidence_cardinality": 0,
            **_difficulty_params(difficulty, ordinal, retrieval_applicable=False),
        },
        lifecycle_logical_fact_id=lifecycle_logical_fact_id,
    )


def _allocate_behavior_knowledge_tasks(
    pools: dict[Split, list[NimbusFact]],
    counts: dict[Split, int],
    documents: list[Document],
    chunks: list[DocumentChunk],
) -> dict[Split, list[tuple[NimbusFact, BehaviorType, int]]]:
    """Allocate concrete behavior+knowledge instances without cross-split reuse.

    Semantic identity is the fact + behavior primitive + evidence cardinality.
    Cardinality changes the actual permitted evidence set and is therefore a
    task-relevant property rather than cosmetic wording.
    """

    used: set[tuple[str, str, int]] = set()
    allocated: dict[Split, list[tuple[NimbusFact, BehaviorType, int]]] = {}
    for split in (Split.train, Split.validation, Split.test):
        candidates: list[tuple[NimbusFact, BehaviorType, int]] = []
        for fact in sorted(pools[split], key=lambda item: item.record_id):
            # Only advertise cardinalities actually available for this fact.
            available = sum(
                1
                for chunk in chunks
                if chunk.version == "v2"
                and chunk.is_authoritative
                and not chunk.is_obsolete
                and fact.record_id in chunk.record_ids
            )
            for behavior in tuple(BehaviorType):
                for cardinality in range(1, min(3, available) + 1):
                    key = (fact.logical_fact_id, behavior.value, cardinality)
                    if key not in used:
                        candidates.append((fact, behavior, cardinality))
        candidates.sort(key=lambda item: (item[2], item[0].record_id, item[1].value))
        need = counts[split]
        if len(candidates) < need:
            raise ValueError(
                f"only {len(candidates)} independent behavior+knowledge tasks are available "
                f"for {split.value}; need {need}"
            )
        chosen = candidates[:need]
        allocated[split] = chosen
        used.update((fact.logical_fact_id, behavior.value, cardinality) for fact, behavior, cardinality in chosen)
    return allocated


def generate_full_tasks(
    world: NimbusWorld,
    documents: list[Document],
    chunks: list[DocumentChunk],
    config: "BenchmarkConfig",
    holdout_policy: "FullHoldoutPolicy",
) -> list[BenchmarkExample]:
    """Generate the exact configured 850-example primary full-v0.0 benchmark."""

    from adaptlab.benchmark.holdout import structural_holdout_for_example

    if world.generation_seed != config.generation_seed:
        raise ValueError("world generation_seed must match benchmark config")
    if holdout_policy.generation_seed != config.generation_seed:
        raise ValueError("holdout policy seed must match benchmark config")

    seed = config.generation_seed
    benchmark_version = config.benchmark_version
    examples: list[BenchmarkExample] = []

    # Test assignments are frozen up front from exact configured quotas.
    test_difficulties = _difficulty_schedule(config)
    test_cursor = 0
    behavior_types: list[BehaviorType] = []
    for behavior_type, count in config.behavior_only_test_behavior_types.by_behavior_type().items():
        behavior_types.extend([behavior_type] * count)
    import random
    random.Random(f"{seed}:behavior-only-types").shuffle(behavior_types)

    # 100 behavior-only test examples, exactly balanced by configured behavior target.
    for i in range(config.test_task_families.behavior_only):
        difficulty = test_difficulties[test_cursor]
        test_cursor += 1
        examples.append(
            _full_behavior_only_example(
                example_id=f"FULL_T_BHV_{i + 1:03d}",
                behavior_type=behavior_types[i],
                difficulty=difficulty,
                split=Split.test,
                seed=seed,
                ordinal=i + 1,
                benchmark_version=benchmark_version,
            )
        )

    test_iid_facts = _eligible_current_facts(world, holdout_policy, Split.test)
    structural_facts = [
        fact
        for fact in world.facts
        if fact.version == "v2" and structural_holdout_for_example(
            BenchmarkExample(
                example_id="probe", benchmark_version=benchmark_version,
                task_family=TaskFamily.knowledge_only, behavior_type=None,
                difficulty=Difficulty.EASY, split=Split.test, split_type=SplitType.iid,
                holdout_dimension=None, holdout_group=None, knowledge_version="v2",
                knowledge_state=KnowledgeState.NOT_APPLICABLE, evidence_status=EvidenceStatus.PRESENT,
                question="probe", expected_output=fact.value, required_record_ids=(fact.record_id,),
                required_logical_fact_ids=(fact.logical_fact_id,), gold_document_ids=("probe",),
                gold_chunk_ids=("probe",), generation_seed=seed, scoring_rule=ScoringRule.FACT_VALUE,
            ),
            world,
            holdout_policy,
        ) is not None
    ]
    test_fact_pool = sorted({f.record_id: f for f in (test_iid_facts + structural_facts) if f.status is FactStatus.ACTIVE}.values(), key=lambda f: f.record_id)
    if not test_fact_pool:
        raise ValueError("full test generation requires eligible current facts")

    bkn_present_counts = {
        Split.train: 75,
        Split.validation: 37,
        Split.test: config.test_task_families.behavior_knowledge - config.evidence_absent.behavior_knowledge,
    }
    bkn_pools = {
        Split.train: _eligible_current_facts(world, holdout_policy, Split.train),
        Split.validation: _eligible_current_facts(world, holdout_policy, Split.validation),
        Split.test: test_fact_pool,
    }
    bkn_allocations = _allocate_behavior_knowledge_tasks(
        bkn_pools, bkn_present_counts, documents, chunks
    )

    def add_test_family(family: TaskFamily, total: int, absent_count: int = 0) -> None:
        nonlocal test_cursor
        present_count = total - absent_count
        knowledge_inventory = (
            _full_knowledge_task_inventory(test_fact_pool, documents, chunks)
            if family is TaskFamily.knowledge_only
            else []
        )
        if family is TaskFamily.knowledge_only and len(knowledge_inventory) < present_count:
            raise ValueError(
                f"only {len(knowledge_inventory)} independent knowledge-only test tasks "
                f"are available for {present_count} PRESENT examples"
            )
        for i in range(present_count):
            question_intent = None
            evidence_cardinality = 1
            if family is TaskFamily.knowledge_only:
                fact, question_intent, evidence_cardinality = knowledge_inventory[i]
                behavior = None
            elif family is TaskFamily.behavior_knowledge:
                fact, behavior, evidence_cardinality = bkn_allocations[Split.test][i]
            else:
                fact = test_fact_pool[i % len(test_fact_pool)]
                behavior = None
            difficulty = _take_compatible_test_difficulty(
                test_difficulties, test_cursor, evidence_cardinality
            )
            test_cursor += 1
            examples.append(
                _full_present_fact_example(
                    example_id=f"FULL_T_{'KNW' if family is TaskFamily.knowledge_only else 'BKN'}_{i + 1:03d}",
                    task_family=family,
                    behavior_type=behavior,
                    difficulty=difficulty,
                    split=Split.test,
                    fact=fact,
                    seed=seed,
                    benchmark_version=benchmark_version,
                    documents=documents,
                    chunks=chunks,
                    ordinal=test_cursor,
                    question_intent=question_intent,
                    evidence_cardinality=evidence_cardinality,
                )
            )
        for i in range(absent_count):
            difficulty = test_difficulties[test_cursor]
            test_cursor += 1
            behavior = BehaviorType.ABSTENTION_BEHAVIOR if family is TaskFamily.behavior_knowledge else None
            prefix = "KNW" if family is TaskFamily.knowledge_only else "BKN"
            examples.append(
                _full_absent_example(
                    example_id=f"FULL_T_{prefix}_ABS_{i + 1:03d}",
                    task_family=family,
                    behavior_type=behavior,
                    difficulty=difficulty,
                    split=Split.test,
                    seed=seed,
                    benchmark_version=benchmark_version,
                    ordinal=test_cursor,
                )
            )

    add_test_family(
        TaskFamily.knowledge_only,
        config.test_task_families.knowledge_only,
        config.evidence_absent.knowledge_only,
    )
    add_test_family(
        TaskFamily.behavior_knowledge,
        config.test_task_families.behavior_knowledge,
        config.evidence_absent.behavior_knowledge,
    )

    changed_allocations, changed_absent_allocations = _allocate_changed_tasks(
        world, holdout_policy, documents, chunks, config
    )
    lifecycle = _full_facts_by_lifecycle(world)
    changed_targets = config.changed_knowledge.by_state()
    changed_absent_remaining = config.evidence_absent.changed_knowledge
    changed_ordinal = 0
    test_present_by_state = {state: [item for item in changed_allocations[Split.test] if item[1] is state] for state in (KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED)}
    absent_cursor = 0
    for state in (KnowledgeState.UNCHANGED, KnowledgeState.UPDATED, KnowledgeState.REMOVED):
        target = changed_targets[state]
        absent_for_state = changed_absent_remaining if state is KnowledgeState.REMOVED else 0
        present_items = test_present_by_state[state]
        assert len(present_items) == target - absent_for_state
        for i, (v2, _, intent, cardinality) in enumerate(present_items):
            difficulty = _take_compatible_test_difficulty(test_difficulties, test_cursor, cardinality)
            test_cursor += 1
            changed_ordinal += 1
            examples.append(_full_present_fact_example(
                example_id=f"FULL_T_CHG_{state.value}_{i + 1:03d}", task_family=TaskFamily.changed_knowledge, behavior_type=None,
                difficulty=difficulty, split=Split.test, fact=v2, seed=seed, benchmark_version=benchmark_version,
                documents=documents, chunks=chunks, ordinal=test_cursor, knowledge_state=state,
                question_intent=intent, evidence_cardinality=cardinality,
            ))
        if absent_for_state:
            for i in range(absent_for_state):
                difficulty = test_difficulties[test_cursor]; test_cursor += 1
                v1, intent = changed_absent_allocations[absent_cursor]; absent_cursor += 1
                examples.append(_full_absent_example(
                    example_id=f"FULL_T_CHG_REMOVED_ABS_{i + 1:03d}", task_family=TaskFamily.changed_knowledge, behavior_type=None,
                    difficulty=difficulty, split=Split.test, seed=seed, benchmark_version=benchmark_version, ordinal=test_cursor,
                    lifecycle_logical_fact_id=v1.logical_fact_id, knowledge_state=KnowledgeState.REMOVED, entity_label=v1.entity_id,
                    question_intent=intent,
                ))

    if test_cursor != config.splits.test:
        raise ValueError(f"generated {test_cursor} test examples, expected {config.splits.test}")

    # Train/validation: balanced family coverage with knowledge-bearing examples
    # restricted to split-eligible current facts. Counts are exact; no model
    # results influence composition.
    split_family_counts = {
        Split.train: [75, 75, 75, 75],
        Split.validation: [38, 38, 37, 37],
    }
    for split, counts in split_family_counts.items():
        target_total = config.splits.train if split is Split.train else config.splits.validation
        if sum(counts) != target_total:
            raise ValueError("internal full split-family plan does not match configured split total")
        fact_pool = _eligible_current_facts(world, holdout_policy, split)
        if not fact_pool:
            raise ValueError(f"no eligible current facts for {split.value}")
        knowledge_inventory = _full_knowledge_task_inventory(fact_pool, documents, chunks)
        required_knowledge_count = counts[1]
        if len(knowledge_inventory) < required_knowledge_count:
            raise ValueError(
                f"only {len(knowledge_inventory)} independent knowledge-only tasks are available "
                f"for {split.value}; need {required_knowledge_count}"
            )
        split_prefix = "TR" if split is Split.train else "VA"
        ordinal = 0
        for family, count in zip(tuple(TaskFamily), counts):
            for i in range(count):
                ordinal += 1
                difficulty = (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[(ordinal + i) % 3]
                if family is TaskFamily.behavior_only:
                    examples.append(
                        _full_behavior_only_example(
                            example_id=f"FULL_{split_prefix}_BHV_{i + 1:03d}",
                            behavior_type=tuple(BehaviorType)[i % len(BehaviorType)],
                            difficulty=difficulty,
                            split=split,
                            seed=seed,
                            ordinal=(101 + i if split is Split.train else 176 + i),
                            benchmark_version=benchmark_version,
                        )
                    )
                elif family is TaskFamily.changed_knowledge:
                    fact, state, intent, cardinality = changed_allocations[split][i]
                    difficulty = _compatible_split_difficulty(difficulty, cardinality)
                    examples.append(
                        _full_present_fact_example(
                            example_id=f"FULL_{split_prefix}_CHG_{i + 1:03d}", task_family=family, behavior_type=None,
                            difficulty=difficulty, split=split, fact=fact, seed=seed, benchmark_version=benchmark_version,
                            documents=documents, chunks=chunks, ordinal=1000 + ordinal, knowledge_state=state,
                            question_intent=intent, evidence_cardinality=cardinality,
                        )
                    )
                else:
                    question_intent = None
                    evidence_cardinality = 1
                    if family is TaskFamily.knowledge_only:
                        fact, question_intent, evidence_cardinality = knowledge_inventory[i]
                        behavior = None
                    elif family is TaskFamily.behavior_knowledge:
                        fact, behavior, evidence_cardinality = bkn_allocations[split][i]
                    else:
                        fact = fact_pool[i % len(fact_pool)]
                        behavior = None
                    difficulty = _compatible_split_difficulty(difficulty, evidence_cardinality)
                    code = "KNW" if family is TaskFamily.knowledge_only else "BKN"
                    examples.append(
                        _full_present_fact_example(
                            example_id=f"FULL_{split_prefix}_{code}_{i + 1:03d}",
                            task_family=family,
                            behavior_type=behavior,
                            difficulty=difficulty,
                            split=split,
                            fact=fact,
                            seed=seed,
                            benchmark_version=benchmark_version,
                            documents=documents,
                            chunks=chunks,
                            ordinal=1000 + ordinal,
                            question_intent=question_intent,
                            evidence_cardinality=evidence_cardinality,
                        )
                    )

    # Annotate structural-test facts without ever changing train/validation.
    annotated: list[BenchmarkExample] = []
    from dataclasses import replace
    for example in examples:
        if example.split is Split.test:
            structural = structural_holdout_for_example(example, world, holdout_policy)
            if structural is not None:
                dimension, group = structural
                example = replace(
                    example,
                    split_type=SplitType.structural_holdout,
                    holdout_dimension=dimension,
                    holdout_group=group,
                )
        annotated.append(example)

    annotated.sort(key=lambda example: example.example_id)
    split_counts = {split: sum(e.split is split for e in annotated) for split in Split}
    expected_split_counts = {
        Split.train: config.splits.train,
        Split.validation: config.splits.validation,
        Split.test: config.splits.test,
    }
    if split_counts != expected_split_counts:
        raise ValueError(f"full task split counts mismatch: {split_counts} != {expected_split_counts}")
    return annotated
