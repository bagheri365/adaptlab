"""Deterministic duplicate and cross-split leakage diagnostics.

This module intentionally uses only local text normalization and token-set
similarity.  It does not use embeddings or model-based similarity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable

from adaptlab.benchmark.holdout import FullHoldoutPolicy, validate_full_holdout_examples
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import Split, TaskFamily
from adaptlab.domain.world import NimbusWorld

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_ARTIFICIAL_SEQUENCE_RE = re.compile(r"\b(?:during\s+review\s+sequence|review\s+sequence|split\s+nonce)\s*[:#-]?\s*\d+\b", re.IGNORECASE)
CROSS_SPLIT_REVIEW_THRESHOLD = 0.80


def strip_artificial_sequence_text(text: str) -> str:
    """Remove legacy artificial sequence/nonces before lexical comparison.

    Current generators must not emit these markers, but stripping them here keeps
    the audit robust to stale artifacts and prevents cosmetic sequence tokens from
    reducing apparent cross-split similarity.
    """

    return _ARTIFICIAL_SEQUENCE_RE.sub(" ", text)


def normalize_text(text: str) -> str:
    """Return a deterministic normalized form suitable for duplicate checks."""

    tokens = _TOKEN_RE.findall(strip_artificial_sequence_text(text).casefold())
    return " ".join(tokens)


def normalized_text_fingerprint(text: str) -> str:
    """Return a stable SHA-256 fingerprint of normalized text."""

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(strip_artificial_sequence_text(text).casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _canonical_output(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _semantic_scoring_parameters(example: BenchmarkExample) -> dict[str, Any]:
    """Return only task-relevant scoring parameters for semantic identity.

    Difficulty construction metadata, example ordering, and other generation-only
    fields are intentionally excluded so superficial generation changes cannot
    make the same concrete task look independent.
    """

    params = dict(example.scoring_parameters or {})
    params.pop("difficulty", None)
    params.pop("question_intent", None)
    params.pop("required_evidence_cardinality", None)
    return params


def semantic_task_payload(example: BenchmarkExample) -> dict[str, Any]:
    """Build a deterministic structured description of the concrete task.

    The payload deliberately excludes ``example_id``, split, prose wording,
    generation order, and any review/nonce marker. Knowledge-bearing tasks use
    logical fact identity and knowledge version; behavior tasks additionally use
    the concrete scoring operands and output contract encoded in scoring metadata.
    """

    payload: dict[str, Any] = {
        "task_family": example.task_family.value,
        "behavior_type": example.behavior_type.value if example.behavior_type else None,
        "scoring_rule": example.scoring_rule.value if example.scoring_rule else None,
        "scoring_parameters": _semantic_scoring_parameters(example),
    }

    if example.task_family is TaskFamily.behavior_only:
        return payload

    logical_ids = tuple(sorted(example.required_logical_fact_ids))
    if not logical_ids and example.lifecycle_logical_fact_id:
        logical_ids = (example.lifecycle_logical_fact_id,)
    payload.update(
        {
            "logical_fact_ids": logical_ids,
            "knowledge_version": example.knowledge_version,
            "knowledge_state": (
                example.knowledge_state.value
                if example.task_family is TaskFamily.changed_knowledge
                else None
            ),
            "evidence_status": example.evidence_status.value,
            "question_intent": (
                (example.scoring_parameters or {}).get("question_intent")
                or (
                    "current_knowledge"
                    if example.task_family in (TaskFamily.knowledge_only, TaskFamily.changed_knowledge)
                    else "current_knowledge_plus_behavior"
                )
            ),
            "required_evidence_cardinality": (
                (example.scoring_parameters or {}).get("required_evidence_cardinality")
                if example.evidence_status.value == "PRESENT"
                else 0
            ) or len(example.gold_chunk_ids),
        }
    )
    return payload


def semantic_task_fingerprint(example: BenchmarkExample) -> str:
    """Return a stable SHA-256 fingerprint of task-relevant structured fields."""

    encoded = json.dumps(
        semantic_task_payload(example),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _group_duplicates(items: Iterable[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, example_id in items:
        grouped[key].append(example_id)
    groups = [tuple(sorted(ids)) for ids in grouped.values() if len(ids) > 1]
    return tuple(sorted(groups))




class WithinSplitDuplicateClass(str, Enum):
    """Review classification for exact/normalized duplicates inside one split.

    A is allowed only when the evaluated system can observe a predeclared,
    task-relevant retrieval/evidence condition that makes the instances
    experimentally distinct. Hidden benchmark metadata alone never qualifies.
    B is redundant duplication with no model-visible experimental distinction.
    C is a generator defect such as identical semantic task fingerprints.
    """

    LEGITIMATE_RETRIEVAL_VARIANT = "A"
    REDUNDANT_DUPLICATE = "B"
    GENERATOR_DEFECT = "C"


WITHIN_SPLIT_DUPLICATE_RULE = (
    "Exact or normalized duplicate questions within one split are disallowed by default. "
    "They may be classified A only when a predeclared task-relevant condition observable "
    "to the evaluated system (for example, a different permitted evidence set that the "
    "system actually receives) makes the instances experimentally distinct. Differences "
    "only in example_id, difficulty metadata, gold IDs, hidden evidence cardinality, or "
    "other evaluator-only metadata are not sufficient; those are B. Identical semantic "
    "task fingerprints are C generator defects."
)


@dataclass(frozen=True, slots=True)
class WithinSplitDuplicateReview:
    example_ids: tuple[str, ...]
    split: str
    task_families: tuple[str, ...]
    difficulties: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    gold_chunk_sets: tuple[tuple[str, ...], ...]
    semantic_fingerprints: tuple[str, ...]
    classification: WithinSplitDuplicateClass
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "example_ids": list(self.example_ids),
            "split": self.split,
            "task_families": list(self.task_families),
            "difficulties": list(self.difficulties),
            "expected_outputs": list(self.expected_outputs),
            "gold_chunk_sets": [list(items) for items in self.gold_chunk_sets],
            "semantic_fingerprints": list(self.semantic_fingerprints),
            "classification": self.classification.value,
            "reason": self.reason,
        }


def _review_within_split_duplicates(
    examples: list[BenchmarkExample],
    groups: tuple[tuple[str, ...], ...],
) -> tuple[WithinSplitDuplicateReview, ...]:
    by_id = {example.example_id: example for example in examples}
    reviews: list[WithinSplitDuplicateReview] = []
    for group in groups:
        members = [by_id[example_id] for example_id in group]
        splits = {item.split.value for item in members}
        if len(splits) != 1:
            continue
        fingerprints = tuple(sorted({semantic_task_fingerprint(item) for item in members}))
        outputs = tuple(sorted({_canonical_output(item.expected_output) for item in members}))
        gold_sets = tuple(sorted({tuple(item.gold_chunk_ids) for item in members}))
        if len(fingerprints) == 1:
            classification = WithinSplitDuplicateClass.GENERATOR_DEFECT
            reason = "same split, same surface question, and identical semantic task fingerprint"
        else:
            # The current benchmark has no per-example model-visible retrieval-corpus
            # condition distinct from the question itself. Therefore hidden gold/evaluator
            # metadata cannot justify duplicate surface questions.
            classification = WithinSplitDuplicateClass.REDUNDANT_DUPLICATE
            reason = (
                "same split and same surface question with distinct semantic metadata, but "
                "no predeclared model-visible retrieval condition justifies the duplicate"
            )
        reviews.append(
            WithinSplitDuplicateReview(
                example_ids=tuple(sorted(group)),
                split=next(iter(splits)),
                task_families=tuple(sorted({item.task_family.value for item in members})),
                difficulties=tuple(sorted({item.difficulty.value for item in members})),
                expected_outputs=outputs,
                gold_chunk_sets=gold_sets,
                semantic_fingerprints=fingerprints,
                classification=classification,
                reason=reason,
            )
        )
    return tuple(sorted(reviews, key=lambda item: item.example_ids))


@dataclass(frozen=True, slots=True)
class NearDuplicatePair:
    left_example_id: str
    right_example_id: str
    left_split: str
    right_split: str
    similarity: float
    cross_split: bool
    behavior_type: str | None
    semantic_fingerprint_match: bool
    parameter_overlap: float
    identifier_overlap: tuple[str, ...]
    template_family_match: bool
    expected_output_structure_match: bool

    @property
    def risk_score(self) -> float:
        """Deterministic review ranking, not a model-performance metric."""

        return (
            0.65 * self.similarity
            + 0.20 * self.parameter_overlap
            + 0.05 * float(bool(self.identifier_overlap))
            + 0.05 * float(self.template_family_match)
            + 0.05 * float(self.expected_output_structure_match)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left_example_id": self.left_example_id,
            "right_example_id": self.right_example_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "similarity": round(self.similarity, 6),
            "cross_split": self.cross_split,
            "behavior_type": self.behavior_type,
            "semantic_fingerprint_match": self.semantic_fingerprint_match,
            "parameter_overlap": round(self.parameter_overlap, 6),
            "identifier_overlap": list(self.identifier_overlap),
            "template_family_match": self.template_family_match,
            "expected_output_structure_match": self.expected_output_structure_match,
            "risk_score": round(self.risk_score, 6),
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    exact_duplicate_questions: tuple[tuple[str, ...], ...]
    within_split_duplicate_reviews: tuple[WithinSplitDuplicateReview, ...]
    normalized_duplicate_questions: tuple[tuple[str, ...], ...]
    suspicious_expected_output_duplicates: tuple[tuple[str, ...], ...]
    suspicious_near_duplicates: tuple[NearDuplicatePair, ...]
    cross_split_near_duplicate_warnings: tuple[str, ...]
    highest_risk_train_validation: tuple[NearDuplicatePair, ...]
    highest_risk_train_test: tuple[NearDuplicatePair, ...]
    cross_split_collisions: tuple[tuple[str, ...], ...]
    semantic_fingerprint_collisions: tuple[tuple[str, ...], ...]
    structural_violations: tuple[str, ...]
    metadata_answer_leakage: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "exact_duplicate_questions": [list(group) for group in self.exact_duplicate_questions],
            "within_split_duplicate_rule": WITHIN_SPLIT_DUPLICATE_RULE,
            "within_split_duplicate_reviews": [item.to_dict() for item in self.within_split_duplicate_reviews],
            "normalized_duplicate_questions": [list(group) for group in self.normalized_duplicate_questions],
            "suspicious_expected_output_duplicates": [
                list(group) for group in self.suspicious_expected_output_duplicates
            ],
            "suspicious_near_duplicates": [pair.to_dict() for pair in self.suspicious_near_duplicates],
            "cross_split_near_duplicate_warnings": list(self.cross_split_near_duplicate_warnings),
            "highest_risk_train_validation": [pair.to_dict() for pair in self.highest_risk_train_validation],
            "highest_risk_train_test": [pair.to_dict() for pair in self.highest_risk_train_test],
            "cross_split_collisions": [list(group) for group in self.cross_split_collisions],
            "semantic_fingerprint_collisions": [
                list(group) for group in self.semantic_fingerprint_collisions
            ],
            "structural_violations": list(self.structural_violations),
            "metadata_answer_leakage": list(self.metadata_answer_leakage),
            "blockers": list(self.blockers),
        }


def _cross_split_groups(
    examples: list[BenchmarkExample],
    normalized_groups: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    by_id = {example.example_id: example for example in examples}
    collisions: list[tuple[str, ...]] = []
    for group in normalized_groups:
        splits = {by_id[example_id].split for example_id in group}
        if len(splits) > 1:
            collisions.append(group)
    return tuple(sorted(collisions))


def _metadata_answer_leaks(examples: list[BenchmarkExample]) -> tuple[str, ...]:
    """Flag answer-bearing benchmark metadata accidentally copied into questions.

    Metadata fields such as lifecycle labels, scoring-rule names, gold IDs, and
    holdout labels are not model inputs in the benchmark contract.  This check
    catches accidental templating that exposes those answer-bearing fields in
    the question text for knowledge-bearing tasks.
    """

    leaks: list[str] = []
    for example in examples:
        if example.task_family is TaskFamily.behavior_only:
            continue
        normalized_question = normalize_text(example.question)
        candidates: list[str] = []
        if example.lifecycle_logical_fact_id:
            candidates.append(example.lifecycle_logical_fact_id)
        candidates.extend(example.gold_document_ids)
        candidates.extend(example.gold_chunk_ids)
        if example.scoring_rule is not None:
            candidates.append(example.scoring_rule.value)
        # Lifecycle labels are metadata for changed-knowledge tasks and should
        # not be the primary answer cue.
        if example.task_family is TaskFamily.changed_knowledge:
            candidates.append(example.knowledge_state.value)
        for candidate in candidates:
            candidate_norm = normalize_text(candidate)
            if candidate_norm and candidate_norm in normalized_question:
                leaks.append(
                    f"example {example.example_id} exposes benchmark metadata {candidate!r} in question"
                )
                break
    return tuple(sorted(leaks))


def _flatten_parameter_tokens(value: Any) -> frozenset[str]:
    tokens: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                tokens.add(f"key:{key}")
                visit(item[key])
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)
        elif item is not None:
            tokens.add(f"value:{_canonical_output(item)}")

    visit(value)
    return frozenset(tokens)


def _parameter_overlap(left: BenchmarkExample, right: BenchmarkExample) -> float:
    return _jaccard(
        _flatten_parameter_tokens(_semantic_scoring_parameters(left)),
        _flatten_parameter_tokens(_semantic_scoring_parameters(right)),
    )


def _identifier_tokens(example: BenchmarkExample) -> frozenset[str]:
    candidates = set(example.required_logical_fact_ids)
    if example.lifecycle_logical_fact_id:
        candidates.add(example.lifecycle_logical_fact_id)
    for token in _TOKEN_RE.findall(strip_artificial_sequence_text(example.question)):
        if "_" in token or any(ch.isdigit() for ch in token):
            candidates.add(token.casefold())
    return frozenset(item.casefold() for item in candidates)


def _template_family(example: BenchmarkExample) -> tuple[str, str | None, str | None]:
    return (
        example.task_family.value,
        example.behavior_type.value if example.behavior_type else None,
        example.scoring_rule.value if example.scoring_rule else None,
    )


def _output_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return ("dict", tuple(sorted((key, _output_structure(item)) for key, item in value.items())))
    if isinstance(value, list):
        return ("list", tuple(_output_structure(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_output_structure(item) for item in value))
    return type(value).__name__


def _top_risk_pairs(
    pairs: list[NearDuplicatePair], left: Split, right: Split, *, limit: int = 10
) -> tuple[NearDuplicatePair, ...]:
    wanted = {left.value, right.value}
    filtered = [pair for pair in pairs if {pair.left_split, pair.right_split} == wanted]
    filtered.sort(key=lambda pair: (-pair.risk_score, -pair.similarity, pair.left_example_id, pair.right_example_id))
    return tuple(filtered[:limit])


def run_leakage_audit(
    examples: Iterable[BenchmarkExample],
    *,
    world: NimbusWorld | None = None,
    holdout_policy: FullHoldoutPolicy | None = None,
    near_duplicate_threshold: float = 0.90,
    expected_output_repeat_threshold: int = 10,
) -> LeakageReport:
    """Run deterministic duplicate/leakage diagnostics for the full benchmark."""

    ordered = sorted(examples, key=lambda item: item.example_id)

    exact = _group_duplicates((example.question, example.example_id) for example in ordered)
    normalized = _group_duplicates(
        (normalized_text_fingerprint(example.question), example.example_id)
        for example in ordered
    )
    cross_split = _cross_split_groups(ordered, normalized)
    within_split_reviews = _review_within_split_duplicates(ordered, normalized)
    semantic_groups = _group_duplicates(
        (semantic_task_fingerprint(example), example.example_id) for example in ordered
    )
    semantic_cross_split = _cross_split_groups(ordered, semantic_groups)

    # Repeated outputs are common for classification/abstention, so this is a
    # diagnostic only and is reported only when unusually concentrated.
    output_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for example in ordered:
        key = (example.task_family.value, _canonical_output(example.expected_output))
        output_groups[key].append(example.example_id)
    suspicious_outputs = tuple(
        sorted(
            tuple(sorted(ids))
            for ids in output_groups.values()
            if len(ids) >= expected_output_repeat_threshold
        )
    )

    token_sets = {example.example_id: _token_set(example.question) for example in ordered}
    semantic_fingerprints = {
        example.example_id: semantic_task_fingerprint(example) for example in ordered
    }
    identifier_sets = {example.example_id: _identifier_tokens(example) for example in ordered}
    near_pairs: list[NearDuplicatePair] = []
    review_pairs: list[NearDuplicatePair] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if normalize_text(left.question) == normalize_text(right.question):
                continue
            score = _jaccard(token_sets[left.example_id], token_sets[right.example_id])
            cross_split_pair = left.split is not right.split
            if score < near_duplicate_threshold and not (
                cross_split_pair and score >= CROSS_SPLIT_REVIEW_THRESHOLD
            ):
                continue
            identifiers = tuple(sorted(identifier_sets[left.example_id] & identifier_sets[right.example_id]))
            pair = NearDuplicatePair(
                left_example_id=left.example_id,
                right_example_id=right.example_id,
                left_split=left.split.value,
                right_split=right.split.value,
                similarity=score,
                cross_split=cross_split_pair,
                behavior_type=(
                    left.behavior_type.value
                    if left.behavior_type is right.behavior_type and left.behavior_type is not None
                    else None
                ),
                semantic_fingerprint_match=(
                    semantic_fingerprints[left.example_id] == semantic_fingerprints[right.example_id]
                ),
                parameter_overlap=_parameter_overlap(left, right),
                identifier_overlap=identifiers,
                template_family_match=_template_family(left) == _template_family(right),
                expected_output_structure_match=(
                    _output_structure(left.expected_output) == _output_structure(right.expected_output)
                ),
            )
            if score >= near_duplicate_threshold:
                near_pairs.append(pair)
            if cross_split_pair and score >= CROSS_SPLIT_REVIEW_THRESHOLD:
                review_pairs.append(pair)
    near_pairs.sort(key=lambda pair: (-pair.similarity, pair.left_example_id, pair.right_example_id))
    review_pairs.sort(key=lambda pair: (-pair.risk_score, -pair.similarity, pair.left_example_id, pair.right_example_id))
    cross_split_warnings = tuple(
        f"high-similarity cross-split pair {pair.left_example_id} ({pair.left_split}) vs "
        f"{pair.right_example_id} ({pair.right_split}): lexical={pair.similarity:.3f}, "
        f"parameter_overlap={pair.parameter_overlap:.3f}, template_match={pair.template_family_match}, "
        f"semantic_fingerprint_match={pair.semantic_fingerprint_match}"
        for pair in review_pairs
    )
    highest_risk_train_validation = _top_risk_pairs(review_pairs, Split.train, Split.validation)
    highest_risk_train_test = _top_risk_pairs(review_pairs, Split.train, Split.test)

    structural_errors: tuple[str, ...] = ()
    if (world is None) ^ (holdout_policy is None):
        raise ValueError("world and holdout_policy must be provided together")
    if world is not None and holdout_policy is not None:
        structural_errors = validate_full_holdout_examples(
            world, ordered, holdout_policy
        ).errors

    metadata_leaks = _metadata_answer_leaks(ordered)

    blockers: list[str] = []
    by_id = {example.example_id: example for example in ordered}
    for group in cross_split:
        splits = {by_id[example_id].split for example_id in group}
        if Split.train in splits and Split.test in splits:
            blockers.append(
                "prohibited exact/normalized train-test question leakage: " + ", ".join(group)
            )
        elif Split.train in splits and Split.validation in splits:
            blockers.append(
                "prohibited exact/normalized train-validation question leakage: "
                + ", ".join(group)
            )
    for group in semantic_cross_split:
        splits = {by_id[example_id].split for example_id in group}
        split_names = ",".join(sorted(split.value for split in splits))
        blockers.append(
            "prohibited cross-split semantic task duplication "
            f"({split_names}): " + ", ".join(group)
        )
    blockers.extend(f"structural holdout violation: {error}" for error in structural_errors)
    blockers.extend(f"metadata leakage: {error}" for error in metadata_leaks)

    return LeakageReport(
        exact_duplicate_questions=exact,
        within_split_duplicate_reviews=within_split_reviews,
        normalized_duplicate_questions=normalized,
        suspicious_expected_output_duplicates=suspicious_outputs,
        suspicious_near_duplicates=tuple(near_pairs),
        cross_split_near_duplicate_warnings=cross_split_warnings,
        highest_risk_train_validation=highest_risk_train_validation,
        highest_risk_train_test=highest_risk_train_test,
        cross_split_collisions=cross_split,
        semantic_fingerprint_collisions=semantic_cross_split,
        structural_violations=tuple(sorted(structural_errors)),
        metadata_answer_leakage=metadata_leaks,
        blockers=tuple(sorted(blockers)),
    )
