"""Deterministic construction rules for benchmark difficulty labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from adaptlab.domain.enums import Difficulty


class DifficultySource(str, Enum):
    """Controlled construction features used to justify declared difficulty."""

    DIRECT_WORDING = "direct_wording"
    PARAPHRASED_WORDING = "paraphrased_wording"
    MULTIPLE_PLAUSIBLE_CANDIDATES = "multiple_plausible_candidates"
    SEMANTIC_DISTRACTORS = "semantic_distractors"
    MULTI_CHUNK_EVIDENCE = "multi_chunk_evidence"
    OBSOLETE_CONFLICT = "obsolete_conflicting_documentation"
    NEAR_DUPLICATE_DISTRACTORS = "near_duplicate_distractors"
    INDIRECT_SEMANTIC_WORDING = "indirect_semantic_wording"
    SIMILAR_IDENTIFIERS = "similar_identifiers"
    VERSION_DISCRIMINATION = "version_discrimination"
    DETERMINISTIC_INFERENCE = "deterministic_inference"


_HARD_CONTROLLED_SOURCES = frozenset(
    {
        DifficultySource.MULTI_CHUNK_EVIDENCE,
        DifficultySource.OBSOLETE_CONFLICT,
        DifficultySource.NEAR_DUPLICATE_DISTRACTORS,
        DifficultySource.INDIRECT_SEMANTIC_WORDING,
        DifficultySource.SIMILAR_IDENTIFIERS,
        DifficultySource.VERSION_DISCRIMINATION,
        DifficultySource.DETERMINISTIC_INFERENCE,
    }
)


@dataclass(frozen=True, slots=True)
class DifficultyPlan:
    """Pre-generation construction metadata for one benchmark example.

    This is intentionally separate from :class:`BenchmarkExample`: difficulty is
    fixed during generation, before any future model evaluation, without changing
    the frozen benchmark-example contract.

    ``required_evidence_cardinality`` is the number of distinct gold chunks the
    task explicitly requires to derive/corroborate the answer.
    ``retrieval_candidate_count`` is the number of relevant or competing chunks
    deliberately present in the constructed retrieval-difficulty scenario. The
    two counts are intentionally different concepts.
    """

    difficulty: Difficulty
    relevant_fact_count: int
    required_evidence_cardinality: int
    retrieval_candidate_count: int
    sources: tuple[DifficultySource, ...]
    multi_hop: bool = False
    retrieval_applicable: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty.value,
            "relevant_fact_count": self.relevant_fact_count,
            "required_evidence_cardinality": self.required_evidence_cardinality,
            "retrieval_candidate_count": self.retrieval_candidate_count,
            "sources": [source.value for source in self.sources],
            "multi_hop": self.multi_hop,
            "retrieval_applicable": self.retrieval_applicable,
        }


def build_difficulty_plan(difficulty: Difficulty, variant: int = 0) -> DifficultyPlan:
    """Return a deterministic construction plan for ``difficulty``.

    ``variant`` is used only to rotate among predeclared controlled constructions;
    it never introduces random post-hoc labeling.
    """

    if variant < 0:
        raise ValueError("variant must be non-negative")

    if difficulty is Difficulty.EASY:
        return DifficultyPlan(
            difficulty=difficulty,
            relevant_fact_count=1,
            required_evidence_cardinality=1,
            retrieval_candidate_count=1,
            sources=(DifficultySource.DIRECT_WORDING,),
            multi_hop=False,
        )

    if difficulty is Difficulty.MEDIUM:
        variants = (
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=1,
                required_evidence_cardinality=1,
                retrieval_candidate_count=3,
                sources=(
                    DifficultySource.PARAPHRASED_WORDING,
                    DifficultySource.MULTIPLE_PLAUSIBLE_CANDIDATES,
                ),
                multi_hop=False,
            ),
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=2,
                required_evidence_cardinality=2,
                retrieval_candidate_count=2,
                sources=(
                    DifficultySource.PARAPHRASED_WORDING,
                    DifficultySource.SEMANTIC_DISTRACTORS,
                ),
                multi_hop=False,
            ),
        )
        return variants[variant % len(variants)]

    if difficulty is Difficulty.HARD:
        variants = (
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=2,
                required_evidence_cardinality=2,
                retrieval_candidate_count=4,
                sources=(
                    DifficultySource.MULTI_CHUNK_EVIDENCE,
                    DifficultySource.NEAR_DUPLICATE_DISTRACTORS,
                ),
                multi_hop=False,
            ),
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=1,
                required_evidence_cardinality=1,
                retrieval_candidate_count=3,
                sources=(
                    DifficultySource.OBSOLETE_CONFLICT,
                    DifficultySource.VERSION_DISCRIMINATION,
                ),
                multi_hop=False,
            ),
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=2,
                required_evidence_cardinality=2,
                retrieval_candidate_count=3,
                sources=(
                    DifficultySource.SIMILAR_IDENTIFIERS,
                    DifficultySource.INDIRECT_SEMANTIC_WORDING,
                ),
                multi_hop=False,
            ),
            DifficultyPlan(
                difficulty=difficulty,
                relevant_fact_count=3,
                required_evidence_cardinality=3,
                retrieval_candidate_count=3,
                sources=(DifficultySource.DETERMINISTIC_INFERENCE,),
                multi_hop=True,
            ),
        )
        return variants[variant % len(variants)]

    raise ValueError(f"unsupported difficulty: {difficulty!r}")


def validate_difficulty_plan(plan: DifficultyPlan) -> tuple[str, ...]:
    """Return deterministic construction-rule violations for ``plan``."""

    errors: list[str] = []
    if plan.relevant_fact_count < 0:
        errors.append("relevant_fact_count must be non-negative")
    if plan.required_evidence_cardinality < 0:
        errors.append("required_evidence_cardinality must be non-negative")
    if plan.retrieval_candidate_count < 0:
        errors.append("retrieval_candidate_count must be non-negative")
    if not plan.retrieval_applicable:
        if plan.required_evidence_cardinality != 0:
            errors.append("retrieval-inapplicable tasks require required_evidence_cardinality=0")
        if plan.retrieval_candidate_count != 0:
            errors.append("retrieval-inapplicable tasks require retrieval_candidate_count=0")
        return tuple(errors)

    if plan.difficulty is Difficulty.EASY:
        if plan.relevant_fact_count != 1:
            errors.append("EASY requires exactly one relevant fact")
        if plan.required_evidence_cardinality != 1:
            errors.append("EASY requires exactly one clear evidence chunk")
        if plan.retrieval_candidate_count != 1:
            errors.append("EASY requires one clear retrieval candidate")
        if DifficultySource.DIRECT_WORDING not in plan.sources:
            errors.append("EASY requires direct wording")
        if plan.multi_hop:
            errors.append("EASY must not require multi-hop reasoning")

    elif plan.difficulty is Difficulty.MEDIUM:
        if plan.relevant_fact_count not in (1, 2):
            errors.append("MEDIUM requires one or two relevant facts")
        if plan.required_evidence_cardinality not in (1, 2):
            errors.append("MEDIUM requires one or two required gold evidence chunks")
        if plan.retrieval_candidate_count < 2:
            errors.append("MEDIUM requires multiple plausible retrieval candidates")
        if DifficultySource.PARAPHRASED_WORDING not in plan.sources:
            errors.append("MEDIUM requires paraphrased wording")
        if not (
            DifficultySource.MULTIPLE_PLAUSIBLE_CANDIDATES in plan.sources
            or DifficultySource.SEMANTIC_DISTRACTORS in plan.sources
        ):
            errors.append("MEDIUM requires retrieval ambiguity or semantic distractors")

    elif plan.difficulty is Difficulty.HARD:
        if not (_HARD_CONTROLLED_SOURCES & set(plan.sources)):
            errors.append("HARD requires at least one controlled source of difficulty")
        if plan.required_evidence_cardinality > 3:
            errors.append("HARD rules allow at most three required gold evidence chunks")
        if plan.relevant_fact_count > 3:
            errors.append("HARD prototype rules allow at most three relevant facts")
    else:
        errors.append(f"unsupported difficulty: {plan.difficulty!r}")

    return tuple(errors)


def difficulty_metadata(difficulty: Difficulty, variant: int = 0) -> dict[str, object]:
    """Convenience metadata for full task generation in the next pipeline stage."""

    plan = build_difficulty_plan(difficulty, variant)
    errors = validate_difficulty_plan(plan)
    if errors:
        raise ValueError("invalid generated difficulty plan: " + "; ".join(errors))
    return plan.to_dict()
