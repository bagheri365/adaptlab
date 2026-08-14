"""Deterministic generalization sentinel with no Nimbus-specific factual knowledge."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from adaptlab.domain.enums import Difficulty, ScoringRule

SENTINEL_VERSION = "0.0.0"
DEFAULT_SENTINEL_SEED = 1729
DEFAULT_SENTINEL_COUNT = 100


class GeneralizationCapability(str, Enum):
    INSTRUCTION_FOLLOWING = "instruction_following"
    SIMPLE_EXTRACTION = "simple_extraction"
    FORMAT_TRANSFORMATION = "format_transformation"
    BASIC_CLASSIFICATION = "basic_classification"
    SHORT_DETERMINISTIC_REASONING = "short_deterministic_reasoning"


@dataclass(frozen=True)
class GeneralizationSentinelExample:
    example_id: str
    sentinel_version: str
    capability: GeneralizationCapability
    difficulty: Difficulty
    question: str
    expected_output: Any
    scoring_rule: ScoringRule
    scoring_parameters: dict[str, Any]
    generation_seed: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capability"] = self.capability.value
        data["difficulty"] = self.difficulty.value
        data["scoring_rule"] = self.scoring_rule.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneralizationSentinelExample":
        return cls(
            example_id=data["example_id"],
            sentinel_version=data["sentinel_version"],
            capability=GeneralizationCapability(data["capability"]),
            difficulty=Difficulty(data["difficulty"]),
            question=data["question"],
            expected_output=data["expected_output"],
            scoring_rule=ScoringRule(data["scoring_rule"]),
            scoring_parameters=dict(data["scoring_parameters"]),
            generation_seed=int(data["generation_seed"]),
        )


@dataclass(frozen=True)
class SentinelValidationResult:
    passed: bool
    errors: tuple[str, ...]
    statistics: dict[str, Any]


# Conservative domain markers used only as a leakage guard. These are not benchmark
# facts and are intentionally broader than the current prototype taxonomy.
_PROHIBITED_TERMS = (
    "nimbus",
    "authentication",
    "projects",
    "deployments",
    "storage",
    "billing",
    "observability",
    "permissions",
    "configuration",
    "access-token",
    "rollback",
)
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:AUTH|PROJ|DEPLOY|STORAGE|BILLING|OBS|PERM|CONFIG)_[A-Z0-9_]+\b"
)


def _difficulty_for(index: int) -> Difficulty:
    return (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)[index % 3]


def _instruction_example(index: int, seed: int) -> GeneralizationSentinelExample:
    words = ("cedar", "glass", "orbit", "violet", "marble", "copper")
    chosen = words[(index + seed) % len(words)]
    repeat = 1 + ((index + seed) % 3)
    return GeneralizationSentinelExample(
        example_id=f"SENT_IF_{index:03d}",
        sentinel_version=SENTINEL_VERSION,
        capability=GeneralizationCapability.INSTRUCTION_FOLLOWING,
        difficulty=_difficulty_for(index),
        question=f"Return the word '{chosen}' exactly {repeat} times, separated by commas and no spaces.",
        expected_output=",".join([chosen] * repeat),
        scoring_rule=ScoringRule.STRUCTURED_EXTRACTION,
        scoring_parameters={"mode": "repeat_token", "token": chosen, "count": repeat, "separator": ","},
        generation_seed=seed,
    )


def _extraction_example(index: int, seed: int) -> GeneralizationSentinelExample:
    base = 100 + ((index * 17 + seed) % 800)
    codes = [f"Q{base}", f"Q{base + 3}", f"Q{base + 7}"]
    question = (
        f"Extract the Q-codes from this sentence in appearance order: "
        f"'Archive {codes[0]}, ignore note alpha, compare {codes[1]}, then close {codes[2]}.' "
        "Return only the list of codes."
    )
    return GeneralizationSentinelExample(
        example_id=f"SENT_EX_{index:03d}",
        sentinel_version=SENTINEL_VERSION,
        capability=GeneralizationCapability.SIMPLE_EXTRACTION,
        difficulty=_difficulty_for(index + 1),
        question=question,
        expected_output=codes,
        scoring_rule=ScoringRule.STRUCTURED_EXTRACTION,
        scoring_parameters={"mode": "literal_list", "items": codes},
        generation_seed=seed,
    )


def _format_example(index: int, seed: int) -> GeneralizationSentinelExample:
    name = ("Ada", "Lin", "Mira", "Owen", "Ravi")[(index + seed) % 5]
    score = 10 + ((index * 11 + seed) % 90)
    output = {"name": name, "score": score}
    return GeneralizationSentinelExample(
        example_id=f"SENT_FMT_{index:03d}",
        sentinel_version=SENTINEL_VERSION,
        capability=GeneralizationCapability.FORMAT_TRANSFORMATION,
        difficulty=_difficulty_for(index + 2),
        question=f"Convert 'name={name}; score={score}' to JSON with keys name and score, preserving score as an integer.",
        expected_output=output,
        scoring_rule=ScoringRule.STRUCTURED_EXTRACTION,
        scoring_parameters={"mode": "literal_object", "value": output},
        generation_seed=seed,
    )


def _classification_example(index: int, seed: int) -> GeneralizationSentinelExample:
    value = (index * 9 + seed) % 101
    threshold = 50
    output = "HIGH" if value >= threshold else "LOW"
    return GeneralizationSentinelExample(
        example_id=f"SENT_CLS_{index:03d}",
        sentinel_version=SENTINEL_VERSION,
        capability=GeneralizationCapability.BASIC_CLASSIFICATION,
        difficulty=_difficulty_for(index),
        question=f"Classification rule: HIGH when value >= {threshold}, otherwise LOW. Value={value}. Answer only HIGH or LOW.",
        expected_output=output,
        scoring_rule=ScoringRule.CLASSIFICATION,
        scoring_parameters={"value": value, "threshold": threshold, "operator": "gte", "true_output": "HIGH", "false_output": "LOW"},
        generation_seed=seed,
    )


def _reasoning_example(index: int, seed: int) -> GeneralizationSentinelExample:
    start = 2 + ((index * 5 + seed) % 20)
    add = 1 + ((index * 7 + seed) % 9)
    subtract = (index + seed) % 5
    output = start + add - subtract
    return GeneralizationSentinelExample(
        example_id=f"SENT_RSN_{index:03d}",
        sentinel_version=SENTINEL_VERSION,
        capability=GeneralizationCapability.SHORT_DETERMINISTIC_REASONING,
        difficulty=_difficulty_for(index + 1),
        question=f"Start with {start}. Add {add}, then subtract {subtract}. Return only the final integer.",
        expected_output=output,
        scoring_rule=ScoringRule.CONDITIONAL_RULE,
        scoring_parameters={"mode": "arithmetic", "start": start, "add": add, "subtract": subtract},
        generation_seed=seed,
    )


_BUILDERS = (
    _instruction_example,
    _extraction_example,
    _format_example,
    _classification_example,
    _reasoning_example,
)


def generate_generalization_sentinel(
    seed: int = DEFAULT_SENTINEL_SEED,
    count: int = DEFAULT_SENTINEL_COUNT,
) -> list[GeneralizationSentinelExample]:
    """Generate a deterministic, Nimbus-free sentinel.

    The canonical v0.0 sentinel contains exactly 100 examples. A custom count is
    supported for focused tests, but the full build should pass the configured
    generalization-sentinel count.
    """
    if count < 0:
        raise ValueError("sentinel count must be non-negative")

    per_capability, remainder = divmod(count, len(_BUILDERS))
    examples: list[GeneralizationSentinelExample] = []
    for capability_index, builder in enumerate(_BUILDERS):
        n = per_capability + (1 if capability_index < remainder else 0)
        for local_index in range(1, n + 1):
            examples.append(builder(local_index, seed))

    return sorted(examples, key=lambda example: example.example_id)


def _derive_expected(example: GeneralizationSentinelExample) -> Any:
    params = example.scoring_parameters
    mode = params.get("mode")

    if example.scoring_rule is ScoringRule.STRUCTURED_EXTRACTION:
        if mode == "repeat_token":
            return params["separator"].join([params["token"]] * int(params["count"]))
        if mode == "literal_list":
            return list(params["items"])
        if mode == "literal_object":
            return dict(params["value"])

    if example.scoring_rule is ScoringRule.CLASSIFICATION:
        value = int(params["value"])
        threshold = int(params["threshold"])
        if params.get("operator") == "gte":
            return params["true_output"] if value >= threshold else params["false_output"]

    if example.scoring_rule is ScoringRule.CONDITIONAL_RULE and mode == "arithmetic":
        return int(params["start"]) + int(params["add"]) - int(params["subtract"])

    raise ValueError(f"unsupported sentinel scoring contract for {example.example_id}")


def validate_generalization_sentinel(
    examples: list[GeneralizationSentinelExample],
    *,
    expected_count: int | None = DEFAULT_SENTINEL_COUNT,
    expected_seed: int | None = None,
) -> SentinelValidationResult:
    errors: list[str] = []
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sentinel example IDs")

    if expected_count is not None and len(examples) != expected_count:
        errors.append(f"sentinel count {len(examples)} != expected {expected_count}")

    capability_counts = {capability.value: 0 for capability in GeneralizationCapability}
    difficulty_counts = {difficulty.value: 0 for difficulty in Difficulty}

    for example in examples:
        capability_counts[example.capability.value] += 1
        difficulty_counts[example.difficulty.value] += 1

        if expected_seed is not None and example.generation_seed != expected_seed:
            errors.append(
                f"{example.example_id}: generation_seed {example.generation_seed} != {expected_seed}"
            )

        searchable = f"{example.question} {example.expected_output}".lower()
        leaked_terms = [term for term in _PROHIBITED_TERMS if term in searchable]
        if leaked_terms:
            errors.append(
                f"{example.example_id}: prohibited Nimbus/domain term(s): {', '.join(leaked_terms)}"
            )
        if _IDENTIFIER_PATTERN.search(f"{example.question} {example.expected_output}"):
            errors.append(f"{example.example_id}: Nimbus-style identifier leakage")

        try:
            derived = _derive_expected(example)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{example.example_id}: invalid scoring contract: {exc}")
        else:
            if derived != example.expected_output:
                errors.append(
                    f"{example.example_id}: expected_output does not match deterministic scoring contract"
                )

    return SentinelValidationResult(
        passed=not errors,
        errors=tuple(errors),
        statistics={
            "count": len(examples),
            "capability_counts": capability_counts,
            "difficulty_counts": difficulty_counts,
        },
    )
