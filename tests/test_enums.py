import json

import pytest

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


EXPECTED_VALUES = {
    TaskFamily: {
        "behavior_only",
        "knowledge_only",
        "behavior_knowledge",
        "changed_knowledge",
    },
    BehaviorType: {
        "SCHEMA_ADHERENCE",
        "CONDITIONAL_DECISION_RULE",
        "TRANSFORMATION_EXTRACTION",
        "CLASSIFICATION_POLICY",
        "ABSTENTION_BEHAVIOR",
    },
    Difficulty: {"EASY", "MEDIUM", "HARD"},
    EvidenceStatus: {"NOT_APPLICABLE", "PRESENT", "ABSENT"},
    KnowledgeState: {"NOT_APPLICABLE", "UNCHANGED", "UPDATED", "REMOVED"},
    Split: {"train", "validation", "test"},
    SplitType: {"iid", "structural_holdout"},
    ScoringRule: {"FACT_VALUE", "RETIRED_STATUS", "STRUCTURED_EXTRACTION", "CONDITIONAL_RULE", "CLASSIFICATION", "ABSTENTION"},
    DocumentStyle: {
        "reference_documentation",
        "troubleshooting_guide",
        "release_note",
        "configuration_guide",
    },
}


@pytest.mark.parametrize("enum_cls, expected", EXPECTED_VALUES.items())
def test_enum_has_exact_expected_values(enum_cls, expected) -> None:
    assert {member.value for member in enum_cls} == expected
    assert len(enum_cls.__members__) == len(expected), "enum aliases are not allowed"


@pytest.mark.parametrize("enum_cls, expected", EXPECTED_VALUES.items())
def test_enum_values_serialize_cleanly_to_json(enum_cls, expected) -> None:
    payload = [member for member in enum_cls]
    assert set(json.loads(json.dumps(payload))) == expected


@pytest.mark.parametrize("enum_cls", EXPECTED_VALUES)
def test_invalid_enum_values_are_rejected(enum_cls) -> None:
    with pytest.raises(ValueError):
        enum_cls("__invalid_contract_value__")
