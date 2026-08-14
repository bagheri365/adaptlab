"""Stable enum contracts for the Nimbus benchmark."""

from enum import Enum


class TaskFamily(str, Enum):
    behavior_only = "behavior_only"
    knowledge_only = "knowledge_only"
    behavior_knowledge = "behavior_knowledge"
    changed_knowledge = "changed_knowledge"


class BehaviorType(str, Enum):
    SCHEMA_ADHERENCE = "SCHEMA_ADHERENCE"
    CONDITIONAL_DECISION_RULE = "CONDITIONAL_DECISION_RULE"
    TRANSFORMATION_EXTRACTION = "TRANSFORMATION_EXTRACTION"
    CLASSIFICATION_POLICY = "CLASSIFICATION_POLICY"
    ABSTENTION_BEHAVIOR = "ABSTENTION_BEHAVIOR"


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class EvidenceStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class KnowledgeState(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCHANGED = "UNCHANGED"
    UPDATED = "UPDATED"
    REMOVED = "REMOVED"


class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


class SplitType(str, Enum):
    iid = "iid"
    structural_holdout = "structural_holdout"


class DocumentStyle(str, Enum):
    reference_documentation = "reference_documentation"
    troubleshooting_guide = "troubleshooting_guide"
    release_note = "release_note"
    configuration_guide = "configuration_guide"


class ScoringRule(str, Enum):
    FACT_VALUE = "FACT_VALUE"
    RETIRED_STATUS = "RETIRED_STATUS"
    STRUCTURED_EXTRACTION = "STRUCTURED_EXTRACTION"
    CONDITIONAL_RULE = "CONDITIONAL_RULE"
    CLASSIFICATION = "CLASSIFICATION"
    ABSTENTION = "ABSTENTION"
