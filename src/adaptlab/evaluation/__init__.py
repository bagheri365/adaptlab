"""Evaluation harness contracts for AdaptLab Milestone 3."""

from adaptlab.evaluation.cache import (
    CACHE_SCHEMA_VERSION,
    ExactRequestCache,
    InferenceCacheKey,
    ResultArtifactIdentity,
)
from adaptlab.evaluation.errors import (
    EvaluationError,
    PermanentProviderError,
    ProviderError,
    TransientProviderError,
)
from adaptlab.evaluation.providers import ModelProvider, ModelRequest, ModelResponse
from adaptlab.evaluation.inputs import (
    EVIDENCE_FORMAT_VERSION,
    ConstructedModelInput,
    canonical_model_input_bytes,
    construct_model_input,
)
from adaptlab.evaluation.provenance import GitState, capture_git_state, require_canonical_git_state
from adaptlab.evaluation.prompts import (
    PROMPT_CONFIG_SCHEMA_VERSION,
    PromptContract,
    load_prompt_contract,
)
from adaptlab.evaluation.metrics import (
    METRICS_SCHEMA_VERSION,
    AccuracyMetric,
    AggregateMetrics,
    aggregate_metrics,
)
from adaptlab.evaluation.runner import (
    METRICS_FILENAME, RESULTS_FILENAME, RUN_MANIFEST_FILENAME, SUMMARY_FILENAME,
    load_benchmark_split, run_evaluation, verify_frozen_benchmark,
)
from adaptlab.evaluation.scoring import (
    NORMALIZER_VERSION,
    SCORER_VERSION,
    SUPPORTED_SCORING_RULES,
    ScoredOutput,
    normalize_output,
    score_output,
)
from adaptlab.evaluation.schemas import (
    EVALUATION_RESULT_SCHEMA_VERSION,
    EVALUATION_RUN_SCHEMA_VERSION,
    FROZEN_BENCHMARK_TAG,
    AdaptationMethod,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunStatus,
    ModelInput,
)

__all__ = [
    "AdaptationMethod",
    "GitState",
    "capture_git_state",
    "require_canonical_git_state",
    "CACHE_SCHEMA_VERSION",
    "ExactRequestCache",
    "InferenceCacheKey",
    "ResultArtifactIdentity",
    "run_evaluation",
    "verify_frozen_benchmark",
    "load_benchmark_split",
    "RESULTS_FILENAME",
    "METRICS_FILENAME",
    "SUMMARY_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "aggregate_metrics",
    "METRICS_SCHEMA_VERSION",
    "AggregateMetrics",
    "AccuracyMetric",
    "ConstructedModelInput",
    "EVIDENCE_FORMAT_VERSION",
    "EVALUATION_RESULT_SCHEMA_VERSION",
    "EVALUATION_RUN_SCHEMA_VERSION",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationRun",
    "EvaluationRunStatus",
    "FROZEN_BENCHMARK_TAG",
    "ModelInput",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "PROMPT_CONFIG_SCHEMA_VERSION",
    "NORMALIZER_VERSION",
    "SCORER_VERSION",
    "SUPPORTED_SCORING_RULES",
    "ScoredOutput",
    "PromptContract",
    "PermanentProviderError",
    "ProviderError",
    "TransientProviderError",
    "canonical_model_input_bytes",
    "construct_model_input",
    "normalize_output",
    "score_output",
    "load_prompt_contract",
]
