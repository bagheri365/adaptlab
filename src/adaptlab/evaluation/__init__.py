"""Evaluation harness contracts for AdaptLab Milestone 3."""

from adaptlab.evaluation.causal_controls import (
    CAUSAL_CONTROL_SCHEMA_VERSION,
    CausalControlReport,
    CausalControlValidationError,
    derive_rag_control_condition,
    load_condition_config,
    require_causal_controls,
    validate_causal_controls,
    write_causal_control_artifact,
)
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
    EVIDENCE_RENDERER_VERSION,
    evidence_renderer_contract,
    evidence_renderer_hash,
    ConstructedModelInput,
    ConstructedRAGInput,
    canonical_model_input_bytes,
    construct_model_input,
    construct_rag_model_input,
)
from adaptlab.evaluation.rag_completeness import (
    RAG_COMPLETENESS_SCHEMA_VERSION,
    CANONICAL_RAG_EXPECTED_COUNT,
    RAGExampleCompletion,
    RAGRunIdentity,
    RAGCompletenessRecord,
    canonical_rag_completeness,
    require_canonical_rag_complete,
    resume_identity_matches,
)
from adaptlab.evaluation.rag_config import (
    CANONICAL_RAG_CONFIG_VERSION,
    CanonicalRAGConfig,
    build_canonical_rag_config,
    load_canonical_rag_config,
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
    "CAUSAL_CONTROL_SCHEMA_VERSION",
    "CausalControlReport",
    "CausalControlValidationError",
    "derive_rag_control_condition",
    "load_condition_config",
    "require_causal_controls",
    "validate_causal_controls",
    "write_causal_control_artifact",
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
    "ConstructedRAGInput",
    "EVIDENCE_FORMAT_VERSION",
    "EVIDENCE_RENDERER_VERSION",
    "evidence_renderer_contract",
    "evidence_renderer_hash",
    "CANONICAL_RAG_CONFIG_VERSION",
    "CanonicalRAGConfig",
    "build_canonical_rag_config",
    "load_canonical_rag_config",
    "RAG_COMPLETENESS_SCHEMA_VERSION",
    "CANONICAL_RAG_EXPECTED_COUNT",
    "RAGExampleCompletion",
    "RAGRunIdentity",
    "RAGCompletenessRecord",
    "canonical_rag_completeness",
    "require_canonical_rag_complete",
    "resume_identity_matches",
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
    "construct_rag_model_input",
    "normalize_output",
    "score_output",
    "load_prompt_contract",
]

from .rag_primary_run import CanonicalRAGRunSummary, canonical_rag_run_id, run_canonical_primary_rag
from .rag_comparison import (
    RAG_COMPARISON_SCHEMA_VERSION,
    RAGComparisonBlockedError,
    analyze_prompt_rag_oracle,
    write_blocked_comparison_artifact,
)
