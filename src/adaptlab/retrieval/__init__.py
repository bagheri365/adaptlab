"""Retrieval experiment contracts for AdaptLab Milestone 4."""

from adaptlab.retrieval.query_policy import (
    QUERY_POLICY_VERSION, RetrievalQuery, construct_retrieval_query,
    is_retrieval_eligible, query_policy_hash, verify_frozen_query_policy,
)
from adaptlab.retrieval.schemas import (
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    RETRIEVAL_RUN_SCHEMA_VERSION,
    RetrievalResult,
    RetrievalRunManifest,
)

__all__ = [
    "QUERY_POLICY_VERSION",
    "RetrievalQuery",
    "construct_retrieval_query",
    "is_retrieval_eligible",
    "query_policy_hash",
    "verify_frozen_query_policy",
    "RETRIEVAL_RESULT_SCHEMA_VERSION",
    "RETRIEVAL_RUN_SCHEMA_VERSION",
    "RetrievalResult",
    "RetrievalRunManifest",
]

from adaptlab.retrieval.bm25 import (
    BM25_RETRIEVER_NAME,
    BM25_RETRIEVER_VERSION,
    BM25Retriever,
    RetrievalHit,
    Retriever,
    bm25_config_hash,
    bm25_config_payload,
    frozen_corpus_hash,
)

__all__ += [
    "BM25_RETRIEVER_NAME", "BM25_RETRIEVER_VERSION", "BM25Retriever",
    "RetrievalHit", "Retriever", "bm25_config_hash", "bm25_config_payload",
    "frozen_corpus_hash",
]

from adaptlab.retrieval.metrics import (
    METRIC_SCHEMA_VERSION,
    PRIMARY_CUTOFFS,
    RetrievalMetricRow,
    RetrievalMetricsReport,
    RetrievalMetricValues,
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
    with_retrieval_metrics,
)

__all__ += [
    "METRIC_SCHEMA_VERSION", "PRIMARY_CUTOFFS", "RetrievalMetricRow",
    "RetrievalMetricsReport", "RetrievalMetricValues", "compute_retrieval_metrics",
    "summarize_retrieval_metrics", "with_retrieval_metrics",
]

from adaptlab.retrieval.version_metrics import (
    VERSION_METRIC_SCHEMA_VERSION,
    VersionDiagnosticRow,
    VersionDiagnosticsReport,
    summarize_version_diagnostics,
    with_version_diagnostics,
)

__all__ += [
    "VERSION_METRIC_SCHEMA_VERSION", "VersionDiagnosticRow", "VersionDiagnosticsReport",
    "summarize_version_diagnostics", "with_version_diagnostics",
]

from adaptlab.retrieval.absent_diagnostics import (
    ABSENT_DIAGNOSTIC_SCHEMA_VERSION,
    AbsentDiagnosticRow,
    AbsentDiagnosticsReport,
    summarize_absent_diagnostics,
    with_absent_diagnostics,
)

__all__ += [
    "ABSENT_DIAGNOSTIC_SCHEMA_VERSION", "AbsentDiagnosticRow", "AbsentDiagnosticsReport",
    "summarize_absent_diagnostics", "with_absent_diagnostics",
]

from adaptlab.retrieval.failure_audit import (
    FAILURE_AUDIT_SCHEMA_VERSION,
    RetrievalFailureAudit,
    RetrievalFailureAuditReport,
    RetrievalFailureCategory,
    RetrievalFailureGroupRow,
    audit_retrieval_failure,
    summarize_retrieval_failures,
)

__all__ += [
    "FAILURE_AUDIT_SCHEMA_VERSION", "RetrievalFailureAudit", "RetrievalFailureAuditReport",
    "RetrievalFailureCategory", "RetrievalFailureGroupRow", "audit_retrieval_failure",
    "summarize_retrieval_failures",
]

from adaptlab.retrieval.top_k_selection import (
    ALL_REQUIRED_TOLERANCE,
    PRIMARY_METRIC,
    SECONDARY_TIE_BREAK,
    TOP_K_CANDIDATES,
    TOP_K_SELECTION_VERSION,
    TopKSelectionDecision,
    select_top_k,
    top_k_selection_policy_hash,
    top_k_selection_policy_payload,
    verify_frozen_top_k_selection_policy,
)

__all__ += [
    "ALL_REQUIRED_TOLERANCE", "PRIMARY_METRIC", "SECONDARY_TIE_BREAK",
    "TOP_K_CANDIDATES", "TOP_K_SELECTION_VERSION", "TopKSelectionDecision",
    "select_top_k", "top_k_selection_policy_hash", "top_k_selection_policy_payload",
    "verify_frozen_top_k_selection_policy",
]

from adaptlab.retrieval.canonical_config import (
    CANONICAL_BM25_CONFIG_FILENAME,
    CANONICAL_BM25_CONFIG_VERSION,
    CanonicalBM25Config,
    build_canonical_bm25_config,
    verify_frozen_canonical_bm25_config,
)

__all__ += [
    "CANONICAL_BM25_CONFIG_FILENAME", "CANONICAL_BM25_CONFIG_VERSION",
    "CanonicalBM25Config", "build_canonical_bm25_config",
    "verify_frozen_canonical_bm25_config",
]

from .primary_test_analysis import analyze_primary_test_retrieval

from adaptlab.retrieval.frozen_artifact import (
    FROZEN_RETRIEVAL_ARTIFACT_VERSION,
    FrozenRetrievalArtifact,
    FrozenRetrievalEntry,
    build_frozen_retrieval_artifact,
    freeze_canonical_retrieval_results,
    load_and_verify_frozen_retrieval_artifact,
)

__all__ += [
    "FROZEN_RETRIEVAL_ARTIFACT_VERSION", "FrozenRetrievalArtifact", "FrozenRetrievalEntry",
    "build_frozen_retrieval_artifact", "freeze_canonical_retrieval_results",
    "load_and_verify_frozen_retrieval_artifact",
]
