"""Frozen primary-test retrieval analysis for Milestone 4 Prompt 13."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.io import canonical_json_bytes, sha256_bytes
from adaptlab.retrieval.schemas import RetrievalResult
from adaptlab.retrieval.metrics import summarize_retrieval_metrics
from adaptlab.retrieval.failure_audit import summarize_retrieval_failures
from adaptlab.retrieval.absent_diagnostics import summarize_absent_diagnostics
from adaptlab.retrieval.version_metrics import summarize_version_diagnostics

ANALYSIS_VERSION = "primary-test-retrieval-analysis-v1"


def analyze_primary_test_retrieval(*, results_path: Path, chunks_path: Path, output_dir: Path) -> dict[str, object]:
    results = tuple(RetrievalResult.from_dict(x) for x in json.loads(results_path.read_text()))
    chunks = tuple(DocumentChunk.from_dict(x) for x in json.loads(chunks_path.read_text()))
    if len(results) != 400 or any(r.split.value != "test" for r in results):
        raise ValueError("analysis requires the complete frozen 400-example primary-test retrieval artifact")
    metrics = summarize_retrieval_metrics(results)
    versions = summarize_version_diagnostics(results, chunks)
    failures = summarize_retrieval_failures(results, chunks)
    absent = summarize_absent_diagnostics(results)
    fail_counts = {name: sum(name in a.categories for a in failures.examples) for name in (
        "OBSOLETE_ONLY", "WRONG_VERSION_TOP1", "CURRENT_AND_OBSOLETE", "GOLD_OUTSIDE_TOP_K")}
    # CURRENT_AND_OBSOLETE comes from version diagnostics rather than failure categories.
    fail_counts["CURRENT_AND_OBSOLETE"] = sum(r.current_and_obsolete_retrieved is True for r in results)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "source_results_hash": sha256_bytes(results_path.read_bytes()),
        "retrieval_quality": metrics.to_dict(),
        "version_diagnostics": versions.to_dict(),
        "version_failure_counts": fail_counts,
        "evidence_absent_diagnostics": absent.to_dict(),
        "claims": {
            "OBSERVATION": "Metrics summarize only the already-frozen canonical primary-test retrieval artifact.",
            "INTERPRETATION": "Differences across slices describe lexical BM25 retrieval behavior under the frozen configuration; they do not establish RAG generation performance.",
            "LIMITATION": "No retriever tuning is permitted from these primary-test results, and evidence-ABSENT returned chunks are unverified context rather than sufficient evidence.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_bytes(canonical_json_bytes(payload))
    text = metrics.to_text() + "\n" + versions.to_text() + "\n" + absent.to_text() + "\nOBSERVATION\n" + payload["claims"]["OBSERVATION"] + "\n\nINTERPRETATION\n" + payload["claims"]["INTERPRETATION"] + "\n\nLIMITATION\n" + payload["claims"]["LIMITATION"] + "\n"
    (output_dir / "analysis.txt").write_text(text)
    return payload
