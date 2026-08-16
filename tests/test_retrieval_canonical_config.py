import json
from pathlib import Path

import pytest

from adaptlab.retrieval.canonical_config import (
    CANONICAL_BM25_CONFIG_VERSION,
    build_canonical_bm25_config,
    verify_frozen_canonical_bm25_config,
)

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "artifacts/retrieval/m4/validation_bm25_candidates_v1/top_k_selection_decision.json"
MANIFEST = ROOT / "artifacts/retrieval/m4/validation_bm25_candidates_v1/run_manifest.json"
FROZEN = ROOT / "config/retrieval/canonical_bm25_v1.json"


def test_frozen_canonical_config_matches_validation_only_selection():
    cfg = verify_frozen_canonical_bm25_config(
        FROZEN,
        selection_decision_path=DECISION,
        validation_manifest_path=MANIFEST,
    )
    assert cfg.config_version == CANONICAL_BM25_CONFIG_VERSION
    assert cfg.retriever_name == "BM25"
    assert cfg.top_k == 10
    assert cfg.tie_break_policy == "equal_bm25_score_then_chunk_id_ascending"


def test_canonical_config_binds_all_required_provenance():
    cfg = build_canonical_bm25_config(
        selection_decision_path=DECISION,
        validation_manifest_path=MANIFEST,
    )
    raw = cfg.to_dict()
    for key in (
        "retriever_name", "retriever_version", "retriever_config_hash",
        "query_policy_version", "query_policy_hash",
        "indexing_policy_version", "indexing_policy_hash",
        "tokenization_policy_version", "tokenization_policy_hash",
        "k1", "b", "top_k", "tie_break_policy", "corpus_hash",
        "benchmark_manifest_hash",
    ):
        assert key in raw
    assert len(cfg.canonical_config_hash) == 64


def test_canonical_config_rejects_non_validation_selection(tmp_path: Path):
    decision = json.loads(DECISION.read_text())
    decision["selection_input"] = "primary_test_metrics"
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="validation retrieval metrics only"):
        build_canonical_bm25_config(selection_decision_path=path, validation_manifest_path=MANIFEST)


def test_frozen_config_detects_tampering(tmp_path: Path):
    raw = json.loads(FROZEN.read_text())
    raw["top_k"] = 5
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        verify_frozen_canonical_bm25_config(
            path,
            selection_decision_path=DECISION,
            validation_manifest_path=MANIFEST,
        )
