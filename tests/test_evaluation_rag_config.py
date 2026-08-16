import json
from pathlib import Path
import yaml
import pytest

from adaptlab.evaluation.inputs import EVIDENCE_RENDERER_VERSION, evidence_renderer_hash
from adaptlab.evaluation.rag_config import (
    CANONICAL_RAG_CONFIG_VERSION, build_canonical_rag_config, load_canonical_rag_config,
)
from adaptlab.retrieval.canonical_config import CanonicalBM25Config
from adaptlab.retrieval.frozen_artifact import load_and_verify_frozen_retrieval_artifact

ROOT=Path(__file__).resolve().parents[1]
PROMPT=ROOT/'configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml'
RAG=ROOT/'configs/evaluation_conditions/milestone4_ollama_rag_v1.yaml'
BM25=ROOT/'config/retrieval/canonical_bm25_v1.json'
RETRIEVAL=ROOT/'artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json'

def _expected():
    prompt=yaml.safe_load(PROMPT.read_text())
    bm25=CanonicalBM25Config.from_dict(json.loads(BM25.read_text()))
    retrieval=load_and_verify_frozen_retrieval_artifact(RETRIEVAL)
    return build_canonical_rag_config(prompt_condition=prompt,bm25=bm25,retrieval=retrieval)

def test_frozen_canonical_rag_matches_bound_artifacts():
    frozen=load_canonical_rag_config(RAG)
    expected=_expected()
    assert frozen.to_dict()==expected.to_dict()
    assert frozen.config_version==CANONICAL_RAG_CONFIG_VERSION
    assert frozen.retrieval_execution=='consume_frozen_artifact_only'

def test_canonical_rag_shares_m3_inference_controls():
    prompt=yaml.safe_load(PROMPT.read_text())
    rag=load_canonical_rag_config(RAG)
    assert rag.model=='qwen3:8b'==prompt['provider']['model_tag']
    assert rag.provider=='ollama'==prompt['provider']['name']
    assert rag.prompt_hash==prompt['prompt']['prompt_hash']
    for k in ('temperature','seed','context_length','max_tokens','stream','think'):
        assert getattr(rag,k)==prompt['request'][k]
    assert rag.scorer_version==prompt['scoring']['scorer_version']
    assert rag.normalizer_version==prompt['scoring']['normalizer_version']
    assert rag.benchmark_manifest_hash==prompt['benchmark']['benchmark_manifest_hash']

def test_canonical_rag_binds_retrieval_and_renderer():
    rag=load_canonical_rag_config(RAG)
    retrieval=load_and_verify_frozen_retrieval_artifact(RETRIEVAL)
    bm25=CanonicalBM25Config.from_dict(json.loads(BM25.read_text()))
    assert rag.retrieval_run_id==retrieval.retrieval_run_id
    assert rag.retrieval_artifact_hash==retrieval.retrieval_artifact_hash
    assert rag.retriever_config_hash==bm25.retriever_config_hash
    assert rag.query_policy_hash==bm25.query_policy_hash
    assert rag.indexing_policy_hash==bm25.indexing_policy_hash
    assert rag.tokenization_policy_hash==bm25.tokenization_policy_hash
    assert rag.top_k==bm25.top_k==10
    assert rag.corpus_hash==bm25.corpus_hash
    assert rag.evidence_renderer_version==EVIDENCE_RENDERER_VERSION
    assert rag.evidence_renderer_hash==evidence_renderer_hash()

def test_tampered_retrieval_identity_cannot_build():
    prompt=yaml.safe_load(PROMPT.read_text())
    bm25=CanonicalBM25Config.from_dict(json.loads(BM25.read_text()))
    retrieval=load_and_verify_frozen_retrieval_artifact(RETRIEVAL)
    from dataclasses import replace
    with pytest.raises(ValueError):
        build_canonical_rag_config(prompt_condition=prompt,bm25=bm25,retrieval=replace(retrieval, corpus_hash='0'*64))
