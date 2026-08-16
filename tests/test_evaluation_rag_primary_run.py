from pathlib import Path
import json

from adaptlab.evaluation.providers import ModelResponse
from adaptlab.evaluation.rag_config import load_canonical_rag_config
from adaptlab.evaluation.rag_primary_run import canonical_rag_run_id, run_canonical_primary_rag
from adaptlab.retrieval.frozen_artifact import load_and_verify_frozen_retrieval_artifact

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'data/generated/v0.0'
PROMPT=ROOT/'configs/prompts/prompt_v1.yaml'
RAGCFG=ROOT/'configs/evaluation_conditions/milestone4_ollama_rag_v1.yaml'
RETR=ROOT/'artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json'

class FakeProvider:
    provider_name='ollama'
    def __init__(self): self.calls=0
    def generate(self, request):
        self.calls += 1
        return ModelResponse(text='x', input_tokens=1, output_tokens=1, latency_ms=1.0, provider_metadata={})


def test_canonical_primary_rag_complete_and_resumes_from_cache(tmp_path):
    cfg=load_canonical_rag_config(RAGCFG)
    art=load_and_verify_frozen_retrieval_artifact(RETR)
    p=FakeProvider()
    summary=run_canonical_primary_rag(benchmark_dir=BENCH,prompt_config=PROMPT,retrieval_artifact=art,config=cfg,provider=p,output_dir=tmp_path)
    assert summary.represented_count == 400
    assert summary.completed_successful_model_responses == 400
    assert summary.unresolved_provider_failures == 0
    assert summary.valid
    assert p.calls == 400
    rows=json.loads((tmp_path/'results.json').read_text())
    assert len(rows)==400
    assert (tmp_path/'metrics.json').exists()
    assert (tmp_path/'summary.txt').exists()
    assert (tmp_path/'run_manifest.json').exists()
    assert (tmp_path/'completion.json').exists()
    assert all('retrieval_result_reference' in r and 'retrieved_chunk_ids' in r and 'retrieved_context_hash' in r and 'input_hash' in r for r in rows)
    assert all('raw_output' in r and 'normalized_output' in r and 'score' in r and 'runtime_provenance' in r and 'cache_metadata' in r for r in rows)
    p2=FakeProvider()
    summary2=run_canonical_primary_rag(benchmark_dir=BENCH,prompt_config=PROMPT,retrieval_artifact=art,config=cfg,provider=p2,output_dir=tmp_path)
    assert p2.calls == 0
    assert summary2.cache_hit_count == 400
    assert summary2.run_id == summary.run_id
    rows2=json.loads((tmp_path/'results.json').read_text())
    assert [r['raw_output'] for r in rows2] == [r['raw_output'] for r in rows]


def test_run_id_is_bound_to_frozen_config():
    cfg=load_canonical_rag_config(RAGCFG)
    rid=canonical_rag_run_id(config=cfg)
    assert rid.startswith('m4-primary-test-rag-')
