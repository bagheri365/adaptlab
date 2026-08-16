from pathlib import Path
import json
from adaptlab.retrieval.primary_test_run import run_primary_test_retrieval

ROOT = Path(__file__).resolve().parents[1]

def test_primary_test_run_represents_all_400_and_bypasses_behavior_only(tmp_path):
    summary = run_primary_test_retrieval(
        test_path=ROOT/'data/generated/v0.0/test.json', chunks_path=ROOT/'data/generated/v0.0/chunks.json',
        benchmark_manifest_path=ROOT/'data/generated/v0.0/manifest.json',
        canonical_config_path=ROOT/'config/retrieval/canonical_bm25_v1.json', output_dir=tmp_path)
    assert summary['example_count'] == summary['represented_count'] == 400
    assert summary['retrieval_errors'] == 0
    rows = json.loads((tmp_path/'results.json').read_text())
    behavior = [r for r in rows if r['task_family'] == 'behavior_only']
    assert behavior and all(not r['retrieval_eligible'] and r['candidate_chunk_ids'] == [] for r in behavior)
    absent = [r for r in rows if r['evidence_status'] == 'ABSENT' and r['retrieval_eligible']]
    assert absent and all(r['any_gold_at_k'] is None for r in absent)
