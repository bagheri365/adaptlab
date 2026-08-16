import json
from pathlib import Path
from adaptlab.retrieval.primary_test_analysis import analyze_primary_test_retrieval

def test_primary_test_analysis_is_read_only_and_complete(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    results = root/'artifacts/retrieval/m4/primary_test_bm25_v1/results.json'
    before = results.read_bytes()
    payload = analyze_primary_test_retrieval(results_path=results, chunks_path=root/'data/generated/v0.0/chunks.json', output_dir=tmp_path)
    assert results.read_bytes() == before
    assert payload['retrieval_quality']['rows'][0]['dimension'] == 'overall'
    dims = {(r['dimension'], r['value']) for r in payload['retrieval_quality']['rows']}
    for value in ('knowledge_only','behavior_knowledge','changed_knowledge'):
        assert ('task_family', value) in dims
    for value in ('EASY','MEDIUM','HARD'):
        assert ('difficulty', value) in dims
    for value in ('iid','structural_holdout'):
        assert ('split_type', value) in dims
    for value in ('UNCHANGED','UPDATED','REMOVED'):
        assert ('knowledge_state', value) in dims
    assert (tmp_path/'analysis.json').exists() and (tmp_path/'analysis.txt').exists()
    assert 'OBSERVATION' in (tmp_path/'analysis.txt').read_text()
