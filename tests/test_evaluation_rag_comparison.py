from dataclasses import replace
from pathlib import Path
import json
import pytest

from adaptlab.domain.enums import Split
from adaptlab.evaluation.rag_comparison import (
    RAGComparisonBlockedError, analyze_prompt_rag_oracle, write_blocked_comparison_artifact,
)
from adaptlab.evaluation.runner import load_benchmark_split

BENCH = Path("data/generated/v0.0")


def _rows(examples, score=1.0):
    return [{"example_id": e.example_id, "input_hash": "same" if e.task_family.value == "behavior_only" else f"h-{e.example_id}",
             "normalized_output": "x", "score": score, "provider_error": None} for e in examples]


def test_complete_three_condition_analysis_and_behavior_control():
    examples = load_benchmark_split(BENCH, Split.test)
    p = _rows(examples, 0.5); r = _rows(examples, 0.75); o = _rows(examples, 1.0)
    report = analyze_prompt_rag_oracle(examples=examples, prompt_rows=p, rag_rows=r, oracle_rows=o)
    overall = next(x for x in report["slices"] if x["dimension"] == "overall")
    assert overall["n"] == 400
    assert overall["rag_minus_prompt"] == pytest.approx(0.25)
    assert overall["gold_evidence_minus_rag"] == pytest.approx(0.25)
    assert report["behavior_only"]["byte_identical_inputs_verified"] is True


def test_incomplete_rag_blocks_analysis():
    examples = load_benchmark_split(BENCH, Split.test)
    p = _rows(examples); r = _rows(examples); o = _rows(examples)
    r[0]["score"] = None; r[0]["provider_error"] = "connection refused"
    with pytest.raises(RAGComparisonBlockedError, match="unsuccessful"):
        analyze_prompt_rag_oracle(examples=examples, prompt_rows=p, rag_rows=r, oracle_rows=o)


def test_behavior_input_mismatch_blocks_analysis():
    examples = load_benchmark_split(BENCH, Split.test)
    p = _rows(examples); r = _rows(examples); o = _rows(examples)
    idx = next(i for i,e in enumerate(examples) if e.task_family.value == "behavior_only")
    r[idx]["input_hash"] = "different"
    with pytest.raises(RAGComparisonBlockedError, match="behavior_only input mismatch"):
        analyze_prompt_rag_oracle(examples=examples, prompt_rows=p, rag_rows=r, oracle_rows=o)


def test_blocked_artifact_is_machine_and_human_readable(tmp_path):
    report = write_blocked_comparison_artifact(output_dir=tmp_path, reasons=["RAG 0/400"], rag_summary={"valid": False})
    assert report["status"] == "BLOCKED"
    assert json.loads((tmp_path / "analysis_blocked.json").read_text())["status"] == "BLOCKED"
    assert "OBSERVATION" in (tmp_path / "analysis_blocked.txt").read_text()
