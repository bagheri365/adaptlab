from pathlib import Path

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_full_tasks
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.benchmark.lexical_overlap import run_lexical_overlap_audit
from adaptlab.domain.enums import Difficulty, EvidenceStatus, TaskFamily


def _fixture():
    config = load_benchmark_config(Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml")
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    return examples, chunks


def test_lexical_overlap_audit_is_deterministic() -> None:
    examples, chunks = _fixture()
    first = run_lexical_overlap_audit(examples, chunks).to_dict()
    second = run_lexical_overlap_audit(reversed(examples), reversed(chunks)).to_dict()
    assert first == second


def test_lexical_overlap_audit_covers_all_knowledge_bearing_examples() -> None:
    examples, chunks = _fixture()
    report = run_lexical_overlap_audit(examples, chunks)
    expected = [
        e for e in examples
        if e.task_family is not TaskFamily.behavior_only
        and e.evidence_status in (EvidenceStatus.PRESENT, EvidenceStatus.ABSENT)
    ]
    assert len(report.examples) == len(expected)
    assert set(report.distributions_by_difficulty) == {d.value for d in Difficulty}
    assert len(report.highest_overlap_examples) <= 10


def test_present_examples_compare_gold_and_distractors() -> None:
    examples, chunks = _fixture()
    report = run_lexical_overlap_audit(examples, chunks)
    by_id = {item.example_id: item for item in report.examples}
    present = next(e for e in examples if e.evidence_status is EvidenceStatus.PRESENT and e.task_family is TaskFamily.knowledge_only)
    row = by_id[present.example_id]
    assert {item.chunk_id for item in row.gold} == set(present.gold_chunk_ids)
    assert row.distractors
    assert all(item.chunk_id not in present.gold_chunk_ids for item in row.distractors)


def test_absent_examples_have_no_gold_overlap_but_still_get_distractor_diagnostics() -> None:
    examples, chunks = _fixture()
    report = run_lexical_overlap_audit(examples, chunks)
    by_id = {item.example_id: item for item in report.examples}
    absent = next(e for e in examples if e.evidence_status is EvidenceStatus.ABSENT)
    row = by_id[absent.example_id]
    assert row.gold == ()
    assert row.best_gold_jaccard == 0.0
    assert row.distractors


def test_human_summary_contains_required_sections() -> None:
    examples, chunks = _fixture()
    summary = run_lexical_overlap_audit(examples, chunks).human_summary()
    assert "Overlap by difficulty" in summary
    assert "Identifier shortcuts" in summary
    assert "Highest-overlap examples" in summary
    assert "Suspicious HARD cases" in summary
