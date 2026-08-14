import json
from pathlib import Path

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_full_tasks
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.benchmark.human_audit import (
    HumanAuditReview,
    build_human_audit_artifact,
    build_pending_human_review_queue,
    select_human_audit_sample,
    write_human_audit_artifact,
    write_human_review_queue,
)

CONFIG = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"


def _examples():
    config = load_benchmark_config(CONFIG)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    return generate_full_tasks(world, documents, chunks, config, policy)


def test_human_audit_sample_is_deterministic_and_stratified():
    examples = _examples()
    first = select_human_audit_sample(examples, sample_size=50)
    second = select_human_audit_sample(reversed(examples), sample_size=50)
    assert [item.example_id for item in first] == [item.example_id for item in second]

    assert {item.task_family.value for item in first} == {
        "behavior_only", "knowledge_only", "behavior_knowledge", "changed_knowledge"
    }
    assert {item.difficulty.value for item in first} == {"EASY", "MEDIUM", "HARD"}
    assert {item.split_type.value for item in first} == {"iid", "structural_holdout"}
    assert {item.evidence_status.value for item in first} >= {"PRESENT", "ABSENT"}
    assert {item.knowledge_state.value for item in first} >= {"UNCHANGED", "UPDATED", "REMOVED"}
    lifecycle_counts = {
        state: sum(1 for item in first if item.knowledge_state.value == state)
        for state in ("UNCHANGED", "UPDATED", "REMOVED")
    }
    assert lifecycle_counts["UNCHANGED"] >= 5
    assert lifecycle_counts["UPDATED"] >= 5
    assert lifecycle_counts["REMOVED"] >= 5
    assert {item.behavior_type.value for item in first if item.behavior_type} == {
        "SCHEMA_ADHERENCE",
        "CONDITIONAL_DECISION_RULE",
        "TRANSFORMATION_EXTRACTION",
        "CLASSIFICATION_POLICY",
        "ABSTENTION_BEHAVIOR",
    }


def test_human_audit_artifact_requires_exact_review_coverage(tmp_path):
    sample = select_human_audit_sample(_examples(), sample_size=40)
    reviews = [
        HumanAuditReview(
            example_id=item.example_id,
            review_result="PASS",
            notes="Reviewed.",
            correction_required=False,
            checks={"gold_answer_uniquely_defensible": True},
        )
        for item in sample
    ]
    artifact = build_human_audit_artifact(sample, reviews)
    output = tmp_path / "human_audit.json"
    write_human_audit_artifact(artifact, output)
    loaded = json.loads(output.read_text())
    assert loaded["sample_size"] == 40
    assert loaded["summary"] == {"correction_required": 0, "failed": 0, "passed": 40}
    assert len(loaded["reviews"]) == 40


def test_pending_human_review_queue_contains_full_context_and_no_auto_pass(tmp_path):
    sample = select_human_audit_sample(_examples(), sample_size=50)
    artifact = build_pending_human_review_queue(sample)
    assert artifact.summary == {
        "passed": 0,
        "failed": 0,
        "correction_required": 0,
        "pending_human_review": 50,
    }
    assert all(review.review_status == "PENDING_HUMAN_REVIEW" for review in artifact.reviews)
    assert all(review.review_notes == "" for review in artifact.reviews)
    assert all(
        review.review_checks
        and set(review.review_checks.values()) == {"PENDING_HUMAN_REVIEW"}
        for review in artifact.reviews
    )
    first = artifact.reviews[0]
    assert first.question
    assert first.expected_output != ""
    assert first.task_family
    assert first.difficulty
    assert first.evidence_status

    # A final queue can embed the exact evidence text needed for manual judgment.
    evidence_map = {chunk_id: f"Evidence for {chunk_id}" for example in sample for chunk_id in example.gold_chunk_ids}
    truth_map = {
        example.example_id: ({"record_id": example.required_record_ids[0], "value": "demo"},)
        for example in sample if example.required_record_ids
    }
    with_text = build_pending_human_review_queue(
        sample, chunk_text_by_id=evidence_map, structured_truth_by_example_id=truth_map
    )
    for review in with_text.reviews:
        assert len(review.gold_evidence_text) == len(review.gold_chunks)
        assert all(review.gold_evidence_text) if review.gold_chunks else review.gold_evidence_text == ()
        if review.required_records:
            assert review.structured_truth

    output = tmp_path / "human_audit.json"
    write_human_review_queue(artifact, output)
    loaded = json.loads(output.read_text())
    assert loaded["summary"]["pending_human_review"] == 50
    assert loaded["summary"]["passed"] == 0
    assert {review["review_status"] for review in loaded["reviews"]} == {"PENDING_HUMAN_REVIEW"}


def test_pending_queue_is_deterministic():
    examples = _examples()
    first = build_pending_human_review_queue(select_human_audit_sample(examples, sample_size=50))
    second = build_pending_human_review_queue(select_human_audit_sample(reversed(examples), sample_size=50))
    assert first.to_dict() == second.to_dict()

from adaptlab.benchmark.human_audit import (
    load_human_review_queue,
    pending_human_review_records,
    update_human_review_record,
)


def _write_pending_queue(tmp_path):
    sample = select_human_audit_sample(_examples(), sample_size=50)
    artifact = build_pending_human_review_queue(sample)
    path = tmp_path / "human_audit.json"
    write_human_review_queue(artifact, path)
    return path


def test_human_review_update_persists_and_preserves_other_records(tmp_path):
    path = _write_pending_queue(tmp_path)
    before = load_human_review_queue(path)
    first_id = before["reviews"][0]["example_id"]
    second_before = dict(before["reviews"][1])

    updated = update_human_review_record(path, first_id, "PASS", "Checked manually.")
    first = next(item for item in updated["reviews"] if item["example_id"] == first_id)
    assert first["review_status"] == "PASS"
    assert first["review_notes"] == "Checked manually."
    assert updated["reviews"][1] == second_before
    assert updated["summary"]["passed"] == 1
    assert updated["summary"]["pending_human_review"] == 49


def test_human_review_resume_returns_only_pending_records(tmp_path):
    path = _write_pending_queue(tmp_path)
    data = load_human_review_queue(path)
    first_id = data["reviews"][0]["example_id"]
    second_id = data["reviews"][1]["example_id"]
    update_human_review_record(path, first_id, "PASS")
    update_human_review_record(path, second_id, "CORRECTION_REQUIRED", "Needs a wording fix.")

    pending = pending_human_review_records(path)
    pending_ids = {item["example_id"] for item in pending}
    assert first_id not in pending_ids
    assert second_id not in pending_ids
    reloaded = load_human_review_queue(path)
    assert reloaded["summary"]["passed"] == 1
    assert reloaded["summary"]["correction_required"] == 1
    assert reloaded["summary"]["pending_human_review"] == 48


def test_finalize_completed_human_audit_marks_complete_and_writes_hash(tmp_path):
    import json
    from adaptlab.benchmark.human_audit import finalize_completed_human_audit

    src = Path('data/generated/v0.0/audits/human_audit.json')
    data = json.loads(src.read_text(encoding='utf-8'))
    for review in data['reviews']:
        review['review_status'] = 'PASS'
    data['summary'] = {'passed': 50, 'failed': 0, 'correction_required': 0, 'pending_human_review': 0}
    path = tmp_path / 'human_audit.json'
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    result, digest = finalize_completed_human_audit(path)
    finalized = json.loads(path.read_text(encoding='utf-8'))
    assert result['complete'] is True
    assert result['passed'] == 50
    assert finalized['complete'] is True
    assert path.with_suffix('.json.sha256').read_text(encoding='utf-8').strip() == digest


def test_validate_completed_human_audit_rejects_pending(tmp_path):
    import json
    import pytest
    from adaptlab.benchmark.human_audit import validate_completed_human_audit

    src = Path('data/generated/v0.0/audits/human_audit.json')
    data = json.loads(src.read_text(encoding='utf-8'))
    data['reviews'][0]['review_status'] = 'PENDING_HUMAN_REVIEW'
    data['summary']['pending_human_review'] = 1
    path = tmp_path / 'human_audit.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        validate_completed_human_audit(path)
