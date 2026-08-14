from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_full_tasks
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.benchmark.leakage import normalized_text_fingerprint, normalize_text, run_leakage_audit
from adaptlab.domain.enums import Split, SplitType


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"


def _full_inputs():
    config = load_benchmark_config(CONFIG_PATH)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    return config, world, policy, examples


def test_normalized_fingerprint_is_stable_and_normalizes_punctuation():
    assert normalize_text("Current  Nimbus-value?!") == "current nimbus value"
    assert normalized_text_fingerprint("A  B") == normalized_text_fingerprint("a-b")


def test_full_generated_benchmark_has_no_prohibited_cross_split_leakage():
    _, world, policy, examples = _full_inputs()
    report = run_leakage_audit(examples, world=world, holdout_policy=policy)
    assert report.passed
    assert not report.cross_split_collisions
    assert not report.semantic_fingerprint_collisions
    assert not report.structural_violations
    assert not report.metadata_answer_leakage
    assert not report.blockers


def test_exact_and_normalized_cross_split_duplicates_are_blockers():
    _, world, policy, examples = _full_inputs()
    train = next(example for example in examples if example.split is Split.train)
    test = next(example for example in examples if example.split is Split.test)
    corrupted_test = replace(
        test,
        question=f"  {train.question.upper()} !!! ",
        split=Split.test,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
        required_record_ids=(),
        required_logical_fact_ids=(),
        gold_document_ids=(),
        gold_chunk_ids=(),
    )
    corpus = [example for example in examples if example.example_id != test.example_id]
    corpus.append(corrupted_test)
    report = run_leakage_audit(corpus)
    assert not report.passed
    assert report.normalized_duplicate_questions
    assert any("train-test" in blocker for blocker in report.blockers)


def test_near_duplicate_detection_is_local_and_deterministic():
    _, _, _, examples = _full_inputs()
    left = examples[0]
    right = replace(
        examples[1],
        question=left.question + " please",
    )
    report_a = run_leakage_audit([left, right], near_duplicate_threshold=0.80)
    report_b = run_leakage_audit([right, left], near_duplicate_threshold=0.80)
    assert report_a.to_dict() == report_b.to_dict()
    assert len(report_a.suspicious_near_duplicates) == 1


def test_structural_group_leakage_is_reported_as_blocker():
    _, world, policy, examples = _full_inputs()
    structural = next(
        example for example in examples if example.split_type is SplitType.structural_holdout
    )
    corrupted = replace(
        structural,
        split=Split.train,
        split_type=SplitType.iid,
        holdout_dimension=None,
        holdout_group=None,
    )
    report = run_leakage_audit([corrupted], world=world, holdout_policy=policy)
    assert not report.passed
    assert report.structural_violations


def test_report_contains_requested_sections_and_output_repeat_diagnostics():
    _, _, _, examples = _full_inputs()
    report = run_leakage_audit(examples)
    payload = report.to_dict()
    assert set(payload) >= {
        "exact_duplicate_questions",
        "normalized_duplicate_questions",
        "suspicious_near_duplicates",
        "cross_split_collisions",
        "structural_violations",
        "suspicious_expected_output_duplicates",
    }
    assert report.suspicious_expected_output_duplicates


def test_semantic_fingerprint_ignores_wording_example_id_and_split():
    _, _, _, examples = _full_inputs()
    original = next(example for example in examples if example.split is Split.train and example.behavior_type is not None)
    rewritten = replace(
        original,
        example_id="REWORDED_COPY",
        split=Split.validation,
        question="Completely different superficial wording for the same concrete task.",
    )
    from adaptlab.benchmark.leakage import semantic_task_fingerprint

    assert semantic_task_fingerprint(original) == semantic_task_fingerprint(rewritten)


def test_semantic_fingerprint_changes_for_genuinely_different_behavior_instance():
    _, _, _, examples = _full_inputs()
    behavior = [example for example in examples if example.task_family.value == "behavior_only"]
    left = behavior[0]
    right = next(
        example
        for example in behavior[1:]
        if example.behavior_type is left.behavior_type and example.split is not left.split
    )
    from adaptlab.benchmark.leakage import semantic_task_fingerprint

    assert semantic_task_fingerprint(left) != semantic_task_fingerprint(right)


def test_cross_split_semantic_duplicate_is_blocker_even_when_wording_changes():
    _, _, _, examples = _full_inputs()
    train = next(example for example in examples if example.split is Split.train and example.behavior_type is not None)
    test = next(example for example in examples if example.split is Split.test)
    semantic_copy = replace(
        train,
        example_id=test.example_id,
        split=Split.test,
        question="A cosmetic rewrite that preserves the exact same task operands.",
    )
    corpus = [example for example in examples if example.example_id != test.example_id]
    corpus.append(semantic_copy)
    report = run_leakage_audit(corpus)

    assert not report.passed
    assert report.semantic_fingerprint_collisions
    assert any("semantic task duplication" in blocker for blocker in report.blockers)


def test_artificial_review_sequence_tokens_are_stripped_before_lexical_comparison():
    from adaptlab.benchmark.leakage import strip_artificial_sequence_text

    left = "During review sequence 11001, compare threshold 52 with candidate 47."
    right = "During review sequence 21001, compare threshold 52 with candidate 47."
    assert normalize_text(left) == normalize_text(right)
    assert "review sequence" not in strip_artificial_sequence_text(left).casefold()


def test_cross_split_high_similarity_pairs_are_prominent_warnings_not_automatic_blockers():
    _, _, _, examples = _full_inputs()
    train = next(example for example in examples if example.split is Split.train and example.behavior_type is not None)
    validation = next(example for example in examples if example.split is Split.validation and example.behavior_type is train.behavior_type)
    # Preserve distinct concrete task parameters but make prose highly similar.
    validation = replace(
        validation,
        question=train.question.replace("value", "value please") if "value" in train.question else train.question + " please",
    )
    report = run_leakage_audit([train, validation], near_duplicate_threshold=0.95)
    assert report.cross_split_near_duplicate_warnings
    assert report.highest_risk_train_validation
    pair = report.highest_risk_train_validation[0]
    assert pair.left_split != pair.right_split
    assert pair.similarity >= 0.80
    assert pair.semantic_fingerprint_match is False
    assert not any("semantic task duplication" in blocker for blocker in report.blockers)


def test_near_duplicate_report_includes_task_risk_diagnostics():
    _, _, _, examples = _full_inputs()
    train = next(example for example in examples if example.split is Split.train and example.behavior_type is not None)
    test = next(example for example in examples if example.split is Split.test and example.behavior_type is train.behavior_type)
    test = replace(test, question=train.question + " for this case")
    report = run_leakage_audit([train, test], near_duplicate_threshold=0.75)
    assert report.suspicious_near_duplicates
    payload = report.suspicious_near_duplicates[0].to_dict()
    assert set(payload) >= {
        "left_example_id",
        "right_example_id",
        "left_split",
        "right_split",
        "similarity",
        "behavior_type",
        "semantic_fingerprint_match",
        "parameter_overlap",
        "identifier_overlap",
        "template_family_match",
        "expected_output_structure_match",
        "risk_score",
    }


def test_highest_risk_train_test_pairs_are_deterministic():
    _, _, _, examples = _full_inputs()
    train = next(example for example in examples if example.split is Split.train and example.behavior_type is not None)
    test = next(example for example in examples if example.split is Split.test and example.behavior_type is train.behavior_type)
    test = replace(test, question=train.question + " for this independent case")
    pair = [train, test]
    report_a = run_leakage_audit(pair)
    report_b = run_leakage_audit(reversed(pair))
    assert [item.to_dict() for item in report_a.highest_risk_train_test] == [
        item.to_dict() for item in report_b.highest_risk_train_test
    ]


def test_current_full_benchmark_has_no_within_split_exact_or_normalized_duplicates():
    _, _, _, examples = _full_inputs()
    report = run_leakage_audit(examples)
    assert report.exact_duplicate_questions == ()
    assert report.normalized_duplicate_questions == ()
    assert report.within_split_duplicate_reviews == ()


def test_within_split_same_semantic_task_is_classified_generator_defect():
    from adaptlab.benchmark.leakage import WithinSplitDuplicateClass

    _, _, _, examples = _full_inputs()
    original = next(example for example in examples if example.split is Split.train)
    duplicate = replace(original, example_id=original.example_id + "_DUP")
    report = run_leakage_audit([original, duplicate])
    assert len(report.within_split_duplicate_reviews) == 1
    review = report.within_split_duplicate_reviews[0]
    assert review.classification is WithinSplitDuplicateClass.GENERATOR_DEFECT
    assert review.semantic_fingerprints


def test_within_split_surface_duplicate_with_hidden_only_difference_is_not_allowed():
    from adaptlab.benchmark.leakage import WithinSplitDuplicateClass, WITHIN_SPLIT_DUPLICATE_RULE

    _, _, _, examples = _full_inputs()
    original = next(
        example for example in examples
        if example.split is Split.train and example.task_family.value == "knowledge_only"
    )
    changed_params = dict(original.scoring_parameters)
    changed_params["question_intent"] = str(changed_params.get("question_intent", "current_value_lookup")) + "_alternate"
    duplicate = replace(
        original,
        example_id=original.example_id + "_META",
        scoring_parameters=changed_params,
    )
    report = run_leakage_audit([original, duplicate])
    review = report.within_split_duplicate_reviews[0]
    assert review.classification is WithinSplitDuplicateClass.REDUNDANT_DUPLICATE
    assert "disallowed by default" in WITHIN_SPLIT_DUPLICATE_RULE
