from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import adaptlab.benchmark.generate_tasks as generate_tasks_module
from adaptlab.benchmark.build import build_full_benchmark
from adaptlab.cli.main import main
from adaptlab.domain.enums import Split, TaskFamily


CONFIG = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"


def test_full_build_writes_clean_candidate_artifacts_and_preliminary_manifest(tmp_path: Path) -> None:
    result = build_full_benchmark(CONFIG, tmp_path)

    assert result.passed
    assert result.manifest["frozen"] is False
    assert result.manifest["candidate_status"] == "VALID_CANDIDATE"
    assert result.manifest["blocking_issue_count"] == 0
    assert result.manifest["blockers"] == []
    assert result.manifest["counts"]["train"] == 300
    assert result.manifest["counts"]["validation"] == 150
    assert result.manifest["counts"]["test"] == 400
    assert result.manifest["counts"]["sentinel"] == 100
    assert not result.leakage_audit.semantic_fingerprint_collisions
    assert not result.leakage_audit.cross_split_collisions
    assert not result.leakage_audit.structural_violations
    assert not result.leakage_audit.metadata_answer_leakage

    expected = {
        "world.json", "documents.json", "chunks.json", "train.json",
        "validation.json", "test.json", "training_subsets.json", "sentinel.json",
        "holdout_policy.json", "holdout_report.json", "preliminary_manifest.json",
        "audits/corpus_composition.json", "audits/leakage.json",
        "audits/lexical_overlap.json", "audits/anti_confounding.json",
    }
    for relative in expected:
        assert (tmp_path / relative).is_file(), relative

    manifest = json.loads((tmp_path / "preliminary_manifest.json").read_text())
    assert manifest["frozen"] is False
    assert manifest["candidate_status"] == "VALID_CANDIDATE"
    assert manifest["blocking_issue_count"] == 0
    assert "preliminary_manifest.json" not in manifest["artifact_hashes"]


def test_full_build_rejects_intentionally_injected_semantic_leakage(
    tmp_path: Path, monkeypatch,
) -> None:
    original_generate_full_tasks = generate_tasks_module.generate_full_tasks

    def generate_with_semantic_leakage(*args, **kwargs):
        examples = list(original_generate_full_tasks(*args, **kwargs))
        train_example = next(
            item
            for item in examples
            if item.split is Split.train and item.task_family is TaskFamily.behavior_only
        )
        validation_index = next(
            index
            for index, item in enumerate(examples)
            if item.split is Split.validation
            and item.task_family is TaskFamily.behavior_only
            and item.behavior_type is train_example.behavior_type
        )
        validation_example = examples[validation_index]
        examples[validation_index] = replace(
            validation_example,
            question=train_example.question,
            expected_output=train_example.expected_output,
            scoring_rule=train_example.scoring_rule,
            scoring_parameters=train_example.scoring_parameters,
        )
        return tuple(examples)

    monkeypatch.setattr(generate_tasks_module, "generate_full_tasks", generate_with_semantic_leakage)

    result = build_full_benchmark(CONFIG, tmp_path)

    assert not result.passed
    assert result.manifest["candidate_status"] == "BLOCKED"
    assert result.manifest["blocking_issue_count"] > 0
    assert result.leakage_audit.semantic_fingerprint_collisions
    assert any("semantic task duplication" in blocker for blocker in result.blockers)


def test_full_build_cli_returns_zero_for_clean_candidate(tmp_path: Path) -> None:
    code = main([
        "benchmark", "build",
        "--config", str(CONFIG),
        "--output", str(tmp_path),
    ])
    assert code == 0
    manifest = json.loads((tmp_path / "preliminary_manifest.json").read_text())
    assert manifest["candidate_status"] == "VALID_CANDIDATE"
    assert manifest["blocking_issue_count"] == 0
