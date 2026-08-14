from adaptlab.benchmark.audit import audit_benchmark
from adaptlab.benchmark.generate_docs import generate_documents
from adaptlab.benchmark.generate_tasks import generate_tasks
from adaptlab.benchmark.generate_world import generate_world
from adaptlab.benchmark.validate import apply_structural_holdout_rules


def _fixture(seed: int = 1729):
    world = generate_world(seed)
    documents, chunks = generate_documents(world)
    examples = generate_tasks(world, documents, chunks)
    examples = apply_structural_holdout_rules(world, examples)
    return world, documents, chunks, examples


def test_audit_reports_all_requested_tables_and_empty_categories() -> None:
    world, documents, _, examples = _fixture()
    result = audit_benchmark(world, documents, examples)

    assert set(result.tables) == {
        "task_family_x_difficulty",
        "task_family_x_component_family",
        "task_family_x_split_type",
        "difficulty_x_split_type",
        "knowledge_state_x_component_family",
        "behavior_type_x_difficulty",
        "behavior_type_x_component_family",
        "evidence_status_x_difficulty",
        "document_style_x_task_family",
    }
    assert result.summary["table_count"] == 9
    assert result.summary["example_count"] == len(examples)
    assert result.summary["empty_cell_count"] > 0
    assert any(table["empty_categories"] for table in result.tables.values())


def test_audit_is_deterministic() -> None:
    world, documents, _, examples = _fixture()
    first = audit_benchmark(world, documents, examples)
    second = audit_benchmark(world, documents, examples)
    assert first == second


def test_audit_detects_intentionally_concentrated_fixture() -> None:
    world, documents, _, examples = _fixture()
    concentrated = [example for example in examples if example.example_id.startswith("BHV_")]
    result = audit_benchmark(world, documents, concentrated)

    assert result.warnings
    assert result.summary["concentration_warning_count"] > 0
    assert any("severely concentrated" in warning for warning in result.warnings)


def test_audit_does_not_treat_imbalance_as_failure() -> None:
    world, documents, _, examples = _fixture()
    result = audit_benchmark(world, documents, examples)
    assert result.summary["prototype_balance_required"] is False


def _full_fixture():
    from pathlib import Path
    from adaptlab.benchmark.config import load_benchmark_config
    from adaptlab.benchmark.generate_docs import generate_full_documents
    from adaptlab.benchmark.generate_tasks import generate_full_tasks
    from adaptlab.benchmark.generate_world import generate_full_world
    from adaptlab.benchmark.holdout import build_full_holdout_policy

    config = load_benchmark_config(Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml")
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    return world, documents, examples


def test_full_scale_audit_adds_proxy_diagnostics_and_dispositions() -> None:
    from adaptlab.benchmark.audit import ImbalanceDisposition

    world, documents, examples = _full_fixture()
    result = audit_benchmark(world, documents, examples, full_scale=True)

    assert set(result.tables).issuperset({
        "task_family_x_difficulty",
        "task_family_x_component_family",
        "task_family_x_split_type",
        "difficulty_x_split_type",
        "knowledge_state_x_component_family",
        "behavior_type_x_difficulty",
        "behavior_type_x_component_family",
        "evidence_status_x_difficulty",
        "document_style_x_task_family",
        "knowledge_state_x_fact_family",
    })
    assert result.summary["full_scale"] is True
    assert result.material_imbalances
    allowed = {item for item in ImbalanceDisposition}
    assert all(finding.disposition in allowed for finding in result.material_imbalances)
    assert all(finding.explanation for finding in result.material_imbalances)


def test_full_scale_audit_is_deterministic_under_input_reordering() -> None:
    world, documents, examples = _full_fixture()
    first = audit_benchmark(world, documents, examples, full_scale=True).to_dict()
    second = audit_benchmark(world, reversed(documents), reversed(examples), full_scale=True).to_dict()
    assert first == second


def test_full_scale_audit_flags_intended_proxy_examples() -> None:
    world, documents, examples = _full_fixture()
    result = audit_benchmark(world, documents, examples, full_scale=True)
    findings = {(f.table, f.row): f for f in result.material_imbalances}

    # Behavior-only tasks intentionally have no component fact provenance.
    finding = findings[("task_family_x_component_family", "behavior_only")]
    assert finding.disposition.value == "ACCEPTED_BY_DESIGN"
    assert finding.dominant_columns == ("NOT_APPLICABLE",)


def test_write_machine_readable_full_audit_artifact(tmp_path) -> None:
    import json
    from adaptlab.benchmark.audit import write_audit_artifact

    world, documents, examples = _full_fixture()
    result = audit_benchmark(world, documents, examples, full_scale=True)
    path = write_audit_artifact(result, tmp_path / "anti_confounding.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == result.to_dict()
    assert payload["summary"]["full_scale"] is True
    assert "material_imbalances" in payload
