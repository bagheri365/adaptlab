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
