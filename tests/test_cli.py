import json
from pathlib import Path

from adaptlab.cli.main import DEFAULT_FIXTURE_DIR, build_parser, main


def test_build_fixture_command(tmp_path, capsys):
    output_dir = tmp_path / "fixture"
    code = main([
        "benchmark",
        "build-fixture",
        "--seed",
        "1729",
        "--output-dir",
        str(output_dir),
    ])
    assert code == 0
    assert (output_dir / "manifest.json").exists()
    assert "Built fixture" in capsys.readouterr().out


def test_validate_command_success(tmp_path, capsys):
    output_dir = tmp_path / "fixture"
    main(["benchmark", "build-fixture", "--output-dir", str(output_dir)])
    capsys.readouterr()

    code = main(["benchmark", "validate", str(output_dir)])
    assert code == 0
    assert "Validation passed" in capsys.readouterr().out


def test_validate_command_failure_returns_nonzero(tmp_path, capsys):
    output_dir = tmp_path / "fixture"
    main(["benchmark", "build-fixture", "--output-dir", str(output_dir)])
    capsys.readouterr()

    examples_path = output_dir / "examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    examples[0]["required_record_ids"] = ["UNKNOWN_RECORD"]
    examples[0]["evidence_status"] = "PRESENT"
    examples[0]["gold_document_ids"] = ["UNKNOWN_DOCUMENT"]
    examples[0]["gold_chunk_ids"] = ["UNKNOWN_CHUNK"]
    examples_path.write_text(json.dumps(examples), encoding="utf-8")

    code = main(["benchmark", "validate", str(output_dir)])
    assert code != 0
    assert "Validation failed" in capsys.readouterr().out


def test_audit_command(tmp_path, capsys):
    output_dir = tmp_path / "fixture"
    main(["benchmark", "build-fixture", "--output-dir", str(output_dir)])
    capsys.readouterr()

    code = main(["benchmark", "audit", str(output_dir)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Audit completed" in output
    assert '"table_count": 9' in output


def test_default_fixture_location_is_documented_by_parser():
    args = build_parser().parse_args(["benchmark", "build-fixture"])
    assert args.output_dir == DEFAULT_FIXTURE_DIR == Path("data/fixtures/prototype")

from adaptlab.benchmark.config import load_benchmark_config
from adaptlab.benchmark.generate_docs import generate_full_documents
from adaptlab.benchmark.generate_tasks import generate_full_tasks
from adaptlab.benchmark.generate_world import generate_full_world
from adaptlab.benchmark.holdout import build_full_holdout_policy
from adaptlab.benchmark.human_audit import build_pending_human_review_queue, select_human_audit_sample, write_human_review_queue
from adaptlab.cli.main import review_human_audit


def test_review_human_audit_saves_each_decision_and_resumes(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "benchmark_v0.0.yaml"
    config = load_benchmark_config(config_path)
    world = generate_full_world(config)
    documents, chunks = generate_full_documents(world, config)
    policy = build_full_holdout_policy(config, world)
    examples = generate_full_tasks(world, documents, chunks, config, policy)
    queue = build_pending_human_review_queue(select_human_audit_sample(examples, sample_size=50))
    audits = tmp_path / "candidate" / "audits"
    audits.mkdir(parents=True)
    path = audits / "human_audit.json"
    write_human_review_queue(queue, path)
    (tmp_path / "candidate" / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks]), encoding="utf-8"
    )

    answers = iter(["PASS", "looks correct", "QUIT"])
    output = []
    code = review_human_audit(path, input_fn=lambda _prompt: next(answers), output_fn=output.append)
    assert code == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["passed"] == 1
    assert data["summary"]["pending_human_review"] == 49

    resumed_output = []
    code = review_human_audit(path, input_fn=lambda _prompt: "QUIT", output_fn=resumed_output.append)
    assert code == 0
    first_display = next(line for line in resumed_output if line.lstrip().startswith("[1/49]"))
    assert data["reviews"][0]["example_id"] not in first_display
