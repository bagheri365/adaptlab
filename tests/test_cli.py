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
