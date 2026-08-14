from dataclasses import replace

from adaptlab.benchmark.sentinel import (
    DEFAULT_SENTINEL_COUNT,
    GeneralizationCapability,
    generate_generalization_sentinel,
    validate_generalization_sentinel,
)


def test_canonical_sentinel_has_exact_count_and_capability_coverage():
    examples = generate_generalization_sentinel(seed=1729)
    assert len(examples) == DEFAULT_SENTINEL_COUNT == 100

    counts = {capability: 0 for capability in GeneralizationCapability}
    for example in examples:
        counts[example.capability] += 1

    assert set(counts.values()) == {20}
    assert [example.example_id for example in examples] == sorted(
        example.example_id for example in examples
    )


def test_sentinel_is_deterministic_and_seed_recorded():
    first = generate_generalization_sentinel(seed=1729)
    second = generate_generalization_sentinel(seed=1729)
    assert [example.to_dict() for example in first] == [example.to_dict() for example in second]
    assert all(example.generation_seed == 1729 for example in first)


def test_sentinel_validation_proves_nimbus_free_and_mechanically_scorable():
    examples = generate_generalization_sentinel(seed=1729)
    result = validate_generalization_sentinel(examples, expected_seed=1729)
    assert result.passed, result.errors
    assert result.statistics["count"] == 100


def test_sentinel_validator_rejects_domain_leakage_and_bad_gold_answer():
    examples = generate_generalization_sentinel(seed=1729)

    leaked = replace(examples[0], question=examples[0].question + " Nimbus authentication")
    bad_answer = replace(examples[1], expected_output="WRONG")

    leaked_result = validate_generalization_sentinel(
        [leaked, *examples[1:]], expected_seed=1729
    )
    bad_result = validate_generalization_sentinel(
        [examples[0], bad_answer, *examples[2:]], expected_seed=1729
    )

    assert not leaked_result.passed
    assert any("prohibited Nimbus/domain" in error for error in leaked_result.errors)
    assert not bad_result.passed
    assert any("deterministic scoring contract" in error for error in bad_result.errors)


def test_sentinel_examples_round_trip():
    examples = generate_generalization_sentinel(seed=1729, count=5)
    rebuilt = [type(example).from_dict(example.to_dict()) for example in examples]
    assert rebuilt == examples
