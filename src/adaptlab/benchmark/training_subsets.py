"""Deterministic nested training subsets for the full Nimbus benchmark.

This module only selects examples. It does not train or invoke any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from adaptlab.benchmark.config import BenchmarkConfig
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import BehaviorType, Split, TaskFamily

TRAINING_SUBSET_VERSION = "1"
_SUBSET_SIZES: tuple[tuple[str, int], ...] = (
    ("train_050", 50),
    ("train_100", 100),
    ("train_200", 200),
    ("train_300", 300),
)


@dataclass(frozen=True)
class TrainingSubsetBundle:
    """Versioned deterministic nested subsets of the configured training split."""

    subset_version: str
    benchmark_version: str
    generation_seed: int
    train_050: tuple[BenchmarkExample, ...]
    train_100: tuple[BenchmarkExample, ...]
    train_200: tuple[BenchmarkExample, ...]
    train_300: tuple[BenchmarkExample, ...]

    def by_name(self) -> dict[str, tuple[BenchmarkExample, ...]]:
        return {
            "train_050": self.train_050,
            "train_100": self.train_100,
            "train_200": self.train_200,
            "train_300": self.train_300,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "subset_version": self.subset_version,
            "benchmark_version": self.benchmark_version,
            "generation_seed": self.generation_seed,
            "subsets": {
                name: [example.to_dict() for example in subset]
                for name, subset in self.by_name().items()
            },
        }


def _round_robin(streams: list[list[BenchmarkExample]]) -> list[BenchmarkExample]:
    """Interleave sorted streams deterministically until all are exhausted."""

    result: list[BenchmarkExample] = []
    cursors = [0] * len(streams)
    while True:
        advanced = False
        for index, stream in enumerate(streams):
            cursor = cursors[index]
            if cursor < len(stream):
                result.append(stream[cursor])
                cursors[index] += 1
                advanced = True
        if not advanced:
            return result


def _balanced_behavior_stream(examples: Iterable[BenchmarkExample]) -> list[BenchmarkExample]:
    """Interleave behavior-only examples so every behavior type appears early."""

    buckets: list[list[BenchmarkExample]] = []
    for behavior_type in BehaviorType:
        bucket = sorted(
            (example for example in examples if example.behavior_type is behavior_type),
            key=lambda example: example.example_id,
        )
        buckets.append(bucket)
    return _round_robin(buckets)


def _selection_order(train_examples: list[BenchmarkExample]) -> list[BenchmarkExample]:
    """Create a deterministic, family-stratified order used to take nested prefixes."""

    streams: list[list[BenchmarkExample]] = []
    for family in TaskFamily:
        family_examples = [example for example in train_examples if example.task_family is family]
        if family is TaskFamily.behavior_only:
            stream = _balanced_behavior_stream(family_examples)
        else:
            stream = sorted(family_examples, key=lambda example: example.example_id)
        streams.append(stream)
    return _round_robin(streams)


def generate_training_subsets(
    examples: list[BenchmarkExample],
    config: BenchmarkConfig,
) -> TrainingSubsetBundle:
    """Generate the frozen nested 50/100/200/300 training subsets.

    Selection is deterministic and performed only from examples already assigned
    to the train split. The selection order is stratified by task family and, for
    behavior-only examples, by behavior type. Returned subsets are canonically
    sorted by ``example_id`` so serialization does not depend on selection order.
    """

    train_examples = sorted(
        (example for example in examples if example.split is Split.train),
        key=lambda example: example.example_id,
    )
    if config.splits.train != 300:
        raise ValueError("v0.0 nested training subsets require configured train size 300")
    if len(train_examples) != config.splits.train:
        raise ValueError(
            f"expected {config.splits.train} train examples, found {len(train_examples)}"
        )
    if len({example.example_id for example in train_examples}) != len(train_examples):
        raise ValueError("training examples must have unique example_id values")
    if any(example.split_type.value == "structural_holdout" for example in train_examples):
        raise ValueError("training subsets cannot contain structural-holdout examples")

    order = _selection_order(train_examples)
    if len(order) != len(train_examples) or len({e.example_id for e in order}) != len(order):
        raise ValueError("deterministic subset selection did not cover training examples exactly once")

    selected: dict[str, tuple[BenchmarkExample, ...]] = {}
    for name, size in _SUBSET_SIZES:
        subset = tuple(sorted(order[:size], key=lambda example: example.example_id))
        selected[name] = subset

    # train_300 is the full canonical configured training set, not merely an
    # equivalent-sized sample.
    if selected["train_300"] != tuple(train_examples):
        raise ValueError("train_300 must equal the full configured training set")

    return TrainingSubsetBundle(
        subset_version=TRAINING_SUBSET_VERSION,
        benchmark_version=config.benchmark_version,
        generation_seed=config.generation_seed,
        train_050=selected["train_050"],
        train_100=selected["train_100"],
        train_200=selected["train_200"],
        train_300=selected["train_300"],
    )
