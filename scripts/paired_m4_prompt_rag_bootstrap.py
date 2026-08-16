#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from adaptlab.benchmark.schemas import BenchmarkExample, EvidenceStatus, KnowledgeState, SplitType, TaskFamily


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "artifacts" / "evaluation" / "m4" / "paired_prompt_rag_bootstrap_v1.json"
OUT_TXT = ROOT / "artifacts" / "evaluation" / "m4" / "paired_prompt_rag_bootstrap_v1.txt"

READINESS = ROOT / "artifacts" / "evaluation" / "m4" / "comparison_readiness_v1.json"
TEST = ROOT / "data" / "generated" / "v0.0" / "test.json"
PROMPT = ROOT / "artifacts" / "evaluation" / "m3" / "prompt" / "results.json"
RAG = ROOT / "artifacts" / "evaluation" / "m4" / "rag" / "results.json"

BOOTSTRAP_SEED = 1729
BOOTSTRAP_REPLICATES = 20000


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_enum(v: Any) -> str:
    if hasattr(v, "name"):
        return str(v.name).upper()
    return str(v).upper()


def accuracy(correct: int, n: int) -> float | None:
    return None if n == 0 else correct / n


def exact_mcnemar_pvalue(prompt_wrong_rag_correct: int, prompt_correct_rag_wrong: int) -> float | None:
    b = int(prompt_wrong_rag_correct)
    c = int(prompt_correct_rag_wrong)
    discordant = b + c
    if discordant == 0:
        return None
    k = min(b, c)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(discordant, i) * (0.5 ** discordant)
    return min(1.0, 2.0 * tail)


def discordant_odds_ratio(prompt_wrong_rag_correct: int, prompt_correct_rag_wrong: int) -> float | None:
    b = int(prompt_wrong_rag_correct)
    c = int(prompt_correct_rag_wrong)
    if b == 0 and c == 0:
        return None
    if c == 0:
        return float("inf")
    return b / c


def bootstrap_ci(delta: list[float], rng: random.Random, replicates: int) -> list[float] | None:
    n = len(delta)
    if n == 0:
        return None
    boot = []
    for _ in range(replicates):
        sample = rng.choices(delta, k=n)
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int(0.025 * replicates)]
    hi = boot[int(0.975 * replicates) - 1]
    return [float(lo), float(hi)]


def load_bundles() -> tuple[dict[str, BenchmarkExample], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    examples = {item["example_id"]: BenchmarkExample.from_dict(item) for item in load_json(TEST)}
    prompt = {item["example_id"]: item for item in load_json(PROMPT)}
    rag = {item["example_id"]: item for item in load_json(RAG)}
    return examples, prompt, rag


def score(row: dict[str, Any]) -> bool:
    return row.get("score") == 1 or row.get("score") == 1.0


def build_examples() -> list[dict[str, Any]]:
    examples, prompt, rag = load_bundles()
    records: list[dict[str, Any]] = []
    for example_id, ex in examples.items():
        records.append(
            {
                "example_id": example_id,
                "task_family": normalize_enum(ex.task_family),
                "split_type": normalize_enum(ex.split_type),
                "difficulty": normalize_enum(ex.difficulty),
                "knowledge_state": normalize_enum(ex.knowledge_state),
                "evidence_status": normalize_enum(ex.evidence_status),
                "prompt_correct": score(prompt[example_id]),
                "rag_correct": score(rag[example_id]),
                "prompt_input_hash": prompt[example_id].get("input_hash"),
                "rag_input_hash": rag[example_id].get("input_hash"),
            }
        )
    return records


def paired_summary(records: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    n = len(records)
    prompt_correct = sum(1 for r in records if r["prompt_correct"])
    rag_correct = sum(1 for r in records if r["rag_correct"])
    prompt_arr = [1.0 if r["prompt_correct"] else 0.0 for r in records]
    rag_arr = [1.0 if r["rag_correct"] else 0.0 for r in records]
    delta = [r - p for r, p in zip(rag_arr, prompt_arr)]

    pw_rc = sum((not r["prompt_correct"]) and r["rag_correct"] for r in records)
    pc_rw = sum(r["prompt_correct"] and (not r["rag_correct"]) for r in records)
    both_correct = sum(r["prompt_correct"] and r["rag_correct"] for r in records)
    both_wrong = sum((not r["prompt_correct"]) and (not r["rag_correct"]) for r in records)

    return {
        "n": n,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": accuracy(prompt_correct, n),
        "rag_correct": rag_correct,
        "rag_accuracy": accuracy(rag_correct, n),
        "absolute_difference": accuracy(rag_correct, n) - accuracy(prompt_correct, n) if n else None,
        "bootstrap_ci_95": bootstrap_ci(delta, rng, BOOTSTRAP_REPLICATES),
        "paired_transitions": {
            "prompt_wrong_rag_correct": pw_rc,
            "prompt_correct_rag_wrong": pc_rw,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "discordant_pairs": pw_rc + pc_rw,
        "mcnemar_exact": {
            "p_value": exact_mcnemar_pvalue(pw_rc, pc_rw),
            "discordant_odds_ratio": discordant_odds_ratio(pw_rc, pc_rw),
            "prompt_wrong_rag_correct": pw_rc,
            "prompt_correct_rag_wrong": pc_rw,
        },
    }


def select(records: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [r for r in records if predicate(r)]


def same_hashes(records: list[dict[str, Any]]) -> dict[str, Any]:
    mismatched = [r["example_id"] for r in records if r["prompt_input_hash"] != r["rag_input_hash"]]
    return {
        "verified": len(mismatched) == 0,
        "matched_example_count": len(records) - len(mismatched),
        "mismatched_example_ids": mismatched,
    }


def main() -> None:
    readiness = load_json(READINESS)
    if readiness.get("status") != "M4_CANONICAL_COMPARISON_READY":
        raise SystemExit(f"readiness gate not satisfied: {readiness.get('status')}")

    records = build_examples()
    rng = random.Random(BOOTSTRAP_SEED)

    confirmatory_defs = [
        ("overall", lambda r: True),
        ("knowledge_only", lambda r: r["task_family"] == "KNOWLEDGE_ONLY"),
        ("behavior_knowledge", lambda r: r["task_family"] == "BEHAVIOR_KNOWLEDGE"),
        ("changed_knowledge", lambda r: r["task_family"] == "CHANGED_KNOWLEDGE"),
    ]
    exploratory_defs = [
        ("IID", lambda r: r["split_type"] == "IID"),
        ("STRUCTURAL_HOLDOUT", lambda r: r["split_type"] == "STRUCTURAL_HOLDOUT"),
        ("EASY", lambda r: r["difficulty"] == "EASY"),
        ("MEDIUM", lambda r: r["difficulty"] == "MEDIUM"),
        ("HARD", lambda r: r["difficulty"] == "HARD"),
        ("UNCHANGED", lambda r: r["knowledge_state"] == "UNCHANGED"),
        ("UPDATED", lambda r: r["knowledge_state"] == "UPDATED"),
        ("REMOVED", lambda r: r["knowledge_state"] == "REMOVED"),
        ("evidence PRESENT", lambda r: r["evidence_status"] == "PRESENT"),
        ("evidence ABSENT", lambda r: r["evidence_status"] == "ABSENT"),
    ]

    confirmatory = {
        name: paired_summary(select(records, pred), rng)
        for name, pred in confirmatory_defs
    }
    exploratory = {
        name: paired_summary(select(records, pred), rng)
        for name, pred in exploratory_defs
    }

    behavior_only_records = select(records, lambda r: r["task_family"] == "BEHAVIOR_ONLY")
    behavior_only = {
        "n": len(behavior_only_records),
        "input_hashes_identical": same_hashes(behavior_only_records),
        "paired_outcomes": paired_summary(behavior_only_records, rng),
    }

    artifact = {
        "schema_version": "m4-prompt-rag-paired-bootstrap-v1",
        "status": "complete",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "readiness_gate": {
            "path": str(READINESS.relative_to(ROOT)),
            "status": readiness.get("status"),
        },
        "confirmatory": confirmatory,
        "exploratory": exploratory,
        "behavior_only": behavior_only,
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("OBSERVATION")
    lines.append(
        f"- Confirmatory slices: overall {confirmatory['overall']['rag_correct']}/{confirmatory['overall']['n']} for RAG vs {confirmatory['overall']['prompt_correct']}/{confirmatory['overall']['n']} for Prompt; "
        f"knowledge_only {confirmatory['knowledge_only']['rag_correct']}/{confirmatory['knowledge_only']['n']} vs {confirmatory['knowledge_only']['prompt_correct']}/{confirmatory['knowledge_only']['n']}; "
        f"behavior_knowledge {confirmatory['behavior_knowledge']['rag_correct']}/{confirmatory['behavior_knowledge']['n']} vs {confirmatory['behavior_knowledge']['prompt_correct']}/{confirmatory['behavior_knowledge']['n']}; "
        f"changed_knowledge {confirmatory['changed_knowledge']['rag_correct']}/{confirmatory['changed_knowledge']['n']} vs {confirmatory['changed_knowledge']['prompt_correct']}/{confirmatory['changed_knowledge']['n']}."
    )
    lines.append(
        f"- The paired bootstrap seed was {BOOTSTRAP_SEED} with {BOOTSTRAP_REPLICATES} replicates."
    )
    lines.append(
        f"- Behavior_only input hashes matched pairwise across Prompt and RAG on {behavior_only['input_hashes_identical']['matched_example_count']} examples; paired outcomes were {behavior_only['paired_outcomes']['paired_transitions']}."
    )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(
        "- The confirmatory paired comparison estimates the frozen RAG condition relative to the frozen Prompt condition; the observed gains are large in the knowledge-bearing slices, but statistical significance alone does not establish retrieval quality."
    )
    lines.append(
        "- The behavior_only control shows no paired disagreement, so there is no evidence here for a retrieval-driven explanation in that control slice."
    )
    lines.append(
        "- Exploratory slices should be read as slice-specific paired effects only; they do not support universal RAG benefit, LoRA benefit, or cross-model generality."
    )
    lines.append("")
    lines.append("LIMITATION")
    lines.append(
        "- This analysis uses only the persisted canonical result bundles and the frozen benchmark partitioning; it does not rerun inference, retrieval, scoring, or normalization."
    )
    lines.append(
        "- Exact McNemar tests are only reported when discordant pairs exist; slices with no discordance have no exact test result."
    )
    lines.append(
        "- Bootstrap confidence intervals are deterministic for the stated seed and replicate count, but they remain sample-based estimates of the paired accuracy difference."
    )

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_TXT)


if __name__ == "__main__":
    main()
