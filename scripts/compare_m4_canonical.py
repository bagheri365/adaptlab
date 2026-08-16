from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptlab.benchmark.documents import DocumentChunk
from adaptlab.benchmark.schemas import BenchmarkExample
from adaptlab.domain.enums import EvidenceStatus, TaskFamily
from adaptlab.retrieval.frozen_artifact import load_and_verify_frozen_retrieval_artifact
from adaptlab.retrieval.schemas import RetrievalResult
from adaptlab.retrieval.version_metrics import with_version_diagnostics


READINESS = ROOT / "artifacts/evaluation/m4/comparison_readiness_v1.json"
BENCHMARK = ROOT / "data/generated/v0.0/test.json"
CHUNKS = ROOT / "data/generated/v0.0/chunks.json"
PROMPT_RESULTS = ROOT / "artifacts/evaluation/m3/prompt/results.json"
ORACLE_RESULTS = ROOT / "artifacts/evaluation/m3/oracle_context/results.json"
RAG_RESULTS = ROOT / "artifacts/evaluation/m4/rag/results.json"
RETRIEVAL_RESULTS = ROOT / "artifacts/retrieval/m4/primary_test_bm25_v1/results.json"
RETRIEVAL_ARTIFACT = ROOT / "artifacts/retrieval/m4/primary_test_bm25_v1/frozen/canonical_retrieval_artifact_v1.json"
OUT_JSON = ROOT / "artifacts/evaluation/m4/prompt_rag_oracle_comparison_v1.json"
OUT_TXT = ROOT / "artifacts/evaluation/m4/prompt_rag_oracle_comparison_v1.txt"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def score(row: dict[str, Any]) -> int:
    return 1 if row.get("score") == 1 or row.get("score") == 1.0 else 0


def mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def pair_counts(a: list[int], b: list[int]) -> dict[str, int]:
    if len(a) != len(b):
        raise ValueError("paired inputs must have equal length")
    out = {"A_wrong_B_correct": 0, "A_correct_B_wrong": 0, "both_correct": 0, "both_wrong": 0}
    for av, bv in zip(a, b):
        if av == 0 and bv == 1:
            out["A_wrong_B_correct"] += 1
        elif av == 1 and bv == 0:
            out["A_correct_B_wrong"] += 1
        elif av == 1 and bv == 1:
            out["both_correct"] += 1
        else:
            out["both_wrong"] += 1
    return out


def main() -> int:
    readiness = load_json(READINESS)
    if readiness.get("status") != "M4_CANONICAL_COMPARISON_READY":
        raise SystemExit(f"readiness gate is not ready: {readiness.get('status')}")

    load_and_verify_frozen_retrieval_artifact(RETRIEVAL_ARTIFACT)

    examples = [BenchmarkExample.from_dict(item) for item in load_json(BENCHMARK)]
    examples.sort(key=lambda e: e.example_id)

    prompt = {row["example_id"]: row for row in load_json(PROMPT_RESULTS)}
    oracle = {row["example_id"]: row for row in load_json(ORACLE_RESULTS)}
    rag = {row["example_id"]: row for row in load_json(RAG_RESULTS)}
    retrieval_rows = {row["example_id"]: RetrievalResult.from_dict(row) for row in load_json(RETRIEVAL_RESULTS)}
    chunks = [DocumentChunk.from_dict(item) for item in load_json(CHUNKS)]
    retrieval_rows = {eid: with_version_diagnostics(row, chunks) for eid, row in retrieval_rows.items()}

    # Basic completeness checks.
    expected_ids = {example.example_id for example in examples}
    if expected_ids != set(prompt) or expected_ids != set(oracle) or expected_ids != set(rag):
        raise ValueError("one or more condition bundles do not cover the full 400-example test split")

    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append({
            "example": example,
            "prompt": prompt[example.example_id],
            "oracle": oracle[example.example_id],
            "rag": rag[example.example_id],
            "retrieval": retrieval_rows[example.example_id],
        })

    def slice_members(example: BenchmarkExample) -> list[tuple[str, str]]:
        return [
            ("overall", "overall"),
            ("task_family", example.task_family.value),
            ("difficulty", example.difficulty.value),
            ("split_type", example.split_type.value.upper()),
            ("knowledge_state", example.knowledge_state.value),
            ("evidence_status", example.evidence_status.value),
        ]

    slice_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in slice_members(row["example"]):
            slice_groups[key].append(row)

    slice_order = [
        ("overall", "overall"),
        ("task_family", TaskFamily.behavior_only.value),
        ("task_family", TaskFamily.knowledge_only.value),
        ("task_family", TaskFamily.behavior_knowledge.value),
        ("task_family", TaskFamily.changed_knowledge.value),
        ("difficulty", "EASY"),
        ("difficulty", "MEDIUM"),
        ("difficulty", "HARD"),
        ("split_type", "IID"),
        ("split_type", "STRUCTURAL_HOLDOUT"),
        ("knowledge_state", "UNCHANGED"),
        ("knowledge_state", "UPDATED"),
        ("knowledge_state", "REMOVED"),
        ("evidence_status", "PRESENT"),
        ("evidence_status", "ABSENT"),
        ("evidence_status", "NOT_APPLICABLE"),
    ]

    slices: list[dict[str, Any]] = []
    for dimension, value in slice_order:
        group = slice_groups[(dimension, value)]
        p_scores = [score(row["prompt"]) for row in group]
        r_scores = [score(row["rag"]) for row in group]
        o_scores = [score(row["oracle"]) for row in group]
        n = len(group)
        p_correct = sum(p_scores)
        r_correct = sum(r_scores)
        o_correct = sum(o_scores)
        slices.append({
            "dimension": dimension,
            "value": value,
            "n": n,
            "prompt_correct": f"{p_correct}/{n}",
            "prompt_accuracy": mean(p_scores),
            "rag_correct": f"{r_correct}/{n}",
            "rag_accuracy": mean(r_scores),
            "oracle_context_correct": f"{o_correct}/{n}",
            "oracle_context_accuracy": mean(o_scores),
            "rag_minus_prompt": None if mean(r_scores) is None or mean(p_scores) is None else mean(r_scores) - mean(p_scores),
            "oracle_minus_rag": None if mean(o_scores) is None or mean(r_scores) is None else mean(o_scores) - mean(r_scores),
            "oracle_minus_prompt": None if mean(o_scores) is None or mean(p_scores) is None else mean(o_scores) - mean(p_scores),
        })

    pair_transitions: dict[str, dict[str, dict[str, int]]] = {}
    pair_specs = [
        ("PROMPT_vs_RAG", "prompt", "rag"),
        ("RAG_vs_ORACLE_CONTEXT", "rag", "oracle"),
    ]
    for pair_name, a_key, b_key in pair_specs:
        pair_transitions[pair_name] = {}
        for dimension, value in slice_order:
            group = slice_groups[(dimension, value)]
            a_scores = [score(row[a_key]) for row in group]
            b_scores = [score(row[b_key]) for row in group]
            pair_transitions[pair_name][f"{dimension}:{value}"] = pair_counts(a_scores, b_scores)

    behavior_only_disagreements: list[dict[str, Any]] = []
    behavior_only_count = 0
    for row in rows:
        if row["example"].task_family is not TaskFamily.behavior_only:
            continue
        behavior_only_count += 1
        hashes = {row["prompt"]["input_hash"], row["oracle"]["input_hash"], row["rag"]["input_hash"]}
        if len(hashes) != 1:
            raise ValueError(f"behavior_only input hash mismatch for {row['example'].example_id}")
        prompt_output = json.dumps(row["prompt"].get("normalized_output"), sort_keys=True, ensure_ascii=False)
        oracle_output = json.dumps(row["oracle"].get("normalized_output"), sort_keys=True, ensure_ascii=False)
        rag_output = json.dumps(row["rag"].get("normalized_output"), sort_keys=True, ensure_ascii=False)
        if len({prompt_output, oracle_output, rag_output}) != 1:
            behavior_only_disagreements.append({
                "example_id": row["example"].example_id,
                "prompt": row["prompt"].get("normalized_output"),
                "oracle_context": row["oracle"].get("normalized_output"),
                "rag": row["rag"].get("normalized_output"),
            })

    retrieval_categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        example = row["example"]
        retrieval = row["retrieval"]
        if example.evidence_status is EvidenceStatus.ABSENT:
            retrieval_categories["evidence ABSENT"].append(row)
            continue
        if example.evidence_status is not EvidenceStatus.PRESENT:
            continue
        retrieved = set(retrieval.candidate_chunk_ids)
        gold = set(example.gold_chunk_ids)
        if retrieved == gold:
            retrieval_categories["exact Oracle gold set retrieved"].append(row)
        elif gold.issubset(retrieved) and len(retrieved) > len(gold):
            retrieval_categories["all required gold retrieved plus extra chunks"].append(row)
        elif retrieved.isdisjoint(gold):
            retrieval_categories["no gold retrieved"].append(row)
        else:
            retrieval_categories["partial gold retrieved"].append(row)

    retrieval_breakdown = []
    for category in [
        "exact Oracle gold set retrieved",
        "all required gold retrieved plus extra chunks",
        "partial gold retrieved",
        "no gold retrieved",
        "evidence ABSENT",
    ]:
        group = retrieval_categories.get(category, [])
        p_scores = [score(row["prompt"]) for row in group]
        r_scores = [score(row["rag"]) for row in group]
        o_scores = [score(row["oracle"]) for row in group]
        retrieval_breakdown.append({
            "category": category,
            "n": len(group),
            "prompt_accuracy": mean(p_scores),
            "rag_accuracy": mean(r_scores),
            "oracle_context_accuracy": mean(o_scores),
            "rag_minus_oracle": None if mean(r_scores) is None or mean(o_scores) is None else mean(r_scores) - mean(o_scores),
        })

    explicit_cases = {
        "all_required_gold_retrieved_plus_RAG_wrong": [],
        "no_gold_retrieved_plus_RAG_correct": [],
        "wrong_version_retrieval_plus_RAG_correct": [],
        "obsolete_only_retrieval_plus_RAG_correct": [],
    }
    for row in rows:
        example = row["example"]
        retrieval = row["retrieval"]
        if example.evidence_status is not EvidenceStatus.PRESENT:
            continue
        retrieved = set(retrieval.candidate_chunk_ids)
        gold = set(example.gold_chunk_ids)
        if gold.issubset(retrieved) and score(row["rag"]) == 0:
            explicit_cases["all_required_gold_retrieved_plus_RAG_wrong"].append(example.example_id)
        if retrieved.isdisjoint(gold) and score(row["rag"]) == 1:
            explicit_cases["no_gold_retrieved_plus_RAG_correct"].append(example.example_id)
        if retrieval.wrong_version_top1 is True and score(row["rag"]) == 1:
            explicit_cases["wrong_version_retrieval_plus_RAG_correct"].append(example.example_id)
        if retrieval.obsolete_only_retrieved is True and score(row["rag"]) == 1:
            explicit_cases["obsolete_only_retrieval_plus_RAG_correct"].append(example.example_id)

    def gap_recovery(slice_name: str) -> float | None:
        item = next(item for item in slices if item["dimension"] + ":" + item["value"] == slice_name)
        p = item["prompt_accuracy"]
        r = item["rag_accuracy"]
        o = item["oracle_context_accuracy"]
        if p is None or r is None or o is None or o <= p:
            return None
        return (r - p) / (o - p)

    report = {
        "schema_version": "m4-canonical-comparison-v1",
        "status": readiness["status"],
        "readiness_gate": READINESS.as_posix(),
        "conditions": {
            "PROMPT": {
                "artifact_path": PROMPT_RESULTS.parent.as_posix(),
                "run_id": readiness["conditions"]["PROMPT"]["run_id"],
                "results_hash": readiness["conditions"]["PROMPT"]["results_hash"],
            },
            "ORACLE_CONTEXT": {
                "artifact_path": ORACLE_RESULTS.parent.as_posix(),
                "run_id": readiness["conditions"]["ORACLE_CONTEXT"]["run_id"],
                "results_hash": readiness["conditions"]["ORACLE_CONTEXT"]["results_hash"],
            },
            "RAG": {
                "artifact_path": RAG_RESULTS.parent.as_posix(),
                "run_id": readiness["conditions"]["RAG"]["run_id"],
                "results_hash": readiness["conditions"]["RAG"]["results_hash"],
            },
        },
        "slices": slices,
        "pairwise_transitions": pair_transitions,
        "behavior_only_control": {
            "inputs_byte_identical": True,
            "behavior_only_count": behavior_only_count,
            "output_disagreement_count": len(behavior_only_disagreements),
            "output_disagreements": behavior_only_disagreements,
        },
        "retrieval_conditioned_rag_accuracy": retrieval_breakdown,
        "explicit_retrieval_cases": explicit_cases,
        "gold_evidence_gap_recovery": {
            "overall": gap_recovery("overall:overall"),
            "knowledge_only": gap_recovery("task_family:knowledge_only"),
            "behavior_knowledge": gap_recovery("task_family:behavior_knowledge"),
            "changed_knowledge": gap_recovery("task_family:changed_knowledge"),
        },
        "sections": {
            "OBSERVATION": "The comparison uses only the persisted canonical Prompt, Oracle, and RAG bundles and the frozen retrieval artifact.",
            "INTERPRETATION": "Differences between PROMPT and RAG can be described as knowledge-access gain where Oracle improves further, while the remaining Oracle-minus-RAG gap is best treated as residual retrieval and generation/reasoning gap rather than a universal model claim.",
            "LIMITATION": "This report does not rerun inference, change retrieval, alter model settings, or establish dense retrieval performance, universal RAG performance, LoRA benefit, or cross-model generality.",
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    text_lines = [
        "OBSERVATION",
        "Canonical Prompt, Oracle, and RAG bundles passed the readiness gate and were compared without rerunning inference.",
        "",
        "INTERPRETATION",
        "RAG shows knowledge-access gain on several evidence-present slices, but the Oracle-minus-RAG gap remains a residual retrieval and generation/reasoning gap rather than proof of a universal RAG benefit.",
        "",
        "LIMITATION",
        "This comparison is bounded to the frozen canonical bundles and does not claim dense retrieval performance, universal RAG performance, LoRA benefit, or cross-model generality.",
        "",
        f"Behavior-only inputs byte-identical: {True}; output disagreements: {len(behavior_only_disagreements)}",
        "",
        "Overall:",
    ]
    overall = next(item for item in slices if item["dimension"] == "overall")
    text_lines.append(
        f"- PROMPT {overall['prompt_correct']} ({overall['prompt_accuracy']:.3f})"
    )
    text_lines.append(
        f"- RAG {overall['rag_correct']} ({overall['rag_accuracy']:.3f})"
    )
    text_lines.append(
        f"- ORACLE_CONTEXT {overall['oracle_context_correct']} ({overall['oracle_context_accuracy']:.3f})"
    )
    text_lines.append(
        f"- gold-evidence gap recovery: {report['gold_evidence_gap_recovery']['overall']}"
    )
    text_lines.append("")
    text_lines.append("Slices:")
    for item in slices:
        text_lines.append(
            f"- {item['dimension']}={item['value']}: "
            f"P {item['prompt_correct']} ({item['prompt_accuracy']:.3f}), "
            f"R {item['rag_correct']} ({item['rag_accuracy']:.3f}), "
            f"O {item['oracle_context_correct']} ({item['oracle_context_accuracy']:.3f})"
        )

    OUT_TXT.write_text("\n".join(text_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
