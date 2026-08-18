from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from adaptlab.domain.enums import Split
from adaptlab.evaluation.inputs import (
    construct_model_input,
    construct_rag_model_input,
)
from adaptlab.evaluation.prompts import load_prompt_contract
from adaptlab.evaluation.runner import (
    load_benchmark_split,
    load_chunks,
)
from adaptlab.evaluation.schemas import AdaptationMethod
from adaptlab.evaluation.scoring import score_output
from adaptlab.retrieval.frozen_artifact import load_and_verify_frozen_retrieval_artifact


CONDITIONS = (
    "M5_PROMPT",
    "M5_RAG",
    "LORA",
    "LORA_RAG",
)


def _sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _model_input_messages(constructed) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": constructed.model_input.system,
        },
        {
            "role": "user",
            "content": constructed.model_input.user,
        },
    ]


def _generate(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[str, str | None]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=None,
        add_generation_prompt=True,
        return_dict=False,
    )

    parts: list[str] = []
    finish_reason = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=256,
        sampler=make_sampler(
            0.0,
            top_p=1.0,
            top_k=0,
        ),
    ):
        parts.append(response.text)
        finish_reason = response.finish_reason

    return "".join(parts), finish_reason


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(r["score"] == 1.0) for r in rows)

    family_names = sorted({r["task_family"] for r in rows})
    by_family: dict[str, dict[str, Any]] = {}

    for family in family_names:
        subset = [r for r in rows if r["task_family"] == family]
        n = len(subset)
        n_correct = sum(int(r["score"] == 1.0) for r in subset)

        by_family[family] = {
            "n": n,
            "correct": n_correct,
            "accuracy": n_correct / n if n else 0.0,
        }

    macro = (
        sum(v["accuracy"] for v in by_family.values()) / len(by_family)
        if by_family
        else 0.0
    )

    runtime_failures = sum(
        int(r["runtime_failure"] is not None)
        for r in rows
    )

    finish_reasons: dict[str, int] = {}
    for row in rows:
        key = row["finish_reason"] or "NONE"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1

    return {
        "n_total": total,
        "correct": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_family_accuracy": macro,
        "by_family": by_family,
        "runtime_failures": runtime_failures,
        "finish_reasons": finish_reasons,
    }


def run_primary_2x2(
    *,
    benchmark_dir: Path,
    prompt_config: Path,
    mlx_base_path: Path,
    selected_adapter_dir: Path,
    selection_decision_path: Path,
    retrieval_artifact_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary only. Do not load sentinel.
    examples = sorted(
        load_benchmark_split(
            benchmark_dir,
            Split.test,
        ),
        key=lambda x: x.example_id,
    )

    if len(examples) != 400:
        raise ValueError(
            f"Expected exactly 400 primary-test examples; got {len(examples)}"
        )

    example_ids = [e.example_id for e in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate primary-test example IDs")

    chunks = load_chunks(benchmark_dir)
    prompt_contract = load_prompt_contract(prompt_config)

    retrieval_artifact = load_and_verify_frozen_retrieval_artifact(
        retrieval_artifact_path
    )

    retrieval_ids = [entry.example_id for entry in retrieval_artifact.entries]

    if len(retrieval_ids) != 400:
        raise ValueError(
            f"Expected 400 retrieval entries; got {len(retrieval_ids)}"
        )

    if set(retrieval_ids) != set(example_ids):
        missing = sorted(set(example_ids) - set(retrieval_ids))
        extra = sorted(set(retrieval_ids) - set(example_ids))
        raise ValueError(
            "Frozen retrieval artifact does not exactly match primary test. "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    selection = json.loads(
        Path(selection_decision_path).read_text()
    )

    selected_candidate_id = selection["selected_candidate_id"]
    selected_checkpoint_iteration = selection[
        "selected_checkpoint_iteration"
    ]

    if selected_candidate_id != "S1_POLICY_B_ATTN_r4_lr1e-05_iters500":
        raise ValueError(
            "Unexpected frozen selected candidate: "
            f"{selected_candidate_id}"
        )

    if selected_checkpoint_iteration != 500:
        raise ValueError(
            "Unexpected frozen selected checkpoint iteration: "
            f"{selected_checkpoint_iteration}"
        )

    expected_checkpoint = (
        Path(selected_adapter_dir)
        / f"{selected_checkpoint_iteration:07d}_adapters.safetensors"
    )
    if not expected_checkpoint.exists():
        raise FileNotFoundError(expected_checkpoint)

    print("Loading base model...")
    base_model, base_tok, _ = load(
        str(mlx_base_path),
        tokenizer_config={"trust_remote_code": True},
        return_config=True,
    )
    base_tok.add_eos_token("<|im_end|>")

    print("Loading selected LoRA model...")
    lora_model, lora_tok, _ = load(
        str(mlx_base_path),
        tokenizer_config={"trust_remote_code": True},
        adapter_path=str(selected_adapter_dir),
        return_config=True,
    )
    lora_tok.add_eos_token("<|im_end|>")

    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: []
        for condition in CONDITIONS
    }

    for idx, example in enumerate(examples, start=1):
        prompt_input = construct_model_input(
            example=example,
            method=AdaptationMethod.PROMPT,
            prompt_contract=prompt_contract,
        )

        rag_input = construct_rag_model_input(
            example=example,
            prompt_contract=prompt_contract,
            chunks=chunks,
            retrieval_artifact=retrieval_artifact,
        )

        condition_inputs = {
            "M5_PROMPT": (
                base_model,
                base_tok,
                prompt_input,
            ),
            "M5_RAG": (
                base_model,
                base_tok,
                rag_input,
            ),
            "LORA": (
                lora_model,
                lora_tok,
                prompt_input,
            ),
            "LORA_RAG": (
                lora_model,
                lora_tok,
                rag_input,
            ),
        }

        for condition in CONDITIONS:
            model, tok, constructed = condition_inputs[condition]

            runtime_failure = None
            output = ""
            finish_reason = None

            try:
                output, finish_reason = _generate(
                    model=model,
                    tokenizer=tok,
                    messages=_model_input_messages(constructed),
                )
                scored = score_output(example, output)
                score = float(scored.score)
                normalized = scored.normalized_output
            except Exception as exc:
                runtime_failure = (
                    f"{type(exc).__name__}: {exc}"
                )
                score = 0.0
                normalized = None

            row = {
                "condition": condition,
                "example_id": example.example_id,
                "task_family": example.task_family.value,
                "expected_output": example.expected_output,
                "output": output,
                "normalized_output": normalized,
                "score": score,
                "finish_reason": finish_reason,
                "runtime_failure": runtime_failure,
            }

            if condition in {"M5_RAG", "LORA_RAG"}:
                row.update(
                    {
                        "evidence_chunk_ids": list(
                            rag_input.evidence_chunk_ids
                        ),
                        "evidence_chunk_hashes": list(
                            rag_input.evidence_chunk_hashes
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "evidence_chunk_ids": [],
                        "evidence_chunk_hashes": [],
                    }
                )

            rows_by_condition[condition].append(row)

        if idx % 10 == 0:
            status = []
            for condition in CONDITIONS:
                correct = sum(
                    int(r["score"] == 1.0)
                    for r in rows_by_condition[condition]
                )
                status.append(
                    f"{condition}={correct}/{idx}"
                )

            print(
                f"primary {idx}/400 "
                + " ".join(status)
            )

    condition_results: dict[str, Any] = {}

    for condition in CONDITIONS:
        rows = rows_by_condition[condition]
        aggregate = _aggregate(rows)

        payload = {
            "schema_version": "m5-primary-condition-result-v1",
            "condition": condition,
            "base_model_path": str(mlx_base_path),
            "selected_candidate_id": (
                selected_candidate_id
                if condition in {"LORA", "LORA_RAG"}
                else None
            ),
            "selected_checkpoint_iteration": (
                selected_checkpoint_iteration
                if condition in {"LORA", "LORA_RAG"}
                else None
            ),
            "retrieval_run_id": (
                retrieval_artifact.retrieval_run_id
                if condition in {"M5_RAG", "LORA_RAG"}
                else None
            ),
            "retrieval_artifact_hash": (
                retrieval_artifact.retrieval_artifact_hash
                if condition in {"M5_RAG", "LORA_RAG"}
                else None
            ),
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_tokens": 256,
                "add_generation_prompt": True,
                "additional_eos_token": "<|im_end|>",
            },
            "aggregate": aggregate,
            "rows": rows,
        }

        payload["result_hash"] = _sha256_json(payload)

        out = output_dir / f"{condition.lower()}_primary_result.json"
        out.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

        condition_results[condition] = payload

        print()
        print(condition)
        print(
            "overall:",
            aggregate["correct"],
            "/",
            aggregate["n_total"],
            "=",
            aggregate["overall_accuracy"],
        )
        print(
            "macro:",
            aggregate["macro_family_accuracy"],
        )
        print(
            "runtime_failures:",
            aggregate["runtime_failures"],
        )
        print(
            "result_hash:",
            payload["result_hash"],
        )

    summary = {
        "schema_version": "m5-primary-2x2-summary-v1",
        "primary_example_count": len(examples),
        "primary_example_ids_hash": _sha256_json(example_ids),
        "selection_run_id": selection["selection_run_id"],
        "selected_candidate_id": selected_candidate_id,
        "selected_checkpoint_id": selection["selected_checkpoint_id"],
        "selected_checkpoint_iteration": selected_checkpoint_iteration,
        "selected_candidate_manifest_hash": selection[
            "selected_candidate_manifest_hash"
        ],
        "selected_candidate_result_hash": selection[
            "selected_candidate_result_hash"
        ],
        "retrieval_run_id": retrieval_artifact.retrieval_run_id,
        "retrieval_artifact_hash": retrieval_artifact.retrieval_artifact_hash,
        "conditions": {
            name: {
                "aggregate": condition_results[name]["aggregate"],
                "result_hash": condition_results[name]["result_hash"],
            }
            for name in CONDITIONS
        },
    }

    summary["summary_hash"] = _sha256_json(summary)

    summary_path = output_dir / "m5_primary_2x2_summary_v1.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print("M5_PRIMARY_2X2_COMPLETE")
    print("summary:", summary_path)
    print("summary_hash:", summary["summary_hash"])

    return summary
