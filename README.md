# AdaptLab

AdaptLab studies where LLM adaptation should live: prompting, retrieval context, or model weights.

## Milestone 1

This milestone builds only the deterministic synthetic benchmark foundation for a fictional Nimbus platform. The goal is to make a small benchmark fixture reproducible byte-for-byte from the same seed while enforcing lifecycle, evidence, behavior, split, structural-holdout, and gold-reference invariants.

This milestone intentionally does **not** implement RAG, embeddings, LoRA, model inference, or LLM-based benchmark generation.

## Requirements

- Python 3.11+
- `pytest` for tests

## Development

```bash
python -m pip install -e '.[test]'
pytest
```

## CLI

The default prototype fixture location is `data/fixtures/prototype/`.

```bash
adaptlab benchmark build-fixture --seed 1729
adaptlab benchmark validate data/fixtures/prototype
adaptlab benchmark audit data/fixtures/prototype
```

The CLI delegates to the same build, validation, and audit functions used by the test suite.

## v0.0 evaluation precommitment

Future prompt selection and LoRA tuning boundaries are frozen before model results in `configs/evaluation_policy_v0.0.yaml`. The file records prompt and validation budgets, allowed search spaces, the generalization guardrail, meaningful-improvement threshold, subgroup policy, architecture-sensitive retention rules, and canonical LoRA seeds. It is policy only: no prompting search, LoRA training, or model evaluation is implemented in this milestone.

## Test suites

Run the normal fast suite without the full v0.0 reproducibility build:

```bash
pytest -m "not full_reproducibility"
```

Run the heavier full benchmark reproducibility integration tests separately:

```bash
pytest -m full_reproducibility
```

The full reproducibility suite compares complete same-config/same-seed builds byte-for-byte, verifies artifact hashes and canonical ordering, and exercises alternate-seed generation. Freeze readiness is evaluated separately from reproducibility and must reflect the current leakage and human-audit gates.

## External-validity precommitment

The future external-validity benchmark is selected under the frozen policy in `configs/external_validity_v0.0.yaml`. The policy records a short public candidate list, deterministic eligibility/ranking rules, sample-size and scoring rules, contamination review, and the directional-transfer claim to test. The external benchmark is not used for AdaptLab prompt selection or hyperparameter tuning, and no external benchmark is run during Milestone 2.
## Benchmark state

The v0.0 benchmark is currently a **candidate and is not frozen**. The current build has unresolved cross-split semantic-leakage blockers, and the 50-example human-review queue is explicitly pending genuine human review.

`data/generated/v0.0/preliminary_manifest.json` is the canonical candidate manifest while blockers remain. `data/generated/v0.0/benchmark_freeze.json` is derived from the current candidate and human-audit gates and must report `V0_0_BENCHMARK_NOT_READY` until both are clean. A final `manifest.json` and Git tag are only valid after those gates pass.


## M5b: Where should adaptation live?

M5b evaluates a controlled 2×2 design on the same Qwen3-8B-Base runtime:

| Condition | Retrieval | LoRA | Primary accuracy |
|---|---:|---:|---:|
| PROMPT | No | No | 18.5% |
| RAG | Yes | No | 51.25% |
| LoRA | No | Yes | 41.25% |
| LoRA + RAG | Yes | Yes | **71.0%** |

The mechanisms specialize differently:

- **Behavior-only:** PROMPT 58%, RAG 58%, LoRA 98%, LoRA+RAG 98%.
- **Knowledge-only:** PROMPT 0%, RAG 53%, LoRA 22%, LoRA+RAG 72%.
- **Changed knowledge:** PROMPT 0%, RAG 41%, LoRA 16%, LoRA+RAG 63%.
- **Behavior + knowledge:** PROMPT 16%, RAG 53%, LoRA 29%, LoRA+RAG 51%.

Paired primary-test comparisons strongly favor the adapted systems. For example, RAG → LoRA+RAG improves by **19.75 percentage points** with 101 paired wins versus 22 losses (exact two-sided McNemar/binomial p ≈ 2.7e-13).

The factorial interaction is mildly negative:

`LoRA+RAG - RAG - LoRA + PROMPT = -3.0 pp`

So retrieval and LoRA are **practically complementary but mildly sub-additive**: retrieval is strongest for external/current knowledge, LoRA is strongest for behavioral compliance, and their combination yields the best overall system.

### M5b correction

The original M5 evaluation used a chat-style inference interface with `Qwen3-8B-Base`, which caused severe continuation/termination artifacts and invalid near-zero scores. Those artifacts are retained for provenance but are not used for scientific conclusions.

M5b froze a corrected continuation-style inference contract before corrected validation selection and primary evaluation. The completed experiment is checkpointed at:

- commit: `1bed76f`
- tag: `v0.3-lora-rag`

Detailed provenance and statistical results are in:

`artifacts/evaluation/m5/M5B_RESULTS.md`

