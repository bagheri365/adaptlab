# M5b — Corrected LoRA × Retrieval Experiment

## Status

M5b is the corrected version of M5.

The original M5 evaluation used a chat-template inference interface with `Qwen3-8B-Base`. Diagnostics showed that the base checkpoint behaved substantially better as a plain continuation model. The original M5 primary result is therefore retained for provenance but is not used for scientific conclusions.

M5b freezes a corrected inference contract before corrected validation selection:

- input: `{system}\n\n{user}\n\nAnswer:`
- temperature: 0
- top-p: 1
- top-k: 0
- max new tokens: 64
- answer extraction: first non-empty line, stripped, with one trailing period removed
- scoring: existing `score_output()`

Inference contract SHA-256:

`921bdf65e7e704ab88dac0f9ae25744efdabdb421161fb0187d0c59bd1a05462`

## Validation selection

Six predeclared LoRA candidates were reevaluated on the frozen 150-example validation split under the corrected M5b inference contract.

Selected candidate:

`S2_POLICY_B_ATTN_r8_lr2e-05_iters250`

Selected checkpoint:

`ckpt-0000250`

Selected candidate validation:

- overall: 44 / 150 = 29.33%
- macro family accuracy: 29.07%

Selection decision hash:

`e142898605ef936f6fe698a688b57c40bf1752b34b4499feb212f63942d1e111`

## Validation 2×2

| Condition | Accuracy |
|---|---:|
| PROMPT | 14.00% |
| RAG | 47.33% |
| LoRA | 29.33% |
| LoRA + RAG | 66.00% |

Validation summary file SHA-256:

`bd108120b1ab521666e95e1c9bd4defc70612f11b719259a51592e17dac09420`

## Primary 2×2

Frozen 400-example primary test:

| Condition | Correct | Accuracy |
|---|---:|---:|
| PROMPT | 74 / 400 | 18.50% |
| RAG | 205 / 400 | 51.25% |
| LoRA | 165 / 400 | 41.25% |
| LoRA + RAG | 284 / 400 | 71.00% |

Primary summary file SHA-256:

`a90ffd1fbfd3ee261af1ec008f2de17ef355041ffaa0d28f779dc574ebdb94a0`

## Paired primary comparisons

Exact two-sided McNemar/binomial tests on paired examples:

| Comparison | Delta | Wins | Losses | p |
|---|---:|---:|---:|---:|
| PROMPT → RAG | +32.75 pp | 132 | 1 | 2.46e-38 |
| PROMPT → LoRA | +22.75 pp | 92 | 1 | 1.90e-26 |
| RAG → LoRA + RAG | +19.75 pp | 101 | 22 | 2.74e-13 |
| LoRA → LoRA + RAG | +29.75 pp | 148 | 29 | 2.00e-20 |
| PROMPT → LoRA + RAG | +52.50 pp | 213 | 3 | 3.19e-59 |

No multiple-testing adjustment is applied.

Paired-analysis logical hash:

`010a30579c99273d43b158282a673105c0ab7e9102d0ed1b11383f226465d7ee`

Paired-analysis file SHA-256:

`21849906630be8670b2bfe7d6c13a553b2883c7189c14d8be8eb89c6c013f8e5`

## Family-level primary results

### Behavior only

- PROMPT: 58%
- RAG: 58%
- LoRA: 98%
- LoRA + RAG: 98%

LoRA provides the dominant gain. Retrieval has no effect, as expected.

### Knowledge only

- PROMPT: 0%
- RAG: 53%
- LoRA: 22%
- LoRA + RAG: 72%

Retrieval supplies the main knowledge-access gain; the combination performs best.

### Changed knowledge

- PROMPT: 0%
- RAG: 41%
- LoRA: 16%
- LoRA + RAG: 63%

Retrieval is substantially stronger than weights alone for changed knowledge, while the combination is strongest.

### Behavior + knowledge

- PROMPT: 16%
- RAG: 53%
- LoRA: 29%
- LoRA + RAG: 51%

This is the important exception: LoRA + RAG does not improve over RAG alone on this family.

## Factorial interpretation

Overall interaction:

`LoRA+RAG - RAG - LoRA + PROMPT = -3.0 percentage points`

The mechanisms are therefore practically complementary but mildly sub-additive rather than super-additive.

## Main conclusion

Retrieval and weight adaptation specialize in different parts of adaptation.

- Retrieval provides the largest gains for external and changed knowledge.
- LoRA provides large gains for behavioral compliance.
- Combining LoRA with retrieval gives the best overall performance.
- The combination is not universally beneficial: behavior+knowledge shows mild interference.

The evidence supports placing adaptation in both retrieval context and model weights when the task requires both knowledge access and learned behavioral compliance.
