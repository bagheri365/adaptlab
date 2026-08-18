# AdaptLab

**Where should LLM adaptation live: prompt, retrieval context, or model weights?**

AdaptLab is a controlled research project for separating the effects of prompting, retrieval-augmented generation, and low-rank weight adaptation.

The project rule is simple:

> Change one adaptation mechanism, measure it, and preserve the evidence — including failed and invalidated experiments.

## At a Glance

- **Research question:** what should be encoded in the prompt, supplied dynamically through retrieval, or learned into model weights?
- **Current milestone:** `v0.3-lora-rag`
- **Controlled primary evaluation:** 400 frozen examples, balanced across four task families
- **Best M5b system:** LoRA + RAG at **71.0%**
- **Main pattern:** LoRA is strongest for behavioral compliance; retrieval is critical for external and changed knowledge
- **Important exception:** on behavior+knowledge, LoRA + RAG does not beat RAG alone
- **Final robustness split:** 100-example sentinel remains untouched

Detailed M5b evidence: [`artifacts/evaluation/m5/M5B_RESULTS.md`](artifacts/evaluation/m5/M5B_RESULTS.md)

## Why This Project Exists

LLM systems are often adapted in several places at once:

- system prompts,
- retrieved context,
- fine-tuned weights,
- output formatting,
- inference scaffolding.

When performance changes, it becomes difficult to tell which mechanism actually helped.

AdaptLab instead asks:

> Where should adaptation live?

The controlled M5b experiment isolates two major mechanisms in a 2x2 design:

| Condition | Retrieval | LoRA |
|---|---:|---:|
| PROMPT | No | No |
| RAG | Yes | No |
| LoRA | No | Yes |
| LoRA + RAG | Yes | Yes |

A separate oracle-context condition was used earlier as a diagnostic for missing-information failures.

## Key Findings

### Controlled M5b Primary Result

All four conditions use the same `Qwen/Qwen3-8B-Base` MLX runtime and corrected continuation-style inference contract.

| Condition | Correct | Accuracy |
|---|---:|---:|
| PROMPT | 74 / 400 | 18.50% |
| RAG | 205 / 400 | 51.25% |
| LoRA | 165 / 400 | 41.25% |
| **LoRA + RAG** | **284 / 400** | **71.00%** |

The combined system is strongest overall.

### Retrieval Adds Substantial Knowledge Access

PROMPT -> RAG:

- **+32.75 percentage points**
- 132 paired wins
- 1 paired loss
- exact two-sided McNemar/binomial `p ≈ 2.46e-38`

### LoRA Adds Substantial Behavioral Capability

PROMPT -> LoRA:

- **+22.75 percentage points**
- 92 paired wins
- 1 paired loss
- `p ≈ 1.90e-26`

The clearest specialization appears on behavior-only tasks:

| Condition | Accuracy |
|---|---:|
| PROMPT | 58% |
| RAG | 58% |
| LoRA | 98% |
| LoRA + RAG | 98% |

Retrieval provides no gain on this family, while LoRA adds 40 percentage points.

This is the clearest evidence in the current benchmark that some adaptation belongs in model weights rather than retrieved context.

### Retrieval Is Stronger for External and Changed Knowledge

Knowledge-only:

| Condition | Accuracy |
|---|---:|
| PROMPT | 0% |
| RAG | 53% |
| LoRA | 22% |
| **LoRA + RAG** | **72%** |

Changed knowledge:

| Condition | Accuracy |
|---|---:|
| PROMPT | 0% |
| RAG | 41% |
| LoRA | 16% |
| **LoRA + RAG** | **63%** |

These families favor retrieval over weight adaptation alone.

### The Mechanisms Complement Each Other Overall

RAG -> LoRA + RAG:

- **+19.75 percentage points**
- 101 paired wins
- 22 losses
- `p ≈ 2.74e-13`

LoRA -> LoRA + RAG:

- **+29.75 percentage points**
- 148 wins
- 29 losses
- `p ≈ 2.00e-20`

PROMPT -> LoRA + RAG:

- **+52.50 percentage points**
- 213 wins
- 3 losses
- `p ≈ 3.19e-59`

The factorial interaction is:

`LoRA+RAG - RAG - LoRA + PROMPT = -3.0 pp`

So the mechanisms are practically complementary, but mildly sub-additive rather than super-additive.

### Combination Is Not Always Better

Behavior + knowledge is the main exception:

| Condition | Accuracy |
|---|---:|
| PROMPT | 16% |
| **RAG** | **53%** |
| LoRA | 29% |
| LoRA + RAG | 51% |

For RAG -> LoRA + RAG:

- 9 wins
- 11 losses
- `p ≈ 0.824`

This suggests a possible interference regime that deserves further study.

The reported paired p-values are unadjusted for multiple comparisons.

## Benchmark

AdaptLab uses a fictional API domain called **Nimbus** so that knowledge state and behavioral requirements can be controlled explicitly.

The benchmark separates four families:

- **behavior_only** — learned behavioral rules without external knowledge requirements
- **knowledge_only** — factual knowledge requirements without additional learned behavior
- **behavior_knowledge** — both knowledge access and behavioral compliance
- **changed_knowledge** — unchanged, updated, or removed facts

Splits:

| Split | Examples |
|---|---:|
| Train | 300 |
| Validation | 150 |
| Primary | 400 |
| Sentinel | 100 |

The primary split contains 100 examples from each family.

The benchmark also includes:

- IID and structural holdouts
- evidence-present and evidence-absent cases
- deterministic hashes
- leakage audits
- human review
- paired per-example scoring

The benchmark and primary split are frozen.

The sentinel split remains untouched.

## Research Evolution

> benchmark -> diagnose -> retrieve -> adapt weights -> combine

| Version | Question | Result | Disposition |
|---|---|---|---|
| `v0.0-benchmark` | Can the benchmark be frozen and audited? | 300/150/400/100 splits; 50/50 human audit PASS | Retain |
| `v0.1-prompt-oracle` | How much failure comes from missing context? | Large Prompt -> Oracle gap on historical Ollama runtime | Retain as diagnostic |
| `v0.2-rag` | Can retrieval close the knowledge gap? | BM25 reached 87.5% on the historical Ollama evaluation | Retain |
| Original M5 | Does LoRA help? | Evaluation interface produced invalid near-zero results | Invalidate for scientific interpretation |
| `v0.3-lora-rag` | Do weights and retrieval complement each other? | LoRA + RAG reaches 71.0% in corrected controlled M5b | Retain |

The v0.1 and v0.2 results used the earlier Ollama `qwen3:8b` evaluation path and should not be compared directly with M5b percentages.

## Negative and Invalidated Results

AdaptLab preserves failed experiments rather than rewriting them away.

The original M5 evaluation used a chat-template inference interface with `Qwen3-8B-Base`.

Diagnostics showed that the model behaved substantially better as a continuation model. The chat path frequently produced:

- continued synthetic turns,
- repeated scaffolding,
- output-boundary failures,
- exact-match failures despite correct answer prefixes.

Additional training-path audits identified and fixed:

- persisted LoRA target-key configuration
- MLX LoRA scale mapping
- causal-loss padding supervision
- incomplete gradient-accumulation tails

After these fixes, all six frozen LoRA candidates were retrained.

The original M5 artifacts remain in the provenance history but are not used for scientific conclusions.

See [`M5_INVALIDATION.md`](artifacts/evaluation/m5/M5_INVALIDATION.md).

## M5b Protocol

The corrected M5b inference contract was frozen before corrected validation selection.

Inference form:

~~~text
{system}

{user}

Answer:
~~~

Evaluation uses:

- temperature `0`
- top-p `1`
- top-k `0`
- max new tokens `64`
- first non-empty output line as the extracted answer
- one trailing period removed before existing exact scoring

Six predeclared LoRA candidates were evaluated on the frozen validation set.

Selected candidate:

`S2_POLICY_B_ATTN_r8_lr2e-05_iters250`

Validation result:

| Condition | Accuracy |
|---|---:|
| PROMPT | 14.00% |
| RAG | 47.33% |
| LoRA | 29.33% |
| **LoRA + RAG** | **66.00%** |

No retraining or retuning occurred after primary evaluation.

## Reproducibility

M5/M5b preserves:

- runtime provenance
- base-model revision and tensor identity
- frozen training formatter
- contamination audit
- LoRA trainable policy
- training configuration
- candidate traces
- validation selection policy
- corrected inference contract
- selection decision
- per-example validation results
- per-example primary results
- paired statistical analysis
- invalidation manifests

Important M5b hashes:

**Inference contract**

`921bdf65e7e704ab88dac0f9ae25744efdabdb421161fb0187d0c59bd1a05462`

**Primary summary file SHA-256**

`a90ffd1fbfd3ee261af1ec008f2de17ef355041ffaa0d28f779dc574ebdb94a0`

**Paired-analysis logical hash**

`010a30579c99273d43b158282a673105c0ab7e9102d0ed1b11383f226465d7ee`

**Paired-analysis file SHA-256**

`21849906630be8670b2bfe7d6c13a553b2883c7189c14d8be8eb89c6c013f8e5`

## Repository Structure

~~~text
adaptlab/
├── artifacts/        # frozen evaluation, retrieval, provenance, and audit artifacts
├── configs/          # experiment configuration
├── data/             # benchmark and supporting data
├── docs/             # methodology and research notes
├── src/adaptlab/     # implementation
├── tests/            # deterministic regression/unit tests
└── README.md
~~~

Large model-weight artifacts are intentionally excluded from Git.

## Running Locally

Create an environment and install the project according to the repository configuration, then run:

~~~bash
python -m pytest -q
~~~

M5-specific tests:

~~~bash
python -m pytest tests/test_m5_*.py -q
~~~

At the `v0.3-lora-rag` milestone:

~~~text
40 passed
~~~

for the M5-specific suite.

## Experimental Discipline

The intended discipline is:

- training split for gradients
- validation split for candidate selection
- frozen candidate search spaces
- deterministic result artifacts
- paired comparisons on identical examples
- explicit invalidation rather than artifact deletion
- no sentinel access before a final frozen protocol

One important qualification:

> The primary split was inspected during diagnostics before M5b was defined.

M5b is therefore a **post-hoc corrected protocol**, not a pristine preregistered confirmatory experiment.

That limitation is explicit rather than hidden.

The sentinel remains untouched and provides a possible final confirmatory evaluation.

## Limitations

- Nimbus is synthetic.
- M5b uses one primary base-model/runtime combination.
- Retrieval is BM25 rather than a learned retriever.
- The LoRA search space is deliberately small.
- Evaluation is exact-match oriented.
- M5b was created after discovering an inference-interface defect.
- The primary split had diagnostic exposure before M5b.
- Family-level and overall paired tests are not multiplicity-adjusted.
- External-domain replication has not yet been performed.

The current results are mechanistic evidence inside this controlled benchmark, not a universal ranking of adaptation strategies.

## What the Experiments Suggest

The current evidence supports a division of labor:

- **prompts** define the immediate task,
- **retrieval** is especially useful for external and mutable knowledge,
- **weight adaptation** is especially useful for learned behavioral compliance.

The combined system performs best overall, but the behavior+knowledge result shows that combining adaptation mechanisms is not automatically additive.

A working design hypothesis from AdaptLab is:

> Keep mutable facts outside the model when possible, use weight adaptation for behaviors worth internalizing, and evaluate their interaction rather than assuming the mechanisms compose cleanly.

## Future Research

- run the untouched sentinel only under a frozen final protocol
- investigate behavior+knowledge interference
- compare stronger retrieval methods
- replicate M5b across additional base models
- test behavioral transfer beyond Nimbus
- distinguish knowledge memorization from behavioral learning more aggressively
- evaluate metrics beyond exact match
- replicate the factorial design on a real external domain

## Milestones

| Tag | Milestone |
|---|---|
| `v0.0-benchmark` | Frozen Nimbus benchmark |
| `v0.1-prompt-oracle` | Prompt / oracle diagnostic |
| `v0.2-rag` | Canonical BM25 RAG evaluation |
| `v0.3-lora-rag` | Corrected LoRA x retrieval experiment |

## Current Conclusion

AdaptLab asks:

> Where should adaptation live?

The current evidence says:

> **Not in only one place.**

Retrieval is strongest where the problem is access to external or changing knowledge.

Weight adaptation is strongest where the problem is behavioral compliance.

Their combination produces the strongest overall M5b system, while the remaining interference case shows why the mechanisms should be measured separately before they are combined.
