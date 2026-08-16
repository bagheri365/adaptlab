# Validation Decision Log

Policy reviewed: `configs/evaluation_policy_v0.0.yaml`

Precommitment status:
- `status`: `PRECOMMITTED_BEFORE_MODEL_RESULTS`
- `selection_split`: `validation`
- `test_set_access_during_selection`: `false`
- `sentinel_access_during_selection`: `false`
- `max_full_validation_passes_per_method`: `20`
- `max_validation_passes_per_candidate`: `1`

Smoke-test artifacts consulted:
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run1-20260815T150236/run_manifest.json`
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run1-20260815T150236/results.json`
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run1-20260815T150236/metrics.json`
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run2-20260815T150236/run_manifest.json`
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run2-20260815T150236/results.json`
- `/private/tmp/adaptlab_ollama_qwen3_8b_smoke/run2-20260815T150236/metrics.json`

Validation-selection budget status:
- valid
- one validation candidate pass was used for the Milestone 3 smoke test
- no primary-test examples were inspected
- no sentinel examples were used for selection
- no repeated prompt-search iteration was performed

## Records

| timestamp | validation examples/results consulted | category | problem observed | change made | why the change was allowed | validation-budget impact | resulting config/prompt version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-15T19:07:34Z | Validation smoke setup for `qwen3:8b`; local Ollama `api/tags`; 24-validation-example run `run1-20260815T150236`; cache replay `run2-20260815T150236` | BUG_FIX | The repo had no Ollama provider path, so a real local Ollama validation smoke test could not be executed through the evaluation harness. | Added `OllamaModelProvider`, exported it from the provider package, added CLI wiring for `--provider ollama`, and added focused provider tests. | This was a validation-driven implementation bug fix. It did not modify benchmark truth, selection data, test data, or sentinel data. | No extra validation-selection budget was consumed beyond the planned smoke test. The actual validation pass count remained `1` for this candidate. | `prompt_v1` + `ollama:qwen3:8b` with `temperature=0`, `seed=1729`, `context_length=40960`, `max_tokens=256`, `stream=false`, `think=false` |
| 2026-08-15T19:07:34Z | 24 validation examples: `FULL_VA_BHV_001`, `FULL_VA_BHV_002`, `FULL_VA_BHV_003`, `FULL_VA_BHV_004`, `FULL_VA_BHV_005`, `FULL_VA_BHV_006`, `FULL_VA_KNW_001`, `FULL_VA_KNW_002`, `FULL_VA_KNW_003`, `FULL_VA_KNW_004`, `FULL_VA_KNW_005`, `FULL_VA_KNW_006`, `FULL_VA_BKN_001`, `FULL_VA_BKN_002`, `FULL_VA_BKN_003`, `FULL_VA_BKN_004`, `FULL_VA_BKN_005`, `FULL_VA_BKN_006`, `FULL_VA_CHG_001`, `FULL_VA_CHG_002`, `FULL_VA_CHG_003`, `FULL_VA_CHG_006`, `FULL_VA_CHG_007`, `FULL_VA_CHG_008`; consulted `results.json` and `metrics.json` from the same run plus cache replay results | MODEL_CONFIGURATION | No prompt/config defect was observed. The current prompt and explicit Ollama settings were acceptable under the precommitted budget, and poor model accuracy alone was not treated as a reason to change prompt content. | Selected the current prompt and Ollama settings as the canonical Milestone 3 candidate: `prompt_v1`, `provider=ollama`, `model_id=qwen3:8b`, `base_url=http://localhost:11434`, `temperature=0`, `seed=1729`, `context_length=40960`, `max_tokens=256`, `stream=false`, `think=false`. | This selection stays entirely within the precommitted validation rules: validation split only, no test-set access, no sentinel access, no repeated search loop, and no benchmark truth changes. | One validation candidate pass; within `max_full_validation_passes_per_method=20` and `max_validation_passes_per_candidate=1`. | `prompt_v1` + `ollama:qwen3:8b` canonical Milestone 3 candidate |
| 2026-08-15T19:07:34Z | Validation smoke results for `run1-20260815T150236` and cache replay `run2-20260815T150236`; consulted the same 24 validation examples and the recorded run manifest/metrics | MODEL_CONFIGURATION | The canonical Milestone 3 PROMPT and ORACLE_CONTEXT conditions were not yet frozen as versioned artifacts, even though the validation smoke test had already confirmed the shared Ollama settings and the Oracle evidence path behaved as intended. | Wrote versioned canonical config artifacts for both conditions: `configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml` and `configs/evaluation_conditions/milestone3_ollama_oracle_context_v1.yaml`. | This freeze is allowed because it preserves the validated settings, does not modify benchmark truth, does not inspect primary-test or sentinel results, and only records the precommitted candidate state for later primary evaluation. | No additional validation-selection budget consumed. The freeze was based on the single already-completed validation candidate pass. | `milestone3_ollama_prompt_v1` and `milestone3_ollama_oracle_context_v1` |
| 2026-08-15T19:22:43Z | Canonical completeness validation tests: `tests/test_canonical_run_completeness.py`, plus the related evaluation/cache/provenance/schema tests that exercise run manifests and resume behavior | BUG_FIX | Canonical runs did not yet enforce explicit `expected_count` versus `completed_successful_responses`, did not suppress canonical accuracy for incomplete canonical runs, and did not surface dropped-example mismatches as hard failures. | Added shared completeness checks, recorded `expected_count` and `completed_successful_responses` in run provenance, suppressed canonical top-line accuracy when completeness is invalid, and added tests for 400/400 primary completion, unresolved failures, resume preservation, retry exhaustion, sentinel 100/100 completion, and dropped-example rejection. | This is a validation-driven bug fix. It changes evaluation bookkeeping and failure handling only, not benchmark truth, selection data, sentinel data, or prompt content. | No additional benchmark-selection budget consumed. The tests used fake providers and local artifact data only. | `prompt_v1` + `ollama:qwen3:8b` canonical Milestone 3 candidate; completeness metadata added to evaluation provenance |

## Observations

- Ollama reachable locally: yes.
- Selected model installed: yes, `qwen3:8b`.
- Validation pass completed: `24/24`.
- Provider errors: `0`.
- Scoring failures caused by infrastructure: `0`.
- Cache behavior: second run hit the exact-request cache for all 24 examples and made `0` provider calls.
- Reasoning/thinking text was not accidentally scored.
- No benchmark truth was modified.

## Milestone 3 Freeze

- Canonical Milestone 3 configuration artifacts were written before any primary-test execution:
  - [configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml](/Users/alireza/research_ideas/09_adaptlab/adaptlab_prompts_01-11/configs/evaluation_conditions/milestone3_ollama_prompt_v1.yaml)
  - [configs/evaluation_conditions/milestone3_ollama_oracle_context_v1.yaml](/Users/alireza/research_ideas/09_adaptlab/adaptlab_prompts_01-11/configs/evaluation_conditions/milestone3_ollama_oracle_context_v1.yaml)
- Exact local model tag recorded: `qwen3:8b`
- Local Ollama metadata supports the digest record, but the tag alone is not treated as proof of immutable weights.
- The only intended difference between the two frozen conditions is the Oracle evidence injection rule when `evidence_status=PRESENT`.
