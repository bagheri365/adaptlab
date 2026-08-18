# Original M5 Evaluation — Invalidated for Scientific Interpretation

The original M5 evaluation artifacts are retained for provenance.

They are not used for scientific conclusions because the evaluation used a chat-template inference interface with `Qwen3-8B-Base`.

Post-run diagnostics established that:

1. the base model produced coherent and often correct answers under plain continuation prompting;
2. frozen BM25 retrieval correctly supplied Nimbus knowledge under that interface;
3. LoRA adapters also produced meaningful outputs under continuation prompting;
4. the chat-template path frequently caused continued synthetic turns, repeated scaffolding, and severe whole-output exact-match failures.

The defect therefore affected the inference/evaluation interface rather than establishing genuine zero capability.

The corrected experiment is M5b. M5b was given a separately frozen inference contract, validation selection, validation 2×2, and primary 2×2.

Original M5 artifacts have not been deleted or rewritten.
