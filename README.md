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
