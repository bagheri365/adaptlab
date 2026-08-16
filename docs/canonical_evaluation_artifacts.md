# Canonical Evaluation Artifacts

The canonical evaluation bundles are the durable, project-owned copies that
should be used for auditing, comparison, and reproduction.

Canonical locations:

- `artifacts/evaluation/m3/prompt`
- `artifacts/evaluation/m3/oracle_context`
- `artifacts/evaluation/m3/sentinel`
- `artifacts/evaluation/m4/rag`

Each canonical run directory should retain:

- `run_manifest.json`
- `results.json` or an equivalent per-example result bundle
- `metrics.json`
- `summary.txt` or a status/completion artifact
- checksum or hash metadata already supported by the runner

For Milestone 4 RAG runs, the durable bundle should also retain or reference:

- `retrieval_run_id`
- `retrieval_artifact_hash`
- retrieved chunk IDs
- retrieved context hashes

Disposable temporary mirrors may exist under `/private/tmp`, but they are not
the canonical source of record and should not be relied on as the only copy of
evaluation outputs.

The Milestone 3 prompt and corrected Oracle bundles in this repository are
post-hoc reproducibility reruns copied from the temporary historical runs.
The Milestone 4 canonical RAG bundle should be written directly into the
repository-backed canonical location once the frozen runtime is available.
