# Data Artifact Guide

Each `data/<dataset>/` directory combines one source log with artifacts produced by preprocessing.
The supported dataset identifiers are `sepsis`, `bpic12`, `bpic17`, and `bpic19`.

## Layout

| Path | Ownership |
| --- | --- |
| `original.csv` | Source event log tracked through Git LFS. |
| `processed/train.csv` | Chronological training split written by preprocessing. |
| `processed/val.csv` | Chronological validation split written by preprocessing. |
| `processed/test.csv` | Chronological test split written by preprocessing. |
| `codec/dataset.json` | Vocabulary and numeric transforms fitted on the training split. |
| `declare/model.decl` | Declare constraints discovered from the training split. |
| `manifest.json` | Bundle hashes and fingerprint. |

## Rules

- Never hand edit an original log, processed split, codec, or Declare model.
- Never replace `original.csv` with a reformatted copy. Preserve its bytes and configure its
  separator and column names under `config/dataset/`.
- Regenerate derived files with `python -m pipelines.preprocess`; do not repair one artifact in
  isolation.
- Fit vocabularies, numeric statistics, and Declare constraints on training data only.
- Preserve chronological split boundaries and the special lower prefix bound for cases that cross
  a boundary.
- Treat regenerated data as a coordinated set. A codec or Declare model from another preprocessing
  run may silently change model inputs or evaluation semantics.
- Confirm Git LFS objects are present before diagnosing CSV parsing failures.

See [`src/datasets/AGENTS.md`](../src/datasets/AGENTS.md) for runtime tensor and codec contracts and
[`pipelines/AGENTS.md`](../pipelines/AGENTS.md) for preprocessing behavior.
