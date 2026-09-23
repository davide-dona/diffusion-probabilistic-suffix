# Source Guide

`src/` contains reusable implementation code. `pipelines/` owns orchestration and Hydra entry
points; source modules should expose typed operations that can be tested without starting a stage.

## Areas

| Area | Responsibility |
| --- | --- |
| `datasets/` | Tensor structures, prefix and suffix cuts, fitted codecs, and split loading. |
| `artifacts/` | Stored artifact locations, dataset manifests, hashes, identities, and provenance. |
| `models/` | Shared model interface, architectures, checkpoint loading, and time derivation. |
| `training/` | Optimization, validation, early stopping, and scalar records. |
| `inference/` | Batch generation, tuning reports, decoded samples, and generations Parquet I/O. |
| `evaluation/` | Metric registration, prefix scoring, aggregation, and evaluation reports. |
| `logs/` | Event-log I/O, preprocessing transforms, Declare discovery, and conformance. |
| `runs/` | Hydra invocation setup, effective configuration, and output directories. |
| `validation/` | Configuration and stage-boundary validation. |
| `visualization/` | Figure and table catalogues, labels, and rendering. |
| `uncertainty/` | Case-level resampling and significance comparisons. |

Read the nested guides for [`datasets/`](datasets/AGENTS.md), [`models/`](models/AGENTS.md), and
[`evaluation/`](evaluation/AGENTS.md) before working in those areas.

## Shared Contracts

- Use tensor comments in the form `[B, T, D]` where `B` is batch, `T` sequence length, `D` model
  width, `S` samples, `V` codec vocabulary size, and `K` diffusion vocabulary size.
- Keep immutable records immutable. Dataset batches and output records use named tuples or frozen
  dataclasses so device moves and transformations return new values.
- Keep architecture decisions behind `SuffixModel`. Training and inference must not branch on a
  concrete model except where a capability is explicitly architecture-specific, such as sampler
  tuning.
- Validate external artifacts at read time: schema, required metadata, run identity, vocabulary,
  and source hashes are part of their contracts.
- Protect replacement of an existing best checkpoint with a temporary file and `Path.replace`.
- Remove directly streamed Parquet destinations when a handled write failure interrupts them.
- Keep random selection reproducible through explicit seeds or generators. Do not depend on prior
  global random state.
- Avoid importing orchestration code from `pipelines/` into `src/`.

## Coding and Validation

Use Python 3.13 typing, four-space indentation, single quotes, and lines no longer than 100
characters. Public operations and non-obvious private contracts need precise docstrings. Comments
may explain current invariants or subtle algorithms, never the history of a change.

Run the narrowest relevant pytest module first, then `uv run ruff check .` and
`uv run ruff format --check .`. Model changes must also satisfy `tests/models`. Artifact changes
must test both valid round trips and rejection of incompatible inputs.
