# Repository Guidelines

## Project Layout

- `pipelines/`: preprocessing, training, tuning, generation, evaluation, visualization entry points.
- `src/`: models, codecs, training, inference, metrics, validation, and artifact provenance.
- `config/`: Hydra datasets, architectures, runtime profiles, and stage defaults.
- `data/`: Git LFS source logs and generated preprocessing artifacts; `outputs/`: run artifacts.
- `scripts/`: checkpoint/dataset utilities; `docs/`: designs.

## Development Commands

See [README.md](../README.md) for the complete pipeline.

```sh
uv sync --locked                          # Install dependencies
git lfs pull                              # Fetch source logs
uv run python -m pipelines.preprocess dataset=sepsis
uv run python -m pipelines.train dataset=sepsis model=head_sampling_transformer
uv run python -m pipelines.generate checkpoint=/path/to/best.pt device=cpu num_samples=100
uv run python -m pipelines.evaluate generations=/path/to/generations.parquet
uv run ruff check .                       # Lint
uv run ruff format --check .               # Check formatting
```

Append `--cfg job --resolve` to inspect Hydra configuration without running a stage.
Training uses W&B; authenticate with `uv run wandb login`.

## Coding and Architecture

Use four-space indentation, single quotes, 100-character lines, type hints,
`snake_case` functions/modules, and `PascalCase` classes. Ruff also sorts imports.
Preserve shared training/generation interfaces and explicit stage artifacts.
Preserve chronological splits, fitted codecs, resolved configurations, seeds, and checkpoint provenance.
Generation must read prefixes only; tune on validation data, never test data.

## Correctness Checks

Check relevant tensor shapes, finite losses, checkpoint reloads, sample counts, and termination.
Run disposable checks with `uv run python tmp/<check_name>.py`.
Inspect small-run artifacts; report commands/results. No test framework or coverage target is configured.

## Commits and Pull Requests

Use short imperative subjects, e.g. `Update loss`, following history.
PRs should explain behavior, configuration changes, validation results, and relevant issues.
