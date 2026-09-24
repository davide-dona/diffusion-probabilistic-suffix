# Pipeline Guide

The six Hydra entry points form an explicit artifact pipeline. No stage discovers an upstream
artifact implicitly. Pass the exact checkpoint, tuning report, generations file, or evaluation
report selected by the caller.

Shared console formatting and Hydra invocation setup live in `pipelines/helpers/`. Stage modules
remain the six entry points.

## Stages

Commands in this table describe production execution. Do not run the heavy stages locally. Use
the safe inspection and test commands below while developing.

| Stage | Command | Reads | Writes | Behavior |
| --- | --- | --- | --- | --- |
| Preprocess | `uv run python -m pipelines.preprocess dataset=sepsis` | `data/<dataset>/original.csv` | processed splits, codec, Declare model, invocation config | Sorts cases, filters configured outliers, derives attributes, makes chronological splits, fits the codec on train, and mines the Declare model on train. |
| Train | `uv run python -m pipelines.train dataset=sepsis model=head_sampling_transformer` | train and validation splits, codec | `outputs/train/<dataset>/<model>/<run-id>/best.pt`, resolved config, W&B run | Optimizes on train, validates on fixed validation subsets, and selects the checkpoint by minimum DLS energy score. |
| Tune | `uv run python -m pipelines.tune checkpoint=/path/to/best.pt device=cpu` | checkpoint, validation split, codec, Declare model | `outputs/tune/<dataset>/<model>/<run-id>/<invocation-id>/tuning.json`, effective config | Searches temperature and top-p for SuTraN-PH and selects minimum DLS energy score. |
| Generate | `uv run python -m pipelines.generate checkpoint=/path/to/best.pt device=cpu num_samples=100` | generation-ready checkpoint, test split, codec | `outputs/generate/<dataset>/<model>/<run-id>/<invocation-id>/generations.parquet`, effective config | Draws suffixes for every test prefix with nested samples and provenance. |
| Evaluate | `uv run python -m pipelines.evaluate generations=/path/to/generations.parquet workers=4` | generations file, Declare model | `evaluation.json`, `prefix_scores.parquet` under the evaluation output | Scores Parquet row groups in worker processes and aggregates scores overall and by prefix and suffix length. |
| Visualize | `uv run python -m pipelines.visualize 'evaluations=[/a/evaluation.json,/b/evaluation.json]'` | reports and adjacent prefix score files | PDF figures and LaTeX tables under `outputs/visualize/` | Compares runs, performs paired case bootstrap analysis, and renders the registered catalogue. |

For SuTraN-PH, generation must use the checkpoint written by tuning:

```sh
uv run python -m pipelines.generate checkpoint=/path/to/tuned.pt \
  device=cpu num_samples=100
```

Visualization can discover reports recursively:

```sh
uv run python -m pipelines.visualize 'evaluations_dir=[outputs/evaluate,pinned]'
```

Hydra multirun accepts comma-separated values after `--multirun`. Each job gets a separate output
directory, but Hydra does not pass artifacts to the next stage. Supply every downstream path.

## Stage Invariants

- Call `start_stage` before stage work and validate the effective configuration before expensive
  reads or model execution.
- Store the fully resolved training configuration in checkpoints. For tuning and generation, save
  the source path, source hash, run identity, and effective runtime overrides.
- Require preprocessing artifacts through `src.artifacts.require_dataset_bundle` before work that
  depends on them.
- Keep split responsibilities separate. Tuning reads validation only; generation reads test only.
- Protect replacement of the best checkpoint atomically. Remove directly streamed Parquet outputs
  after handled write failures.
- Preserve `RunIdentity` and the checkpoint SHA-256 through tuning, generation, and evaluation.
- Give each tune, generate, and evaluate invocation a separate output directory below the training
  run. The source artifact still determines the training run identity.
- Generation batches may be sorted for efficiency, but prefix keys must align results across runs.

## Safe Local Development

Inspect a resolved configuration without executing its stage:

```sh
uv run python -m pipelines.train dataset=sepsis model=diffusion_transformer --cfg job --resolve
uv run python -m pipelines.preprocess dataset=sepsis --cfg job --resolve
```

Use focused CPU checks:

```sh
uv run pytest tests/models
uv run ruff check .
uv run ruff format --check .
```

When changing one stage, inspect small fixtures or existing artifact metadata. Tests must cover the
input contract, output contract, split selection, and provenance fields affected by the change.
