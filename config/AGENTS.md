# Hydra Configuration Guide

Hydra composes stage files at `config/*.yaml` with dataset, model, training, runtime, and output
groups. The result is validated at the stage boundary and stored with every durable run artifact.

## Groups

| Path | Contract |
| --- | --- |
| `dataset/*.yaml` | Raw columns, split fractions, filtering, features, scaling, and Declare discovery. |
| `model/*.yaml` | Architecture kind, public model name, dimensions, architecture parameters, and sampler where supported. |
| `training/default.yaml` | Optimizer, step schedules, validation cadence, early stopping, sampling counts, and per-dataset regimes. |
| `runtime/cuda.yaml` | Seed, per-dataset batch sizes, loader workers, device, and W&B settings. |
| `output/default.yaml` | Hydra run and sweep directories with `hydra.job.chdir: false`. |
| Stage YAML | Defaults composition, required inputs, runtime overrides, and output resolvers. |

## Rules

- Keep `model.kind` aligned with `src.models.factory.build_model` and `model.name` aligned with
  `RunIdentity`, output paths, and visualization labels.
- The diffusion model separates `diffusion.steps` noise levels from `diffusion.sampler.calls`
  denoiser calls. The sampler uses DDIM, calls must not exceed noise levels, and
  `diffusion.sampler.eta` lies in `[0, 1]`. The default is 100 levels, 50 calls, and zero Gaussian
  sampling stochasticity.
- Add or change fields together with their checks in `src/config_validation/`. Reject invalid values before
  reading large artifacts or starting model work.
- Keep dataset-specific training values under `training.regimes.<dataset>` and batch sizes under
  `dataloader.batch_sizes.<dataset>` so a dataset override selects a complete regime.
- Express training duration, warmup, and validation cadence in optimizer steps. Express early
  stopping patience in validation checks.
- Keep optional CLI overrides as `null` in stage configuration and apply them explicitly to the
  checkpoint configuration at runtime.
- Do not encode machine-specific absolute paths in committed YAML.
- Preserve `_self_` placement when composition order matters and use `# @package _global_` for
  groups that populate the root configuration.

## Safe Inspection and Tests

```sh
uv run python -m pipelines.train dataset=sepsis model=head_sampling_transformer --cfg job --resolve
uv run python -m pipelines.train dataset=sepsis model=diffusion_transformer --cfg job --resolve
uv run python -m pipelines.preprocess dataset=sepsis --cfg job --resolve
uv run ruff check config src/config_validation
```

Run the focused tests for the consumer of any changed field. Do not launch training merely to
validate configuration composition.
