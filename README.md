# Suffix Generation

Probabilistic suffix generation for predictive process monitoring, with a framework prepared for
diffusion models.

This repository provides a Head-sampling Transformer baseline together with preprocessing,
training, inference, evaluation, and visualization pipelines. Additional architectures can be
added through the shared model interface and configuration group.

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/).

The datasets under `data/` are tracked with Git LFS. Pull them before preprocessing:

```bash
git lfs install
git lfs pull
uv sync --locked
```

Training logs to [W&B](https://wandb.ai/). Sign in once per machine:

```bash
uv run wandb login
```

## Run the pipeline

The pipeline stages below run in sequence, each reading an explicit artifact written by the
previous stage. Hydra composes the YAML files under `config/` and accepts overrides directly on
the command line.

Every invocation records its resolved configuration beside the stage artifacts described below.

### Run multiple jobs

Hydra multirun executes the Cartesian product of comma-separated values, one job at a time.
Use it to process a batch at any pipeline stage:

```bash
uv run python -m pipelines.preprocess --multirun dataset=sepsis,bpic12,bpic17,bpic19

uv run python -m pipelines.train --multirun \
  dataset=sepsis,bpic12,bpic17,bpic19 \
  model=head_sampling_transformer
```

Generation and evaluation can be queued in the same way by listing their input artifacts:

```bash
uv run python -m pipelines.generate --multirun \
  checkpoint=/path/to/first.pt,/path/to/second.pt device=cpu num_samples=100

uv run python -m pipelines.evaluate --multirun \
  generations=/path/to/first/generations.parquet,/path/to/second/generations.parquet workers=4
```

Each multirun job receives its own output directory. Batch runs do not
transfer artifacts between stages automatically, so supply each stage's input artifact
explicitly.

### 1. Preprocessing

Run once per dataset:

```bash
uv run python -m pipelines.preprocess dataset=sepsis
```

The original log is read from `data/sepsis/original.csv`. The out-of-time splits, fitted codec,
declarative model, and dataset manifest are written under `data/sepsis/`. The manifest records the
hash of each bundle file and one fingerprint for the complete bundle. Invocation records, including
the resolved preprocessing configuration, are written under `outputs/preprocess/sepsis/<timestamp>/`.

> [!WARNING]
> Training, tuning, generation, and evaluation stop if the preprocessing manifest is missing or
> any dataset artifact differs from its recorded hash. Artifacts written before this format change
> must be regenerated through their pipeline stages.

Checkpoints, tuning reports, generations, and evaluation outputs carry the same `provenance`
record: the training run, dataset fingerprint, checkpoint hash, and immediate source file hash.
Hashes that do not yet apply are `null`. This lets each stage reject a dataset bundle different
from the one used for training.

### 2. Training

Choose the dataset and architecture independently:

```bash
uv run python -m pipelines.train dataset=sepsis model=head_sampling_transformer
```

Available model configs are `head_sampling_transformer` (SuTraN-PH), `u_ed_sutran`
(U-ED-SuTraN), `diffusion_transformer`, `diffusion_transformer_wide_shallow`, and
`diffusion_transformer_prefix_encoder`.
U-ED-SuTraN shares SuTraN-PH's encoder and causal
decoder, adding MC dropout and learned activity-logit and time variances. Its defaults use 20
categorical likelihood draws and log-variance bounds of `[-10, 10]`; these are configurable under
`model.uncertainty`. Both SuTraN models train on complete suffixes with activity-only decoder inputs.
Validation uses isolated seeded draws and selects checkpoints by the existing generation metric.
The diffusion model uses absorbing MASK activity corruption and Gaussian time noise. Its default
configuration has 1000 noise levels and 50 DDIM sampling calls starting at level 990. The activity
loss supervises real events and all EOT positions in the fixed suffix canvas. The
`diffusion_transformer_wide_shallow` model config uses width 48 and four layers instead of width
32 and eight layers, keeping its parameter count close to the other models. Compare sampler and
model settings on validation data before final test
generation. Older checkpoints without a sampler start level retain their configured terminal start.
The `diffusion_transformer_prefix_encoder` config uses four prefix encoder layers and four
bidirectional suffix decoder layers. The decoder cross-attends to the encoded event sequence, which
is cached across sampling calls and samples for each prefix. It retains the same activity and time
corruption, losses, and sampler settings as the joint baseline.
Train the stable width-32 baseline and both variants with the same dataset and seed:

```bash
uv run python -m pipelines.train dataset=sepsis model=diffusion_transformer
uv run python -m pipelines.train dataset=sepsis model=diffusion_transformer_wide_shallow
uv run python -m pipelines.train dataset=sepsis model=diffusion_transformer_prefix_encoder
```

Training writes the best validation checkpoint to
`outputs/train/<dataset>/<model>/<run-id>/best.pt`. Runs cannot be resumed, but an interrupted run
retains its last successfully saved best checkpoint.

Training curves are logged to the `diffusion-probabilistic-suffix` W&B project. On normal completion, the
selected checkpoint is also uploaded to W&B.
Diffusion runs also log `train/masked_real_activity_loss`, `train/masked_eot_activity_loss`, and
their `val/` counterparts. Together they equal the logged activity loss; each uses the full-canvas
denominator.

### 3. Sampler tuning

Tune a Head-sampling Transformer on the validation split before test generation:

```bash
uv run python -m pipelines.tune checkpoint=/path/to/best.pt device=cpu
```

The selected sampler and full search are written to
`outputs/tune/<dataset>/<model>/<training-run-id>/<invocation-id>/tuning.json`. The same directory
contains `tuned.pt`, a self-contained checkpoint required for Head-sampling Transformer generation.

U-ED-SuTraN samples its learned distribution directly. It does not support temperature or top-p
tuning, and its `best.pt` can be used for generation without this stage.

### 4. Inference

Generate suffixes for every prefix of the test split:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/checkpoint.pt device=cpu num_samples=100
```

For a Head-sampling Transformer, pass the checkpoint produced by sampler tuning:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/tuned.pt device=cpu num_samples=100
```

The generations are written to
`outputs/generate/<dataset>/<model>/<training-run-id>/<invocation-id>/generations.parquet`.

### 5. Evaluation

Evaluate a generations file:

```bash
uv run python -m pipelines.evaluate generations=/path/to/generations.parquet workers=4
```

The report and its per-prefix scores are written under
`outputs/evaluate/<dataset>/<model>/<training-run-id>/<invocation-id>/` as `evaluation.json` and
`prefix_scores.parquet`. The JSON summary groups scores under `scores.activity`,
`scores.suffix_length`, `scores.remaining_time`, `scores.inter_event_time`, and
`scores.conformance`, both overall and within each length bucket.

DLS sample mean and suffix-length MAE are validation diagnostics, logged to W&B under
`diagnostic_activity/dls_sample_mean` and `diagnostic_suffix_length/suffix_length_mae`.
They are excluded from final reports, score files, and publication comparisons.

Generation metrics are logged under `generation_<group>/<metric>`. Only model-owned metrics are
logged during training validation; log-owned values remain in the evaluation report.

### 6. Visualization

Plot and tabulate one or more evaluation reports:

```bash
uv run python -m pipelines.visualize \
  'evaluations=[/path/to/first/evaluation.json,/path/to/second/evaluation.json]'
```

Keep every `evaluation.json` beside its `prefix_scores.parquet`. Figures are written as PDF under
`outputs/visualize/<date>/<time>/figures/`, and comparison tables as LaTeX under
`outputs/visualize/<date>/<time>/tables/`.

To visualize every report below one or more directories instead:

```bash
uv run python -m pipelines.visualize 'evaluations_dir=[outputs/evaluate,pinned]'
```

## Published checkpoints

Download all published checkpoints to `pretrained/<dataset>/<model>.pt`:

```bash
uv run python -m scripts.fetch
```

## Configuration

Datasets, models, training defaults, and runtime profiles live in the corresponding groups under
`config/`. Training duration, warmup, and validation cadence are expressed in optimizer steps;
early stopping is expressed in validation checks. The CUDA profile selects a batch size and training
regime for each dataset automatically.
All model configs use activity/resource/attribute embedding widths of 32/16/8. The wider
diffusion variant projects these to width 48; the other configs project to width 32. Checkpoints
retain their own embedding configuration.
Override individual settings with dotted keys:

```bash
uv run python -m pipelines.train dataset=bpic17 model=head_sampling_transformer \
  optimizer.lr=0.0005 training.device=cuda:0
```

Inspect the fully resolved configuration without starting a run:

```bash
uv run python -m pipelines.train dataset=sepsis model=head_sampling_transformer --cfg job --resolve
```

## Maintainer operations

### Publish a checkpoint

Once a run has been evaluated, propose its generation-ready checkpoint as a published model. Use
`tuned.pt` for a Head-sampling Transformer and `best.pt` for a diffusion model:

```bash
uv run python -m scripts.publish -m /path/to/checkpoint.pt
```

This opens a pull request against the Hugging Face model repository.
