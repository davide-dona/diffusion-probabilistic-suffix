# Diffusion activity self-conditioning experiment

## Hypothesis

Carrying the previous call's soft clean-activity probabilities at still-masked suffix positions
improves validation activity energy score by making activity and EOT interpretations more coherent.
The changed variable is self-conditioning. Revealed activities stay fixed, and both arms use the
same datasets, schedules, losses, sampler, training budgets, and validation samples.

## Run handoff

Use a suitable training machine. Do not run full training or generation locally. Use the current
processed bundles for Sepsis, BPIC12, BPIC17, and BPIC19 after verifying their manifests. Existing
diffusion checkpoints lack the required self-conditioning configuration, so train both arms anew.

```sh
uv sync --locked
uv run python -m pipelines.train --multirun \
  dataset=sepsis,bpic12,bpic17,bpic19 \
  model=diffusion_transformer,diffusion_transformer_self_conditioned \
  seed=42,43
```

Each job writes its best validation checkpoint under
`outputs/train/<dataset>/<model>/<run-id>/best.pt`. Record the checkpoint path, run ID, dataset
fingerprint, resolved configuration, selected step, and checkpoint hash for every job. Match arms
within each dataset and seed. Do not select a checkpoint or adjust sampling settings using test
prefixes.

## Analysis

Compare the selected checkpoints' validation `generation_activity/energy_score_dls` values. The
variant meets the primary criterion if it is lower on the same three or more datasets for both
seeds. Inspect activity DLS sample mean, exact and bigram energy scores, suffix-length and time
CRPS, coverage gaps, conformance, mean pairwise activity distance, and validation generation time.

Treat a secondary regression as material when it recurs on both seeds of a dataset and exceeds 5%
relative for CRPS, 0.05 absolute for coverage gap or conformance, or 10% relative for mean pairwise
activity distance. Check whether activity gains coincide with reduced diversity, changed EOT
behavior, or worse time calibration. If the primary criterion holds without material secondary
regressions, review the result before final test generation. If gains are isolated or inconsistent,
retain the baseline and record the observed tradeoffs.

After the validation decision, generate and evaluate selected checkpoints using their explicit
paths. The existing generation pipeline reads test prefixes, so it must not be used for selecting
the experimental arm.
