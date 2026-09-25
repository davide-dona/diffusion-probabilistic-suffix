# Model Guide

All architectures implement `SuffixModel`. Training, validation, checkpoint restoration, and batch
generation depend on this interface rather than on architecture internals.

## Shared Interface

| Operation | Contract |
| --- | --- |
| `forward(item)` | Run the architecture's stochastic or teacher-forced training pass and return a `ModelOutput`. |
| `compute_loss(output, batch)` | Return a scalar mean loss for backpropagation and a `Loss` whose fields are sums over batch rows. |
| `generate(item, num_samples=S)` | Read `item.prefix` only and return `GeneratedSuffix` with leading shape `[B, S]`. |
| `pad_activity_index` | Activity index ignored by reconstruction loss and used after generated termination. |
| `eot_activity_index` | Activity token that terminates a suffix. |

`GeneratedSuffix.activities` and `inter_event_times` have shape `[B, S, T]`; `lengths`,
`remaining_time`, and optional `used_sentinel` have shape `[B, S]`. Length counts real generated
events before EOT, or the generated canvas length when no EOT appears.

`DecoderOutput` holds activity logits `[B, T, V]` and standardized time predictions `[B, T]`.
`DiffusionOutput` also carries the clean and noisy states, sampled timesteps, Gaussian noise, and
the real-event time mask needed to evaluate one stochastic diffusion pass. Its activity loss spans
the full fixed canvas, including trailing EOT targets.
`UncertaintyAwareDecoderOutput` carries Gaussian activity-logit means and log-variances `[B, T, V]`
and standardized time means and log-variances `[B, T]`.

## Construction and Persistence

- `build_model` is the only architecture selection point. A new `model.kind` requires a factory
  branch, Hydra model configuration, validation, visualization label, and shared contract tests.
- Checkpoints contain the resolved run configuration, provenance, state dictionary, optimizer
  step, selection score, selection metric, and direction. `load_checkpoint` uses safe globals,
  loads on CPU, validates required keys, and checks identity against configuration.
- `model_from_checkpoint` rebuilds from the stored model configuration, loads weights, moves to the
  requested device, and returns evaluation mode. Do not reconstruct from a current YAML file.
- Save the repeatedly replaced best checkpoint through a temporary `.pt.tmp` file. Checkpoints
  written once may be written directly to their final destination.

## Shared Semantics

- Generation must be invariant to every true suffix field. Keep
  `test_generation_reads_prefix_only` passing for all architectures.
- PAD and SOS are structural tokens and cannot be sampled as suffix activities. EOT determines
  generated length. UNK remains a valid modeled activity.
- Inter-event times are modeled in the codec's standardized space. Remaining time is derived from
  generated inter-event times after inverse scaling, nonnegative clamping, summation, and
  standardization through the remaining-time codec. It is not an independent generated head.
- Average position losses within each trace before averaging traces so long suffixes do not receive
  unintended batch weight.
- Report finite losses and generations and keep gradients finite for every trainable model.

## Architecture Routing

| Model kind | Guide | Status |
| --- | --- | --- |
| `diffusion_transformer` | [`architectures/diffusion_transformer/AGENTS.md`](architectures/diffusion_transformer/AGENTS.md) | Implemented main model |
| `head_sampling_transformer` | [`architectures/head_sampling_transformer/AGENTS.md`](architectures/head_sampling_transformer/AGENTS.md) | Implemented SuTraN-PH baseline |
| `u_ed_sutran` | [`architectures/u_ed_sutran/AGENTS.md`](architectures/u_ed_sutran/AGENTS.md) | Implemented uncertainty-aware SuTraN |

Run `uv run pytest tests/models` after model interface, architecture, checkpoint, or configuration
changes. Use the small fixtures and reduced diffusion steps in the tests. Do not start training.
