# Model Guide

All architectures implement `SuffixModel`, through one of two subclasses. A
`TrainableSuffixModel` is optimized by the training loop. A `FittedSuffixModel` is estimated once
from the train split without an optimizer. Training, validation, checkpoint restoration, and batch
generation depend on these interfaces rather than on architecture internals.

## Shared Interface

| Operation | Contract |
| --- | --- |
| `generate(item, num_samples=S)` | Read `item.prefix` only and return `GeneratedSuffix` with leading shape `[B, S]`. |
| `forward(item)` | Trainable only. Run the architecture's stochastic or teacher-forced training pass and return a `ModelOutput`. |
| `compute_loss(output, batch)` | Trainable only. Return a scalar mean loss for backpropagation and a `Loss` whose fields are sums over batch rows. |
| `fit(dataset)` | Fitted only. Estimate the model from every cut of the train split and keep everything it learns in persistent buffers. |
| `pad_activity_index` | Activity index ignored by reconstruction loss and used after generated termination. |
| `eot_activity_index` | Activity token that terminates a suffix. |

`GeneratedSuffix.activities` and `inter_event_times` have shape `[B, S, T]`; `lengths` and
`used_sentinel` have shape `[B, S]`. Length counts real generated
events before EOT, or the generated canvas length when no EOT appears.

`DecoderOutput` holds activity logits `[B, T, V]` and standardized time means and log-variances
`[B, T]`, each time conditioned on the activity at its position.
`DiffusionOutput` also carries the clean and noisy states, sampled timesteps, Gaussian noise, and
the real-event mask needed to evaluate one stochastic diffusion pass. Its activity and time losses
span the full fixed canvas, including trailing EOT targets with standardized zero time.
`UncertaintyAwareDecoderOutput` carries Gaussian activity-logit means and log-variances `[B, T, V]`
and standardized time means and log-variances `[B, T]`.

## Construction and Persistence

- `SuffixModel.from_config` resolves `model._target_` from the Hydra configuration. The class path
  is the only model identity: `src.models.architectures.architecture_of` reads the architecture
  package from it, and that name is the run identity model. A new architecture requires a package
  under `architectures/`, an entry in `ARCHITECTURES`, a Hydra configuration, validation, a
  visualization label, and shared contract support.
- Checkpoints contain the resolved run configuration, provenance, state dictionary, optimizer
  step, selection score, selection metric, and direction. `persistence.io.load_checkpoint` loads
  plain data and tensors on CPU; `persistence.validation` checks the stored model configuration,
  provenance, and run identity.
- `SuffixModel.from_checkpoint` rebuilds from the stored model configuration, loads weights, moves
  to the requested device, and returns evaluation mode. Do not reconstruct from a current YAML file.
- `src.artifacts` owns `RunIdentity` and `Provenance`.
- Diffusion checkpoints require the current prefix encoder configuration and an explicit sampler
  start level.
- Save the repeatedly replaced best checkpoint through a temporary `.pt.tmp` file.

## Shared Semantics

- Generation must be invariant to every true suffix field for all architectures.
- PAD and SOS are structural tokens and cannot be sampled as suffix activities. EOT determines
  generated length. UNK remains a valid modeled activity.
- Inter-event times are modeled in the codec's standardized space. Models do not produce remaining
  time; evaluation derives it by summing decoded, nonnegative inter-event minutes. Do not add an
  independent remaining-time head.
- `src/models/embeddings.py` holds event-content and positional embeddings used across model
  families. `src/models/backbones/autoregressive` holds the causal prefix encoder, decoder trunk,
  attention, cache, and loss normalization reused by the two SuTraN baselines. Neither location
  defines a selectable architecture. Model-specific heads, losses, and sampling policies stay in
  their architecture packages.
- Average position losses within each trace before averaging traces so long suffixes do not receive
  unintended batch weight.
- Report finite losses and generations and keep gradients finite for every trainable model.

## Architecture Routing

| Architecture | Guide | Status |
| --- | --- | --- |
| `diffusion_transformer` | [`architectures/diffusion_transformer/AGENTS.md`](architectures/diffusion_transformer/AGENTS.md) | Implemented main model |
| `head_sampling_transformer` | [`architectures/head_sampling_transformer/AGENTS.md`](architectures/head_sampling_transformer/AGENTS.md) | Implemented SuTraN-PH baseline |
| `u_ed_sutran` | [`architectures/u_ed_sutran/AGENTS.md`](architectures/u_ed_sutran/AGENTS.md) | Implemented uncertainty-aware SuTraN |
| `case_based` | `architectures/case_based/model.py` | Implemented non-neural case-based baseline |

For local verification, use small inputs and reduced diffusion steps under the root guide's
disposable-check policy. Do not start training.
