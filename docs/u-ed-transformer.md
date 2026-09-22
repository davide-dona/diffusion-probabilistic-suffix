# U-ED-Transformer implementation plan

## Summary

Add `u_ed_transformer`, an uncertainty-aware encoder-decoder Transformer for
probabilistic suffix prediction. It preserves the U-ED-LSTM method's Monte Carlo
dropout, learned aleatoric uncertainty, autoregressive five-step scheduled-sampling
training, and Monte Carlo suffix sampling, while replacing recurrent encoder and
decoder cells with Transformer attention.

The design follows the [U-ED-LSTM paper](https://arxiv.org/abs/2505.21339) and its
[reference repository](https://github.com/ProbabilisticSuffixPredictionLab/probabilistic_suffix_prediction_U-ED-LSTM).
No preprocessing change is needed: `ts_start`, weekday, and time-of-day features
are already computed and configured for every dataset.

## Implementation changes

- Add `config/model/u_ed_transformer.yaml` with a four-layer, 128-wide encoder and
  decoder Transformer, eight attention heads, 0.1 MC-dropout probability, five-step
  decoder unroll, and paper-faithful teacher-forcing schedule: 0.8 initially,
  linearly decaying to 0 from 20% through 100% of training steps.
- Register `u_ed_transformer` in model validation, the factory, exports, checkpoint
  restoration, configuration tests, README architecture list, and visualization
  labels. It remains independent from the existing head-sampling Transformer so
  neither baseline changes behavior.
- Implement a dedicated architecture under
  `src/models/architectures/u_ed_transformer/`:
  - The encoder reads the full existing prefix representation: activity, resource,
    inter-event time, configured event attributes, case elapsed time, and cyclic
    calendar features.
  - The causal decoder cross-attends to encoded prefix events and autoregressively
    reads only the previous activity and inter-event time. Its initial input is the
    final prefix event, matching the U-ED-LSTM decoder contract.
  - Training unrolls up to five future events. At each step it uses the ground-truth
    prior event with the scheduled teacher-forcing probability, otherwise a detached
    draw from its own predictive distribution.
  - Generation continues until EOT or the existing maximum trace length, returns the
    shared `GeneratedSuffix` format, excludes EOT from event durations, and derives
    remaining time through the existing shared conversion.
- Add an uncertainty-aware model output type with:
  - Activity logits and a per-class activity log-variance.
  - Inter-event-time mean and log-variance.
  - Bounded log-variances before exponentiation for finite, stable losses and
    sampling.
- Train with learned loss attenuation:
  - Activity loss uses the target class's predicted log-variance to attenuate
    cross-entropy.
  - Time loss uses heteroscedastic Gaussian negative log-likelihood over valid
    non-EOT events.
  - Preserve paper-style two-task GradNorm balancing between activity and time
    losses, using the shared decoder trunk as the gradient reference and recording
    the balanced losses in existing training metrics.
- Add an optional training-progress hook to `SuffixModel`, implemented as a no-op by
  existing architectures and called by the training loop before each forward pass.
  U-ED-Transformer uses it only to compute scheduled teacher forcing.
- Use a dedicated MC-dropout mode during `generate`: dropout is active for every
  sampled suffix while the rest of the model stays in evaluation behavior. For each
  decoding step, perturb activity logits by their learned Gaussian variance before
  categorical sampling and draw time from its learned Gaussian. Do not add
  temperature or nucleus tuning.

## Five-step scheduled sampling

The paper-faithful training path predicts a short five-event horizon
autoregressively:

1. Seed the decoder with the final observed prefix event.
2. Predict the next suffix event.
3. For the next decoder input, choose the ground-truth prior event with probability
   `p`, or a detached sample from the prediction with probability `1 - p`.
4. Continue through at most five suffix events, masking positions after EOT.
5. Start with `p = 0.8`. From 20% of the configured optimizer steps, linearly decay
   `p` to zero by the final step.

This limits the exposure bias caused by training only on correct prior events. It is
deliberately sequential, therefore it needs five decoder calls per training batch and
only directly supervises the next five events of each prefix. Long suffix generation
remains fully autoregressive.

## Full single-pass causal alternative

A Transformer-native alternative shifts the complete target suffix once and scores
all suffix positions concurrently under a causal attention mask. Each position reads
only true earlier suffix events during training, never later ones.

| Property | Five-step scheduled sampling | Full causal single pass |
| --- | --- | --- |
| Training horizon | Next five events | Entire suffix |
| Decoder execution | Sequential | Parallel across positions |
| Prior decoder inputs | Mix of targets and samples | Targets only |
| Exposure to own mistakes | Explicitly trained | Only encountered at generation |
| Compute efficiency | Lower | Higher |
| Fidelity to U-ED-LSTM | High | Lower |

Full causal training is faster and provides supervision at every suffix position,
which is especially attractive for a Transformer. Its main drawback is teacher-forcing
mismatch: it conditions on ground-truth history in training but on sampled history in
generation, allowing early errors to compound in long suffixes.

A future hybrid variant can train with full causal passes for most optimization steps,
then fine-tune with scheduled sampling. This is intentionally out of scope for the
initial paper-faithful implementation.

## Test plan

- Extend configuration, factory, checkpoint, contract, and feature-embedding
  parameterizations to include `u_ed_transformer`.
- Verify teacher forcing starts from the final prefix event, honors the configured
  schedule, and limits training prediction targets to five valid suffix positions.
- Verify causal cached decoding matches full causal decoding with dropout disabled.
- Verify EOT termination, output shapes, padding, remaining-time conversion, and
  repeated MC samples through the shared generation contract.
- Verify activity and time attenuation losses mask padding correctly, remain finite at
  extreme bounded log-variances, and expose the existing scale-loss metrics.
- Verify MC-dropout generation produces distinct draws from a fixed checkpoint while
  deterministic validation forward passes remain reproducible with dropout disabled.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and the model test suite,
  followed by a small CPU train, checkpoint reload, generation, and evaluation smoke
  test on Sepsis.

## Assumptions

- "U" means full uncertainty awareness: both epistemic uncertainty via MC dropout
  and aleatoric uncertainty via learned output scales.
- The paper-faithful decoder predicts only activity and inter-event time; resources
  and other attributes condition the prefix encoder but are not generated.
- Existing preprocessing artifacts remain valid because the paper's required
  elapsed-time and calendar inputs are already present in every dataset configuration.
- Sampling behavior is intentionally limited to the learned predictive distribution,
  without post-hoc temperature or top-p controls.
