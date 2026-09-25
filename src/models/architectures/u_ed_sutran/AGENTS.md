# U-ED-SuTraN

## Purpose and Reference

`UEDSuTraN` is the implemented uncertainty-aware SuTraN baseline, registered as `u_ed_sutran`.
It estimates a distribution over complete future activity and inter-event-time sequences from an
observed prefix. Both uncertainty sources are required:

- Epistemic uncertainty through Monte Carlo dropout during suffix generation.
- Aleatoric uncertainty through learned activity-logit and time means and log-variances.

The uncertainty formulation follows
[An Uncertainty-Aware ED-LSTM for Probabilistic Suffix Prediction](https://arxiv.org/abs/2505.21339)
and its
[official repository](https://github.com/ProbabilisticSuffixPredictionLab/probabilistic_suffix_prediction_U-ED-LSTM).
The backbone and training structure follow the local SuTraN-PH model. This is a controlled
uncertainty-aware extension of SuTraN-PH, not a reproduction of every U-ED-LSTM design choice.
`u_ed_lstm` is not a registered runtime architecture.

Repeated stochastic trajectories approximate the predictive distribution over suffixes. Dropout
samples effective model parameterizations, while Gaussian predictive heads represent observation
uncertainty. Activity-logit variance remains part of this formulation even though categorical
sampling itself also represents ambiguity over activities. Use the likelihoods below rather than
substituting a generic loss-attenuation formula.

## Shared Backbone and Controlled Comparison

`UEDSuTraN` uses the shared SuTraN event embeddings, prefix encoder, causal decoder trunk,
attention, KV cache, and generation loop under `shared_components/sutran`. Prediction heads,
uncertainty losses, and MC dropout belong in this package.

- Keep structural defaults in `config/model/u_ed_sutran.yaml` equal to those in
  `config/model/head_sampling_transformer.yaml`: embedding sizes, encoder and decoder topology,
  widths, attention heads, feedforward sizes, dropout, and prediction-head widths.
- Preserve matching structural configuration fields. Capacity changes require a separate
  experiment rather than a silent change to the controlled comparison.
- Preserve `HeadSamplingTransformer` behavior, checkpoint restoration, and sampler tuning when
  editing shared components.
- Keep model-specific prediction heads and generation policies attached to shared hidden states.
- Only activities feed back into the decoder. Do not feed generated times, resources, or other
  attributes back into it. SOS opens decoding.
- Reuse existing datasets and fitted codecs. Do not import the reference repository's data formats
  or notebook workflow across the shared dataset, checkpoint, and artifact boundaries.

SuTraN-PH uses categorical activity logits and fixed unit time variance in standardized space.
U-ED-SuTraN learns both variances and enables inference-time dropout. It samples its learned
predictive distribution directly, without temperature, top-p, or sampler tuning.

## Full-Suffix Causal Training

Training uses one teacher-forced causal decoder pass over the complete padded suffix:

1. Encode the observed prefix.
2. Shift target activities behind SOS using the shared SuTraN operation.
3. Apply the configured teacher-forced activity dropout.
4. Run causal self-attention and prefix cross-attention once over all suffix positions.
5. Predict activity and time distribution parameters at every position.

Activity supervision includes real events and EOT, excluding PAD. Time supervision includes only
real events, excluding EOT and PAD. Positions after EOT contribute to neither loss.

Do not introduce five-event truncation, scheduled sampling, sampled decoder inputs, teacher-forcing
decay, detached training samples, or training-progress hooks on `SuffixModel`.

## Outputs and Likelihoods

Use the immutable `UncertaintyAwareDecoderOutput` in the shared `ModelOutput` union:

- Activity-logit means and log-variances have shape `[B, T, V]`.
- Standardized inter-event-time means and log-variances have shape `[B, T]`.

Clamp log-variances to configured finite bounds before exponentiation. Sampling uses standard
deviation `exp(0.5 * log_variance)` for both distributions.

For activity loss, draw Gaussian logits, apply log-softmax, and gather each target activity's
log probability. Compute the negative log of the Monte Carlo mean target probability using stable
log-mean-exp across the configured likelihood draws. PAD and SOS remain output vocabulary classes
for training but cannot be emitted during generation.

For time loss, use heteroscedastic Gaussian negative log likelihood without the constant:

\[
L_{time} = \frac{1}{2}\left(\exp(-s)(y-\mu)^2+s\right),
\]

where `mu` is the standardized time mean and `s` is bounded log-variance.

Sum valid activity and time terms within each trace and divide by `2 * suffix.length - 1`, then
average traces across the batch. Preserve the shared loss metrics and summed-per-row reporting
contract. Do not add GradNorm or independent task weighting.

Likelihood draws use a device-local advancing generator during training and a freshly seeded local
generator during validation. Validation generation uses the training runner's isolated seeded RNG
context and the existing generation-based selection metric. Repeated validation of an unchanged
checkpoint must be reproducible without depending on prior global random state.

## Monte Carlo Suffix Generation

Generation reads only `TraceCut.prefix`. Expand prefixes into `num_samples` independent rows before
encoding so each suffix receives its own encoder dropout draw. Enable MC dropout for every sample
row, including functional encoder and decoder attention dropout. Leave other evaluation behavior
unchanged and restore every module's prior mode on exit, including after failure.

At each cached autoregressive step:

1. Predict activity-logit and time means and bounded log-variances.
2. Draw Gaussian activity logits, mask PAD and SOS, apply softmax, and sample an activity.
3. Draw standardized inter-event time from its learned Gaussian distribution.
4. Feed the sampled activity into the next decoder step.
5. Stop each row at EOT or the configured generation cap.

Dropout remains stochastic at each encoder and decoder application. Expanded sample rows must have
independent masks and random draws; cached decoding must not share randomness across rows.

Return the shared `GeneratedSuffix` shapes from [the model guide](../../AGENTS.md). EOT and later
positions become PAD with zero standardized durations. Length counts retained events before EOT;
rows reaching the cap set `used_sentinel`. Derive remaining time from retained inter-event times
through inverse scaling, nonnegative clamping, summation, and remaining-time standardization.
Do not add an independently generated remaining-time head or expose true suffix fields to sampling.

## Integration and Provenance

Keep configuration validation, factory registration, exports, checkpoint restoration, visualization
labels, README documentation, and shared model contract parametrization consistent with
`u_ed_sutran`. Preserve the existing `DecoderOutput` contract for other architectures.

Preserve chronological splits, fitted codecs, seeds, resolved configurations, checkpoint hashes,
and run identity across pipeline handoffs. Checkpoint selection and any future inference-setting
comparisons use validation data only. Test prefixes are reserved for final generation and evaluation.

## Validation

Preserve these behaviors:

- Matching structural defaults and unchanged SuTraN-PH forward outputs, seeded generation, state
  dictionary keys, and checkpoint round trips after shared-component edits.
- One full-suffix causal training pass and cached/full decoding agreement without stochasticity.
- Activity and time masks, both log-variance bounds, finite losses, and finite gradients.
- Deterministic validation under its dedicated random generators.
- Epistemic variation with aleatoric draws fixed, aleatoric variation with dropout disabled, and
  independent dropout masks and draws across sample rows.
- Prefix-only generation, event-feature handling, output shapes, EOT termination, padding, sentinel
  behavior, remaining-time conversion, checkpoint reload, and mode restoration after failure.

Follow the root guide's disposable-check policy for local verification and inspect Hydra
configuration. Never launch training, sampler tuning, full test generation, or full evaluation
locally.

## Explicit Departures from U-ED-LSTM

- The local SuTraN Transformer replaces the recurrent encoder and decoder.
- Complete suffixes use parallel causal teacher forcing rather than a five-event autoregressive
  unroll or scheduled sampling.
- Activity-only feedback replaces generated event-attribute feedback.
- Equal per-target loss aggregation replaces GradNorm.
- SOS opens decoding instead of duplicating the final prefix event.
