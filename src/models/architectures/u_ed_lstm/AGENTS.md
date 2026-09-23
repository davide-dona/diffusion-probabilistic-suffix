# U-ED-LSTM: Planned Comparison Model

## Status

U-ED-LSTM is a planned architecture and an external comparison. This directory intentionally
contains documentation only. There is currently no model implementation, Hydra model configuration,
factory registration, checkpoint support, pipeline capability, or test parametrization for
`u_ed_lstm`. The existing visualization label does not imply runtime support.

Do not claim the model is available until all shared integration requirements below are implemented.

## Published Method

The intended reference is
[An Uncertainty-Aware ED-LSTM for Probabilistic Suffix Prediction](https://arxiv.org/abs/2505.21339)
and its
[official repository](https://github.com/ProbabilisticSuffixPredictionLab/probabilistic_suffix_prediction_U-ED-LSTM).
The paper defines probabilistic suffix prediction as approximating a distribution over complete
future activity and time sequences instead of returning one most likely continuation.

At method level, an encoder LSTM maps the observed prefix `c` to recurrent state. A decoder LSTM
then emits a suffix autoregressively until EOT. Its uncertainty model combines:

- Epistemic uncertainty through Monte Carlo dropout. Dropout remains active while drawing suffixes,
  so each stochastic forward trajectory corresponds to a sampled effective parameterization.
- Aleatoric uncertainty through learned loss attenuation. Predictive heads estimate observation
  uncertainty and use it to scale task residuals or likelihood terms during training.
- Monte Carlo suffix sampling. Repeated stochastic decoder trajectories approximate
  `p(activity suffix, time suffix | prefix, training data)`.

If `w` denotes model parameters and `D` the training data, the predictive distribution motivates
the approximation

\[
p(y\mid c,D) = \int p(y\mid c,w)p(w\mid D)\,dw
\approx \frac{1}{M}\sum_{m=1}^{M} p(y\mid c,w_m),
\]

where `w_m` is induced by an independent dropout realization. Learned attenuation commonly has the
form `exp(-s) * L_task + s`, so uncertain observations contribute less without allowing uncertainty
to grow without penalty. The future implementation must follow the exact published activity and
time likelihoods rather than treating this generic form as a complete specification.

## Requirements for Future Implementation

Before registering `u_ed_lstm`, make its design decision complete against the paper and reference
code, then satisfy all of the following:

- Implement `SuffixModel.forward`, `compute_loss`, and prefix-only `generate` with the shared tensor
  shapes from [`src/models/AGENTS.md`](../../AGENTS.md).
- Define exactly which recurrent connections use dropout during training and sampling, how one
  dropout realization persists through a decoded trajectory, and which sources of randomness
  distinguish samples.
- Define activity, inter-event-time, and uncertainty heads, their distributions, loss attenuation,
  target masks, normalization, and EOT semantics from the primary source.
- Derive remaining time from generated inter-event times through the shared codec path unless a
  deliberate interface change is made for every architecture.
- Add `config/model/u_ed_lstm.yaml`, configuration validation, a factory branch, checkpoint round
  trips, visualization identity, and model contract parametrization.
- Preserve chronological splits and use validation data for checkpoint and inference-setting
  selection. Generate final samples from test prefixes only.
- Add focused CPU tests for finite stochastic loss, finite gradients, prefix-only generation,
  output shapes, EOT and sentinel behavior, event-feature handling, checkpoint reload, and both
  epistemic and aleatoric stochasticity.

Any future implementation plan must state where it intentionally follows or differs from the paper.
Do not copy the external repository's data formats or notebook workflow into this architecture when
the shared dataset, checkpoint, and artifact contracts already provide the corresponding boundary.
