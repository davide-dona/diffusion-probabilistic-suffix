# Research Objective and Models

Develop a diffusion model for probabilistic suffix prediction in predictive process monitoring:
given an observed process prefix, predict a distribution of future activities and inter-event times.
Compare against SuTraN-PH and U-ED-LSTM on Sepsis, BPIC13, BPIC17, and BPIC19.

## Joint Diffusion Transformer (Planned)

Jointly denoises categorical activities and Gaussian time noise using a shared bidirectional
Transformer conditioned on the clean prefix. EOT determines termination; remaining time is
the sum of generated durations.

Read the [diffusion design](../docs/diffusion-model-design.md) before implementation.
It specifies intended behavior; the architecture is not yet registered in the model factory.

## SuTraN-PH (Implemented Baseline)

`head_sampling_transformer` encodes the prefix and generates activities autoregressively
from categorical heads, with a Gaussian inter-event-time head. Temperature and nucleus sampling can be
tuned on the validation split.

Currently predicts remaining time separately; the diffusion design specifies removing that head
and deriving remaining time from generated durations through shared decoding.
Implementation: `src/model/architectures/head_sampling_transformer.py`.

## U-ED-LSTM (External Comparison)

An encoder-decoder LSTM using Monte Carlo dropout and learned loss attenuation to sample
uncertain suffixes. See the [U-ED-LSTM paper](https://arxiv.org/abs/2505.21339).

The `u_ed_lstm` visualization label exists in `src/visualization/labels/models.py`;
its implementation and training configuration are not included in this repository.
