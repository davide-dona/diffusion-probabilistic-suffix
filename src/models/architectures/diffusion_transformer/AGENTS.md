# Joint Diffusion Transformer

## Purpose and Notation

The model approximates the joint conditional distribution

\[
p_\theta(a_{1:L}, \tau_{1:L} \mid c),
\]

where `c` is an observed process prefix, `a` are future activities, `tau` are standardized
inter-event times, and the first EOT token determines the effective suffix length. The fixed canvas
width is `L = codec.max_trace_length - 1`.

This is a repository-specific joint model built from categorical diffusion in the style of
[D3PM](https://papers.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)
and Gaussian diffusion with the cosine schedule from
[Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html). Those works explain the base
processes; this file defines the implemented combination and is authoritative for code changes.

Use `B` for batch size, `T` for canvas width, `K` for the compact diffusion activity vocabulary,
`P` for padded prefix width, and `D` for model width.

## Clean Representation

The categorical process contains EOT, UNK, and observed activity rows. It excludes PAD and SOS.
`CategoricalDiffusion` owns stable maps between codec indices and this compact vocabulary.

For a suffix with `n` real future events:

- `clean_activity[:, :n]` contains the real activities.
- `clean_activity[:, n:]` contains EOT, not PAD.
- `clean_time[:, :n]` contains standardized inter-event times.
- `clean_time[:, n:]` contains the standardized value corresponding to zero minutes.
- `activity_mask` is true through the first EOT, at positions `0..n`.
- `time_mask` is true only for real events, at positions `0..n-1`.

The dataset suffix itself includes EOT, so `n = suffix.length - 1`. Activity loss includes one
termination target while time loss never charges an EOT or padded position.

## Forward Processes

A training row samples one timestep `t` uniformly from `1..S`, shared by all positions and both
channels. Let `beta_t` be the cosine schedule and
`alpha_bar_t = product_{s=1}^t (1 - beta_s)`.

For activities, uniform categorical corruption is

\[
q(x_t = j \mid x_0 = i) = \bar\alpha_t [j=i] + (1-\bar\alpha_t)/K.
\]

The final categorical beta is forced to one, so `x_S` is uniform and matches the generation prior.
`CategoricalDiffusion.posterior` computes the exact
`q(x_{t-1} | x_t, x_0)` for this transition process.

For standardized times,

\[
y_t = \sqrt{\bar\alpha_t} y_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon \sim \mathcal N(0,I).
\]

The time schedule keeps the ordinary clipped cosine endpoint. `GaussianDiffusion.sample_reverse`
uses the fixed posterior variance implied by that schedule.

## Conditional Denoiser

`DiffusionDenoiser` concatenates two segments and applies bidirectional Transformer encoder layers:

1. Clean prefix rows `[B, P, D]` embed activity, resource, standardized inter-event time,
   categorical attributes, numeric values, numeric-presence indicators, position, and prefix
   segment.
2. Noisy suffix rows `[B, T, D]` embed the compact activity through the shared activity table,
   noisy time through a linear projection, suffix position, suffix segment, and projected sinusoidal
   timestep.

Only padded prefix positions are masked. Every non-padded prefix and suffix position may attend to
every other non-padded position. Heads over the suffix rows predict clean-activity logits
`[B, T, K]` and Gaussian noise `[B, T]`.

## Objective

Let `pi_theta(x_0 | x_t, y_t, c, t)` be the activity softmax after removing EOT probability at
position zero. The reverse distribution marginalizes the exact categorical posterior over the
predicted clean state:

\[
p_\theta(x_{t-1}\mid x_t,c) =
\sum_{\hat x_0} q(x_{t-1}\mid x_t,\hat x_0)\,
\pi_\theta(\hat x_0\mid x_t,y_t,c,t).
\]

At `t = 1`, categorical loss is clean-state cross entropy. At later timesteps it is
`KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t,c))`. The auxiliary term is clean-state cross
entropy at every timestep. Time loss is squared error between sampled and predicted noise.

For each trace, after the masks above average positions separately,

\[
L = S L_{cat} + L_{aux} + L_{time}.
\]

`compute_loss` averages this trace loss across the batch. The logged reconstruction loss equals the
full loss; logged activity loss is `S L_cat + L_aux`; logged inter-event-time loss is `L_time`.

## Generation

Generation repeats each prefix `num_samples` times, initializes activities uniformly over `K` and
times from `N(0,I)`, then iterates `t = S..1`:

1. Predict clean-activity probabilities and time noise jointly.
2. Sample the exact categorical reverse transition, using the predicted clean distribution
   directly at `t = 1`.
3. Sample the Gaussian reverse transition, using its mean directly at `t = 1`.

At the final categorical step, EOT is prohibited at position zero, so every generated suffix has at
least one event. The first EOT gives the generated length. If no EOT appears, length is `T` and
`used_sentinel` is true. Positions from EOT onward become PAD activities and standardized zero
times. Remaining time is derived through `remaining_time_from_inter_event_times`.

Generation may read `TraceCut.prefix` only. Do not use the clean suffix to initialize, clamp, guide,
or score the reverse process.

## Configuration and Tests

`config/model/diffusion_transformer.yaml` defines `d_model`, event embedding widths, Transformer
depth, attention heads, feedforward width, dropout, diffusion steps, and separate cosine offsets.
`d_model` must be divisible by the number of attention heads. Both schedules must use the configured
number of steps.

Run these focused CPU tests after changes:

```sh
uv run pytest tests/models/test_diffusion_transformer.py
uv run pytest tests/models/test_contracts.py -k diffusion_transformer
uv run pytest tests/models/test_configuration.py -k diffusion_transformer
```

Keep tests for normalized posterior and reverse probabilities, initial EOT exclusion, exact activity
and time masks, finite loss and gradients, output shapes, event-feature use, and prefix-only
generation. Use reduced diffusion steps in tests and never launch training locally.
