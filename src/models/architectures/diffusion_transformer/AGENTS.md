# Joint Diffusion Transformer

## Contract

The model learns the distribution of future activities and standardized inter-event times given an
observed prefix. Its fixed suffix canvas width is `codec.max_trace_length - 1`. The first generated
EOT determines the length. Dataset cuts always retain at least one real future event, so EOT is
forbidden at position zero in both training predictions and sampling. PAD and SOS are structural;
UNK and EOT are clean activity targets. MASK is a separate noisy suffix state and is never emitted.

`generate` may read `TraceCut.prefix` only. No sampling bound, denoiser input, or stopping decision
may use a true suffix field. The denoiser uses full bidirectional attention over clean prefix rows
and its own noisy or revealed suffix canvas.

## Clean Canvas and Forward Processes

For `n = suffix.length - 1` real events, clean activities contain the `n` real activities followed
by EOT through the end of the canvas. All positions receive activity supervision. Clean times
contain standardized inter-event times for real events and standardized zero afterward. Time loss
applies to real events only.

Training draws one level `t` uniformly from `1..diffusion.steps` per row. The activity process uses
an absorbing MASK state. At level `t`, each clean activity remains visible with probability
`alpha_t`, the cumulative cosine signal, or becomes MASK. At the terminal level `alpha_t = 0`, so
all activities are MASK. Do not force masking when a row happens to remain entirely visible.
The Gaussian channel corrupts time with the same sampled level and its own cosine schedule:
`y_t = sqrt(alpha_t) y_0 + sqrt(1 - alpha_t) epsilon`.

The suffix activity embedding has one row for each clean compact activity and a separate MASK row.
The activity prediction head has only clean compact activities. Prefix events use the shared event
content embedding.

## Loss

At a MASK position, a reverse jump from level `t` to level `s` reveals the clean activity with
probability `(alpha_s - alpha_t) / (1 - alpha_t)`. The activity loss uses that adjacent-step reveal
probability times clean-token cross entropy at MASK positions, multiplied by `diffusion.steps` for
uniform timestep sampling. Each row averages over the entire canvas, including trailing EOT
positions. A row with no MASK has zero activity loss. The Gaussian noise-prediction squared error
averages real event positions only. The batch loss averages row losses, and `Loss` fields store row
sums. There is no auxiliary categorical loss or uniform categorical posterior.

## Sampling

The sampler uses DDIM. `diffusion.sampler.calls` selects a descending grid from
`diffusion.sampler.start_level` to level one and a final jump to level zero. New runs use 50 calls
from level 99 across 100 noise levels, avoiding the near-zero terminal time signal. Checkpoints
without `start_level` start from the terminal level, preserving their original sampling behavior.
Activities start as all MASK and times start as standard Gaussian noise. At each jump, still-masked
positions reveal with the cumulative probability above and already revealed positions stay fixed.
The final jump reveals every MASK. The Gaussian channel uses a DDIM jump based on the cumulative
signal at both endpoints. `diffusion.sampler.eta` controls its stochasticity, with zero as the
deterministic default.

Initial EOT is forbidden when a masked activity is sampled. After sampling, the first EOT sets the
length; positions from EOT onward become PAD and standardized zero time. If no EOT appears,
`used_sentinel` is true and the length is the full canvas. Remaining time is derived from decoded,
nonnegative inter-event times. A residual MASK is an error. Generation metadata stores the
sampling and noise schedule configuration together with checkpoint provenance.

## Validation

For local verification of terminal masking, reveal probabilities, fixed visible states, EOT canvas
and loss masks, finite gradients, timestep grids, Gaussian endpoints, EOT truncation, sentinel
behavior, shape and length bounds, and prefix-only generation, follow the root guide's
disposable-check policy. Do not run training, sampler tuning, full generation, or full evaluation
locally. Select later sampling settings on validation data only.
