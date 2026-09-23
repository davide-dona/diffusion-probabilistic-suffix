# SuTraN-PH: Head-Sampling Transformer

## Purpose and Relation to SuTraN

The implemented `head_sampling_transformer` is the probabilistic SuTraN-PH baseline. It follows the
encoder-decoder suffix modeling direction of
[SuTraN](https://ieeexplore.ieee.org/document/10680671/) and samples the repository's activity and
time heads autoregressively. The [reference implementation](https://github.com/BrechtWts/SuffixTransformerNetwork)
provides background for SuTraN. This guide describes the local model exactly; do not assume a
component from the paper exists here unless it appears below.

For observed prefix `c`, the model factorizes a suffix with real length `n` and terminal EOT as

\[
p_\theta(a_{1:n}, \mathrm{EOT}, \tau_{1:n}\mid c)
= \prod_{t=1}^{n+1} p_\theta(a_t\mid a_{<t},c)
  \prod_{t=1}^{n} p_\theta(\tau_t\mid a_{<t},c).
\]

The implementation samples activities categorically and standardized inter-event times from a
unit-variance Gaussian centered on the time head. Remaining time is derived from sampled durations.

Use `B` for batch size, `P` for padded prefix width, `T` for suffix width, `V` for the activity
vocabulary, `S` for samples per prefix, and `D` for model width.

## Event Representation and Prefix Encoder

`EventContentEmbedding` concatenates activity and resource embeddings, standardized inter-event
time, embedded categorical attributes, standardized numeric attributes, and numeric-presence
indicators, then projects them to `D`. `EventEmbeddings` adds fixed sinusoidal positions. The same
content and position modules are shared by encoder and decoder, with separate input layer norms.

`TraceEncoder` prepends a learned CLS token and applies pre-norm Transformer encoder layers with
full self-attention. PAD prefix positions are masked. It returns:

- `summary`: the encoded CLS row `[B, D]`, currently not consumed by the decoder.
- `events`: encoded prefix rows `[B, P, D]`, used as cross-attention memory.

Do not silently start using the CLS summary or remove it without updating the architecture contract
and checkpoint compatibility.

## Autoregressive Decoder

Training shifts ground-truth suffix activities right behind SOS. Activity dropout independently
replaces eligible teacher-forced activity inputs with PAD during training. EOT remains a target and
the decoder never receives future target activities.

Each pre-norm decoder layer applies:

1. Causal self-attention over suffix inputs.
2. Cross-attention over encoded prefix events with the prefix PAD mask.
3. A ReLU feedforward block.

A shared hidden layer feeds an activity-logit head `[B, T, V]` and a standardized time-mean head
`[B, T]`. The decoder input supplies generated activities; model-unknown decoder event channels use
their neutral or padding representation rather than predicted resources or attributes.

Incremental generation preprojects prefix keys and values once and preallocates a suffix KV cache
for every layer. A cached one-position pass must match the corresponding rows of an uncached causal
teacher-forced pass in evaluation mode.

## Objective

Activity loss is cross entropy over every real suffix activity and EOT, ignoring PAD. Time loss is
squared error over real events only, which is Gaussian negative log likelihood up to constants for
fixed unit variance. If `m = suffix.length` includes EOT, the number of supervised scalar targets is
`2m - 1`.

For each trace,

\[
L_{act} = \sum_{t=1}^{m} -\log p_\theta(a_t), \qquad
L_{time} = \sum_{t=1}^{m-1}(\hat\tau_t-\tau_t)^2,
\]

\[
L = \frac{L_{act}+L_{time}}{2m-1}.
\]

`compute_loss` averages `L` across batch rows. Logged activity, time, and reconstruction fields are
the corresponding normalized values summed across rows.

## Probabilistic Generation

The prefix is encoded once, then repeated into `B * S` independent rows. Generation starts with SOS
and repeatedly:

1. Runs one cached decoder step.
2. Masks PAD and SOS activity logits.
3. Applies temperature, softmax, and top-p nucleus truncation.
4. Samples the next activity from the resulting categorical distribution.
5. Samples standardized time as `mean + epsilon`, with `epsilon ~ N(0,1)`.

A row finishes when it samples EOT. Its length excludes EOT, and downstream decoding reads only
positions before that length. Raw activity and time tensors retain EOT and subsequent sampled
values; those durations do not contribute to remaining time. Unfinished rows stop at the maximum
decoder steps, which equals the batch's padded prefix width. This baseline leaves `used_sentinel`
unset. Preserve these raw tensor semantics for compatibility.

Generation may read `TraceCut.prefix` only. Never use the true suffix for teacher forcing or stopping
during sampling.

## Sampler Tuning

`temperature > 0` rescales logits. `0 < top_p <= 1` keeps the smallest descending-probability set
whose cumulative mass reaches the threshold and renormalizes it. Masked structural tokens remain
impossible under both controls.

`pipelines.tune` evaluates the Cartesian grid of temperatures and top-p values on one seeded fixed
validation subset. Every point uses the same number of prefixes, samples, seed, checkpoint, codec,
and Declare model. It selects the minimum activity DLS energy score. A tuning report is valid for
generation only when both `RunIdentity` and checkpoint SHA-256 match.

## Configuration and Tests

The encoder, embeddings, attention, causal decoder trunk, cache, and generation loop live in
`shared_components/sutran` and are shared with U-ED-SuTraN. The local decoder owns baseline heads
and sampler controls. Keep parameter registration paths and initialization order stable.

`config/model/head_sampling_transformer.yaml` defines model and embedding widths, encoder and
decoder depth, attention heads, feedforward sizes, dropout, teacher-forced activity dropout, shared
head width, temperature, and top-p. Model width must be divisible by both attention head counts.

Run:

```sh
uv run pytest tests/models/test_head_sampling_transformer.py
uv run pytest tests/models/test_contracts.py -k head_sampling_transformer
uv run pytest tests/models/test_configuration.py -k head_sampling_transformer
```

Keep tests for cached versus full decoding, finite loss and gradients, structural token masking,
termination and sentinel behavior, output shapes, event-feature use, and prefix-only generation.
Never launch training or a full tuning grid locally.
