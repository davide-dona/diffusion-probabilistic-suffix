# Evaluation Guide

Evaluation treats each generated prefix as one probabilistic forecast represented by sampled
suffixes. Metrics operate per prefix first, then reports average prefixes with equal weight.

## Package Layout

- `scoring.py` owns immutable score records, prefix scoring, and aggregation.
- `reports.py` owns JSON reports and the dataframe view used by visualization.
- `score_store.py` owns streaming Parquet writes, score readers, and adjacent-file discovery.
- `prepared.py` owns the shared per-prefix arrays, conformance checks, signature Gram matrices,
  and PIT noise.
- `signature.py` owns the path embedding, the batched Goursat PDE solver of the signature kernel,
  and the train-split calibration of its time scale and bandwidth.
- `goursat.py` owns the Triton kernel that solves the Goursat PDE on CUDA, one lane per pair
  sweeping its grid row by row. It is imported only on CUDA, since Triton is unavailable elsewhere.
- `Draws.mean` owns the draw-weighted sample mean. `metrics/helpers.py` owns the
  distance-parametrized energy score, CRPS, its accuracy (MAE) and spread terms, and the
  randomized PIT.
- `metrics/definitions/activity.py` defines each activity distance inside its registered metric.
- `metrics/metadata.py` defines metric records, groups, display labels, units and bounds, owners,
  and ranking directions.
- `metrics/registry.py` owns ordered registration and report/diagnostic selection.
- `metrics/definitions/` contains the registered activity, conformance, joint, remaining-time,
  and suffix-length metrics. Importing `metrics` registers these groups in that order.

Import scoring and report operations from `src.evaluation`; metric definitions and their
numerical helpers remain under `src.evaluation.metrics`. Keep package initializers limited to
imports and exports.

Public operations use a concise summary followed by `Args`, `Returns` or `Yields`, and explicit
contract errors under `Raises` where applicable. Document array shapes, units, draw multiplicities,
and empty-input behavior where they affect the result; keep private documentation brief.

## Input and Preparation

`GenerationWriter` stores one Parquet row per `(case_id, prefix_len)`. Repeated activity suffixes
are folded into distinct strings plus draw records containing the suffix index, times, and sentinel
flag. The file also stores the prefix inter-event times, truth, activity vocabulary, sampler
settings, and provenance.
`Generations.block` decodes a row group column by column: a prefix's `Draws` holds its draw
times as one flat float64 array, a view of the block's, rather than one object per draw.

`PreparedPrefix.of_batch` expands shared values: suffix lengths, remaining times, Declare
conformance, the normalized signature Gram matrix of the draws with the observed suffix last, and
one PIT noise uniform. It checks every full trace of a batch in one checker call, and solves every
signature kernel of a batch in one solver call, since both have a fixed cost per call that
outweighs their work per item; score batches rather than single prefixes. Remaining
time is the sum of the decoded inter-event minutes, `Draws.remaining_time_minutes` for draws and
`DecodedEvents.remaining_time_minutes` for truth; the generations file does not store it. The validation-only DLS similarity is computed only when requested.
Preserve draw multiplicity when working with folded suffixes.

## Metric Semantics

| Group | Metrics | Preferred value |
| --- | --- | --- |
| Joint | Signature kernel score over activities and inter-event times | Lower score |
| Activity | DLS, exact-match, and bigram energy scores | Higher similarity; lower energy |
| Suffix length | CRPS | Lower CRPS |
| Remaining time | Remaining-time CRPS in days | Lower CRPS |
| Conformance | Full-conformance sample rate and its observed-log reference | Higher model conformance |

Energy scores combine distance to truth with a diversity correction between independent draws.
CRPS uses the same fair finite-sample energy estimator with absolute distances, integrated over
gaps between sorted samples with nonnegative weights. Never compute it as accuracy minus spread:
extreme draws make both totals huge and their difference is floating-point noise.
Inter-event times have no metric of their own: a positional score would align draws to the
observed suffix length, which depends on the realization and is not proper. The joint signature
kernel score covers them together with the activities.

The signature kernel score is the fair kernel score `1/2 E k(X, X') - E k(X, y) + 1/2` under the
cosine-normalized signature kernel `k`, with an RBF static kernel of bandwidth `sigma`. A suffix
is embedded as a path through `[tau, activity counts]`, where each event first advances
`tau = lambda * sum log1p(minutes)` and then steps its activity count. Preprocessing fits both
constants on the train split, from the suffix after each training case's first event, the longest
suffix a cut can produce: `lambda` is the inverse mean `log1p` of their inter-event minutes, and
`sigma` the exact median distance over pairs of different points of all their paths, selected by
radix passes over float64 bit patterns on `device`. The PDE is solved with dyadic
order `DYADIC_ORDER`, on CUDA by the Triton kernel or else on the CPU one anti-diagonal at a time;
both run the same float32 scheme and agree up to rounding. Identical paths of a prefix are solved
once. The
signature kernel score selects checkpoints.

Declare checks evaluate the full trace formed by prefix plus suffix against constraints mined from
the training split by the FCFDeclare library, which `src/logs/declare.py` wraps. A constraint
is kept when its non-vacuous support and its confidence reach `min_support` and `min_confidence`.
The checker counts a trace without a constraint's activation as satisfying it. Unary and choice
templates have no activation, so their confidence equals their support. Observed conformance
metrics belong to the log and have no model ranking direction. Training validation logs only model-owned report and diagnostic metrics to W&B.
Log-owned metrics are still computed during validation and remain in report scores, JSON reports,
and Parquet columns; ownership filtering applies only to metric logging.

Metric registration order is part of report and Parquet column order. A new metric requires a
unique stable key, label, group, optional publication label, display unit and bounds, owner,
direction, compute function, and visualization handling.

## Reports and Aggregation

`EvaluationReport` holds provenance and the summary.
`PrefixSummary` stores prefix length, true suffix length, grouped report scores, and diagnostics.
`EvaluationSummary` reports the unweighted mean over all prefixes and separate means bucketed by
prefix length and true suffix length. Each summary holds one grouped `scores` record and one flat
`diagnostics` record. Aggregation consumes summaries once, retaining only metric totals and counts
overall and per length bucket.
Each prefix is flattened once for aggregation; summation follows input order and length buckets
are emitted in ascending order. Empty aggregates contain zero scores. `evaluation.json` contains
this summary and provenance.
`prefix_scores.parquet` stores the prefix key, lengths, all report metric values for paired
analysis, and every diagnostic value. Its default reader returns the report columns only.

Keep the two files adjacent. `read_reports` holds the report scores and the diagnostics.
Visualization rejects duplicate model reports within one dataset and requires score files for
significance analysis and PIT histograms. A table with a reference model reports the skill
score `1 - mean / reference mean` of each negatively oriented metric and leaves the reference row
out; the skill table uses the `unconditional` model and needs its report for every dataset.
Ranking by skill equals ranking by the mean score, so significance tests run on the means. Comparisons align identical prefix populations,
resample whole cases, use 10,000 paired bootstrap draws with seed 42, and apply Holm correction over
all model pairs for each dataset and ranked metric. Table emphasis means observed best or no detected
difference from it; it does not establish equivalence.

## Parallelism

Evaluation assigns Parquet row groups to spawned worker processes, so each can open its own CUDA
context on the shared `device`. Each worker opens its own file, Declare checker, and signature
kernel, scores each row group as one batch, returns compact `PrefixSummary` values, and
preserves row-group order. Stream
scores to Parquet while aggregating so full decoded generations never accumulate in the parent.

Metric formulas must handle folded-draw weighting, empty or degenerate cases, schema rejection,
prefix alignment, and aggregation buckets. Significance analysis must be reproducible. Do not run
full evaluation locally.

## Diagnostics

Register debugging metrics with `diagnostic=True`. They are logged as `diagnostic_<group>/<metric>`
in W&B during training validation, and final evaluation stores them in the `diagnostics` records
of `evaluation.json` and in `prefix_scores.parquet`. They do not enter default Parquet views or
significance comparisons, never select a checkpoint, and appear in publication figures and tables
only without emphasis.

| Group | Diagnostics |
| --- | --- |
| Activity | DLS sample mean; DLD spread, the mean DLD between draws; spread error gap, that spread minus the mean DLD to truth, negative when overconfident; unique suffix ratio, unique activity suffixes per draw; best-of-5 and best-of-10 DLS, the expected highest DLS among 5 or 10 draws |
| Suffix length | Signed error, sampled minus true length; truncated draw rate, the share of draws closed at the length cap without generating EOT; truth quantile; CRPS error and spread terms |
| Remaining time | Log error, the signed difference of `log1p` days, sampled minus true; truth quantile; CRPS error and spread terms |
| Conformance | Mean satisfied-constraint share of the draws and of the observed trace; their signed bias, sampled minus observed |

Pairwise quantities retain the multiplicity of repeated sampled suffixes. Every diagnostic is
linear in its per-prefix values, so prefix means compose across buckets; the signed errors and
biases can cancel overall and are read per length bucket.

Best-of-k DLS sorts the K draw similarities ascending and weights the i-th by
`C(i - 1, k - 1) / C(K, k)`, the probability that it is the best of a random k-subset. This is
unbiased for any `k <= K`, so validation and evaluation agree despite different draw counts; fewer
than k draws is a contract error. It rewards spreading draws, so it never ranks.

The truth quantile of each CRPS-scored quantity is its randomized PIT
`(#draws below truth + V * (#draws equal to truth + 1)) / (K + 1)` with `V ~ U[0, 1]` from a
fixed seed, so integer lengths and zero remaining times do not skew it. Its histogram, pooled
over prefixes, is read from the per-prefix values in `prefix_scores.parquet`; calibration never
ranks. The CRPS error term is the draw mean of `|x - y|` and the spread term is half the mean
`|x - x'|` over pairs of different draws; their difference is the CRPS, but report each term on
its own and keep computing the CRPS itself with the sorted-sample estimator.

`SuffixModel.generation_diagnostics` returns a model's own per-prefix values, empty by default.
Training validation averages them over the generation prefixes and logs them as
`diagnostic_model/<key>`. The case-based model reports its retrieval: matched depth, its share of
the prefix, dropped prefix activities, exact and fallback match rates, log10 candidate count, and
single-candidate rate.
