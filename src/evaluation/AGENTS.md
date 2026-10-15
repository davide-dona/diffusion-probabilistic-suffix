# Evaluation Guide

Evaluation treats each generated prefix as one probabilistic forecast represented by sampled
suffixes. Metrics operate per prefix first, then reports average prefixes with equal weight.

## Package Layout

- `scoring.py` owns immutable score records, prefix scoring, and aggregation.
- `reports.py` owns JSON reports and the dataframe view used by visualization.
- `score_store.py` owns streaming Parquet writes, score readers, and adjacent-file discovery.
- `prepared.py` owns the shared per-prefix arrays and conformance checks.
- `Draws.mean` owns the draw-weighted sample mean. `metrics/helpers.py` owns the
  distance-parametrized energy score, CRPS, MAE, and coverage gap.
- `metrics/definitions/activity.py` defines each activity distance inside its registered metric.
- `metrics/metadata.py` defines metric records, groups, display labels, units and bounds, owners,
  and ranking directions.
- `metrics/registry.py` owns ordered registration and report/diagnostic selection.
- `metrics/definitions/` contains the registered activity, conformance, suffix-length, and time
  metrics. Importing `metrics` registers these groups in that order.

Import scoring and report operations from `src.evaluation`; metric definitions and their
numerical helpers remain under `src.evaluation.metrics`. Keep package initializers limited to
imports and exports.

Public operations use a concise summary followed by `Args`, `Returns` or `Yields`, and explicit
contract errors under `Raises` where applicable. Document array shapes, units, draw multiplicities,
and empty-input behavior where they affect the result; keep private documentation brief.

## Input and Preparation

`GenerationWriter` stores one Parquet row per `(case_id, prefix_len)`. Repeated activity suffixes
are folded into distinct strings plus draw records containing the suffix index, times, and sentinel
flag. The file also stores truth, activity vocabulary, sampler settings, and provenance.
`Generations.block` decodes a row group column by column: a prefix's `Draws` holds its draw
times as one flat float64 array, a view of the block's, rather than one object per draw.

`PreparedPrefix.of_batch` expands shared values: suffix lengths, aligned inter-event times,
remaining times, the central-interval bounds at each `COVERAGE_LEVELS` level, and Declare
conformance. It checks every full trace of a batch in one checker call, since the checker's fixed
cost per call outweighs its work per trace; score batches rather than single prefixes. Remaining
time is the sum of the decoded inter-event minutes, `Draws.remaining_time_minutes` for draws and
`DecodedEvents.remaining_time_minutes` for truth; the generations file does not store it. The validation-only DLS similarity is computed only when requested.
Preserve draw multiplicity when working with folded suffixes.

## Metric Semantics

| Group | Metrics | Preferred value |
| --- | --- | --- |
| Activity | DLS, exact-match, and bigram energy scores | Higher similarity; lower energy |
| Suffix length | CRPS, central interval coverage gaps at 50%, 75%, and 95% | Lower error and CRPS; gap closest to zero |
| Remaining time | Remaining-time CRPS in days; central interval coverage gaps at 50%, 75%, and 95% | Lower CRPS; gap closest to zero |
| Inter-event time | Aligned inter-event-time CRPS in days; central interval coverage gaps at 50%, 75%, and 95% | Lower CRPS; gap closest to zero |
| Conformance | Mean satisfied-constraint share, full-conformance sample rate, and observed-log references | Higher model conformance |

Energy scores combine distance to truth with a diversity correction between independent draws.
CRPS uses the same fair finite-sample energy estimator with absolute distances, integrated over
gaps between sorted samples with nonnegative weights. Never compute it as accuracy minus spread:
extreme draws make both totals huge and their difference is floating-point noise. Coverage gap is empirical central-interval
coverage minus its nominal level. Inter-event metrics compare against the observed suffix width;
sampled time sequences are truncated or zero-padded to that width during preparation.

Declare checks evaluate the full trace formed by prefix plus suffix against constraints mined from
the training split by the FCFDeclare library, which `src/logs/declare.py` wraps. The checker
reads `vacuity` from the model's `settings`: a model mined with vacuity counts a trace without a constraint's activation as satisfying it. Unary and
choice templates have no activation, so vacuity never affects them. Observed conformance metrics belong to the
log and have no model ranking direction. Training validation logs only model-owned report and diagnostic metrics to W&B.
Log-owned metrics are still computed during validation and remain in report scores, JSON reports,
and Parquet columns; ownership filtering applies only to metric logging.

Metric registration order is part of report and Parquet column order. A new metric requires a
unique stable key, label, group, optional publication label, display unit and bounds, owner,
direction, compute function, and visualization handling.

## Reports and Aggregation

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

Keep the two files adjacent. Visualization rejects duplicate model reports within one dataset and
requires score files for significance analysis. Comparisons align identical prefix populations,
resample whole cases, use 10,000 paired bootstrap draws with seed 42, and apply Holm correction over
all model pairs for each dataset and ranked metric. Table emphasis means observed best or no detected
difference from it; it does not establish equivalence.

## Parallelism

Evaluation assigns Parquet row groups to worker processes. Each worker opens its own file and
Declare checker, scores each row group as one batch, returns compact `PrefixSummary` values, and
preserves row-group order. Stream
scores to Parquet while aggregating so full decoded generations never accumulate in the parent.

Metric formulas must handle folded-draw weighting, empty or degenerate cases, schema rejection,
prefix alignment, and aggregation buckets. Significance analysis must be reproducible. Do not run
full evaluation locally.

## Diagnostics

Register debugging metrics with `diagnostic=True`. They are logged as `diagnostic_<group>/<metric>`
in W&B during training validation, and final evaluation stores them in the `diagnostics` records
of `evaluation.json` and in `prefix_scores.parquet`. They do not enter default Parquet views,
publication figures or tables, or significance comparisons, and never select a checkpoint.

| Group | Diagnostics |
| --- | --- |
| Activity | DLS sample mean; mean pairwise DLD between draws; spread skill gap, that pairwise DLD minus the mean DLD to truth, negative when overconfident; distinct suffix share; truth hit rate, whether any draw reproduces the true activities |
| Suffix length | MAE; signed bias, sampled minus true length; EOT sentinel rate |
| Remaining time | Signed bias of `log1p` days, sampled minus true |
| Conformance | Signed bias of the satisfied-constraint share, sampled minus observed |

Pairwise quantities retain the multiplicity of repeated sampled suffixes. Every diagnostic is
linear in its per-prefix values, so prefix means compose across buckets; the signed biases can
cancel overall and are read per length bucket.

`SuffixModel.generation_diagnostics` returns a model's own per-prefix values, empty by default.
Training validation averages them over the generation prefixes and logs them as
`diagnostic_model/<key>`. The case-based model reports its retrieval: matched depth, its share of
the prefix, dropped prefix activities, exact and fallback match rates, log10 candidate count, and
single-candidate rate.
