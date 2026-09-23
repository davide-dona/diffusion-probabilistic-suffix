# Evaluation Guide

Evaluation treats each generated prefix as one probabilistic forecast represented by sampled
suffixes. Metrics operate per prefix first, then reports average prefixes with equal weight.

## Package Layout

- `scoring.py` owns immutable score records, prefix scoring, and aggregation.
- `reports.py` owns JSON reports and the dataframe view used by visualization.
- `score_store.py` owns streaming Parquet writes, score readers, and adjacent-file discovery.
- `metrics/` owns preparation, ordered metric declarations, and numerical helpers.

Import shared operations from `src.evaluation`; metric declarations remain under
`src.evaluation.metrics`.

## Input and Preparation

`GenerationWriter` stores one Parquet row per `(case_id, prefix_len)`. Repeated activity suffixes
are folded into distinct strings plus draw indices; time sequences remain one per draw. The file
schema also stores truth, activity vocabulary, sampler settings, run identity, and checkpoint hash.

`PreparedPrefix` expands only the values needed by metrics: draw weights, activity similarities,
suffix lengths, aligned inter-event times, remaining times, and Declare conformance. Preserve draw
multiplicity when working with folded suffixes.

## Metric Semantics

| Group | Metrics | Preferred value |
| --- | --- | --- |
| Activity | DLS, exact-match, and bigram energy scores | Higher similarity; lower energy |
| Suffix length | CRPS, central interval coverage gaps at 50%, 75%, and 95% | Lower error and CRPS; gap closest to zero |
| Time | Remaining-time and aligned inter-event-time CRPS in days; central interval coverage gaps at 50%, 75%, and 95% | Lower CRPS; gap closest to zero |
| Conformance | Mean satisfied-constraint share, full-conformance sample rate, and observed-log references | Higher model conformance |

Energy scores combine distance to truth with a diversity correction between independent draws.
CRPS uses the same fair finite-sample energy estimator with absolute distances, calculated
efficiently through sorted samples. Coverage gap is empirical central-interval
coverage minus its nominal level. Inter-event metrics compare against the observed suffix width;
sampled time sequences are truncated or zero-padded to that width during preparation.

Declare checks evaluate the full trace formed by prefix plus suffix against constraints mined from
the training split. Observed conformance metrics belong to the log and have no model ranking
direction.

Metric registration order is part of report and Parquet column order. A new metric requires a
unique stable key, label, group, unit, owner, direction, compute function, visualization handling,
and compatibility consideration for old score files.

## Reports and Aggregation

`PrefixSummary` stores prefix length, true suffix length, and every report metric.
`EvaluationSummary` reports the unweighted mean over all prefixes and separate means bucketed by
prefix length and true suffix length. Aggregation consumes summaries once, retaining only metric
totals and counts overall and per length bucket. Each prefix is flattened once for aggregation;
summation follows input order and length buckets are emitted in ascending order. Empty aggregates
contain zero scores. `evaluation.json` contains this summary and provenance.
`prefix_scores.parquet` stores the prefix key, lengths, and all report metric values for paired analysis.

Keep the two files adjacent. Visualization rejects duplicate model reports within one dataset and
requires score files for significance analysis. Comparisons align identical prefix populations,
resample whole cases, use 10,000 paired bootstrap draws with seed 42, and apply Holm correction over
all model pairs for each dataset and ranked metric. Table emphasis means observed best or no detected
difference from it; it does not establish equivalence.

## Parallelism and Tests

Evaluation assigns Parquet row groups to worker processes. Each worker opens its own file and
Declare checker, returns compact `PrefixSummary` values, and preserves row-group order. Stream
scores to Parquet while aggregating so full decoded generations never accumulate in the parent.

Test metric formulas on small deterministic samples, folded-draw weighting, empty or degenerate
cases, schema rejection, prefix alignment, aggregation buckets, and reproducible significance
analysis. Do not run full evaluation locally.

## Diagnostics

Register validation-only metrics with `diagnostic=True`. DLS sample mean and suffix-length MAE
are diagnostics, logged as `diagnostic/<group>/<metric>` in W&B during training validation.
They are not computed during final evaluation and do not enter JSON reports, default Parquet
views, publication figures or tables, or significance comparisons. Historical score files may
contain diagnostic columns; readers require only report columns.
