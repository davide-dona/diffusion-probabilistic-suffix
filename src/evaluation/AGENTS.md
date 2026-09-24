# Evaluation Guide

Evaluation treats each generated prefix as one probabilistic forecast represented by sampled
suffixes. Metrics operate per prefix first, then reports average prefixes with equal weight.

## Package Layout

- `scoring.py` owns immutable score records, prefix scoring, and aggregation.
- `reports.py` owns JSON reports and the dataframe view used by visualization.
- `score_store.py` owns streaming Parquet writes, score readers, and adjacent-file discovery.
- `prepared.py` owns the shared per-prefix arrays and conformance checks.
- `metrics/helpers.py` owns the draw-weighted sample mean, distance-parametrized energy score,
  CRPS, MAE, and coverage gap.
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
are folded into distinct strings plus draw indices; time sequences remain one per draw. The file
schema also stores truth, activity vocabulary, sampler settings, run identity, and checkpoint hash.

`PreparedPrefix` expands shared values: suffix lengths, aligned inter-event times, remaining times,
and Declare conformance. The validation-only DLS similarity is computed only when requested.
Preserve draw multiplicity when working with folded suffixes.

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
direction. Training validation logs only model-owned report and diagnostic metrics to W&B.
Log-owned metrics are still computed during validation and remain in report scores, JSON reports,
and Parquet columns; ownership filtering applies only to metric logging.

Metric registration order is part of report and Parquet column order. A new metric requires a
unique stable key, label, group, optional publication label, display unit and bounds, owner,
direction, compute function, and visualization handling.

## Reports and Aggregation

`PrefixSummary` stores prefix length, true suffix length, and grouped report scores.
`EvaluationSummary` reports the unweighted mean over all prefixes and separate means bucketed by
prefix length and true suffix length. Each summary holds one grouped `scores` record. Aggregation
consumes summaries once, retaining only metric totals and counts overall and per length bucket.
Each prefix is flattened once for aggregation; summation follows input order and length buckets
are emitted in ascending order. Empty aggregates contain zero scores. `evaluation.json` contains
this summary and provenance.
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
are diagnostics, logged as `diagnostic-<group>/<metric>` in W&B during training validation.
They are not computed during final evaluation and do not enter JSON reports, default Parquet
views, publication figures or tables, or significance comparisons.
