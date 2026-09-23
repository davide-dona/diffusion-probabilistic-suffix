from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from src.evaluation.metrics import METRICS, PreparedPrefix
from src.evaluation.metrics.definitions import MetricGroup
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True)
class ScoreGroups:
    """Report metric values for a prefix or an aggregate, grouped by evaluation question."""

    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, values: dict[str, float]) -> 'ScoreGroups':
        """Group a complete mapping of report metric values."""
        metrics = METRICS.report
        expected = set(metrics)
        missing = expected - set(values)
        extra = set(values) - expected
        if missing or extra:
            raise ValueError(
                'metric values differ from the registry: '
                f'missing {sorted(missing)}, extra {sorted(extra)}.'
            )
        grouped: dict[str, dict[str, float]] = {group: {} for group in MetricGroup}
        for key, metric in metrics.items():
            grouped[metric.group][key] = values[key]
        return cls(**grouped)

    @classmethod
    def mean(cls, values: Sequence['ScoreGroups']) -> 'ScoreGroups':
        """Average complete score mappings, field by field."""
        accumulator = _ScoreAccumulator()
        for scores in values:
            accumulator.add(scores.flatten())
        return accumulator.mean()

    def flatten(self) -> dict[str, float]:
        """Return all values in registry declaration order."""
        groups = {
            MetricGroup.ACTIVITY: self.activity,
            MetricGroup.SUFFIX_LENGTH: self.suffix_length,
            MetricGroup.TIME: self.time,
            MetricGroup.CONFORMANCE: self.conformance,
        }
        return {key: groups[metric.group][key] for key, metric in METRICS.report.items()}


@dataclass(frozen=True)
class PrefixSummary:
    """Scores for one generated prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    scores: ScoreGroups
    diagnostics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        generation: Generation,
        *,
        checker: ConformanceChecker,
        include_diagnostics: bool = False,
    ) -> 'PrefixSummary':
        """Score one generated suffix against truth and constraints."""
        context = PreparedPrefix.of(generation, checker=checker)
        scores = {key: metric.compute(context) for key, metric in METRICS.report.items()}
        diagnostics = {}
        if include_diagnostics:
            diagnostics = {
                key: metric.compute(context) for key, metric in METRICS.diagnostics.items()
            }
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            scores=ScoreGroups.of(scores),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> 'LengthSummary':
        """Aggregate scores for prefixes with a common length."""
        accumulator = _ScoreAccumulator()
        for prefix in prefixes:
            accumulator.add(prefix.scores.flatten())
        return accumulator.length_summary(length)


@dataclass
class _ScoreAccumulator:
    """Metric totals for an equally weighted population of prefixes."""

    totals: dict[str, float] = field(default_factory=lambda: dict.fromkeys(METRICS.report, 0.0))
    count: int = 0

    def add(self, values: Mapping[str, float]) -> None:
        for key in self.totals:
            self.totals[key] += values[key]
        self.count += 1

    def mean(self) -> ScoreGroups:
        if self.count == 0:
            return ScoreGroups.of(self.totals.copy())
        return ScoreGroups.of({key: total / self.count for key, total in self.totals.items()})

    def length_summary(self, length: int) -> LengthSummary:
        scores = self.mean()
        return LengthSummary(
            length=length,
            prefixes=self.count,
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
        )


def _add_to_bucket(
    buckets: dict[int, _ScoreAccumulator], length: int, values: Mapping[str, float]
) -> None:
    if length not in buckets:
        buckets[length] = _ScoreAccumulator()
    buckets[length].add(values)


def _by_length(buckets: dict[int, _ScoreAccumulator]) -> list[LengthSummary]:
    """Summarize length buckets in ascending order."""
    return [buckets[length].length_summary(length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate prefix scores overall and by prefix and suffix length."""
        overall = _ScoreAccumulator()
        prefix_buckets: dict[int, _ScoreAccumulator] = {}
        suffix_buckets: dict[int, _ScoreAccumulator] = {}
        for prefix in prefixes:
            values = prefix.scores.flatten()
            overall.add(values)
            _add_to_bucket(prefix_buckets, prefix.prefix_len, values)
            _add_to_bucket(suffix_buckets, prefix.suffix_len, values)
        scores = overall.mean()
        return cls(
            prefixes=overall.count,
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


type Summarized = PrefixSummary | LengthSummary | EvaluationSummary


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's grouped scores into a registry-ordered mapping."""
    if isinstance(summary, PrefixSummary):
        return summary.scores.flatten()
    return ScoreGroups(
        activity=summary.activity,
        suffix_length=summary.suffix_length,
        time=summary.time,
        conformance=summary.conformance,
    ).flatten()
