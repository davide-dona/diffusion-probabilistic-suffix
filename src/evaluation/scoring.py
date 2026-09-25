from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import MetricGroup
from src.evaluation.prepared import PreparedPrefix
from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True)
class ScoreGroups:
    """Report metric values for a prefix or an aggregate, grouped by evaluation question."""

    activity: dict[str, float]
    suffix_length: dict[str, float]
    remaining_time: dict[str, float]
    inter_event_time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, values: dict[str, float]) -> 'ScoreGroups':
        """Group a complete mapping of report metric values.

        Args:
            values: Exactly the report metric keys and their scalar values.

        Returns:
            Scores grouped by their registered evaluation question.

        Raises:
            ValueError: If report keys are missing or unregistered or diagnostic keys are present.
        """
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
        """Average complete score mappings, field by field.

        Args:
            values: Report score groups, each receiving equal weight.

        Returns:
            Mean report scores, or zero for every metric if the input is empty.
        """
        accumulator = _ScoreAccumulator()
        for scores in values:
            accumulator.add(scores.flatten())
        return accumulator.mean()

    def flatten(self) -> dict[str, float]:
        """Return all values in registry declaration order.

        Returns:
            A new mapping from each report metric key to its score.
        """
        groups = {
            MetricGroup.ACTIVITY: self.activity,
            MetricGroup.SUFFIX_LENGTH: self.suffix_length,
            MetricGroup.REMAINING_TIME: self.remaining_time,
            MetricGroup.INTER_EVENT_TIME: self.inter_event_time,
            MetricGroup.CONFORMANCE: self.conformance,
        }
        return {key: groups[metric.group][key] for key, metric in METRICS.report.items()}


@dataclass(frozen=True)
class PrefixSummary:
    """Scores for one generated prefix, returned by a worker.

    prefix_len counts observed prefix events; suffix_len counts true suffix events.
    Report scores and optional validation diagnostics are stored separately.
    """

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
        """Score one generated suffix against truth and constraints.

        Args:
            generation: One observed prefix with its sampled and true suffixes.
            checker: Declare checker for full traces formed from the prefix and each suffix.
            include_diagnostics: Whether to compute validation-only metrics alongside report scores.

        Returns:
            Report scores and sequence lengths for the prefix, with diagnostics stored separately.
        """
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
    """Mean scores for prefixes with one shared length.

    length identifies the prefix-length or true-suffix-length bucket; prefixes counts its
    members. Scores hold equally weighted means over those members.
    """

    length: int
    prefixes: int
    scores: ScoreGroups


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
            scores=scores,
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
    scores: ScoreGroups
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate prefix scores overall and by prefix and suffix length.

        Args:
            prefixes: Prefix summaries consumed once in input order.

        Returns:
            Equally weighted report means overall and by observed lengths, with buckets sorted
            in ascending order. Empty input gives zero overall scores and no buckets.
        """
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
            scores=scores,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )
