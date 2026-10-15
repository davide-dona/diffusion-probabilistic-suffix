from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from fcfdeclare import Checker

from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import MetricGroup
from src.evaluation.prepared import PreparedPrefix
from src.inference.generation import Generation


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
    Report scores and diagnostics are stored separately.
    """

    prefix_len: int
    suffix_len: int
    scores: ScoreGroups
    diagnostics: dict[str, float]

    @classmethod
    def of_batch(
        cls, generations: Sequence[Generation], *, checker: Checker
    ) -> list['PrefixSummary']:
        """Score a batch of generated suffixes against truth and constraints.

        Args:
            generations: Observed prefixes with their sampled and true suffixes.
            checker: Declare checker for full traces formed from each prefix and its suffixes.

        Returns:
            Report scores, diagnostics, and sequence lengths for each prefix in input order.
            Empty input gives an empty list.
        """
        report = METRICS.report
        diagnostic = METRICS.diagnostics
        return [
            cls(
                prefix_len=context.generation.prefix_len,
                suffix_len=len(context.generation.truth),
                scores=ScoreGroups.of(
                    {key: metric.compute(context) for key, metric in report.items()}
                ),
                diagnostics={key: metric.compute(context) for key, metric in diagnostic.items()},
            )
            for context in PreparedPrefix.of_batch(generations, checker=checker)
        ]


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length.

    length identifies the prefix-length or true-suffix-length bucket; prefixes counts its
    members. Scores and diagnostics hold equally weighted means over those members.
    """

    length: int
    prefixes: int
    scores: ScoreGroups
    diagnostics: dict[str, float]


@dataclass
class _ScoreAccumulator:
    """Report and diagnostic metric totals for an equally weighted population of prefixes."""

    totals: dict[str, float] = field(default_factory=lambda: dict.fromkeys(METRICS.entries, 0.0))
    count: int = 0

    def add(self, values: Mapping[str, float]) -> None:
        for key in self.totals:
            self.totals[key] += values[key]
        self.count += 1

    def mean(self) -> tuple[ScoreGroups, dict[str, float]]:
        """Return mean report scores and mean diagnostics, zero for an empty population."""
        means = {key: total / max(self.count, 1) for key, total in self.totals.items()}
        return (
            ScoreGroups.of({key: means[key] for key in METRICS.report}),
            {key: means[key] for key in METRICS.diagnostics},
        )

    def length_summary(self, length: int) -> LengthSummary:
        scores, diagnostics = self.mean()
        return LengthSummary(
            length=length, prefixes=self.count, scores=scores, diagnostics=diagnostics
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
    """Aggregate evaluation scores and diagnostics for one run."""

    prefixes: int
    scores: ScoreGroups
    diagnostics: dict[str, float]
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate prefix scores overall and by prefix and suffix length.

        Args:
            prefixes: Prefix summaries consumed once in input order.

        Returns:
            Equally weighted report and diagnostic means overall and by observed lengths, with
            buckets sorted in ascending order. Empty input gives zero overall values and no
            buckets.
        """
        overall = _ScoreAccumulator()
        prefix_buckets: dict[int, _ScoreAccumulator] = {}
        suffix_buckets: dict[int, _ScoreAccumulator] = {}
        for prefix in prefixes:
            values = prefix.scores.flatten() | prefix.diagnostics
            overall.add(values)
            _add_to_bucket(prefix_buckets, prefix.prefix_len, values)
            _add_to_bucket(suffix_buckets, prefix.suffix_len, values)
        scores, diagnostics = overall.mean()
        return cls(
            prefixes=overall.count,
            scores=scores,
            diagnostics=diagnostics,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )
