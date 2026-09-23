from collections.abc import Callable

from src.evaluation.metrics.metadata import (
    Direction,
    Metric,
    MetricCompute,
    MetricGroup,
    Owner,
    Unit,
)


class MetricRegistry:
    """Ordered declarations of every evaluation metric."""

    def __init__(self) -> None:
        self.entries: dict[str, Metric] = {}

    def register(
        self,
        key: str,
        *,
        label: str,
        group: MetricGroup,
        unit: Unit,
        direction: Direction = Direction.NONE,
        owner: Owner = Owner.MODEL,
        diagnostic: bool = False,
    ) -> Callable[[MetricCompute], MetricCompute]:
        """Register a prefix-scoring function under a stable metric key.

        Args:
            key: Unique identifier used in score mappings and artifact columns.
            label: Human-readable metric name.
            group: Evaluation question used to group scores.
            unit: Physical unit and display bounds.
            direction: Preferred value when comparing models.
            owner: Whether the value belongs to a model or the observed log.
            diagnostic: Whether to exclude the metric from final reports and score files.

        Returns:
            Decorator that records metadata in declaration order and returns the function.

        Raises:
            ValueError: When the decorator encounters an already registered key.
        """

        def decorator(compute: MetricCompute) -> MetricCompute:
            if key in self.entries:
                raise ValueError(f'metric {key!r} is registered more than once.')
            self.entries[key] = Metric(
                key=key,
                label=label,
                group=group,
                unit=unit,
                direction=direction,
                owner=owner,
                diagnostic=diagnostic,
                compute=compute,
            )
            return compute

        return decorator

    @property
    def report(self) -> dict[str, Metric]:
        """Return reportable metrics in declaration order.

        Returns:
            A new mapping containing only metrics without the diagnostic flag.
        """
        return {key: metric for key, metric in self.entries.items() if not metric.diagnostic}

    @property
    def diagnostics(self) -> dict[str, Metric]:
        """Return validation-only metrics in declaration order.

        Returns:
            A new mapping containing only metrics with the diagnostic flag.
        """
        return {key: metric for key, metric in self.entries.items() if metric.diagnostic}

    def __getitem__(self, key: str) -> Metric:
        """Look up a registered metric.

        Args:
            key: Stable metric identifier, including diagnostic identifiers.

        Returns:
            The registered metadata and compute function.

        Raises:
            ValueError: If no metric has this key.
        """
        if key not in self.entries:
            raise ValueError(f'no evaluation metric is registered as {key!r}.')
        return self.entries[key]


METRICS = MetricRegistry()
