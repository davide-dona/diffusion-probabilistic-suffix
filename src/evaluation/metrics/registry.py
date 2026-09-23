from collections.abc import Callable

from src.evaluation.metrics.definitions import Direction, Metric, MetricGroup, Owner, Unit


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
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Register a prefix-scoring function under a stable metric key."""

        def decorator(compute: Callable[..., float]) -> Callable[..., float]:
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
        """Return reportable metrics in declaration order."""
        return {key: metric for key, metric in self.entries.items() if not metric.diagnostic}

    @property
    def diagnostics(self) -> dict[str, Metric]:
        """Return validation-only metrics in declaration order."""
        return {key: metric for key, metric in self.entries.items() if metric.diagnostic}

    def __getitem__(self, key: str) -> Metric:
        if key not in self.entries:
            raise ValueError(f'no evaluation metric is registered as {key!r}.')
        return self.entries[key]


METRICS = MetricRegistry()
