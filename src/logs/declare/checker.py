from dataclasses import dataclass
from functools import lru_cache

from src.datasets.codec import ActivityCodec
from src.logs.declare.model import DeclareModel
from src.logs.declare.templates import Trace


@dataclass(frozen=True, slots=True)
class Conformance:
    """How one trace scores against the Declare model mined from its dataset.
    - satisfied: How many constraints the trace satisfies.
    - total: How many constraints there are in the model.
    """

    satisfied: int
    total: int

    @property
    def share(self) -> float:
        """The fraction of constraints the trace satisfies, in `[0, 1]`"""
        return self.satisfied / self.total if self.total else 0.0

    @property
    def full(self) -> float:
        """1.0 if the trace satisfies every constraint and 0.0 otherwise.
        A mean over traces is the share of them that are conformant."""
        return float(self.total > 0 and self.satisfied == self.total)


class ConformanceChecker:
    """Checks encoded traces against a Declare model."""

    def __init__(self, model: DeclareModel, codes: ActivityCodec) -> None:
        """
        Args:
            model: The model to check against, under the vacuity it was mined with.
            codes: The codebook the traces are encoded with.

        Raises:
            ValueError: If the codebook lacks an activity the model names, which means the two
                belong to different datasets.
        """
        self._vacuity: bool = model.settings.consider_vacuity
        # Map the constraints labels to the codebook's characters
        try:
            self._constraints = tuple(
                constraint.relabel(codes.codes) for constraint in model.constraints
            )
        except KeyError as error:
            raise ValueError(
                f'The codebook lacks the activity {error.args[0]!r} of the Declare model.'
            ) from error

    @lru_cache(maxsize=100_000)  # noqa: B019 -- one checker per scoring process
    def check(self, trace: str) -> Conformance:
        """
        Check one trace against every constraint.

        Args:
            trace: The whole case, prefix included, one character per event.
        Returns:
            The satisfied and total constraint counts.
        """
        case = Trace(trace)
        satisfied = sum(
            constraint.holds(case, vacuity=self._vacuity) for constraint in self._constraints
        )
        return Conformance(satisfied=satisfied, total=len(self._constraints))
