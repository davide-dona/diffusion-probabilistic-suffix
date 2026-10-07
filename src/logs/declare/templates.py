from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal


class Trace:
    """A finished trace, with where each of its activities occurs.

    - activities: One character per event, encoding the event's activity.
    - positions: Each activity that occurs, to its ascending positions in `activities`.
    """

    def __init__(self, activities: str) -> None:
        self.activities = activities
        self.positions: dict[str, list[int]] = {}
        for index, activity in enumerate(activities):
            self.positions.setdefault(activity, []).append(index)


@dataclass(frozen=True, slots=True)
class Constraint:
    """A template instantiated on one or two activities, written `Template(a, b)`.

    `first` is `a` and `second` is `b`, the names every template check is documented with.
    """

    template: '_Template'
    first: str
    # None for a unary template
    second: str | None
    # The count of a cardinality template, and 1 for the rest
    n: int

    def holds(self, trace: Trace, *, vacuity: bool) -> bool:
        """
        Whether a finished trace satisfies this constraint.

        Args:
            trace: The trace to check.
            vacuity: Whether a trace without the constraint's activation satisfies it.
        """
        activation = self.template.activation
        if vacuity and activation is not None:
            activity = self.first if activation == 'first' else self.second
            if activity not in trace.positions:
                return True
        return self.template.holds(self, trace)

    @property
    def activities(self) -> tuple[str, ...]:
        """The one or two activities the constraint is on, in template order."""
        return (self.first,) if self.second is None else (self.first, self.second)

    def relabel(self, labels: Mapping[str, str]) -> 'Constraint':
        """
        The same constraint on other labels for its activities.

        Raises:
            KeyError: If `labels` lacks one of its activities.
        """
        return replace(
            self,
            first=labels[self.first],
            second=None if self.second is None else labels[self.second],
        )


@dataclass(frozen=True, slots=True)
class _Template:
    """A Declare template: its check and the shape of its constraints."""

    name: str
    # The check of a trace that contains the activation; `Constraint.holds` applies vacuity
    holds: Callable[[Constraint, Trace], bool]
    is_binary: bool = True
    supports_cardinality: bool = False
    # Which of `first` and `second` is the activation, or None for a template without one, which
    # vacuity never affects
    activation: Literal['first', 'second'] | None = None


def _existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs at least `n` times."""
    return len(trace.positions.get(constraint.first, ())) >= constraint.n


def _absence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs fewer than `n` times."""
    return len(trace.positions.get(constraint.first, ())) < constraint.n


def _exactly(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs exactly `n` times."""
    return len(trace.positions.get(constraint.first, ())) == constraint.n


def _init(constraint: Constraint, trace: Trace) -> bool:
    """`a` is the first event of the trace."""
    return trace.activities.startswith(constraint.first)


def _end(constraint: Constraint, trace: Trace) -> bool:
    """`a` is the last event of the trace."""
    return trace.activities.endswith(constraint.first)


def _choice(constraint: Constraint, trace: Trace) -> bool:
    """`a` or `b` occurs."""
    return constraint.first in trace.positions or constraint.second in trace.positions


def _exclusive_choice(constraint: Constraint, trace: Trace) -> bool:
    """`a` or `b` occurs, and never both."""
    return (constraint.first in trace.positions) != (constraint.second in trace.positions)


def _responded_existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and so does `b`."""
    return constraint.first in trace.positions and constraint.second in trace.positions


def _not_responded_existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and `b` does not."""
    return constraint.first in trace.positions and constraint.second not in trace.positions


def _response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and every occurrence of it is followed by a `b`."""
    activations = trace.positions.get(constraint.first)
    targets = trace.positions.get(constraint.second)
    return bool(activations) and bool(targets) and activations[-1] < targets[-1]


def _precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and every occurrence of it is preceded by an `a`."""
    activations = trace.positions.get(constraint.second)
    targets = trace.positions.get(constraint.first)
    return bool(activations) and bool(targets) and targets[0] < activations[0]


def _not_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and no occurrence of it is followed by a `b`."""
    activations = trace.positions.get(constraint.first)
    targets = trace.positions.get(constraint.second)
    return bool(activations) and (not targets or activations[0] > targets[-1])


def _not_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and no occurrence of it is preceded by an `a`."""
    activations = trace.positions.get(constraint.second)
    targets = trace.positions.get(constraint.first)
    return bool(activations) and (not targets or targets[0] > activations[-1])


def _adjacent(constraint: Constraint) -> str:
    """`a` immediately followed by `b`, as it reads in `Trace.activities`.

    The two activities of a binary constraint differ, so its occurrences never overlap.
    """
    return constraint.first + constraint.second


def _chain_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` follows it immediately every time."""
    activations = trace.positions.get(constraint.first)
    return bool(activations) and trace.activities.count(_adjacent(constraint)) == len(activations)


def _chain_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` precedes it immediately every time."""
    activations = trace.positions.get(constraint.second)
    return bool(activations) and trace.activities.count(_adjacent(constraint)) == len(activations)


def _not_chain_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` never follows it immediately."""
    return constraint.first in trace.positions and _adjacent(constraint) not in trace.activities


def _not_chain_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` never precedes it immediately."""
    return constraint.second in trace.positions and _adjacent(constraint) not in trace.activities


def _alternate_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` follows each occurrence of it before `a` recurs."""
    pending = False
    for activity in trace.activities:
        if activity == constraint.first:
            if pending:
                return False
            pending = True
        elif activity == constraint.second:
            pending = False
    return constraint.first in trace.positions and not pending


def _alternate_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` precedes each occurrence of it since the previous `b`."""
    preceded = False
    for activity in trace.activities:
        if activity == constraint.first:
            preceded = True
        elif activity == constraint.second:
            if not preceded:
                return False
            preceded = False
    return constraint.second in trace.positions


# The minable templates, keyed by the name the model file stores.
TEMPLATES: dict[str, _Template] = {
    template.name: template
    for template in (
        _Template('Existence', _existence, is_binary=False, supports_cardinality=True),
        _Template('Absence', _absence, is_binary=False, supports_cardinality=True),
        _Template('Exactly', _exactly, is_binary=False, supports_cardinality=True),
        _Template('Init', _init, is_binary=False),
        _Template('End', _end, is_binary=False),
        _Template('Choice', _choice),
        _Template('Exclusive Choice', _exclusive_choice),
        _Template('Responded Existence', _responded_existence, activation='first'),
        _Template('Not Responded Existence', _not_responded_existence, activation='first'),
        _Template('Response', _response, activation='first'),
        _Template('Precedence', _precedence, activation='second'),
        _Template('Not Response', _not_response, activation='first'),
        _Template('Not Precedence', _not_precedence, activation='second'),
        _Template('Chain Response', _chain_response, activation='first'),
        _Template('Chain Precedence', _chain_precedence, activation='second'),
        _Template('Not Chain Response', _not_chain_response, activation='first'),
        _Template('Not Chain Precedence', _not_chain_precedence, activation='second'),
        _Template('Alternate Response', _alternate_response, activation='first'),
        _Template('Alternate Precedence', _alternate_precedence, activation='second'),
    )
}
