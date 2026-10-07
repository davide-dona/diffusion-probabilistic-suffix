from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

# Where each activity occurs in a trace, keyed by its character.
Positions = dict[str, list[int]]


def positions_of(trace: str) -> Positions:
    """Map each activity of a trace, one character per event, to its ascending positions."""
    positions: Positions = {}
    for index, activity in enumerate(trace):
        positions.setdefault(activity, []).append(index)
    return positions


@dataclass(frozen=True, slots=True)
class Constraint:
    """A template instantiated on one or two activities."""

    template: '_Template'
    first: str
    # None for a unary template
    second: str | None
    # The count of a cardinality template, and 1 for the rest
    n: int

    def holds(self, trace: str, positions: Positions, *, vacuity: bool) -> bool:
        """
        Whether a finished trace satisfies this constraint.

        Args:
            trace: One character per activity.
            positions: The trace's `positions_of`.
            vacuity: Whether a trace that never activates the constraint satisfies it.
        """
        activation = self.template.activation
        if vacuity and activation is not None:
            activator = self.first if activation == 'first' else self.second
            if activator not in positions:
                return True
        return self.template.holds(self, trace, positions)

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
    holds: Callable[[Constraint, str, Positions], bool]
    is_binary: bool
    supports_cardinality: bool
    # Which activity activates the constraint, or None if every trace does
    activation: Literal['first', 'second'] | None


def _existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs at least `n` times."""
    return len(positions.get(constraint.first, ())) >= constraint.n


def _absence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs fewer than `n` times."""
    return len(positions.get(constraint.first, ())) < constraint.n


def _exactly(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs exactly `n` times."""
    return len(positions.get(constraint.first, ())) == constraint.n


def _init(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` is the first event of the trace."""
    return bool(trace) and trace[0] == constraint.first


def _end(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` is the last event of the trace."""
    return bool(trace) and trace[-1] == constraint.first


def _choice(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` or `b` occurs."""
    return constraint.first in positions or constraint.second in positions


def _exclusive_choice(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` or `b` occurs, and never both."""
    return (constraint.first in positions) != (constraint.second in positions)


def _responded_existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and so does `b`."""
    return constraint.first in positions and constraint.second in positions


def _not_responded_existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and `b` does not."""
    return constraint.first in positions and constraint.second not in positions


def _response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and every occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and bool(targets) and activations[-1] < targets[-1]


def _precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and every occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and bool(earlier) and earlier[0] < activations[0]


def _not_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and no occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and (not targets or activations[0] > targets[-1])


def _not_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and no occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and (not earlier or earlier[0] > activations[-1])


def _chain_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` follows it immediately every time."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and all(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _chain_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` precedes it immediately every time."""
    activations = positions.get(constraint.second)
    return bool(activations) and all(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _not_chain_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` never follows it immediately."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and not any(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _not_chain_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` never precedes it immediately."""
    activations = positions.get(constraint.second)
    return bool(activations) and not any(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _alternate_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` follows each occurrence of it before `a` recurs."""
    activations = fulfillments = 0
    pending = False
    for activity in trace:
        if activity == constraint.first:
            pending = True
            activations += 1
        if pending and activity == constraint.second:
            pending = False
            fulfillments += 1
    return activations > 0 and activations == fulfillments


def _alternate_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` precedes each occurrence of it since the previous `b`."""
    activations = fulfillments = 0
    preceding = 0
    for activity in trace:
        if activity == constraint.first:
            preceding += 1
        if activity == constraint.second:
            activations += 1
            if preceding:
                fulfillments += 1
            preceding = 0
    return activations > 0 and activations == fulfillments


# The minable templates, keyed by the name the model file stores.
TEMPLATES: dict[str, _Template] = {
    template.name: template
    for template in (
        _Template(
            name='Existence',
            holds=_existence,
            is_binary=False,
            supports_cardinality=True,
            activation=None,
        ),
        _Template(
            name='Absence',
            holds=_absence,
            is_binary=False,
            supports_cardinality=True,
            activation=None,
        ),
        _Template(
            name='Exactly',
            holds=_exactly,
            is_binary=False,
            supports_cardinality=True,
            activation=None,
        ),
        _Template(
            name='Init', holds=_init, is_binary=False, supports_cardinality=False, activation=None
        ),
        _Template(
            name='End', holds=_end, is_binary=False, supports_cardinality=False, activation=None
        ),
        _Template(
            name='Choice',
            holds=_choice,
            is_binary=True,
            supports_cardinality=False,
            activation=None,
        ),
        _Template(
            name='Exclusive Choice',
            holds=_exclusive_choice,
            is_binary=True,
            supports_cardinality=False,
            activation=None,
        ),
        _Template(
            name='Responded Existence',
            holds=_responded_existence,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Not Responded Existence',
            holds=_not_responded_existence,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Response',
            holds=_response,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Precedence',
            holds=_precedence,
            is_binary=True,
            supports_cardinality=False,
            activation='second',
        ),
        _Template(
            name='Not Response',
            holds=_not_response,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Not Precedence',
            holds=_not_precedence,
            is_binary=True,
            supports_cardinality=False,
            activation='second',
        ),
        _Template(
            name='Chain Response',
            holds=_chain_response,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Chain Precedence',
            holds=_chain_precedence,
            is_binary=True,
            supports_cardinality=False,
            activation='second',
        ),
        _Template(
            name='Not Chain Response',
            holds=_not_chain_response,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Not Chain Precedence',
            holds=_not_chain_precedence,
            is_binary=True,
            supports_cardinality=False,
            activation='second',
        ),
        _Template(
            name='Alternate Response',
            holds=_alternate_response,
            is_binary=True,
            supports_cardinality=False,
            activation='first',
        ),
        _Template(
            name='Alternate Precedence',
            holds=_alternate_precedence,
            is_binary=True,
            supports_cardinality=False,
            activation='second',
        ),
    )
}
