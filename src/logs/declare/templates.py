from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal


class Trace:
    """A finished trace, with where each of its activities occurs.

    - activities: One character per event, encoding the event's activity.
    - occurrences: Each activity that occurs, to its ascending indices in `activities`.
    """

    def __init__(self, activities: str) -> None:
        self.activities = activities
        self.occurrences: dict[str, list[int]] = {}
        for index, activity in enumerate(activities):
            self.occurrences.setdefault(activity, []).append(index)


@dataclass(frozen=True, slots=True)
class Constraint:
    """A template instantiated on one or two activities, written `Template(a, b)`."""

    template: '_Template'
    a: str
    # None for a unary template
    b: str | None
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
        # If vacuity is considered, a trace without the activation satisfies the constraint
        if vacuity and activation is not None:
            activity = self.a if activation == 'a' else self.b
            if activity not in trace.occurrences:
                return True
        return self.template.holds(self, trace)

    @property
    def activities(self) -> tuple[str, ...]:
        """The one or two activities the constraint is on, in template order."""
        return (self.a,) if self.b is None else (self.a, self.b)

    def relabel(self, labels: Mapping[str, str]) -> 'Constraint':
        """
        The same constraint with each activity replaced by its entry in `labels`.

        Templates are checked on encoded traces, so a constraint must name its activities with
        the characters of the trace's codebook (`ActivityCodec.codes`), while a saved model names
        them by their activity names (`ActivityCodec.names`).

        Raises:
            KeyError: If `labels` lacks one of its activities.
        """
        return replace(
            self,
            a=labels[self.a],
            b=None if self.b is None else labels[self.b],
        )


@dataclass(frozen=True, slots=True)
class _Template:
    """A Declare template: its check and the shape of its constraints."""

    name: str
    # The check of a trace that contains the activation; `Constraint.holds` applies vacuity
    holds: Callable[[Constraint, Trace], bool]
    is_binary: bool = True
    supports_cardinality: bool = False
    # Which of `a` and `b` is the activation, or None for a template without one, which vacuity
    # never affects
    activation: Literal['a', 'b'] | None = None


def _existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs at least `n` times."""
    return len(trace.occurrences.get(constraint.a, ())) >= constraint.n


def _absence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs fewer than `n` times."""
    return len(trace.occurrences.get(constraint.a, ())) < constraint.n


def _exactly(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs exactly `n` times."""
    return len(trace.occurrences.get(constraint.a, ())) == constraint.n


def _init(constraint: Constraint, trace: Trace) -> bool:
    """`a` is the first event of the trace."""
    return trace.activities.startswith(constraint.a)


def _end(constraint: Constraint, trace: Trace) -> bool:
    """`a` is the last event of the trace."""
    return trace.activities.endswith(constraint.a)


def _choice(constraint: Constraint, trace: Trace) -> bool:
    """`a` or `b` occurs."""
    return constraint.a in trace.occurrences or constraint.b in trace.occurrences


def _exclusive_choice(constraint: Constraint, trace: Trace) -> bool:
    """`a` or `b` occurs, and never both."""
    return (constraint.a in trace.occurrences) != (constraint.b in trace.occurrences)


def _responded_existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and so does `b`."""
    return constraint.a in trace.occurrences and constraint.b in trace.occurrences


def _not_responded_existence(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and `b` does not."""
    return constraint.a in trace.occurrences and constraint.b not in trace.occurrences


def _response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and every occurrence of it is followed by a `b`."""
    activations = trace.occurrences.get(constraint.a)
    targets = trace.occurrences.get(constraint.b)
    return bool(activations) and bool(targets) and activations[-1] < targets[-1]


def _precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and every occurrence of it is preceded by an `a`."""
    activations = trace.occurrences.get(constraint.b)
    targets = trace.occurrences.get(constraint.a)
    return bool(activations) and bool(targets) and targets[0] < activations[0]


def _not_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and no occurrence of it is followed by a `b`."""
    activations = trace.occurrences.get(constraint.a)
    targets = trace.occurrences.get(constraint.b)
    return bool(activations) and (not targets or activations[0] > targets[-1])


def _not_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and no occurrence of it is preceded by an `a`."""
    activations = trace.occurrences.get(constraint.b)
    targets = trace.occurrences.get(constraint.a)
    return bool(activations) and (not targets or targets[0] > activations[-1])


def _adjacent(constraint: Constraint) -> str:
    """`a` immediately followed by `b`, as it reads in `Trace.activities`.

    The two activities of a binary constraint differ, so its occurrences never overlap.
    """
    return constraint.a + constraint.b


def _chain_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` follows it immediately every time."""
    activations = trace.occurrences.get(constraint.a)
    return bool(activations) and trace.activities.count(_adjacent(constraint)) == len(activations)


def _chain_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` precedes it immediately every time."""
    activations = trace.occurrences.get(constraint.b)
    return bool(activations) and trace.activities.count(_adjacent(constraint)) == len(activations)


def _not_chain_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` never follows it immediately."""
    return constraint.a in trace.occurrences and _adjacent(constraint) not in trace.activities


def _not_chain_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` never precedes it immediately."""
    return constraint.b in trace.occurrences and _adjacent(constraint) not in trace.activities


def _alternate_response(constraint: Constraint, trace: Trace) -> bool:
    """`a` occurs, and a `b` follows each occurrence of it before `a` recurs."""
    pending = False
    for activity in trace.activities:
        if activity == constraint.a:
            if pending:
                return False
            pending = True
        elif activity == constraint.b:
            pending = False
    return constraint.a in trace.occurrences and not pending


def _alternate_precedence(constraint: Constraint, trace: Trace) -> bool:
    """`b` occurs, and an `a` precedes each occurrence of it since the previous `b`."""
    preceded = False
    for activity in trace.activities:
        if activity == constraint.a:
            preceded = True
        elif activity == constraint.b:
            if not preceded:
                return False
            preceded = False
    return constraint.b in trace.occurrences


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
        _Template('Responded Existence', _responded_existence, activation='a'),
        _Template('Not Responded Existence', _not_responded_existence, activation='a'),
        _Template('Response', _response, activation='a'),
        _Template('Precedence', _precedence, activation='b'),
        _Template('Not Response', _not_response, activation='a'),
        _Template('Not Precedence', _not_precedence, activation='b'),
        _Template('Chain Response', _chain_response, activation='a'),
        _Template('Chain Precedence', _chain_precedence, activation='b'),
        _Template('Not Chain Response', _not_chain_response, activation='a'),
        _Template('Not Chain Precedence', _not_chain_precedence, activation='b'),
        _Template('Alternate Response', _alternate_response, activation='a'),
        _Template('Alternate Precedence', _alternate_precedence, activation='b'),
    )
}
