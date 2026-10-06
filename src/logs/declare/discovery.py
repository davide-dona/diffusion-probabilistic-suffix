from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from itertools import permutations

import pandas as pd
from omegaconf import DictConfig

from src.logs.declare.model import DeclareModel
from src.logs.declare.templates import TEMPLATES, Constraint, Positions, positions_of
from src.logs.keys import ACTIVITY_KEY, CASE_KEY

# The first code point handed to an activity while mining, inside the Private Use Area so that no
# activity name can collide with a code.
_FIRST_CODE = 0xE000


def discover_declare_model(
    train: pd.DataFrame,
    *,
    dataset: str,
    declare_config: DictConfig,
) -> int:
    """
    Discover a declarative model from the train split and write it beside the dataset.

    Mines the same constraints as Declare4Py's `DeclareMiner`. The candidates are the unary
    templates over every activity present in a share of at least `itemsets_support` of the traces,
    and the binary templates, in both orders, over every pair of activities present together in
    that share. A candidate is kept when it holds in a share of at least `min_support` of the
    traces, read through `Constraint.holds`, the same check conformance uses.

    Args:
        train: The train split, as preprocessing holds it. The only log discovery reads, so the
            constraints never carry anything from the validation or test split.
        dataset: The dataset the split came from, naming where the model goes.
        declare_config: The `declare` section: which constraints are looked for and how much of
            the log has to support one.

    Returns:
        The number of constraints written.
    """
    activities = sorted(train[ACTIVITY_KEY].astype(str).unique())

    codes = {activity: chr(_FIRST_CODE + index) for index, activity in enumerate(activities)}
    names = {code: activity for activity, code in codes.items()}
    traces = train[ACTIVITY_KEY].astype(str).map(codes).groupby(train[CASE_KEY], sort=False)
    variants = Counter(''.join(trace) for _, trace in traces)
    log = [(positions_of(variant), variant, count) for variant, count in variants.items()]
    total = variants.total()

    present: Counter[str] = Counter()
    for variant, count in variants.items():
        present.update(dict.fromkeys(set(variant), count))
    frequent = sorted(
        code for code, count in present.items() if count / total >= declare_config.itemsets_support
    )
    together: Counter[tuple[str, str]] = Counter()
    for variant, count in variants.items():
        together.update(dict.fromkeys(permutations(set(variant) & set(frequent), 2), count))
    pairs = [
        pair
        for pair in permutations(frequent, 2)
        if together[pair] / total >= declare_config.itemsets_support
    ]

    mined = [
        constraint
        for constraint in _candidates(frequent, pairs, declare_config.max_cardinality)
        if _support(constraint, log, vacuity=declare_config.consider_vacuity) / total
        >= declare_config.min_support
    ]

    DeclareModel(
        settings=declare_config,
        constraints=tuple(
            replace(
                constraint,
                first=names[constraint.first],
                second=None if constraint.second is None else names[constraint.second],
            )
            for constraint in mined
        ),
    ).save(dataset)

    return len(mined)


def _candidates(
    frequent: list[str], pairs: list[tuple[str, str]], max_cardinality: int
) -> Iterator[Constraint]:
    """Yield every constraint discovery tests, unary templates first, in `TEMPLATES` order."""
    for template in TEMPLATES.values():
        if template.is_binary:
            continue
        cardinalities = range(1, max_cardinality + 1) if template.supports_cardinality else (1,)
        for activity in frequent:
            for n in cardinalities:
                yield Constraint(template=template, first=activity, second=None, n=n)

    for template in TEMPLATES.values():
        if not template.is_binary:
            continue
        for first, second in pairs:
            yield Constraint(template=template, first=first, second=second, n=1)


def _support(
    constraint: Constraint, log: list[tuple[Positions, str, int]], *, vacuity: bool
) -> int:
    """Count the traces of the log that satisfy one constraint.

    Args:
        constraint: The candidate.
        log: Each distinct trace once, with its positions and how many traces share it.
        vacuity: Whether a trace that never activates the constraint satisfies it.
    Returns:
        The number of traces, repeats included, that satisfy it.
    """
    return sum(
        count
        for positions, trace, count in log
        if constraint.holds(trace, positions, vacuity=vacuity)
    )
