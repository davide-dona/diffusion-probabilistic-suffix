import json
from collections import Counter
from collections.abc import Iterator
from itertools import permutations

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import artifacts
from src.logs.declare.constraints import COMMENT, SETTINGS_LINE
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

    # A constraint line names its activities inside brackets, separated by `, `, and closes on
    # the conditions behind a `|`. An activity carrying any of that would be read back as two
    # activities, or as a condition, so the model could not say what it was mined to say.
    unwritable = [
        activity
        for activity in activities
        if any(marker in activity for marker in ('[', ']', '|', ', '))
    ]
    if unwritable:
        raise ValueError(
            f'{dataset} has activities a declarative model cannot name: {unwritable}. '
            'Rename them in the raw log, or drop them from the preprocessed split.'
        )

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

    # What the constraints below were mined under, so a reader of the file can tell a model mined
    # one way from one mined another, and so the checker reads vacuity the way mining did.
    lines = [
        f'{COMMENT} discovered from the train split of {dataset} by pipelines.preprocess',
        f'{SETTINGS_LINE}{json.dumps(OmegaConf.to_container(declare_config, resolve=True))}',
    ]
    lines += [f'activity {activity}' for activity in activities]
    lines += [_serialize(constraint, names) for constraint in mined]

    path = artifacts.DECLARE_MODEL.prepare(dataset)
    path.write_text('\n'.join(lines) + '\n')

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


def _serialize(constraint: Constraint, names: dict[str, str]) -> str:
    """Write one constraint as a Declare4Py line, with empty activation, target, and time
    conditions for a binary template and empty activation and time conditions for a unary one.
    """
    template = constraint.template
    name = next(key for key, value in TEMPLATES.items() if value is template)
    cardinality = str(constraint.n) if template.supports_cardinality else ''
    if template.is_binary:
        return f'{name}[{names[constraint.first]}, {names[constraint.second]}] | | |'
    return f'{name}{cardinality}[{names[constraint.first]}] | |'
