from dataclasses import dataclass

from src.evaluation import TRACE_METRICS, Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Owner


@dataclass(frozen=True)
class Column:
    """One table column: a frame metric, its header, and its preferred direction.

    `observed` names a log-owned metric shown under this column in a separate Log row.
    """

    key: str
    label: str
    direction: Direction
    observed: str | None = None


@dataclass(frozen=True)
class ColumnGroup:
    """A contiguous group of table columns sharing one header."""

    label: str
    span: int


@dataclass(frozen=True)
class Table:
    """Definition of one comparison table and its columns.

    With a reference model, each cell is the skill score `1 - mean / reference mean` of its
    metric, and the reference model's own row is left out. Emphasized tables bold the cells
    that are best or not detectably worse; the others only describe. Omitted models get no row.
    """

    name: str
    axis: Axis
    columns: tuple[Column, ...]
    column_groups: tuple[ColumnGroup, ...] = ()
    reference: str | None = None
    emphasis: bool = True
    omitted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.emphasis:
            unranked = [column.key for column in self.columns if column.key not in METRICS.report]
            undirected = [
                column.key for column in self.columns if column.direction is Direction.NONE
            ]
            if unranked or undirected:
                raise ValueError(
                    f'{self.name} emphasizes columns without a ranking: {unranked + undirected}'
                )
        if self.reference is not None:
            unskilled = [c.key for c in self.columns if c.direction is not Direction.LOWER]
            if unskilled:
                raise ValueError(f'{self.name} skill needs negatively oriented scores: {unskilled}')
        unowned = [
            column.observed
            for column in self.columns
            if column.observed is not None and METRICS[column.observed].owner is not Owner.LOG
        ]
        if unowned:
            raise ValueError(f'{self.name} Log row needs log-owned metrics: {unowned}')
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(f'{self.name} column groups do not cover its columns.')


def metric_column(key: str, *, label: str | None = None, observed: str | None = None) -> Column:
    """Return the column of a registered metric, headed by its publication label by default."""
    metric = METRICS[key]
    return Column(
        key=key,
        label=label or metric.publication_label or metric.label,
        direction=metric.direction,
        observed=observed,
    )


# The prefix-blind forecast every skill score is measured against.
SKILL_REFERENCE = 'unconditional'

_PROPER_SCORES = (
    metric_column('signature_kernel_score'),
    metric_column('energy_score_dls'),
    metric_column('energy_score_exact'),
    metric_column('energy_score_bigram'),
    metric_column('suffix_length_crps', label='CRPS'),
    metric_column('remaining_time_crps_days', label='CRPS'),
)
_PROPER_GROUPS = (
    ColumnGroup(label='Joint', span=1),
    ColumnGroup(label='Activity', span=3),
    ColumnGroup(label='Length', span=1),
    ColumnGroup(label='Time', span=1),
)

# Each table answers one evaluation question.
TABLES = (
    Table(
        name='skill',
        axis=Axis.OVERALL,
        columns=_PROPER_SCORES,
        column_groups=_PROPER_GROUPS,
        reference=SKILL_REFERENCE,
    ),
    Table(
        name='plausibility',
        axis=Axis.OVERALL,
        columns=(
            metric_column(
                'full_conformance_sample_rate',
                label='Full',
                observed='full_conformance_observed',
            ),
            metric_column('best_of_5_dls', label='$k=5$'),
            metric_column('best_of_10_dls', label='$k=10$'),
            Column(key=TRACE_METRICS[0], label='Precision', direction=Direction.HIGHER),
            Column(key=TRACE_METRICS[1], label='Recall', direction=Direction.HIGHER),
        ),
        column_groups=(
            ColumnGroup(label='Conformance', span=1),
            ColumnGroup(label='Best-of-$k$ DLS', span=2),
            ColumnGroup(label='Traces', span=2),
        ),
        emphasis=False,
        omitted=(SKILL_REFERENCE,),
    ),
    Table(
        name='means',
        axis=Axis.OVERALL,
        columns=_PROPER_SCORES,
        column_groups=_PROPER_GROUPS,
    ),
    Table(
        name='crps-decomposition',
        axis=Axis.OVERALL,
        columns=(
            metric_column('suffix_length_crps_error', label='Error'),
            metric_column('suffix_length_crps_spread', label='Spread'),
            metric_column('remaining_time_crps_error_days', label='Error'),
            metric_column('remaining_time_crps_spread_days', label='Spread'),
        ),
        column_groups=(
            ColumnGroup(label='Suffix length [events]', span=2),
            ColumnGroup(label='Remaining time [days]', span=2),
        ),
        emphasis=False,
        omitted=(SKILL_REFERENCE,),
    ),
)
