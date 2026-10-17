from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Metric
from src.evaluation.traces import TRACE_MEASURES, TraceMeasure


@dataclass(frozen=True)
class ColumnGroup:
    """A contiguous group of table columns sharing one header."""

    label: str
    span: int


@dataclass(frozen=True)
class Table:
    """Definition of one comparison table and its columns.

    With a reference model, every cell is the skill `1 - score / reference score` of the model's
    mean against the reference's on the same dataset, the reference row is left out, and every
    column must be a score to minimize.
    """

    name: str
    axis: Axis
    columns: tuple[Metric | TraceMeasure, ...]
    column_groups: tuple[ColumnGroup, ...] = ()
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.reference is not None:
            unscored = [
                metric.key for metric in self.columns if metric.direction is not Direction.LOWER
            ]
            if unscored:
                raise ValueError(f'{self.name} has skill columns that are not scores: {unscored}')
        undirected = [metric.key for metric in self.columns if metric.direction is Direction.NONE]
        if undirected:
            raise ValueError(f'{self.name} has metrics without a ranking direction: {undirected}')
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(f'{self.name} column groups do not cover its columns.')


# Each table answers one evaluation question with directional metrics.
TABLES = (
    Table(
        name='sample-prediction',
        axis=Axis.OVERALL,
        columns=(
            METRICS.report['kernel_score_joint'],
            METRICS.report['energy_score_dls'],
            METRICS.report['energy_score_exact'],
            METRICS.report['energy_score_bigram'],
            METRICS.report['suffix_length_crps'],
        ),
    ),
    Table(
        name='skill',
        axis=Axis.OVERALL,
        columns=(
            METRICS.report['kernel_score_joint'],
            METRICS.report['energy_score_dls'],
            METRICS.report['suffix_length_crps'],
            METRICS.report['remaining_time_crps_days'],
            METRICS.report['teacher_forced_inter_event_time_crps_days'],
        ),
        column_groups=(ColumnGroup('Skill over the unconditional reference', 5),),
        reference='unconditional',
    ),
    Table(name='trace-precision-recall', axis=Axis.OVERALL, columns=TRACE_MEASURES),
)
