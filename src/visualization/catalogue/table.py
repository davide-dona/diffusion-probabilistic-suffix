from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Metric


@dataclass(frozen=True)
class ColumnGroup:
    """A contiguous group of table columns sharing one header."""

    label: str
    span: int


@dataclass(frozen=True)
class Table:
    """Definition of one comparison table and its columns.

    With a reference model, each cell is the skill score `1 - mean / reference mean` of its
    metric, and the reference model's own row is left out.
    """

    name: str
    axis: Axis
    columns: tuple[Metric, ...]
    column_groups: tuple[ColumnGroup, ...] = ()
    reference: str | None = None

    def __post_init__(self) -> None:
        undirected = [metric.key for metric in self.columns if metric.direction is Direction.NONE]
        if undirected:
            raise ValueError(f'{self.name} has metrics without a ranking direction: {undirected}')
        if self.reference is not None:
            unskilled = [m.key for m in self.columns if m.direction is not Direction.LOWER]
            if unskilled:
                raise ValueError(f'{self.name} skill needs negatively oriented scores: {unskilled}')
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(f'{self.name} column groups do not cover its columns.')


# Each table answers one evaluation question with directional metrics.
TABLES = (
    Table(
        name='skill',
        axis=Axis.OVERALL,
        columns=(
            METRICS.report['signature_kernel_score'],
            METRICS.report['energy_score_dls'],
            METRICS.report['energy_score_exact'],
            METRICS.report['energy_score_bigram'],
            METRICS.report['suffix_length_crps'],
            METRICS.report['remaining_time_crps_days'],
        ),
        reference='unconditional',
    ),
)
