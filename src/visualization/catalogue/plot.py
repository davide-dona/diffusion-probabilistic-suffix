from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Direction, Metric, Owner
from src.visualization.catalogue.table import SKILL_REFERENCE


@dataclass(frozen=True)
class Plot:
    """Definition of one multi-dataset figure and its panels.

    With a reference model, each line is the skill score `1 - mean / reference mean` of its
    metric in each length bucket, and the reference model's own line is left out. Omitted models
    get no line.
    """

    name: str
    breakdowns: tuple[Axis, ...]
    panels: tuple[tuple[Metric, ...], ...]
    reference: str | None = None
    omitted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reference is not None:
            unskilled = [
                metric.key
                for panel in self.panels
                for metric in panel
                if metric.direction is not Direction.LOWER or metric.owner is Owner.LOG
            ]
            if unskilled:
                raise ValueError(f'{self.name} skill needs negatively oriented scores: {unskilled}')


# Catalogue figures: panels by breakdown and dataset.
FIGURES = (
    Plot(
        name='skill-by-length',
        breakdowns=(Axis.PREFIX, Axis.SUFFIX),
        panels=((METRICS['signature_kernel_score'],),),
        reference=SKILL_REFERENCE,
    ),
    Plot(
        name='full-conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=((METRICS['full_conformance_sample_rate'], METRICS['full_conformance_observed']),),
        omitted=(SKILL_REFERENCE,),
    ),
)

# Randomized PIT values drawn as histograms, one per CRPS-scored quantity.
PIT_METRICS = (
    METRICS['suffix_length_truth_quantile'],
    METRICS['remaining_time_truth_quantile'],
)
