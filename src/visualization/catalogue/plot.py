from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.metrics import METRICS
from src.evaluation.metrics.metadata import Metric


@dataclass(frozen=True)
class Plot:
    """Definition of one multi-dataset figure and its panels."""

    name: str
    breakdowns: tuple[Axis, ...]
    panels: tuple[tuple[Metric, ...], ...]


@dataclass(frozen=True)
class CalibrationPlot:
    """Definition of one figure of PIT calibration curves, one column per PIT metric."""

    name: str
    metrics: tuple[Metric, ...]


# Catalogue figures: panels by breakdown and dataset.
FIGURES = (
    Plot(
        name='conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=((METRICS.report['conformance_sample_mean'],),),
    ),
)

# Calibration figures: PIT curves by metric and dataset, read from per-prefix scores.
CALIBRATION_FIGURES = (
    CalibrationPlot(
        name='pit-calibration',
        metrics=(
            METRICS.report['suffix_length_pit'],
            METRICS.report['remaining_time_pit'],
            METRICS.report['teacher_forced_remaining_time_pit'],
        ),
    ),
)
