from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation import read_prefix_scores, score_files
from src.evaluation.metrics.metadata import Metric
from src.visualization import labels
from src.visualization.catalogue import PIT_METRICS, SKILL_REFERENCE
from src.visualization.style import (
    DATASET_HEIGHT,
    FIGURE_OVERHEAD,
    PAGE_WIDTH,
    REFERENCE_LINE,
    Y_HEADROOM,
)

PIT_BINS = 10
PIT_COLUMNS = ('dataset', 'model', 'metric', 'value')
_EDGES = np.linspace(0.0, 1.0, PIT_BINS + 1)


def read_pit(reports: Sequence[Path]) -> pd.DataFrame:
    """Read every prefix's randomized PIT from the score files beside the reports.

    Args:
        reports: Evaluation reports with per-prefix score files beside them.

    Returns:
        Long-form values under `PIT_COLUMNS`, one row per prefix and `PIT_METRICS` entry.

    Raises:
        ValueError: If a score file is missing, invalid, or lacks a PIT column.
    """
    keys = [metric.key for metric in PIT_METRICS]
    frames = [
        read_prefix_scores(path, columns=keys)
        .melt(var_name='metric', value_name='value')
        .assign(dataset=dataset, model=model)
        for dataset, files in score_files(reports).items()
        for model, path in files.items()
    ]
    if not frames:
        return pd.DataFrame(columns=list(PIT_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(PIT_COLUMNS)]


def _density(values: pd.Series) -> np.ndarray:
    """Return the histogram density of PIT values over `PIT_BINS` equal bins, `[PIT_BINS]`."""
    counts, _ = np.histogram(values.to_numpy(), bins=_EDGES)
    return counts * PIT_BINS / max(1, counts.sum())


def _title(metric: Metric) -> str:
    """Name the quantity whose truth quantile the metric holds."""
    return metric.group.replace('_', ' ').capitalize()


def compose_pit_grid(frame: pd.DataFrame, metric: Metric) -> Figure:
    """Draw one quantity's PIT histogram per model, with datasets as rows and models as columns.

    The skill reference is left out, since a prefix-blind forecast is calibrated by construction.

    Args:
        frame: PIT values from `read_pit`.
        metric: The PIT metric to draw, one of `PIT_METRICS`.

    Returns:
        Composed Matplotlib figure, where a calibrated model follows the uniform density.
    """
    values = frame[(frame['metric'] == metric.key) & (frame['model'] != SKILL_REFERENCE)]
    datasets = labels.ordered(values['dataset'], labels.DATASETS, kind='dataset')
    models = labels.ordered(values['model'], labels.MODELS, kind='model')
    figure, grid = plt.subplots(
        nrows=len(datasets),
        ncols=len(models),
        figsize=(PAGE_WIDTH, len(datasets) * DATASET_HEIGHT + FIGURE_OVERHEAD),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    top = 1.0
    for dataset, row in zip(datasets, grid, strict=True):
        for model, axes in zip(models, row, strict=True):
            drawn = values.loc[(values['dataset'] == dataset) & (values['model'] == model), 'value']
            if drawn.empty:
                axes.text(0.5, 0.5, 'no report', ha='center', va='center', transform=axes.transAxes)
                continue
            density = _density(drawn)
            top = max(top, float(density.max()))
            axes.stairs(density, _EDGES, fill=True, color=labels.MODELS[model].color, alpha=0.8)
    for axes in grid.flat:
        axes.axhline(1.0, **REFERENCE_LINE)
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, top * (1.0 + Y_HEADROOM))
        axes.set_xticks([0.0, 0.5, 1.0])
    for axes in grid[:-1].flat:
        axes.tick_params(labelbottom=False)
    for axes in grid[:, 1:].flat:
        axes.tick_params(labelleft=False)
    for model, axes in zip(models, grid[0], strict=True):
        axes.set_title(labels.MODELS[model].label)
    for dataset, axes in zip(datasets, grid[:, 0], strict=True):
        axes.set_ylabel(labels.DATASETS[dataset], rotation=0, ha='right', va='center', labelpad=12)
    figure.align_ylabels(grid[:, 0])
    figure.supxlabel('PIT')
    figure.suptitle(_title(metric))
    return figure
