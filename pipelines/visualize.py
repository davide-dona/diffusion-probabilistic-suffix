from collections.abc import Sequence
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from pipelines.helpers.console import banner, step
from pipelines.helpers.invocation import output_path, start_stage
from src.config_validation import validate_visualization_config
from src.evaluation import read_reports
from src.uncertainty import test_significance
from src.visualization import (
    CALIBRATION_FIGURES,
    FIGURES,
    TABLES,
    apply_style,
    compose_calibration_figure,
    compose_figure,
    latex_table,
)


def _save_figure(figure: Figure, path: Path) -> None:
    """Write one finished figure and close it.

    Args:
        figure: The figure to write, closed afterwards so a run drawing dozens does not hold them
            all open.
        path: Where to write it, inside the active Hydra output directory.
    """
    figure.savefig(path)
    plt.close(figure)


def _draw_figures(frame: pd.DataFrame, evaluation_files: Sequence[Path]) -> int:
    """Draw every figure of the catalogue, each covering every log the reports cover at once.

    Args:
        frame: Every report read, from `read_reports`.
        evaluation_files: The reports read, whose per-prefix scores the calibration figures
            draw.
    Returns:
        How many figures were written, under the invocation's `figures/`.
    """
    for plot in FIGURES:
        _save_figure(
            figure=compose_figure(frame[frame['axis'].isin(plot.breakdowns)], plot),
            path=output_path(f'figures/{plot.name}.pdf'),
        )
    for calibration in CALIBRATION_FIGURES:
        _save_figure(
            figure=compose_calibration_figure(evaluation_files, calibration),
            path=output_path(f'figures/{calibration.name}.pdf'),
        )
    return len(FIGURES) + len(CALIBRATION_FIGURES)


def _write_tables(frame: pd.DataFrame, evaluation_files: Sequence[Path]) -> int:
    """Write every comparison table, over every log at once, under the invocation's `tables/`.

    A skill table's emphasis compares the models it shows, leaving its reference out.

    Args:
        frame: Every report read, from `read_reports`.
        evaluation_files: The reports read, whose per-prefix scores the emphasis is tested on.
    Returns:
        How many tables were written.
    """
    significance = {
        reference: test_significance(evaluation_files, excluded=reference)
        for reference in dict.fromkeys(table.reference for table in TABLES)
    }
    for table in TABLES:
        output_path(f'tables/{table.name}.tex').write_text(
            latex_table(frame, table, significance[table.reference])
        )
    return len(TABLES)


def run(evaluation_files: Sequence[Path]) -> None:
    """Draw a set of evaluation reports and tabulate them, under the active Hydra output directory.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`. These draw
            the metric figures and comparison tables, and the per-prefix scores beside each report
            are what the tables' emphasis is tested on.
    Raises:
        ValueError: If a file is not a report, if a report has no per-prefix scores beside it, if a
            model has no look declared in `src.visualization.labels`, or if one log is given two
            runs of the same model.
    """
    apply_style()
    banner(
        'Drawing the figures and tables',
        {
            'reports': f'{len(evaluation_files)} file(s), with their per-prefix scores beside them',
            'figures': output_path('figures'),
            'tables': output_path('tables'),
        },
    )

    with step(f'Reading {len(evaluation_files)} evaluation report(s)'):
        reports = read_reports(evaluation_files)

    logs = sorted(set(reports['dataset']))
    with step(f'Drawing {", ".join(logs)}'):
        drawn = _draw_figures(reports, evaluation_files)

    with step('Comparing means with a paired case bootstrap and writing the comparison tables'):
        tables = _write_tables(reports, evaluation_files)

    print(
        f'\nWrote {drawn} figures in pdf to {output_path("figures")} '
        f'and {tables} tables in tex to {output_path("tables")}'
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='visualize')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    validate_visualization_config(evaluations=cfg.evaluations, evaluations_dir=cfg.evaluations_dir)
    files = [Path(path) for path in cfg.evaluations]
    if cfg.evaluations_dir:
        files = sorted(
            path for folder in cfg.evaluations_dir for path in Path(folder).rglob('evaluation.json')
        )
    if not files:
        raise ValueError('No evaluation reports found')
    run(files)


if __name__ == '__main__':
    main()
