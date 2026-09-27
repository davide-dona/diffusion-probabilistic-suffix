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
from src.evaluation import read_reports, summarize_runs
from src.uncertainty import test_significance
from src.visualization import (
    FIGURES,
    TABLES,
    apply_style,
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


def _draw_figures(frame: pd.DataFrame) -> int:
    """Draw every figure of the catalogue, each covering every log the reports cover at once.

    Args:
        frame: Every report summarized over runs, from `summarize_runs`.
    Returns:
        How many figures were written, under the invocation's `figures/`.
    """
    written = 0
    for plot in FIGURES:
        _save_figure(
            figure=compose_figure(frame[frame['axis'].isin(plot.breakdowns)], plot),
            path=output_path(f'figures/{plot.name}.pdf'),
        )
        written += 1
    return written


def _write_tables(frame: pd.DataFrame, significance: pd.DataFrame) -> int:
    """Write every comparison table, over every log at once, under the invocation's `tables/`.

    Args:
        frame: Every report summarized over runs, from `summarize_runs`.
        significance: Table emphasis and adjusted comparisons from `test_significance`.
    Returns:
        How many tables were written.
    """
    for table in TABLES:
        output_path(f'tables/{table.name}.tex').write_text(latex_table(frame, table, significance))
    return len(TABLES)


def _run_counts(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count the runs behind each dataset and model.

    Args:
        frame: Every report summarized over runs, from `summarize_runs`.
    Returns:
        Run counts keyed by dataset and model, in report order.
    """
    counts = frame.groupby(['dataset', 'model'], sort=False)['runs'].first()
    return {(str(dataset), str(model)): int(runs) for (dataset, model), runs in counts.items()}


def run(evaluation_files: Sequence[Path]) -> None:
    """Draw a set of evaluation reports and tabulate them, under the active Hydra output directory.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`. These draw
            the metric figures and comparison tables, and the per-prefix scores beside each report
            are what the tables' emphasis is tested on. Several reports of one model on one log
            are seeds of one configuration, summarized by their mean and standard deviation.
    Raises:
        ValueError: If a file is not a report, if a report has no per-prefix scores beside it, if a
            model has no look declared in `src.visualization.labels`, if one training run is given
            twice, or if the runs of one model do not score the same prefixes.
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
        reports = summarize_runs(read_reports(evaluation_files))
    for (dataset, model), runs in _run_counts(reports).items():
        print(f'  {dataset}: {model}, {runs} run(s)')

    logs = sorted(set(reports['dataset']))
    with step(f'Drawing {", ".join(logs)}'):
        drawn = _draw_figures(reports)

    with step('Comparing means with a paired case bootstrap'):
        significance = test_significance(evaluation_files)

    with step('Writing the comparison tables'):
        tables = _write_tables(reports, significance)

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
