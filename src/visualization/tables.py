import math
from collections.abc import Container, Sequence

import pandas as pd

from src.evaluation.metrics.metadata import Direction, Metric
from src.visualization import labels
from src.visualization.catalogue import Table

TABLE_ARROWS = {
    Direction.HIGHER: r' ($\uparrow$)',
    Direction.LOWER: r' ($\downarrow$)',
    Direction.ZERO: r' ($\rightarrow 0$)',
    Direction.NONE: '',
}


def _escape_latex(text: str) -> str:
    """Escape text for LaTex.

    Args:
        text: Unescaped text.

    Returns:
        Text safe for LaTex text mode.
    """
    for character in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(character, f'\\{character}')
    return text


def _value(frame: pd.DataFrame, key: str) -> tuple[float, float]:
    """Read a metric's summary over runs.

    Args:
        frame: Summary rows for one model and dataset.
        key: Metric key.

    Returns:
        Mean over runs and its sample standard deviation, NaN for a single run.
    """
    rows = frame.loc[frame['metric'] == key, ['mean', 'std']]
    if len(rows) != 1:
        raise ValueError(f'Expected one value for {key}, found {len(rows)}.')
    return float(rows['mean'].iloc[0]), float(rows['std'].iloc[0])


def _cell(mean: float, std: float, *, bold: bool) -> str:
    """Render one table cell as a value, or as mean and standard deviation over runs.

    Args:
        mean: Mean over runs.
        std: Sample standard deviation over runs, NaN for a single run.
        bold: Whether to emphasize the cell.

    Returns:
        LaTex cell contents.
    """
    written = f'{mean:.3f}'
    if math.isnan(std):
        return f'\\textbf{{{written}}}' if bold else written
    spread = f'{std:.3f}'
    if bold:
        # `\textbf` does not reach math mode.
        written, spread = f'\\mathbf{{{written}}}', f'\\mathbf{{{spread}}}'
    return f'${written} \\pm {spread}$'


def _row(
    columns: Sequence[Metric],
    frame: pd.DataFrame,
    *,
    label: str,
    best: Container[str],
) -> str:
    """Render one model row of a LaTex table.

    Args:
        columns: Metrics to render.
        frame: Summary rows for the model.
        label: Model display label.
        best: Metrics to render in bold.

    Returns:
        One LaTex table row.
    """
    cells = [_cell(*_value(frame, metric.key), bold=metric.key in best) for metric in columns]
    return '   & ' + ' & '.join([_escape_latex(label), *cells]) + ' \\\\'


def _block(
    table: Table,
    frame: pd.DataFrame,
    significance: pd.DataFrame,
    models: Sequence[str],
) -> list[str]:
    """Render rows for all reported models of one dataset.

    Args:
        table: Table definition.
        frame: Summary rows for the dataset.
        significance: Metrics to emphasize by model.
        models: Models in display order.

    Returns:
        LaTex rows for the dataset.
    """
    rows = []
    for model in models:
        if not (frame['model'] == model).any():
            continue
        marked = significance[(significance['model'] == model) & significance['best']]
        rows.append(
            _row(
                table.columns,
                frame[frame['model'] == model],
                label=labels.MODELS[model].label,
                best=set(marked['metric']),
            )
        )
    return rows


def _headers(table: Table) -> list[str]:
    """Render one or two header rows for a table.

    Args:
        table: Table definition.

    Returns:
        LaTex header lines.
    """
    headers = [
        f'{metric.publication_label or metric.label}{TABLE_ARROWS[metric.direction]}'
        for metric in table.columns
    ]
    if not table.column_groups:
        return ['  Dataset & Model & ' + ' & '.join(headers) + ' \\\\']

    headers = [metric.publication_label or metric.label for metric in table.columns]
    groups = ' & '.join(
        f'\\multicolumn{{{group.span}}}{{c}}{{{group.label}}}' for group in table.column_groups
    )
    spans: list[str] = []
    start = 3
    for group in table.column_groups:
        end = start + group.span - 1
        spans.append(f'\\cmidrule(lr){{{start}-{end}}}')
        start = end + 1
    return [
        '  \\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model} & ' + groups + ' \\\\',
        *spans,
        '  & & ' + ' & '.join(headers) + ' \\\\',
    ]


def latex_table(frame: pd.DataFrame, table: Table, significance: pd.DataFrame) -> str:
    """Render a booktabs LaTex table with descriptive and inferential emphasis notes.

    A model with several runs shows its mean and sample standard deviation over them.

    Args:
        frame: Summary rows for all datasets and models, from `summarize_runs`.
        table: Table definition.
        significance: Metrics to emphasize by dataset and model.

    Returns:
        Complete `tabularx` environment.
    """
    overall = frame[frame['axis'] == table.axis]
    # Keep models and datasets in catalogue order.
    models = labels.ordered(overall['model'], labels.MODELS, kind='model')
    datasets = labels.ordered(overall['dataset'], labels.DATASETS, kind='dataset')
    lines = ['\\toprule', *_headers(table), '\\midrule']
    for index, dataset in enumerate(datasets):
        if index > 0:
            lines.append('\\midrule')
        rows = _block(
            table,
            overall[overall['dataset'] == dataset],
            significance[significance['dataset'] == dataset],
            models,
        )
        # Let the dataset column size itself to its label.
        name = _escape_latex(labels.DATASETS[dataset])
        lines.append(f'  \\multirow{{{len(rows)}}}{{*}}{{{name}}}')
        lines.extend(rows)
    lines.append('\\bottomrule')

    if table.column_groups:
        value_columns = f'*{{{len(table.columns)}}}{{r}}'
        preamble = (
            f'\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}ll|{value_columns}}}'
        )
        environment = 'tabular*'
    else:
        value_columns = f'*{{{len(table.columns)}}}{{>{{\\centering\\arraybackslash}}X}}'
        preamble = f'\\begin{{tabularx}}{{\\linewidth}}{{ll|{value_columns}}}'
        environment = 'tabularx'
    return '\n'.join((preamble, *lines, f'\\end{{{environment}}}')) + '\n'
