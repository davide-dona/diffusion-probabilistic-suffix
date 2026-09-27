from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import read_prefix_scores, require_columns

PREFIX_KEYS = ('case_id', 'prefix_len')


def _aligned(
    dataset: str, files: dict[str, tuple[Path, ...]], metrics: Sequence[str]
) -> tuple[tuple[np.ndarray, ...], pd.MultiIndex]:
    """Align every run's scores on identical sorted prefix keys.

    Returns:
        One `[prefixes, runs, metrics]` array per model in model order, and the shared prefix keys.
    """
    columns = (*PREFIX_KEYS, *metrics)
    prefixes = None
    models = []
    for runs in files.values():
        frames = []
        for file in runs:
            require_columns(path=file, columns=columns)
            frame = read_prefix_scores(path=file, columns=columns)
            if frame[list(PREFIX_KEYS)].isna().any().any():
                raise ValueError(f'{file} contains missing prefix keys.')
            frame = frame.set_index(list(PREFIX_KEYS)).sort_index()
            if frame.index.has_duplicates:
                raise ValueError(f'{file} scores the same prefix twice, so it cannot be compared.')
            if prefixes is not None and not frame.index.equals(prefixes):
                raise ValueError(f'The runs of {dataset} do not score the same prefixes: {file}')
            prefixes = frame.index
            frames.append(frame[list(metrics)].to_numpy(dtype=np.float64))
        models.append(np.stack(frames, axis=1))

    assert prefixes is not None
    return tuple(models), prefixes


def by_case(
    dataset: str, files: dict[str, tuple[Path, ...]], metrics: Sequence[str]
) -> Iterator[tuple[list[str], tuple[np.ndarray, ...], np.ndarray]]:
    """Yield case totals and prefix counts for each reporting population.

    Args:
        dataset: Dataset name used in validation errors.
        files: Per-prefix score files of each model's runs, in model order.
        metrics: Metrics to aggregate, in output order.

    Yields:
        Metric names, one score total array `[cases, runs, metrics]` per model in model order, and
        prefix counts `[cases]`.

    Raises:
        ValueError: If required columns are missing, prefix keys are invalid or differ between
            runs, or scores are nonfinite.
    """
    values, prefixes = _aligned(dataset=dataset, files=files, metrics=metrics)
    if not all(np.isfinite(model).all() for model in values):
        raise ValueError(f'{dataset} has nonfinite scores for {", ".join(metrics)}.')
    cases = prefixes.get_level_values('case_id').to_numpy()
    starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    totals = tuple(np.add.reduceat(model, starts, axis=0) for model in values)
    counts = np.diff(np.r_[starts, len(cases)])
    yield list(metrics), totals, counts
