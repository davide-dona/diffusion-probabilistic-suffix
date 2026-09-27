from collections.abc import Iterator, Sequence

import numpy as np

BATCH_SIZE = 250


def resample_means(
    totals: Sequence[np.ndarray],
    counts: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield two-level paired bootstrap means in batches of at most `BATCH_SIZE`.

    Each draw resamples cases once for all models, keeping the comparison paired, then resamples
    each model's runs independently. A model with one run draws no runs, so with one run per model
    the draws reduce to a paired case bootstrap.

    Args:
        totals: Summed prefix scores per case and run, one `[cases, runs, ...]` array per model.
        counts: Eligible prefix counts per case, `[cases]`, all positive.
        resamples: Positive number of bootstrap draws.
        generator: Random source shared by all models and metrics in each draw.

    Yields:
        `[batch, models, ...]` means, with every eligible prefix weighted equally.
    """
    cases = len(counts)
    flat = [model.reshape(cases, model.shape[1], -1) for model in totals]
    for start in range(0, resamples, BATCH_SIZE):
        size = min(BATCH_SIZE, resamples - start)
        picks = generator.integers(low=0, high=cases, size=(size, cases))
        picks += (np.arange(size) * cases)[:, None]
        multiplicities = np.bincount(picks.ravel(), minlength=size * cases).reshape(size, cases)
        prefixes = (multiplicities @ counts)[:, None]
        means = []
        for model in flat:
            runs = model.shape[1]
            if runs == 1:
                means.append((multiplicities @ model[:, 0]) / prefixes)
                continue
            weights = _run_weights(runs=runs, size=size, generator=generator)
            means.append(np.einsum('bc,br,crk->bk', multiplicities, weights, model) / prefixes)
        yield np.stack(means, axis=1).reshape(size, len(flat), *totals[0].shape[2:])


def _run_weights(*, runs: int, size: int, generator: np.random.Generator) -> np.ndarray:
    """Draw `[size, runs]` run weights that resample runs with replacement and sum to one."""
    picks = generator.integers(low=0, high=runs, size=(size, runs))
    picks += (np.arange(size) * runs)[:, None]
    counts = np.bincount(picks.ravel(), minlength=size * runs).reshape(size, runs)
    return counts / runs
