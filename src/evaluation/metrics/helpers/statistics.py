import numpy as np


def energy_score(*, truth_sum: float, pair_sum: float, draws: float) -> float:
    """Reduce distance sums to the fair finite-sample energy score.

    ``truth_sum`` sums distances from each draw to the observation. ``pair_sum`` sums
    distances over ordered pairs of distinct draws. Folded samples use multiplicities,
    not normalized weights. A singleton has no diversity correction.
    """
    if draws <= 0:
        raise ValueError('energy score requires a positive draw count.')
    accuracy = truth_sum / draws
    if draws > 1:
        return accuracy - pair_sum / (2.0 * draws * (draws - 1.0))
    return accuracy


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean fair CRPS over columns using sorted absolute-distance sums."""
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    truth_sum = float(np.abs(draws - truth).sum(axis=0).mean())
    ordered = np.sort(draws, axis=0)
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return energy_score(truth_sum=truth_sum, pair_sum=2.0 * float(spread.mean()), draws=count)


def mae(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean absolute error over draws and predicted quantities."""
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    return float(np.abs(draws - truth).mean())


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage."""
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
