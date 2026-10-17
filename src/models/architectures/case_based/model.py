import numpy as np
import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.backbones.retrieval import TrainCutModel
from src.models.contracts import GeneratedSuffix

# Marks the start of a trace in a context, below every activity index.
_START = -1


class CaseBased(TrainCutModel):
    """Sample real train suffixes whose preceding activities match the prefix's latest ones.

    A context is the run of activities right before a cut point, read backwards and closed by a
    start-of-trace marker. For a prefix, the matched context is the longest one shared with at
    least one train cut point: the whole prefix when a train case began with exactly these
    activities, otherwise its last k activities for the largest k some cut point shares, and
    every cut point when not even the last activity was seen. Each sample is then the observed
    suffix, activities and inter-event times, of a cut point drawn uniformly from those matches.
    Teacher-forced times ignore the match and draw from the train times of each activity.
    """

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        """Create empty train tables, filled by `fit` or by loading a checkpoint."""
        super().__init__(codec=codec)
        self._index: tuple[np.ndarray, np.ndarray] | None = None

    def _invalidate(self) -> None:
        super()._invalidate()
        self._index = None

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Draw `num_samples` retrieved suffixes for every prefix in `item`.

        Args:
            item: A batch from `TraceDataset`, read for its prefix activities only.
            num_samples: How many suffixes to draw per prefix, independently and with replacement,
                so a context matched by one cut point repeats that cut point's suffix.
        Returns:
            The suffixes, `[batch_size, num_samples, steps]`. A suffix longer than the padded
            prefix width, the cap every generator shares, is cut there and marked as such.
        """
        bounds = self._matches(item)[:, :2]  # [B, 2]
        order, _ = self._contexts()
        lower = torch.from_numpy(bounds[:, :1])
        span = torch.from_numpy(bounds[:, 1:] - bounds[:, :1])
        batch_size = len(bounds)
        # float64 keeps the scaled draw strictly below `span` for any realistic split size.
        offsets = (torch.rand(size=(batch_size, num_samples), dtype=torch.float64) * span).long()
        drawn = torch.from_numpy(order)[lower + offsets].flatten()  # [B * S]
        return self._per_sample(
            generated=self._suffixes(
                cut_ids=drawn.to(self.cuts.device), cap=item.prefix.activities.size(dim=1)
            ),
            batch_size=batch_size,
        )

    @torch.no_grad()
    def generation_diagnostics(self, item: TraceCut) -> dict[str, torch.Tensor]:
        """Describe the train cut points each prefix's draws are retrieved from.

        Args:
            item: A batch from `TraceDataset`, read for its prefix activities only.
        Returns:
            Per-prefix values, `[batch_size]` each: `match_depth`, how many of the latest prefix
            activities matched; `match_depth_share`, that count over the prefix length, zero for
            an empty prefix; `dropped_activities`, how many of the oldest prefix activities were
            left out of the match, zero when all of them matched; `exact_match`, 1.0 when the
            whole prefix matched from the trace start; `fallback`, 1.0 when not even the last
            activity matched; `log10_candidates`, the base-10 log of how many cut points matched;
            and `single_candidate`, 1.0 when one cut point matched, so every draw repeats its
            suffix.
        """
        matches = self._matches(item)  # [B, 3]
        prefix_lengths = item.prefix.length.cpu().numpy()  # [B]
        candidates = matches[:, 1] - matches[:, 0]  # [B]
        depth = np.minimum(matches[:, 2], prefix_lengths)  # [B]
        values = {
            'match_depth': depth,
            'match_depth_share': depth / np.maximum(prefix_lengths, 1),
            'dropped_activities': prefix_lengths - depth,
            'exact_match': matches[:, 2] > prefix_lengths,
            'fallback': depth == 0,
            'log10_candidates': np.log10(candidates),
            'single_candidate': candidates == 1,
        }
        return {key: torch.from_numpy(value.astype(np.float64)) for key, value in values.items()}

    def _matches(self, item: TraceCut) -> np.ndarray:
        """Match every prefix of `item` against the sorted train contexts.

        Returns:
            One `[lower, upper, depth]` row per prefix, `[B, 3]`, as `_match` returns them.
        """
        self._require_fitted()
        _, contexts = self._contexts()
        prefix_activities = item.prefix.activities.cpu().numpy()
        prefix_lengths = item.prefix.length.cpu().numpy()
        return np.array(
            [
                self._match(contexts, prefix_activities[row, :length])
                for row, length in enumerate(prefix_lengths)
            ],
            dtype=np.int64,
        ).reshape(-1, 3)

    def _contexts(self) -> tuple[np.ndarray, np.ndarray]:
        """Sort the train cut points by context, so each context prefix is a contiguous range.

        Returns:
            The cut ids in sorted order `[N]`, and their contexts `[N, W]` in the same order and
            column-major, so each depth reads as one contiguous column. Position `d` holds the
            activity `d + 1` events before the cut, `_START` from the trace start onwards.
        """
        if self._index is None:
            cuts = self.cuts.cpu().numpy()
            activities = self.case_activities.cpu().numpy()
            case, cut = cuts[:, 0], cuts[:, 1]
            depth = np.arange(int(cut.max()) + 1)
            positions = cut[:, None] - 1 - depth[None, :]  # [N, W]
            contexts = np.where(
                positions >= 0,
                activities[case[:, None], np.maximum(positions, 0)],
                _START,
            ).astype(np.int32)
            # `lexsort` sorts by its last key first, so depth 0 goes last.
            order = np.lexsort(contexts.T[::-1])
            self._index = order, np.asfortranarray(contexts[order])
        return self._index

    @staticmethod
    def _match(contexts: np.ndarray, prefix: np.ndarray) -> tuple[int, int, int]:
        """Narrow the sorted contexts to the longest run shared with `prefix`.

        Args:
            contexts: The sorted contexts, from `_contexts`.
            prefix: One prefix's real activities, oldest first.
        Returns:
            The `[lower, upper)` range of sorted cut points sharing the longest matched context,
            every cut point when not even the last activity matches, and the matched depth: how
            many context positions matched, counting the start-of-trace marker after the whole
            prefix.
        """
        lower, upper, matched = 0, len(contexts), 0
        query = np.append(prefix[::-1], _START)
        for depth, token in enumerate(query[: contexts.shape[1]]):
            column = contexts[lower:upper, depth]
            first = lower + int(np.searchsorted(column, token, side='left'))
            last = lower + int(np.searchsorted(column, token, side='right'))
            if first == last:
                break
            lower, upper, matched = first, last, depth + 1
        return lower, upper, matched
