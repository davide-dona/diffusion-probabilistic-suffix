import numpy as np
import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut, TraceDataset
from src.models.base import FittedSuffixModel
from src.models.contracts import GeneratedSuffix

# Marks the start of a trace in a context, below every activity index.
_START = -1


class PrefixRetrieval(FittedSuffixModel):
    """Sample real train suffixes whose preceding activities match the prefix's latest ones.

    A context is the run of activities right before a cut point, read backwards and closed by a
    start-of-trace marker. For a prefix, the matched context is the longest one shared with at
    least one train cut point: the whole prefix when a train case began with exactly these
    activities, otherwise its last k activities for the largest k some cut point shares, and
    every cut point when not even the last activity was seen. Each sample is then the observed
    suffix, activities and inter-event times, of a cut point drawn uniformly from those matches.
    """

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        """Create empty train tables, filled by `fit` or by loading a checkpoint."""
        super().__init__(codec=codec)
        self.register_buffer(
            name='case_activities', tensor=torch.zeros(size=(0, 0), dtype=torch.long)
        )
        self.register_buffer(name='case_inter_event_times', tensor=torch.zeros(size=(0, 0)))
        self.register_buffer(name='case_lengths', tensor=torch.zeros(size=(0,), dtype=torch.long))
        self.register_buffer(name='cuts', tensor=torch.zeros(size=(0, 2), dtype=torch.long))
        self._index: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, dataset: TraceDataset) -> None:
        """Store every case and cut point of the train split."""
        tables = dataset.case_cuts()
        if not len(tables.cuts):
            raise ValueError('Prefix retrieval needs at least one train cut point')
        device = self.cuts.device
        self.case_activities = tables.activities.to(device)
        self.case_inter_event_times = tables.inter_event_times.to(device)
        self.case_lengths = tables.lengths.to(device)
        self.cuts = tables.cuts.to(device)
        self._index = None

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Resize the train tables to the stored ones, whose sizes depend on the split."""
        for name, buffer in self._buffers.items():
            stored = state_dict.get(prefix + name)
            if buffer is not None and stored is not None:
                self._buffers[name] = torch.empty_like(stored, device=buffer.device)
        self._index = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
        if not len(self.cuts):
            raise RuntimeError('Prefix retrieval must be fitted before it generates')
        order, contexts = self._contexts()
        prefix_activities = item.prefix.activities.cpu().numpy()
        prefix_lengths = item.prefix.length.cpu().numpy()
        bounds = np.array(
            [
                self._match(contexts, prefix_activities[row, :length])
                for row, length in enumerate(prefix_lengths)
            ]
        )  # [B, 2]
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
    def _match(contexts: np.ndarray, prefix: np.ndarray) -> tuple[int, int]:
        """Narrow the sorted contexts to the longest run shared with `prefix`.

        Args:
            contexts: The sorted contexts, from `_contexts`.
            prefix: One prefix's real activities, oldest first.
        Returns:
            The `[lower, upper)` range of sorted cut points sharing the longest matched context,
            every cut point when not even the last activity matches.
        """
        lower, upper = 0, len(contexts)
        query = np.append(prefix[::-1], _START)
        for depth, token in enumerate(query[: contexts.shape[1]]):
            column = contexts[lower:upper, depth]
            first = lower + int(np.searchsorted(column, token, side='left'))
            last = lower + int(np.searchsorted(column, token, side='right'))
            if first == last:
                break
            lower, upper = first, last
        return lower, upper

    def _suffixes(self, cut_ids: torch.Tensor, *, cap: int) -> GeneratedSuffix:
        """Read the observed suffix after each cut point, closed by EOT and padded.

        Args:
            cut_ids: Which cut points to read, `[R]`.
            cap: The most positions a suffix may take; a longer one is cut there without EOT.
        Returns:
            The suffixes, `[R, steps]`, with `steps` the widest of them plus its EOT, at most
            `cap`.
        """
        case = self.cuts[cut_ids, 0]  # [R]
        start = self.cuts[cut_ids, 1]  # [R]
        lengths = self.case_lengths[case] - start  # [R]
        steps = min(cap, int(lengths.max()) + 1)
        positions = torch.arange(end=steps, device=cut_ids.device)  # [T]
        source = (start.unsqueeze(dim=1) + positions.unsqueeze(dim=0)).clamp(
            max=self.case_activities.size(dim=1) - 1
        )  # [R, T]
        rows = case.unsqueeze(dim=1)  # [R, 1]
        real = positions.unsqueeze(dim=0) < lengths.unsqueeze(dim=1)  # [R, T]
        activities = (
            self.case_activities[rows, source]
            .masked_fill(mask=~real, value=self.pad_activity_index)
            .masked_fill(
                mask=positions.unsqueeze(dim=0) == lengths.unsqueeze(dim=1),
                value=self.eot_activity_index,
            )
        )
        return GeneratedSuffix(
            activities=activities,
            lengths=lengths.clamp(max=steps),
            inter_event_times=self.case_inter_event_times[rows, source].masked_fill(
                mask=~real, value=0.0
            ),
            used_sentinel=lengths >= cap,
        )
