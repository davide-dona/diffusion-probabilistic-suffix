import numpy as np
import torch

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut, TraceDataset
from src.models.base import FittedSuffixModel
from src.models.contracts import GeneratedSuffix


class TrainCutModel(FittedSuffixModel):
    """A model that samples observed train suffixes and inter-event times, without parameters.

    It stores every train case and cut point in persistent buffers. Subclasses choose which cut
    points a prefix's suffixes are read from; teacher-forced times are drawn, for each true
    activity, from the train inter-event times of that activity.
    """

    def __init__(self, codec: DatasetCodec) -> None:
        """Create empty train tables, filled by `fit` or by loading a checkpoint."""
        super().__init__(codec=codec)
        self.register_buffer(
            name='case_activities', tensor=torch.zeros(size=(0, 0), dtype=torch.long)
        )
        self.register_buffer(name='case_inter_event_times', tensor=torch.zeros(size=(0, 0)))
        self.register_buffer(name='case_lengths', tensor=torch.zeros(size=(0,), dtype=torch.long))
        self.register_buffer(name='cuts', tensor=torch.zeros(size=(0, 2), dtype=torch.long))
        self._time_index: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, dataset: TraceDataset) -> None:
        """Store every case and cut point of the train split."""
        tables = dataset.case_cuts()
        if not len(tables.cuts):
            raise ValueError(f'{type(self).__name__} needs at least one train cut point')
        device = self.cuts.device
        self.case_activities = tables.activities.to(device)
        self.case_inter_event_times = tables.inter_event_times.to(device)
        self.case_lengths = tables.lengths.to(device)
        self.cuts = tables.cuts.to(device)
        self._invalidate()

    def _invalidate(self) -> None:
        """Drop every lookup derived from the train tables, after they change."""
        self._time_index = None

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
        self._invalidate()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _require_fitted(self) -> None:
        if not len(self.cuts):
            raise RuntimeError(f'{type(self).__name__} must be fitted before it generates')

    @torch.no_grad()
    def generate_times(self, item: TraceCut, *, num_samples: int) -> torch.Tensor:
        """Draw every true activity's time from the train times of the same activity.

        Draws are independent across positions and samples, uniform with replacement over the
        train events after each case's first, whose time is zero by construction. An activity
        no such event holds, EOT and PAD included, draws from all of them.

        Args:
            item: A batch from `TraceDataset`, read for its true suffix activities only.
            num_samples: How many time sequences to draw per prefix.
        Returns:
            Standardized times `[batch_size, num_samples, T]`, `T` the padded suffix width.
        """
        self._require_fitted()
        times, bounds = self._times_by_activity()
        suffix = item.suffix.activities.cpu().numpy()  # [B, T]
        lower, upper = bounds[suffix], bounds[suffix + 1]  # [B, T]
        empty = upper == lower
        lower = np.where(empty, 0, lower)
        upper = np.where(empty, len(times), upper)
        lower_t = torch.from_numpy(lower).unsqueeze(dim=1)  # [B, 1, T]
        span = torch.from_numpy(upper - lower).unsqueeze(dim=1)  # [B, 1, T]
        batch_size, width = suffix.shape
        # float64 keeps the scaled draw strictly below `span` for any realistic split size.
        offsets = (
            torch.rand(size=(batch_size, num_samples, width), dtype=torch.float64) * span
        ).long()
        drawn = torch.from_numpy(times)[lower_t + offsets]  # [B, S, T]
        return drawn.to(device=self.case_inter_event_times.device)

    def _times_by_activity(self) -> tuple[np.ndarray, np.ndarray]:
        """Group the train events after each case's first by activity.

        Returns:
            Their standardized times sorted by activity `[E]`, and the `[V + 1]` offsets where
            each activity index's run starts, then the end.
        """
        if self._time_index is None:
            activities = self.case_activities.cpu().numpy()
            times = self.case_inter_event_times.cpu().numpy()
            lengths = self.case_lengths.cpu().numpy()
            positions = np.arange(activities.shape[1])[None, :]
            timed = (positions >= 1) & (positions < lengths[:, None])  # [C, W]
            event_activities, event_times = activities[timed], times[timed]
            order = np.argsort(event_activities, kind='stable')
            event_activities, event_times = event_activities[order], event_times[order]
            bounds = np.searchsorted(
                event_activities, np.arange(self.codec.activity.num_rows + 1), side='left'
            )
            self._time_index = event_times, bounds
        return self._time_index

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
