from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from src.datasets.codec.continuous import NumericColumn
from src.datasets.dataset import Events
from src.models.backbones.autoregressive.cache import LayerCache
from src.models.backbones.autoregressive.decoder_layer import DecoderLayer
from src.models.backbones.autoregressive.embeddings import EventEmbeddings
from src.models.contracts import GeneratedSuffix


class CausalDecoder[OutputT](nn.Module, ABC):
    """Transformer decoder that predicts sampled suffixes from encoded prefixes."""

    def __init__(
        self,
        config: DictConfig,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        sos_activity_index: int,
        pad_activity_index: int,
        pad_resource_index: int,
        eot_activity_index: int,
        input_inter_event_time: NumericColumn,
        inter_event_time: NumericColumn,
    ) -> None:
        """
        Args:
            config: The decoder's own hyperparameters.
            embeddings: The event embeddings, shared with the encoder.
            d_model: The width the stack runs at.
            sos_activity_index: What the first decoder input is, in both teacher forcing and
                free-running generation.
            pad_activity_index: What `activity_dropout` blanks a teacher-forced activity to.
            pad_resource_index: The resource row every decoder input carries, the channel being
                one the decoder never feeds itself.
            eot_activity_index: What ends a generated suffix.
            input_inter_event_time: The encoding of the time channel the decoder reads.
            inter_event_time: The encoding of the times the decoder predicts, which `generate`
                re-encodes through `input_inter_event_time` before feeding them back.
        """
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        # The embeddings are shared with the encoder but this norm is not: the two stacks read one
        # embedding space at whatever scale each of them settles on.
        self.embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.activity_dropout = config.activity_dropout

        self.sos_activity_index = sos_activity_index
        self.pad_activity_index = pad_activity_index
        self.pad_resource_index = pad_resource_index
        self.eot_activity_index = eot_activity_index
        self.input_inter_event_time = input_inter_event_time
        self.inter_event_time = inter_event_time

        self.layers = nn.ModuleList(
            DecoderLayer(config=config, d_model=d_model) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
        # with a norm of its own.
        self.norm = nn.LayerNorm(normalized_shape=d_model)

        # EOT and UNK remain valid output classes; only structural PAD and SOS are masked.
        self.register_buffer(
            name='unemittable_activities',
            tensor=torch.tensor(data=[pad_activity_index, sos_activity_index], dtype=torch.long),
            persistent=False,
        )

        # A trunk shared by every head, so the heads can be smaller.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )

    def forward(
        self,
        suffix: Events,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> OutputT:
        """Predict an event for every position of a suffix at once.

        Args:
            suffix: The ground-truth suffix events, `[batch_size, seq_len]` per field. Their
                activities and input-encoded times are read one step behind, so the decoder sees
                the truth up to each position rather than its own predictions.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            The per-position predictions.
        """
        activities = self._teacher_forced(suffix.activities, start=self.sos_activity_index)
        times = self._teacher_forced(suffix.inter_event_times, start=0.0)
        if self.training and self.activity_dropout > 0.0:
            activities = self._drop_activities(activities)
        hidden, _ = self._run_layers(
            activities=activities,
            times=times,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            start=prefix_pad_mask.logical_not().sum(dim=1),
            caches=None,
        )  # [batch_size, seq_len, d_model]
        features = self.shared_layer(hidden)  # [batch_size, seq_len, head_hidden_dim]
        return self.predict(features, suffix.activities)

    @staticmethod
    def _teacher_forced(channel: torch.Tensor, *, start: float) -> torch.Tensor:
        """Shift a suffix channel `[B, T]` right behind `start` without changing its width."""
        return F.pad(input=channel[:, :-1], pad=(1, 0), value=start)  # [B, T - 1] -> [B, T]

    def _drop_activities(self, activities: torch.Tensor) -> torch.Tensor:
        """Replace teacher-forced activities with PAD at the configured rate, preserving SOS."""
        dropped = (
            torch.rand_like(input=activities, dtype=torch.float32) < self.activity_dropout
        ) & (activities != self.sos_activity_index)  # [batch_size, seq_len]
        return activities.masked_fill(mask=dropped, value=self.pad_activity_index)

    def _run_layers(
        self,
        *,
        activities: torch.Tensor,
        times: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        start: torch.Tensor,
        caches: list[LayerCache] | None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed a run of decoder inputs and push it through the stack.

        The one pass both teacher forcing and `generate` go through; `caches` is all that
        differs between them. Without one this reads a whole suffix under a causal mask, with
        one it reads the single event that follows what the caches already hold.

        Args:
            activities: The decoder input activities to read, `[batch_size, seq_len]`.
            times: The decoder input times in the input encoding, `[batch_size, seq_len]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            start: The trace position of each row's first input, `[batch_size]`. A step sits
                at the trace position of the event it predicts.
            caches: One per layer, from a previous call, or None to read from the beginning.
        Returns:
            The stack's output for the positions read, and the caches carrying them.
        """
        hidden = self.embedding_norm(
            self.dropout(
                self.embeddings(events=self._decoder_events(activities, times), start=start)
            )
        )  # [batch_size, seq_len, d_model]

        # No suffix padding mask: under the causal mask a padded position is visible only to
        # later positions, themselves padded and dropped from the loss. Masking it here would
        # leave a row with nothing to attend to, whose softmax is a NaN.
        layer_caches = caches if caches is not None else [None] * len(self.layers)
        new_caches: list[LayerCache] = []
        for layer, cache in zip(self.layers, layer_caches, strict=True):
            hidden, layer_cache = layer(
                hidden=hidden,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                cache=cache,
            )
            new_caches.append(layer_cache)

        return self.norm(hidden), new_caches

    def _decoder_events(self, activities: torch.Tensor, times: torch.Tensor) -> Events:
        """Wrap decoder input activities and times as the events `EventEmbeddings` reads.

        Activity and time are the channels the decoder feeds itself, so they are the only ones
        that may carry real content here: teacher forcing would otherwise hand it ground-truth
        values that `generate` has no prediction of, and the two would read different things.
        Every other channel is blanked to a PAD row or 0.0 scalar.

        Args:
            activities: Vocabulary indices to read as the activity channel, `[batch_size, seq_len]`.
            times: Input-encoded times to read as the time channel, `[batch_size, seq_len]`.
        Returns:
            The events, ready for `EventEmbeddings`.
        """
        batch_size, seq_len = activities.shape
        device = activities.device
        return Events(
            activities=activities,
            resources=torch.full(
                size=(batch_size, seq_len),
                fill_value=self.pad_resource_index,
                dtype=torch.long,
                device=device,
            ),
            inter_event_times=times,
            categorical_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_categorical),
                dtype=torch.long,
                device=device,
            ),
            numeric_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            numeric_attributes_present=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            # Every position is read: the causal mask is what keeps a position from seeing
            # what follows it, and no caller asks these events for a padding mask.
            length=torch.full(
                size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device
            ),
        )

    def _input_time(self, times: torch.Tensor) -> torch.Tensor:
        """Re-encode predicted standardized times `[...]` as the input time channel.

        A negative duration is read as zero minutes, since the log1p input encoding is undefined
        below that. A log-scaled prediction is clamped in log space instead of passing through
        expm1, which overflows on large draws.
        """
        target = self.inter_event_time
        scaled = (times * target.std + target.mean).clamp_min(0.0)
        log_minutes = scaled if target.log else scaled.log1p()
        return (log_minutes - self.input_inter_event_time.mean) / self.input_inter_event_time.std

    @abstractmethod
    def predict(self, features: torch.Tensor, activities: torch.Tensor) -> OutputT:
        """Read distribution parameters from shared decoder features.

        Args:
            features: The shared decoder features, `[batch_size, seq_len, head_hidden_dim]`.
            activities: The target activity at each position, `[batch_size, seq_len]`, for heads
                that condition on the activity being written there.
        """

    @abstractmethod
    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one activity and standardized duration per row."""

    @abstractmethod
    def sample_time(self, features: torch.Tensor, activities: torch.Tensor) -> torch.Tensor:
        """Draw one standardized duration per row for the activity `[rows]` written there."""

    def generate_times(
        self,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        activities: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a duration for every given activity, reading the activities as the history.

        Each step reads the given activity of the previous step and the duration drawn for it,
        exactly as `generate` reads its own draws, so only the activities are teacher-forced.

        Args:
            prefix_encoded: The encoded prefix events, `[rows, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            activities: The suffix activities to time, `[rows, steps]`.
        Returns:
            Standardized durations, `[rows, steps]`.
        """
        rows, steps = activities.shape
        prefix_length = prefix_pad_mask.logical_not().sum(dim=1)  # [rows]
        next_activity = torch.full(
            size=(rows, 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=activities.device,
        )
        next_time = torch.zeros(size=(rows, 1), device=activities.device)
        times = torch.zeros(
            size=(rows, steps), dtype=prefix_encoded.dtype, device=activities.device
        )
        caches = [
            layer.init_cache(prefix_encoded=prefix_encoded, max_steps=steps)
            for layer in self.layers
        ]
        for position in range(steps):
            hidden, caches = self._run_layers(
                activities=next_activity,
                times=next_time,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                start=prefix_length + position,
                caches=caches,
            )
            features = self.shared_layer(hidden[:, 0])  # [rows, head_hidden_dim]
            sampled = self.sample_time(features, activities[:, position])  # [rows]
            times[:, position] = sampled
            next_activity = activities[:, position : position + 1]  # [rows, 1]
            next_time = self._input_time(sampled).unsqueeze(dim=1)  # [rows, 1]
        return times

    def generate(
        self,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        max_steps: int,
    ) -> GeneratedSuffix:
        """Decode independent rows until EOT or the generation cap."""
        batch_size = prefix_encoded.size(dim=0)
        device = prefix_encoded.device
        prefix_length = prefix_pad_mask.logical_not().sum(dim=1)  # [batch_size]
        # What the decoder reads at each step: SOS at time 0.0 first, exactly how teacher forcing
        # opens, then the event the previous step predicted.
        next_activity = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=device,
        )
        next_time = torch.zeros(size=(batch_size, 1), device=device)

        generated_activities = torch.zeros(
            size=(batch_size, max_steps), dtype=torch.long, device=device
        )
        generated_inter_event_times = torch.zeros(
            size=(batch_size, max_steps), dtype=prefix_encoded.dtype, device=device
        )
        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

        steps_taken = max_steps
        # Seeded before the loop rather than left None for its first iteration: every layer's
        # suffix cache is preallocated to `max_steps` right away, so even the first step writes
        # into it in place instead of starting the cache off at its exact size.
        caches: list[LayerCache] = [
            layer.init_cache(prefix_encoded=prefix_encoded, max_steps=max_steps)
            for layer in self.layers
        ]
        for position in range(max_steps):
            # Only this one position is new; everything before it is in `caches`.
            hidden, caches = self._run_layers(
                activities=next_activity,
                times=next_time,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                start=prefix_length + position,
                caches=caches,
            )
            features = self.shared_layer(hidden[:, 0])  # [batch_size, head_hidden_dim]
            activities, times = self.sample(features)
            generated_activities[:, position] = activities
            generated_inter_event_times[:, position] = times
            next_activity = activities.unsqueeze(dim=1)  # [batch_size, 1]
            next_time = self._input_time(times).unsqueeze(dim=1)  # [batch_size, 1]

            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activities == self.eot_activity_index)
            lengths = lengths.masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            # Reading this stalls the device queue once per step, but suffixes are far shorter
            # than `max_steps` on every log here, so most of the loop is skipped outright.
            if bool(finished.all()):
                steps_taken = position + 1
                break

        return GeneratedSuffix(
            activities=generated_activities[:, :steps_taken],  # [batch_size, steps]
            lengths=lengths,
            inter_event_times=generated_inter_event_times[:, :steps_taken],  # [batch_size, steps]
            used_sentinel=lengths.eq(max_steps),
        )
