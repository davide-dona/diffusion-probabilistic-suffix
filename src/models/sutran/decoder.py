from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from torch import nn

from src.datasets.dataset import Events
from src.models.contracts import GeneratedSuffix
from src.models.sutran.cache import LayerCache
from src.models.sutran.decoder_layer import DecoderLayer
from src.models.sutran.embeddings import EventEmbeddings


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
        suffix_activities: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> OutputT:
        """Predict an event for every position of a suffix at once.

        Args:
            suffix_activities: The ground-truth suffix activities, `[batch_size, seq_len]`. Read
                one step behind, so the decoder sees the truth up to each position rather than
                its own predictions.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            The per-position predictions.
        """
        decoder_input = self._teacher_forced_input(suffix_activities)
        if self.training and self.activity_dropout > 0.0:
            decoder_input = self._drop_activities(decoder_input)
        hidden, _ = self._run_layers(
            activities=decoder_input,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            start_position=0,
            caches=None,
        )  # [batch_size, seq_len, d_model]
        features = self.shared_layer(hidden)  # [batch_size, seq_len, head_hidden_dim]
        return self.predict(features)

    def _teacher_forced_input(self, suffix_activities: torch.Tensor) -> torch.Tensor:
        """Shift suffix activities `[B, T]` right behind SOS without changing the padded width."""
        start = torch.full(
            size=(suffix_activities.size(dim=0), 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=suffix_activities.device,
        )
        return torch.cat(
            tensors=(start, suffix_activities[:, :-1]), dim=1
        )  # [B, 1] + [B, T - 1] -> [B, T]

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
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        start_position: int,
        caches: list[LayerCache] | None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed a run of decoder inputs and push it through the stack.

        The one pass both teacher forcing and `generate` go through; `caches` is all that
        differs between them. Without one this reads a whole suffix under a causal mask, with
        one it reads the single event that follows what the caches already hold.

        Args:
            activities: The decoder input activities to read, `[batch_size, seq_len]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            start_position: Where in the suffix `activities` starts, for the positional encoding.
            caches: One per layer, from a previous call, or None to read from the beginning.
        Returns:
            The stack's output for the positions read, and the caches carrying them.
        """
        hidden = self.embedding_norm(
            self.dropout(
                self.embeddings(
                    events=self._blank_events(activities), start_position=start_position
                )
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

    def _blank_events(self, activities: torch.Tensor) -> Events:
        """Wrap decoder input activities as the events `EventEmbeddings` reads.

        The activity is the one channel the decoder feeds itself, so it is the one channel that
        may carry real content here: teacher forcing would otherwise hand it a ground-truth value
        that `generate` has only its own prediction of, and the two would read different things.
        Every other channel is blanked to the same PAD row or 0.0 scalar `generate` starts from,
        the predicted times included.

        Args:
            activities: Vocabulary indices to read as the activity channel, `[batch_size, seq_len]`.
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
            inter_event_times=torch.zeros(size=(batch_size, seq_len), device=device),
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

    @abstractmethod
    def predict(self, features: torch.Tensor) -> OutputT:
        """Read distribution parameters from shared decoder features."""

    @abstractmethod
    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one activity and standardized duration per row."""

    def generate(
        self,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        max_steps: int,
    ) -> GeneratedSuffix:
        """Decode independent rows until EOT or the generation cap."""
        batch_size = prefix_encoded.size(dim=0)
        device = prefix_encoded.device
        # What the decoder reads at each step: SOS first, exactly how `_teacher_forced_input`
        # opens, then the activity the previous step predicted.
        next_input = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=device,
        )

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
                activities=next_input,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                start_position=position,
                caches=caches,
            )
            features = self.shared_layer(hidden[:, 0])  # [batch_size, head_hidden_dim]
            activities, times = self.sample(features)
            generated_activities[:, position] = activities
            generated_inter_event_times[:, position] = times
            next_input = activities.unsqueeze(dim=1)  # [batch_size, 1]

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
            remaining_time=generated_inter_event_times.new_zeros(size=(batch_size,)),
            used_sentinel=lengths.eq(max_steps),
        )
