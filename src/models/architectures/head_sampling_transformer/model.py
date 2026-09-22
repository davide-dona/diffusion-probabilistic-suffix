from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.architectures.head_sampling_transformer.components.decoder import Decoder
from src.models.architectures.head_sampling_transformer.components.embeddings import EventEmbeddings
from src.models.architectures.head_sampling_transformer.components.trace_encoder import TraceEncoder
from src.models.contracts import DecoderOutput, GeneratedSuffix
from src.models.models import SuffixModel
from src.models.time import remaining_time_from_inter_event_times
from src.training import Loss


def _timed_positions(batch: TraceCut) -> torch.Tensor:
    """Mark suffix positions containing real events and time targets, `[B, T]`."""
    positions = torch.arange(
        end=batch.suffix.activities.size(dim=1), device=batch.suffix.length.device
    )  # [T]
    return positions.unsqueeze(dim=0) < (batch.suffix.length - 1).unsqueeze(dim=1)  # [B, T]


class HeadSamplingTransformer(SuffixModel):
    """An encoder-decoder transformer that samples every suffix channel from output heads."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        """Build the shared embeddings, prefix encoder, and autoregressive decoder."""
        super().__init__(codec=codec)
        self.codec = codec
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        # The encoder reads the prefix that conditions each generated suffix.
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.decoder = Decoder(
            config=config.decoder,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
            sampling=config.sampling,
        )

    def forward(self, item: TraceCut) -> DecoderOutput:
        """
        Args:
            item: A batch from `TraceDataset`, read for its prefix and for the suffix the
                decoder is teacher-forced on.
        Returns:
            The decoder's teacher-forced predictions.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        decoder_output = self.decoder(
            suffix_activities=item.suffix.activities,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return decoder_output

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix. Every row decodes independently,
                so `num_samples` rows of one prefix give `num_samples` different suffixes.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # The prefix is encoded once and repeated per sample, so the decoder writes every sample
        # of the batch in one pass.
        prefix_events = prefix.events.repeat_interleave(
            repeats=num_samples, dim=0
        )  # [B, P, D] -> [B * S, P, D]
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(
            repeats=num_samples, dim=0
        )  # [B, P] -> [B * S, P]

        generated = self.decoder.generate(
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
        )
        positions = torch.arange(
            generated.inter_event_times.size(dim=1), device=generated.activities.device
        )  # [T]
        kept = positions.unsqueeze(dim=0) < generated.lengths.unsqueeze(dim=1)  # [B * S, T]
        remaining = remaining_time_from_inter_event_times(
            times=generated.inter_event_times, keep=kept, codec=self.codec
        )  # [B * S]
        generated = replace(generated, remaining_time=remaining)
        return self._per_sample(generated=generated, batch_size=item.prefix.length.size(dim=0))

    def compute_loss(self, output: DecoderOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Score a teacher-forced pass by equal-weight trace reconstruction loss."""
        # Every suffix position has one activity target, including EOT; each real event has one
        # Gaussian inter-event-time target. `suffix.length` includes EOT.
        target_counts = (2 * batch.suffix.length - 1).to(dtype=output.activity_logits.dtype)

        activity_loss = F.cross_entropy(
            input=output.activity_logits.transpose(1, 2),  # [B, T, V] -> [B, V, T]
            target=batch.suffix.activities,
            ignore_index=self.pad_activity_index,
            reduction='none',
        ).sum(dim=1)

        inter_event_time_loss = (
            (output.inter_event_times - batch.inter_event_times)
            .square()
            .masked_fill(~_timed_positions(batch), 0.0)
            .sum(dim=1)
        )

        reconstruction_loss = activity_loss + inter_event_time_loss
        normalized_activity_loss = activity_loss / target_counts
        normalized_inter_event_time_loss = inter_event_time_loss / target_counts
        normalized_reconstruction_loss = reconstruction_loss / target_counts

        metrics = Loss(
            loss=normalized_reconstruction_loss.sum().item(),
            reconstruction_loss=normalized_reconstruction_loss.sum().item(),
            activity_loss=normalized_activity_loss.sum().item(),
            inter_event_time_loss=normalized_inter_event_time_loss.sum().item(),
        )
        return normalized_reconstruction_loss.mean(), metrics
