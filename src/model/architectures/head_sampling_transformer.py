from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.model.components.decoder import Decoder, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.trace_encoder import TraceEncoder
from src.model.models import ModelOutput, SuffixModel, time_loss
from src.training import Loss


class HeadSamplingTransformer(SuffixModel):
    """An encoder-decoder transformer that samples every suffix channel from output heads."""

    def __init__(self, config: DictConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
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

    def forward(self, item: TraceCut) -> ModelOutput:
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
        return ModelOutput(decoder=decoder_output)

    @torch.no_grad()
    def generate(
        self, item: TraceCut, *, num_samples: int
    ) -> GeneratedSuffix:
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
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        generated = self.decoder.generate(
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
        )
        return self._per_sample(generated=generated, batch_size=item.prefix.length.size(dim=0))

    def compute_loss(
        self, output: ModelOutput, batch: TraceCut
    ) -> tuple[torch.Tensor, Loss]:
        """Score a teacher-forced pass by equal-weight trace reconstruction loss."""
        # Every suffix position has one activity target, including EOT; every real event has two
        # time targets. `suffix.length` includes EOT.
        target_counts = (3 * batch.suffix.length - 2).to(
            dtype=output.decoder.activity_logits.dtype
        )

        activity_loss = F.cross_entropy(
            input=output.decoder.activity_logits.transpose(1, 2),
            target=batch.suffix.activities,
            ignore_index=self.pad_activity_index,
            reduction='none',
        ).sum(dim=1)

        inter_event_time_loss, inter_event_time_scale = time_loss(
            prediction=output.decoder.inter_event_times,
            target=batch.inter_event_times,
            batch=batch,
        )
        remaining_time_loss, remaining_time_scale = time_loss(
            prediction=output.decoder.remaining_times,
            target=batch.remaining_times,
            batch=batch,
        )

        reconstruction_loss = activity_loss + inter_event_time_loss + remaining_time_loss
        normalized_activity_loss = activity_loss / target_counts
        normalized_inter_event_time_loss = inter_event_time_loss / target_counts
        normalized_remaining_time_loss = remaining_time_loss / target_counts
        normalized_reconstruction_loss = reconstruction_loss / target_counts

        metrics = Loss(
            loss=normalized_reconstruction_loss.sum().item(),
            reconstruction_loss=normalized_reconstruction_loss.sum().item(),
            activity_loss=normalized_activity_loss.sum().item(),
            inter_event_time_loss=normalized_inter_event_time_loss.sum().item(),
            remaining_time_loss=normalized_remaining_time_loss.sum().item(),
            inter_event_time_scale_loss=(inter_event_time_scale / target_counts).sum().item(),
            remaining_time_scale_loss=(remaining_time_scale / target_counts).sum().item(),
        )
        return normalized_reconstruction_loss.mean(), metrics
