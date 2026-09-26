import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.architectures.head_sampling_transformer.decoder import Decoder
from src.models.contracts import DecoderOutput, GeneratedSuffix
from src.models.sutran.loss import (
    reconstruction_loss,
    timed_positions,
)
from src.models.sutran.model import SuTraNModel
from src.training.loss import Loss


class HeadSamplingTransformer(SuTraNModel[DecoderOutput]):
    """An encoder-decoder transformer that samples every suffix channel from output heads."""

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        """Build the shared embeddings, prefix encoder, and autoregressive decoder."""
        super().__init__(config=config, codec=codec)
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
        prefix_events = prefix.repeat_interleave(
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
        return self._finish_generation(
            generated=generated, batch_size=item.prefix.length.size(dim=0)
        )

    def compute_loss(self, output: DecoderOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Score a teacher-forced pass by equal-weight trace reconstruction loss."""
        activity_loss = F.cross_entropy(
            input=output.activity_logits.transpose(dim0=1, dim1=2),  # [B, T, V] -> [B, V, T]
            target=batch.suffix.activities,
            ignore_index=self.pad_activity_index,
            reduction='none',
        ).sum(dim=1)

        inter_event_time_loss = (
            (output.inter_event_times - batch.inter_event_times)
            .square()
            .masked_fill(mask=~timed_positions(batch), value=0.0)
            .sum(dim=1)
        )

        return reconstruction_loss(
            activity_loss=activity_loss,
            time_loss=inter_event_time_loss,
            lengths=batch.suffix.length,
        )
