from dataclasses import replace

import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.base import SuffixModel
from src.models.contracts import GeneratedSuffix
from src.models.sutran.decoder import CausalDecoder
from src.models.sutran.embeddings import EventEmbeddings
from src.models.sutran.trace_encoder import TraceEncoder


class SuTraNModel[OutputT](SuffixModel):
    """Shared prefix encoding, teacher forcing, and duration conversion for SuTraN models."""

    decoder: CausalDecoder[OutputT]

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        super().__init__(codec=codec)
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )

    def forward(self, item: TraceCut) -> OutputT:
        """Predict all suffix positions in one causal teacher-forced pass."""
        prefix_pad_mask = item.prefix.pad_mask()
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)
        return self.decoder(
            suffix_activities=item.suffix.activities,
            prefix_encoded=prefix,
            prefix_pad_mask=prefix_pad_mask,
        )

    def _finish_generation(self, generated: GeneratedSuffix, *, batch_size: int) -> GeneratedSuffix:
        """Pad terminated rows, derive remaining time, and group independent sample rows."""
        positions = torch.arange(
            end=generated.inter_event_times.size(dim=1), device=generated.activities.device
        )
        kept = positions.unsqueeze(dim=0) < generated.lengths.unsqueeze(dim=1)  # [B * S, T]
        remaining = self._remaining_time(times=generated.inter_event_times, keep=kept)
        return self._per_sample(
            generated=replace(
                generated,
                activities=generated.activities.masked_fill(~kept, self.pad_activity_index),
                inter_event_times=generated.inter_event_times.masked_fill(~kept, 0.0),
                remaining_time=remaining,
                used_sentinel=generated.used_sentinel,
            ),
            batch_size=batch_size,
        )
