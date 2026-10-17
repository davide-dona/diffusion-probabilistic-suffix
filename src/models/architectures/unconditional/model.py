import torch
from omegaconf import DictConfig

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models.backbones.retrieval import TrainCutModel
from src.models.contracts import GeneratedSuffix


class Unconditional(TrainCutModel):
    """Sample real train suffixes while ignoring the prefix: the reference of skill scores.

    Each sample is the observed suffix, activities and inter-event times, of a train cut point
    drawn uniformly from all of them, so every prefix receives the same forecast. Teacher-forced
    times draw from the train times of each activity.
    """

    def __init__(self, config: DictConfig, codec: DatasetCodec) -> None:
        """Create empty train tables, filled by `fit` or by loading a checkpoint."""
        super().__init__(codec=codec)

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Draw `num_samples` train suffixes for every prefix in `item`, independent of it.

        Args:
            item: A batch from `TraceDataset`, read for its batch size and padded width only.
            num_samples: How many suffixes to draw per prefix, independently and with
                replacement.
        Returns:
            The suffixes, `[batch_size, num_samples, steps]`. A suffix longer than the padded
            prefix width, the cap every generator shares, is cut there and marked as such.
        """
        self._require_fitted()
        batch_size = item.prefix.length.size(dim=0)
        drawn = torch.randint(
            high=len(self.cuts), size=(batch_size * num_samples,), device=self.cuts.device
        )  # [B * S]
        return self._per_sample(
            generated=self._suffixes(cut_ids=drawn, cap=item.prefix.activities.size(dim=1)),
            batch_size=batch_size,
        )
