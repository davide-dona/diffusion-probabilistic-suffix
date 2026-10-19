import torch

from src.datasets.dataset import TraceCut
from src.models.architectures.case_based.model import CaseBased
from src.models.contracts import GeneratedSuffix


class Unconditional(CaseBased):
    """Sample real train suffixes regardless of the prefix.

    Each sample is the observed suffix, activities and inter-event times, of a train cut point
    drawn uniformly from all of them, so the forecast is the marginal suffix distribution over
    train prefixes. It is the reference forecast of skill scores, and shares the case-based
    model's train tables with a match of depth zero for every prefix.
    """

    @torch.no_grad()
    def generate(self, item: TraceCut, *, num_samples: int) -> GeneratedSuffix:
        """Draw `num_samples` train suffixes for every prefix in `item`.

        Args:
            item: A batch from `TraceDataset`, read for its prefix shape only.
            num_samples: How many suffixes to draw per prefix, independently and with replacement.
        Returns:
            The suffixes, `[batch_size, num_samples, steps]`. A suffix longer than the padded
            prefix width, the cap every generator shares, is cut there and marked as such.
        """
        if not len(self.cuts):
            raise RuntimeError('The unconditional model must be fitted before it generates')
        batch_size = item.prefix.activities.size(dim=0)
        drawn = torch.randint(high=len(self.cuts), size=(batch_size * num_samples,))  # [B * S]
        return self._per_sample(
            generated=self._suffixes(
                cut_ids=drawn.to(self.cuts.device), cap=item.prefix.activities.size(dim=1)
            ),
            batch_size=batch_size,
        )

    def generation_diagnostics(self, item: TraceCut) -> dict[str, torch.Tensor]:
        """Return no diagnostics, since no draw depends on the prefix."""
        return {}
