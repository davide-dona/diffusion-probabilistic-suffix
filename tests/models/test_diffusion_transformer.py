import torch

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models import build_model
from src.models.architectures.diffusion_transformer.components.process import CategoricalDiffusion
from src.models.contracts import DiffusionOutput
from tests.conftest import model_config


# Checks that diffusion posterior and reverse distributions are normalized and exclude initial EOT.
def test_diffusion_probabilities(codec: DatasetCodec) -> None:
    process = CategoricalDiffusion(codec=codec, steps=3, cosine_offset=0.008)
    clean = torch.tensor([[0, 1, 2], [2, 0, 1]])
    noisy = torch.tensor([[1, 1, 0], [0, 2, 1]])
    timestep = torch.tensor([1, 3])
    predicted = torch.full((2, 3, process.num_activities), 1 / process.num_activities)

    posterior = process.posterior(noisy=noisy, clean=clean, timestep=timestep)
    reverse = process.reverse_probabilities(noisy=noisy, predicted=predicted, timestep=timestep)
    constrained = process.prohibit_initial_eot(predicted)

    assert torch.allclose(posterior.sum(dim=-1), torch.ones((2, 3)), atol=1e-6)
    assert torch.allclose(reverse.sum(dim=-1), torch.ones((2, 3)), atol=1e-6)
    assert torch.allclose(constrained.sum(dim=-1), torch.ones((2, 3)), atol=1e-6)
    assert torch.equal(constrained[:, 0, process.eot_index], torch.zeros(2))


# Checks that the diffusion loss includes activities through EOT and times before EOT only.
def test_diffusion_canvas_masks(codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=model_config('diffusion_transformer'), codec=codec)
    output = model(batch)
    assert isinstance(output, DiffusionOutput)
    assert torch.equal(
        output.activity_mask, torch.tensor([[True, True, False], [True, True, True]])
    )
    assert torch.equal(output.time_mask, torch.tensor([[True, False, False], [True, True, False]]))
