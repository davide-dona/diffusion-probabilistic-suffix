import pytest
import torch

from src.datasets.codec import FEATURE_TOKENS, CategoricalColumn, DatasetCodec, NumericColumn
from src.datasets.dataset import Events, TraceCut
from src.models import build_model
from tests.conftest import model_config


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that each model can produce finite loss and valid generated suffixes for a small batch.
def test_model_smoke(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=model_config(name), codec=codec).eval()

    with torch.no_grad():
        output = model(batch)
        loss, _ = model.compute_loss(output, batch)
        generated = model.generate(batch, num_samples=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert generated.activities.shape[:2] == (2, 2)
    assert generated.inter_event_times.shape == generated.activities.shape
    assert generated.lengths.shape == (2, 2)
    assert torch.all(generated.lengths >= 0)
    assert torch.all(generated.lengths <= generated.activities.size(-1))
    assert generated.remaining_time.shape == (2, 2)
    assert torch.isfinite(generated.remaining_time).all()


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that loss gradients are present and finite for every model.
def test_model_loss_backpropagates(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=model_config(name), codec=codec).train()
    output = model(batch)
    loss, _ = model.compute_loss(output, batch)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that generated suffixes do not depend on the ground-truth suffix supplied in the batch.
def test_generation_reads_prefix_only(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=model_config(name), codec=codec).eval()
    changed_suffix = batch.suffix._replace(
        activities=torch.full_like(batch.suffix.activities, fill_value=-1)
    )
    changed_batch = batch._replace(suffix=changed_suffix)

    torch.manual_seed(17)
    original = model.generate(batch, num_samples=2)
    torch.manual_seed(17)
    changed = model.generate(changed_batch, num_samples=2)

    assert torch.equal(original.activities, changed.activities)
    assert torch.equal(original.inter_event_times, changed.inter_event_times)
    assert torch.equal(original.lengths, changed.lengths)


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
# Checks that categorical and numeric event features are embedded by every model.
def test_event_features_embed_in_both_models(
    name: str, codec: DatasetCodec, batch: TraceCut
) -> None:
    feature = CategoricalColumn(
        column='feature', vocab=('x',), special_tokens=FEATURE_TOKENS, offset=1
    )
    featured_codec = codec.model_copy(
        update={
            'categorical_features': (feature,),
            'numeric_features': (NumericColumn(column='value', log=False, mean=0.0, std=1.0),),
        }
    )

    def with_features(events: Events) -> Events:
        shape = events.activities.shape
        return events._replace(
            categorical_attributes=torch.full((*shape, 1), 2, dtype=torch.long),
            numeric_attributes=torch.ones((*shape, 1)),
            numeric_attributes_present=torch.ones((*shape, 1)),
        )

    featured_batch = batch._replace(
        prefix=with_features(batch.prefix), suffix=with_features(batch.suffix)
    )
    model = build_model(config=model_config(name), codec=featured_codec).eval()
    output = model(featured_batch)
    loss, _ = model.compute_loss(output, featured_batch)
    assert torch.isfinite(loss)
