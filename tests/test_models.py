from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.datasets.codec import (
    ACTIVITY_TOKENS,
    FEATURE_TOKENS,
    RESOURCE_TOKENS,
    CategoricalColumn,
    DatasetCodec,
    NumericColumn,
)
from src.datasets.dataset import Events, TraceCut
from src.models import build_model, load_checkpoint, model_from_checkpoint, save_checkpoint
from src.models.architectures.diffusion_transformer.components.process import CategoricalDiffusion
from src.models.contracts import DecoderOutput, DiffusionOutput
from src.runs.identity import RunIdentity
from src.validation.model import validate_model


@pytest.fixture
def codec() -> DatasetCodec:
    return DatasetCodec(
        activity=CategoricalColumn(
            column='activity', vocab=('a', 'b'), special_tokens=ACTIVITY_TOKENS, offset=0
        ),
        resource=CategoricalColumn(
            column='resource', vocab=('r',), special_tokens=RESOURCE_TOKENS, offset=0
        ),
        inter_event_time=NumericColumn(column='inter_event_time', log=False, mean=0.0, std=1.0),
        remaining_time=NumericColumn(column='remaining_time', log=False, mean=0.0, std=1.0),
        categorical_features=(),
        numeric_features=(),
        max_trace_length=4,
        dataset='test',
    )


@pytest.fixture
def batch(codec: DatasetCodec) -> TraceCut:
    def events(
        activities: list[list[int]], resources: list[list[int]], lengths: list[int]
    ) -> Events:
        shape = (len(activities), len(activities[0]))
        return Events(
            activities=torch.tensor(activities),
            resources=torch.tensor(resources),
            inter_event_times=torch.zeros(shape),
            categorical_attributes=torch.zeros((*shape, 0), dtype=torch.long),
            numeric_attributes=torch.zeros((*shape, 0)),
            numeric_attributes_present=torch.zeros((*shape, 0)),
            length=torch.tensor(lengths),
        )

    return TraceCut(
        case_id=('one', 'two'),
        prefix=events([[4, 5, 0, 0], [5, 0, 0, 0]], [[3, 3, 0, 0], [3, 0, 0, 0]], [2, 1]),
        suffix=events(
            [[4, codec.activity.eot_index, 0, 0], [5, 4, codec.activity.eot_index, 0]],
            [[3, codec.resource.eot_index, 0, 0], [3, 3, codec.resource.eot_index, 0]],
            [2, 3],
        ),
        inter_event_times=torch.zeros((2, 4)),
        remaining_times=torch.zeros((2, 4)),
    )


def _composed_model_config(name: str) -> DictConfig:
    config_dir = Path(__file__).parents[1] / 'config'
    with initialize_config_dir(version_base='1.3', config_dir=str(config_dir)):
        config = compose(config_name='train', overrides=[f'model={name}', 'dataset=sepsis'])
    return config.model


def _model_config(name: str) -> DictConfig:
    config = _composed_model_config(name)
    OmegaConf.update(config, 'd_model', 8)
    OmegaConf.update(config, 'embeddings.activity_dim', 4)
    OmegaConf.update(config, 'embeddings.resource_dim', 4)
    OmegaConf.update(config, 'embeddings.feature_dim', 4)
    if name == 'head_sampling_transformer':
        for section in ('encoder', 'decoder'):
            OmegaConf.update(config, f'{section}.num_layers', 1)
            OmegaConf.update(config, f'{section}.num_heads', 2)
            OmegaConf.update(config, f'{section}.feedforward_dim', 16)
            OmegaConf.update(config, f'{section}.dropout', 0.0)
        OmegaConf.update(config, 'decoder.head_hidden_dim', 8)
    else:
        OmegaConf.update(config, 'transformer.num_layers', 1)
        OmegaConf.update(config, 'transformer.num_heads', 2)
        OmegaConf.update(config, 'transformer.feedforward_dim', 16)
        OmegaConf.update(config, 'transformer.dropout', 0.0)
        OmegaConf.update(config, 'diffusion.steps', 2)
    return config


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
def test_model_config_composes_and_validates(name: str) -> None:
    validate_model(_composed_model_config(name))


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
def test_model_smoke(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=_model_config(name), codec=codec).eval()

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
def test_model_loss_backpropagates(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=_model_config(name), codec=codec).train()
    output = model(batch)
    loss, _ = model.compute_loss(output, batch)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
def test_generation_reads_prefix_only(name: str, codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=_model_config(name), codec=codec).eval()
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
def test_new_checkpoint_reload(name: str, codec: DatasetCodec, tmp_path: Path) -> None:
    config = _model_config(name)
    model = build_model(config=config, codec=codec)
    run = RunIdentity(dataset='test', model=name, run_id='20260922-120000-000000')
    path = tmp_path / 'best.pt'
    save_checkpoint(
        model=model,
        config={'model': OmegaConf.to_container(config), 'data': {'name': 'test'}},
        step=1,
        selection_score=0.0,
        wandb_id=None,
        run=run,
        path=path,
    )
    restored = model_from_checkpoint(checkpoint=load_checkpoint(path), codec=codec)

    assert type(restored) is type(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


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


def test_diffusion_canvas_masks(codec: DatasetCodec, batch: TraceCut) -> None:
    model = build_model(config=_model_config('diffusion_transformer'), codec=codec)
    output = model(batch)
    assert isinstance(output, DiffusionOutput)
    assert torch.equal(
        output.activity_mask, torch.tensor([[True, True, False], [True, True, True]])
    )
    assert torch.equal(output.time_mask, torch.tensor([[True, False, False], [True, True, False]]))


@pytest.mark.parametrize('name', ['head_sampling_transformer', 'diffusion_transformer'])
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
    model = build_model(config=_model_config(name), codec=featured_codec).eval()
    output = model(featured_batch)
    loss, _ = model.compute_loss(output, featured_batch)
    assert torch.isfinite(loss)


def test_baseline_cached_decoder_matches_full_pass(codec: DatasetCodec, batch: TraceCut) -> None:
    config = _model_config('head_sampling_transformer')
    OmegaConf.update(config, 'decoder.activity_dropout', 0.0)
    model = build_model(config=config, codec=codec).eval()
    prefix_mask = batch.prefix.pad_mask()
    prefix = model.encoder(events=batch.prefix, pad_mask=prefix_mask)
    output = model(batch)
    assert isinstance(output, DecoderOutput)
    activities = model.decoder._teacher_forced_input(batch.suffix.activities)
    caches = [
        layer.init_cache(prefix_encoded=prefix.events, max_steps=activities.size(dim=1))
        for layer in model.decoder.layers
    ]
    logits = []
    times = []
    with torch.no_grad():
        for position in range(activities.size(dim=1)):
            hidden, caches = model.decoder._run_layers(
                activities=activities[:, position : position + 1],
                prefix_encoded=prefix.events,
                prefix_pad_mask=prefix_mask,
                start_position=position,
                caches=caches,
            )
            features = model.decoder.shared_layer(hidden)
            logits.append(model.decoder.activity_head(features))
            times.append(model.decoder.inter_event_time_head(features).squeeze(dim=-1))

    assert torch.allclose(torch.cat(logits, dim=1), output.activity_logits, atol=1e-5)
    assert torch.allclose(torch.cat(times, dim=1), output.inter_event_times, atol=1e-5)
