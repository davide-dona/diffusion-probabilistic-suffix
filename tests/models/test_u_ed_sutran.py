from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models import UEDSuTraN, build_model
from src.models.architectures.u_ed_sutran.distributions import sample_gaussian
from src.models.architectures.u_ed_sutran.dropout import monte_carlo_dropout
from src.models.contracts import UncertaintyAwareDecoderOutput
from src.training.randomness import validation_randomness
from tests.conftest import model_config


def _model(codec: DatasetCodec, *, dropout: float = 0.0) -> UEDSuTraN:
    config = model_config('u_ed_sutran')
    for section in ('encoder', 'decoder'):
        OmegaConf.update(config, f'{section}.dropout', dropout)
    OmegaConf.update(config, 'decoder.activity_dropout', 0.0)
    return build_model(config=config, codec=codec).eval()


def test_cached_decoder_matches_full_pass(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec)
    mask = batch.prefix.pad_mask()
    with torch.no_grad():
        prefix = model.encoder(events=batch.prefix, pad_mask=mask)
        expected = model(batch)
        activities = model.decoder._teacher_forced_input(batch.suffix.activities)
        caches = [
            layer.init_cache(prefix_encoded=prefix.events, max_steps=activities.size(dim=1))
            for layer in model.decoder.layers
        ]
        outputs = []
        for position in range(activities.size(dim=1)):
            hidden, caches = model.decoder._run_layers(
                activities=activities[:, position : position + 1],
                prefix_encoded=prefix.events,
                prefix_pad_mask=mask,
                start_position=position,
                caches=caches,
            )
            outputs.append(model.decoder.predict(model.decoder.shared_layer(hidden)))
    for field in fields(expected):
        actual = torch.cat(tensors=[getattr(output, field.name) for output in outputs], dim=1)
        torch.testing.assert_close(actual=actual, expected=getattr(expected, field.name))


def test_full_suffix_causal_supervision(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec.model_copy(update={'max_trace_length': 8}))
    suffix = batch.suffix._replace(
        activities=torch.full(size=(2, 8), fill_value=4, dtype=torch.long),
        length=torch.tensor([8, 7]),
    )
    suffix.activities[0, 7] = codec.activity.eot_index
    suffix.activities[1, 6] = codec.activity.eot_index
    suffix.activities[1, 7] = codec.activity.pad_index
    batch = batch._replace(suffix=suffix, inter_event_times=torch.zeros(size=(2, 8)))
    with patch.object(
        model.decoder.layers[0], 'forward', wraps=model.decoder.layers[0].forward
    ) as run:
        output = model(batch)
    assert run.call_count == 1
    output.activity_logit_means.retain_grad()
    loss, _ = model.compute_loss(output=output, batch=batch)
    loss.backward()
    assert output.activity_logit_means.grad[0, 6:].abs().sum() > 0
    assert output.activity_logit_means.grad[1, 7].abs().sum() == 0
    changed = suffix.activities.clone()
    changed[:, 5:] = 5
    later = model(batch._replace(suffix=suffix._replace(activities=changed)))
    torch.testing.assert_close(
        actual=later.activity_logit_means[:, :6], expected=output.activity_logit_means[:, :6]
    )


@pytest.mark.parametrize('log_variance', [-10.0, 10.0])
def test_loss_bounds_masks_and_gradients(
    codec: DatasetCodec, batch: TraceCut, log_variance: float
) -> None:
    model = _model(codec)
    output = UncertaintyAwareDecoderOutput(
        activity_logit_means=torch.zeros(size=(2, 4, codec.activity.num_rows), requires_grad=True),
        activity_log_variances=torch.full(
            size=(2, 4, codec.activity.num_rows), fill_value=log_variance, requires_grad=True
        ),
        inter_event_time_means=torch.ones(size=(2, 4), requires_grad=True),
        inter_event_time_log_variances=torch.full(
            size=(2, 4), fill_value=log_variance, requires_grad=True
        ),
    )
    loss, _ = model.compute_loss(output=output, batch=batch)
    assert torch.isfinite(loss)
    loss.backward()
    for field in fields(output):
        gradient = getattr(output, field.name).grad
        assert gradient is not None and torch.isfinite(gradient).all()
        for row, length in enumerate(batch.suffix.length.tolist()):
            stop = length if field.name.startswith('activity') else length - 1
            assert gradient[row, stop:].count_nonzero() == 0
    assert output.activity_logit_means.grad[0, 1].abs().sum() > 0


def test_loss_matches_marginal_likelihood_and_trace_weighting(
    codec: DatasetCodec, batch: TraceCut
) -> None:
    model = _model(codec)
    output = model(batch)
    generator = torch.Generator().manual_seed(model.uncertainty.validation_seed)
    means = output.activity_logit_means.unsqueeze(dim=0).expand(20, -1, -1, -1)
    sampled = sample_gaussian(
        mean=means, log_variance=output.activity_log_variances.unsqueeze(dim=0), generator=generator
    )
    probabilities = sampled.softmax(dim=-1).mean(dim=0)
    row_losses = []
    for row, length in enumerate(batch.suffix.length.tolist()):
        activity = (
            -probabilities[row, :length]
            .gather(dim=-1, index=batch.suffix.activities[row, :length].unsqueeze(dim=-1))
            .log()
            .sum()
        )
        mu = output.inter_event_time_means[row, : length - 1]
        s = output.inter_event_time_log_variances[row, : length - 1]
        time = (0.5 * ((-s).exp() * mu.square() + s)).sum()
        row_losses.append((activity + time) / (2 * length - 1))
    actual, metrics = model.compute_loss(output=output, batch=batch)
    torch.testing.assert_close(actual=actual, expected=torch.stack(row_losses).mean())
    assert metrics.loss == pytest.approx(actual.item() * 2)
    assert metrics.loss == pytest.approx(metrics.activity_loss + metrics.inter_event_time_loss)


def test_local_likelihood_rng(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec)
    output = model(batch)
    state = torch.random.get_rng_state()
    first, _ = model.compute_loss(output=output, batch=batch)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(size=(100,))
    repeated, _ = model.compute_loss(output=output, batch=batch)
    assert torch.equal(first, repeated)
    model.train()
    state = torch.random.get_rng_state()
    first, _ = model.compute_loss(output=output, batch=batch)
    second, _ = model.compute_loss(output=output, batch=batch)
    assert not torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('raise_error', [False, True])
def test_dropout_restores_mixed_modes(codec: DatasetCodec, raise_error: bool) -> None:
    model = _model(codec, dropout=0.5)
    model.decoder.train()
    modes = [(module, module.training) for module in model.modules()]
    try:
        with monte_carlo_dropout(model):
            assert not model.training
            assert model.encoder.encoder.layers[0].training
            assert model.encoder.encoder.layers[0].self_attn.training
            assert model.decoder.layers[0].self_attention.training
            if raise_error:
                raise RuntimeError('test')
    except RuntimeError:
        assert raise_error
    assert all(module.training == mode for module, mode in modes)


def test_attention_dropout_is_stochastic(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec, dropout=0.5)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    with torch.no_grad(), monte_carlo_dropout(model):
        first_prefix = model.encoder(events=batch.prefix, pad_mask=batch.prefix.pad_mask())
        second_prefix = model.encoder(events=batch.prefix, pad_mask=batch.prefix.pad_mask())
        first = model.decoder(
            suffix_activities=batch.suffix.activities,
            prefix_encoded=first_prefix.events,
            prefix_pad_mask=batch.prefix.pad_mask(),
        )
        second = model.decoder(
            suffix_activities=batch.suffix.activities,
            prefix_encoded=first_prefix.events,
            prefix_pad_mask=batch.prefix.pad_mask(),
        )
    assert not torch.equal(first_prefix.events, second_prefix.events)
    assert not torch.equal(first.activity_logit_means, second.activity_logit_means)


def test_independent_encoder_draws_and_epistemic_variation(
    codec: DatasetCodec, batch: TraceCut
) -> None:
    model = _model(codec, dropout=0.5)
    encodings = []
    handle = model.encoder.register_forward_hook(
        lambda module, args, output: encodings.append(output.events.detach().clone())
    )

    def fixed_draw(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = model.decoder.predict(features)
        return torch.full_like(
            input=output.inter_event_time_means, fill_value=4, dtype=torch.long
        ), (output.inter_event_time_means)

    with patch.object(model.decoder, 'sample', side_effect=fixed_draw):
        first = model.generate(item=batch, num_samples=8)
        second = model.generate(item=batch, num_samples=8)
    handle.remove()
    assert encodings[0].size(dim=0) == 16
    assert not torch.equal(encodings[0][0], encodings[0][1])
    assert not torch.equal(first.inter_event_times, second.inter_event_times)
    assert not torch.equal(first.inter_event_times[0, 0], first.inter_event_times[0, 1])


def test_aleatoric_sampling_and_structural_mask(codec: DatasetCodec) -> None:
    model = _model(codec)
    features = torch.zeros(size=(512, 8))
    with torch.no_grad():
        activities, times = model.decoder.sample(features)
    assert activities.unique().numel() > 1
    assert times.unique().numel() > 1
    assert not (activities == codec.activity.pad_index).any()
    assert not (activities == codec.activity.sos_index).any()
    generator = torch.Generator().manual_seed(7)
    actual = sample_gaussian(
        mean=torch.zeros(size=(128,)),
        log_variance=torch.full(size=(128,), fill_value=2.0),
        generator=generator,
    )
    expected = torch.randn(size=(128,), generator=torch.Generator().manual_seed(7)) * torch.e
    torch.testing.assert_close(actual=actual, expected=expected)


@pytest.mark.parametrize('terminate', [False, True])
def test_termination_padding_sentinel_and_remaining_time(
    codec: DatasetCodec, batch: TraceCut, terminate: bool
) -> None:
    codec = codec.model_copy(
        update={
            'inter_event_time': codec.inter_event_time.model_copy(update={'mean': 3.0, 'std': 2.0}),
            'remaining_time': codec.remaining_time.model_copy(update={'mean': 1.0, 'std': 2.0}),
        }
    )
    model = _model(codec)
    position = 0

    def draw(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal position
        token = codec.activity.eot_index if terminate and position == 1 else 4
        position += 1
        return (
            torch.full(size=(features.size(dim=0),), fill_value=token, dtype=torch.long),
            torch.ones(size=(features.size(dim=0),)),
        )

    with patch.object(model.decoder, 'sample', side_effect=draw):
        generated = model.generate(item=batch, num_samples=3)
    length = 1 if terminate else 4
    assert (generated.lengths == length).all()
    assert (generated.used_sentinel == (not terminate)).all()
    assert (generated.activities[:, :, length:] == codec.activity.pad_index).all()
    assert (generated.inter_event_times[:, :, length:] == 0).all()
    assert (generated.remaining_time == (length * 5 - 1) / 2).all()


def test_seeded_generation_restores_rng(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec, dropout=0.5)
    device = torch.device('cpu')
    state = torch.random.get_rng_state()
    with validation_randomness(seed=13, device=device):
        first = model.generate(item=batch, num_samples=5)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(size=(100,))
    with validation_randomness(seed=13, device=device):
        second = model.generate(item=batch, num_samples=5)
    for field in fields(first):
        assert torch.equal(getattr(first, field.name), getattr(second, field.name))


def test_mixed_termination_retains_first_eot(codec: DatasetCodec, batch: TraceCut) -> None:
    model = _model(codec)
    position = 0
    stops = torch.tensor([0, 1, 4, 2])

    def draw(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal position
        tokens = torch.full_like(input=stops, fill_value=4)
        tokens[stops == position] = codec.activity.eot_index
        position += 1
        return tokens, torch.ones_like(input=stops, dtype=torch.float32)

    with patch.object(model.decoder, 'sample', side_effect=draw):
        generated = model.generate(item=batch, num_samples=2)
    assert torch.equal(generated.lengths.flatten(), stops)
    assert torch.equal(generated.used_sentinel.flatten(), stops == 4)
    for row, length in enumerate(stops.tolist()):
        assert (generated.activities.reshape((4, 4))[row, length:] == 0).all()
        assert (generated.inter_event_times.reshape((4, 4))[row, length:] == 0).all()


def test_validation_rng_restores_after_error() -> None:
    state = torch.random.get_rng_state()
    with pytest.raises(RuntimeError, match='test'):
        with validation_randomness(seed=17, device=torch.device('cpu')):
            torch.rand(size=(10,))
            raise RuntimeError('test')
    assert torch.equal(state, torch.random.get_rng_state())
