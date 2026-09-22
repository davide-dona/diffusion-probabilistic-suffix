import torch
from omegaconf import OmegaConf

from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceCut
from src.models import build_model
from src.models.contracts import DecoderOutput
from tests.conftest import model_config


# Checks that incremental cached decoding matches teacher-forced decoding at every position.
def test_baseline_cached_decoder_matches_full_pass(codec: DatasetCodec, batch: TraceCut) -> None:
    config = model_config('head_sampling_transformer')
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
