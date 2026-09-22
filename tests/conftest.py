from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.datasets.codec import (
    ACTIVITY_TOKENS,
    RESOURCE_TOKENS,
    CategoricalColumn,
    DatasetCodec,
    NumericColumn,
)
from src.datasets.dataset import Events, TraceCut


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


def composed_model_config(name: str) -> DictConfig:
    config_dir = Path(__file__).parents[1] / 'config'
    with initialize_config_dir(version_base='1.3', config_dir=str(config_dir)):
        config = compose(config_name='train', overrides=[f'model={name}', 'dataset=sepsis'])
    return config.model


def model_config(name: str) -> DictConfig:
    config = composed_model_config(name)
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
