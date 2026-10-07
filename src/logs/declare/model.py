import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from omegaconf import DictConfig, OmegaConf

from src import artifacts
from src.logs.declare.templates import TEMPLATES, Constraint

_MODEL_KEYS = {'settings', 'constraints'}
_CONSTRAINT_KEYS = {'template', 'activities', 'n'}


@dataclass(frozen=True, slots=True)
class DeclareModel:
    """A Declare model mined from a train split, with the settings it was mined under."""

    # The `declare` config section, whose `consider_vacuity` checking must reuse
    settings: DictConfig
    constraints: tuple[Constraint, ...]

    def save(self, dataset: str) -> Path:
        """Save the model among the dataset's artifacts and return its path."""
        payload = {
            'settings': OmegaConf.to_container(self.settings, resolve=True),
            'constraints': [
                {
                    'template': constraint.template.name,
                    'activities': list(constraint.activities),
                    'n': constraint.n,
                }
                for constraint in self.constraints
            ],
        }
        path = artifacts.DECLARE_MODEL.prepare(dataset)
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + '\n')
        return path

    @classmethod
    def load(cls, dataset: str) -> Self:
        """
        Load the model saved for a dataset.

        Raises:
            ValueError: If the file or any of its constraints is malformed.
        """
        path = artifacts.DECLARE_MODEL.require(dataset)
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or set(payload) != _MODEL_KEYS
            or not isinstance(payload['settings'], dict)
            or not isinstance(payload['constraints'], list)
        ):
            raise ValueError(f'{path} is not a Declare model.')
        return cls(
            settings=OmegaConf.create(payload['settings']),
            constraints=tuple(_constraint(entry) for entry in payload['constraints']),
        )


def _constraint(entry: Any) -> Constraint:
    """Parse one saved constraint, rejecting any that discovery cannot produce."""
    if not isinstance(entry, dict) or set(entry) != _CONSTRAINT_KEYS:
        raise ValueError(f'{entry} is not a constraint.')

    template = TEMPLATES.get(entry['template'])
    if template is None:
        raise ValueError(f'{entry} uses an unknown template.')

    activities = entry['activities']
    expected = 2 if template.is_binary else 1
    if (
        not isinstance(activities, list)
        or len(activities) != expected
        or not all(isinstance(activity, str) for activity in activities)
    ):
        raise ValueError(f'{entry} does not name {expected} activities.')
    if template.is_binary and activities[0] == activities[1]:
        raise ValueError(f'{entry} names one activity twice.')

    n = entry['n']
    if type(n) is not int or n < 1 or (n != 1 and not template.supports_cardinality):
        raise ValueError(f'{entry} asks for a count its template does not take.')

    return Constraint(
        template=template,
        first=activities[0],
        second=activities[1] if template.is_binary else None,
        n=n,
    )
