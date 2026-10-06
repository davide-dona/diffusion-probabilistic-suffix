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
    """A declarative model mined from a dataset's train split, and the settings it was mined
    under."""

    # The `declare` config section discovery ran with. The checker reads `consider_vacuity` from
    # it, so a model is always checked the way it was mined.
    settings: DictConfig
    constraints: tuple[Constraint, ...]

    def save(self, dataset: str) -> Path:
        """Write this model where the dataset's artifacts live, and return where it went."""
        payload = {
            'settings': OmegaConf.to_container(self.settings, resolve=True),
            'constraints': [
                {
                    'template': constraint.template.name,
                    'activities': [
                        constraint.first,
                        *([] if constraint.second is None else [constraint.second]),
                    ],
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
        """Read the model preprocessing wrote for a dataset.

        Raises:
            ValueError: If the file is not a model this module writes, or if a constraint names a
                template `TEMPLATES` does not hold, the wrong number of activities, one activity
                twice, or a count its template does not take. Each would silently change every
                conformance number in a report, so none is skipped.
        """
        path = artifacts.DECLARE_MODEL.require(dataset)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or set(payload) != _MODEL_KEYS:
            raise ValueError(f'{path} is not a declarative model.')
        if not isinstance(payload['settings'], dict) or not isinstance(
            payload['constraints'], list
        ):
            raise ValueError(f'{path} is not a declarative model.')
        return cls(
            settings=OmegaConf.create(payload['settings']),
            constraints=tuple(_constraint(entry) for entry in payload['constraints']),
        )


def _constraint(entry: Any) -> Constraint:
    """Build one constraint from its written form, rejecting any it could not have been mined as."""
    if not isinstance(entry, dict) or set(entry) != _CONSTRAINT_KEYS:
        raise ValueError(f'{entry} is not a constraint.')

    template = TEMPLATES.get(entry['template'])
    if template is None:
        raise ValueError(
            f'{entry} uses the {entry["template"]} template, which src.logs.declare.templates '
            'does not check. Add it to TEMPLATES there, or mine the model without it.'
        )

    activities = entry['activities']
    expected = 2 if template.is_binary else 1
    if (
        not isinstance(activities, list)
        or len(activities) != expected
        or not all(isinstance(activity, str) for activity in activities)
    ):
        raise ValueError(f'{entry} does not name {expected} activities.')
    if template.is_binary and activities[0] == activities[1]:
        raise ValueError(f'{entry} names one activity twice, which no template is defined for.')

    n = entry['n']
    if type(n) is not int or n < 1 or (n != 1 and not template.supports_cardinality):
        raise ValueError(f'{entry} asks for a count its template does not take.')

    return Constraint(
        template=template,
        first=activities[0],
        second=activities[1] if template.is_binary else None,
        n=n,
    )
