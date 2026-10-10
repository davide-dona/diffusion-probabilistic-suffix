from collections.abc import Iterable, Sequence
from pathlib import Path

from fcfdeclare import Checker, Constraint, DeclareModel, mine
from omegaconf import DictConfig

from src import artifacts
from src.datasets.codec import ActivityCodec


def mine_model(traces: Iterable[Sequence[str]], *, settings: DictConfig) -> DeclareModel:
    """Mine a Declare model from the train split.

    Args:
        traces: The train split's cases, the only data mining reads, one sequence of activity
            names per case.
        settings: The `declare` config section.

    Returns:
        The mined model, its constraints on activity names and each with its support.
    """
    return mine(
        traces,
        activity_support=settings.activity_support,
        min_support=settings.min_support,
        max_cardinality=settings.max_cardinality,
        vacuity=settings.vacuity,
    )


def save_model(model: DeclareModel, dataset: str) -> Path:
    """Save the model among the dataset's artifacts and return its path."""
    path = artifacts.DECLARE_MODEL.prepare(dataset)
    model.save(path)
    return path


def load_model(dataset: str) -> DeclareModel:
    """
    Load the model saved for a dataset.

    Raises:
        ValueError: If the file is not a well-formed model.
    """
    return DeclareModel.load(artifacts.DECLARE_MODEL.require(dataset))


def code_checker(model: DeclareModel, codes: ActivityCodec) -> Checker:
    """
    A checker of traces encoded with `codes`, one character per event.

    Args:
        model: The model to check against, under the vacuity it was mined with.
        codes: The codebook the traces are encoded with.

    Raises:
        ValueError: If the codebook lacks an activity the model names, which means the two
            belong to different datasets.
    """
    try:
        constraints = {
            Constraint(
                constraint.template,
                tuple(codes.codes[name] for name in constraint.activities),
                constraint.n,
            ): support
            for constraint, support in model.constraints.items()
        }
    except KeyError as error:
        raise ValueError(
            f'The codebook lacks the activity {error.args[0]!r} of the Declare model.'
        ) from error
    return Checker(DeclareModel(settings=model.settings, constraints=constraints))
