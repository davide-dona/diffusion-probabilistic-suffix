"""Selectable suffix-model architectures, one package each."""

ARCHITECTURES = (
    'diffusion_transformer',
    'head_sampling_transformer',
    'u_ed_sutran',
    'case_based',
)

_PACKAGE = 'src.models.architectures'


def architecture_of(target: object) -> str:
    """Name the architecture a model class path belongs to.

    The name is the architecture's package, and it identifies the model in run identities,
    output paths, and visualization labels.

    Args:
        target: A `model._target_` value, `src.models.architectures.<architecture>.model.<Class>`.
    Returns:
        The architecture name, one of `ARCHITECTURES`.
    Raises:
        ValueError: If the target is not a class path inside a known architecture's model module.
    """
    parts = target.split('.') if isinstance(target, str) else []
    prefix = _PACKAGE.split('.')
    if (
        len(parts) != len(prefix) + 3
        or parts[: len(prefix)] != prefix
        or parts[len(prefix)] not in ARCHITECTURES
        or parts[len(prefix) + 1] != 'model'
        or not parts[-1]
    ):
        raise ValueError(
            f'model._target_ must name a class in {_PACKAGE}.<architecture>.model with '
            f'<architecture> one of {", ".join(ARCHITECTURES)}, got {target!r}'
        )
    return parts[len(prefix)]
