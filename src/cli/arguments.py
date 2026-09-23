import argparse
from pathlib import Path


def existing_file(value: str) -> Path:
    """Read a path argument that must name an existing file."""
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'no such file: {path}')
    return path
