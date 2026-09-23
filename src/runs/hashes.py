import hashlib
import re
from pathlib import Path

_SHA256 = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def validate_sha256(value: object, field: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest."""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f'Invalid {field}: {value!r}')
    return value
