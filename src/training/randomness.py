from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def validation_randomness(*, seed: int | None, device: torch.device) -> Iterator[None]:
    """Isolate repeatable validation draws from the CPU and accelerator training streams."""
    if seed is None:
        yield
        return
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if (device.type == 'cuda')
        else []
    )
    with torch.random.fork_rng(devices=devices):
        mps_state = torch.mps.get_rng_state() if device.type == 'mps' else None
        try:
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            if mps_state is not None:
                torch.mps.manual_seed(seed)
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
