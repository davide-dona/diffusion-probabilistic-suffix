import torch
import triton
import triton.language as tl

# Pairs swept by one program, one per lane.
_BLOCK = 128


def sweep(increments: torch.Tensor, order: int) -> torch.Tensor:
    """Solve the Goursat PDE of each pair in one CUDA kernel, one lane per pair.

    Each lane fills its refined grid row by row and keeps only the latest row of points, so a
    refined cell reads its increment and one stored point and writes one point.

    Args:
        increments: `[B, H, W]` static kernel second differences of each pair's coarse cells,
            on CUDA, with `H, W >= 1` and fewer than `2**31` elements.
        order: Dyadic order: each coarse cell is split into `2**order` steps along both paths.

    Returns:
        `[B]` float32 kernel values on the device of `increments`.
    """
    batch, height, width = increments.shape
    cells = increments.float().div_(4**order).permute(1, 2, 0).contiguous()  # [H, W, B]
    latest = torch.ones(width << order, batch, dtype=torch.float32, device=increments.device)
    values = torch.empty(batch, dtype=torch.float32, device=increments.device)
    _sweep[(triton.cdiv(batch, _BLOCK),)](
        cells, latest, values, batch, height, width, order=order, block=_BLOCK
    )
    return values


@triton.jit
def _sweep(cells, latest, values, batch, height, width, order: tl.constexpr, block: tl.constexpr):
    """Fill the refined grids of one block of pairs and store each corner point.

    Before refined row `i`, `latest[j]` holds point `(i, j + 1)`; the cell `(i, j)` overwrites
    it with point `(i + 1, j + 1)`. Points `(i, 0)` lie on the boundary and equal one.
    """
    pairs = tl.program_id(0) * block + tl.arange(0, block)
    valid = pairs < batch
    corner = tl.full([block], 1.0, tl.float32)
    for i in range(height << order):
        earlier = tl.full([block], 1.0, tl.float32)  # point (i, j)
        current = tl.full([block], 1.0, tl.float32)  # point (i + 1, j)
        for j in range(width << order):
            cell = (i >> order) * width + (j >> order)
            increment = tl.load(cells + cell * batch + pairs, mask=valid, other=0.0)
            second = 1.0 - increment * increment * (1.0 / 12.0)
            first = increment * 0.5 + 2.0 - second
            above = tl.load(latest + j * batch + pairs, mask=valid, other=1.0)
            current = (current + above) * first - earlier * second
            tl.store(latest + j * batch + pairs, current, mask=valid)
            earlier = above
        corner = current
    tl.store(values + pairs, corner, mask=valid)
