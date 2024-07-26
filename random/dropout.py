import triton
import triton.language as tl
import torch
import torch.nn as nn

from utils import philox_cuda_seed_offset

@triton.jit
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = tl.uint32_to_uniform_float(r0)
    r1 = tl.uint32_to_uniform_float(r1)
    r2 = tl.uint32_to_uniform_float(r2)
    r3 = tl.uint32_to_uniform_float(r3)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    p = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0)
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0)
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0)
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0)

    y0 = x0 * p * mask0  # tl.where(mask0, x0 * p, 0.0)
    y1 = x1 * p * mask1  # tl.where(mask1, x1 * p, 0.0)
    y2 = x2 * p * mask2  # tl.where(mask2, x2 * p, 0.0)
    y3 = x3 * p * mask3  # tl.where(mask3, x3 * p, 0.0)

    tl.store(Y + off_0, y0, mask=off_0 < N)
    tl.store(Y + off_1, y1, mask=off_1 < N)
    tl.store(Y + off_2, y2, mask=off_2 < N)
    tl.store(Y + off_3, y3, mask=off_3 < N)

UNROLL = 4

def dropout(x, p):
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = x.device
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    with torch.cuda.device(device):
        philox_seed, philox_offset = philox_cuda_seed_offset(increment)
        dropout_forward_kernel[grid_fn](x, out, N, p, philox_seed, philox_offset, BLOCK=128)
    return out, None

if __name__ == "__main__":
    m = nn.Dropout(p=0.2)
    input = torch.randn(20, 16, device="cuda")
    output = m(input)
    print(output)
    print(dropout(input, p = 0.2))
