import logging

import torch
import triton

from rand import rand_kernel
from utils import philox_cuda_seed_offset

UNROLL = 4

def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logging.debug("GEMS RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_cuda_seed_offset(increment)
    with torch.cuda.device(x.device):
        rand_kernel[grid_fn](out, N, philox_seed, philox_offset, BLOCK=8)
    return out

if __name__ == "__main__":
    a = torch.rand([4, 4], device="cuda")
    print(torch.rand_like(a))
    print(rand_like(a))
