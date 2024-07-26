import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


@triton.jit
def reduce_mul(a, b):
    return a * b

@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))

@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch.cuda.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
        prod_kernel_result[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out

