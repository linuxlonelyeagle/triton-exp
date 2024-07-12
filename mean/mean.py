import torch
import triton
import triton.language as tl
import math

@triton.jit
def mean_kernel_1(
  inp,
  mid,
  M,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(0)
  offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  inp_ptrs = inp + offset
  mask = offset < M
  inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
  sum_val = tl.sum(inp_val, axis=0)
  mid_ptr = mid + pid
  tl.store(mid_ptr, sum_val)

@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)

def mean(inp, *, dtype=None):
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    '''
    M: 2024
    block_size: 64
    mid_size: 32
    block_mid = 32
    '''
    
    print(f"M: {M}")
    print(f"block_size: {block_size}")
    print(f"mid_size: {mid_size}")
    print(f"block_mid = {block_mid}")

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch.cuda.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out

input = torch.arange(0, 2024, device="cuda")
output = mean(input)
print(output)