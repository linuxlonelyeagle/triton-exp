import triton
import triton.language as tl
import torch
from utils import volume
from utils import philox_cuda_seed_offset

@triton.jit
def rand_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
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
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N)
    tl.store(out_ptr + off_1, r1, mask=off_1 < N)
    tl.store(out_ptr + off_2, r2, mask=off_2 < N)
    tl.store(out_ptr + off_3, r3, mask=off_3 < N)


def rand(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cuda")

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    philox_seed, philox_offset = philox_cuda_seed_offset(N)
    with torch.cuda.device(device):
      rand_kernel[grid_fn](out, N, philox_seed, philox_offset, 8)
    return out

if __name__ == "__main__":
    print(rand([4, 4], device="cuda", dtype=torch.float64))
    