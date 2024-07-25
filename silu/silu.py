import triton
import triton.language as tl
import torch
import math

@triton.jit
def silu_kernel(x_ptr, out_ptr , n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_value = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    x_value = tl.fdiv(x_value, (1.0 + tl.exp(-x_value)))
    tl.store(out_ptr + offset, x_value, mask=mask)

def silu(A: torch.Tensor):
    out = torch.empty_like(A)
    n_elements = A.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    silu_kernel[(grid_size, 1, 1)](A, out, n_elements, block_size)
    return out

if __name__ == "__main__":
    a = torch.randn([4, 4], device="cuda")
    print(silu(a))