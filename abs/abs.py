import triton
import triton.language as tl
import torch

import math

@triton.jit
def abs_kernel(x_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.abs(x)   
    tl.store(output_ptr + offsets, result, mask=mask)

def abs(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    abs_kernel[(grid_size, 1, 1)](x, output, n_elements, block_size)
    return output


if __name__ == "__main__":
    a = torch.randn((4, 4), device = "cuda")
    print(a)
    print(abs(a))
