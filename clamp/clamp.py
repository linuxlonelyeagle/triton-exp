import torch
import triton
import triton.language as tl
import math

@triton.jit
def clamp_func(A, B, mini, maxi, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements 
    A_value = tl.load(A + offset, mask=mask)
    result = tl.minimum(maxi, tl.maximum(mini, A_value.to(tl.float32)))
    tl.store(B + offset, result, mask=mask)

@triton.jit
def clamp_func_min(A, B, maxi, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements 
    A_value = tl.load(A + offset, mask=mask)
    result = tl.maximum(maxi, A_value.to(tl.float32))
    tl.store(B + offset, result, mask=mask)

@triton.jit
def clamp_func_max(A, B, maxi, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements 
    A_value = tl.load(A + offset, mask=mask)
    result = tl.minimum(maxi, A_value.to(tl.float32))
    tl.store(B + offset, result, mask=mask)


def clamp(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        B = torch.empty_like(A)
        n_elements = B.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        clamp_func_max[(grid_size, 1, 1)](A, B, maxi, n_elements, block_size)
        return B
    elif maxi is None:
        B = torch.empty_like(A)
        n_elements = B.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        clamp_func_min[(grid_size, 1, 1)](A, B, mini, n_elements, block_size)
        return B
    else:
        B = torch.empty_like(A)
        n_elements = B.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        clamp_func[(grid_size, 1, 1)](A, B, mini, maxi, n_elements, block_size)
        return B

if __name__ == "__main__":
    a = torch.randn((4, 4), device = "cuda")
    print(a)
    print(torch.clamp(a, max=1))
    print(clamp(a, maxi=1))
    print(clamp(a, mini=-1))
    print(clamp(a, maxi=1, mini=-1))
