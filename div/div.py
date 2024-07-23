import triton
import triton.language as tl
import torch
import math

@triton.jit
def true_div_func(A_ptr, B_ptr, C_ptr,n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A / B
    tl.store(C_ptr + offsets, C, mask=mask)

@triton.jit
def true_div_func_tensor_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A / B
    tl.store(C_ptr + offsets, C, mask=mask)

def true_divide(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        C = torch.empty_like(A)
        n_elements = C.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        true_div_func[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
        return C
    if isinstance(A, torch.Tensor):
        C = torch.empty_like(A)
        n_elements = C.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        true_div_func_tensor_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
        return C
    if isinstance(B, torch.Tensor):
        mag = "The case where the divisor is a scalar is not supported"
        raise ValueError(mag)

def div(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, but found {rounding_mode}."
        raise ValueError(msg)

if __name__ == "__main__":
    A = torch.randn([4, 4], device="cuda")
    B = torch.randn([4, 4], device="cuda")
    print(torch.div(A, B))
    print(div(A, B))
    print(torch.div(A, 2))
    print(div(A, 2))
    print(torch.div(2, A))
    