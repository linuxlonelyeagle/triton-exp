import triton
import triton.language as tl
import torch
import math

@triton.jit
def sub_func(x, y, z, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_value = tl.load(x + offset, mask=mask)
    y_value = tl.load(y + offset, mask=mask)
    z_value = x_value - y_value * alpha
    tl.store(z + offset, z_value, mask=mask)

@triton.jit
def sub_func_tensor_scalar(x, y, z, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_value = tl.load(x + offset, mask=mask)
    z_value = x_value - y * alpha
    tl.store(z + offset, z_value, mask=mask)    

@triton.jit
def sub_func_scalar_tensor(x, y, z, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    y_value = tl.load(y + offset, mask=mask)
    z_value = x - y_value * alpha
    tl.store(z + offset, z_value, mask=mask)   


def sub(A, B, *, alpha=1):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        C = torch.empty_like(A)
        n_elements = A.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        sub_func[(grid_size, 1, 1)](A, B, C, alpha, n_elements, block_size)
        return C
    elif isinstance(A, torch.Tensor):
        C = torch.empty_like(A)
        n_elements = A.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        sub_func_tensor_scalar[(grid_size, 1, 1)](A, B, C, alpha, n_elements, block_size)
        return C
    elif isinstance(B, torch.Tensor):
        C = torch.empty_like(B)
        n_elements = B.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        sub_func_scalar_tensor[(grid_size, 1, 1)](A, B, C, alpha, n_elements, block_size)
        return C
    else:
        return A + B * alpha
    
if __name__ == "__main__":
    x0 = torch.arange(0, 9, device="cuda")
    y0 = torch.arange(0, 9, device="cuda")
    print(sub(x0, y0, alpha=2))
    x1 = torch.arange(0, 9, device="cuda")
    y1 = 1
    print(sub(x1, y1, alpha=2))
    x2 = 1
    y2 = torch.arange(0, 9, device="cuda")
    print(sub(x2, y2, alpha=2))
