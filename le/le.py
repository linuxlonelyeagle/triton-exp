import triton
import triton.language as tl
import torch
import math

@triton.jit
def le_func_tensor(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A <= B
    tl.store(C_ptr + offsets, C, mask=mask)

@triton.jit
def le_func_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A <= B
    tl.store(C_ptr + offsets, C, mask=mask)

def le(A, B):
    C = torch.empty_like(A)
    n_elements = C.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    if isinstance(B, torch.Tensor):
      le_func_tensor[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
    else:
      le_func_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)   
    return C.to(torch.bool)

if __name__ == "__main__":
    A = torch.randn([4, 4], device = "cuda")
    B = torch.randn([4, 4], device = "cuda")
    print(torch.le(A, A))
    print(le(A, A))
    print(torch.le(A, B))
    print(le(A, B))
    print(torch.le(A, 1))
    print(le(A, 1))
