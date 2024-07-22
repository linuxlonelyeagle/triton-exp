import triton
import triton.language as tl
import torch
import math

@triton.jit
def bitwise_and_func_tensor(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

@triton.jit
def bitwise_and_func_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

def bitwise_and(A, B):
    C = torch.empty_like(A)
    n_elements = C.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    if isinstance(B, torch.Tensor):
      bitwise_and_func_tensor[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
    else:
      bitwise_and_func_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)   
    return C

if __name__ == "__main__":
    # tensor or tensor
    A = torch.arange(9, device="cuda")
    B = torch.arange(9, device="cuda")
    print(torch.bitwise_and(A, B))
    print(bitwise_and(A, B))
    print(bitwise_and(A, 0))
    print(bitwise_and(A, 0))
    A = torch.tensor([1, 1, 1, 1], device="cuda")
    B = torch.tensor([2, 2, 2, 2], device="cuda")
    print(torch.bitwise_and(A, B))
    print(bitwise_and(A, B))
    print(bitwise_and(A, 2))
