import triton
import triton.language as tl
import torch
import math

@triton.jit
def eq_func_tensor(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  a = tl.load(a_ptr + offsets, mask=mask)
  b = tl.load(b_ptr + offsets, mask=mask)
  c = a == b
  tl.store(c_ptr + offsets, c, mask=mask)

@triton.jit
def eq_func_scalar(a_ptr, b: tl.constexpr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  a = tl.load(a_ptr + offsets, mask=mask)
  c = a == b
  tl.store(c_ptr + offsets, c, mask=mask)

def eq(A, B):
  C = torch.empty_like(A)
  n_elements = A.numel()
  block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
  grid_size = triton.cdiv(n_elements, block_size)
  if isinstance(B, torch.Tensor):
    eq_func_tensor[(grid_size, 1, 1)](A, B, C, n_elements, block_size)  
  else:
    eq_func_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
  return C.to(torch.bool)

if __name__ == "__main__":
  A = torch.tensor([1, 2, 3, 4], device="cuda")
  B = torch.tensor([1, 2, 3, 4], device="cuda")
  print(torch.eq(A, B))
  print(eq(A, B))
  print(torch.eq(A, 1))
  print(eq(A, 1))