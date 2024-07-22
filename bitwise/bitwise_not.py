import torch
import triton
import triton.language as tl
import math

@triton.jit
def bitwise_not_func(A_ptr, B_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  A = tl.load(A_ptr + offsets, mask=mask)
  B = ~A   
  tl.store(B_ptr + offsets, B, mask=mask)

def bitwise_not(A):
  B = torch.empty_like(A)
  n_elements = B.numel()
  block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
  grid_size = triton.cdiv(n_elements, block_size)
  bitwise_not_func[(grid_size, 1, 1)](A, B, n_elements, block_size)  
  return B

if __name__ == "__main__":
  A = torch.tensor([-1, -2, -3], device="cuda")
  print(torch.bitwise_not(A))
  print(bitwise_not(A))
  