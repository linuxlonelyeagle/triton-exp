import triton
import triton.language as tl
import torch
import math
import pdb

@triton.jit
def where_tensor_tensor_kernel(condititon_ptr, self_ptr, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    condition = tl.load(condititon_ptr + offset, mask=mask)
    self = tl.load(self_ptr + offset, mask=mask)
    other = tl.load(other_ptr + offset, mask=mask)
    out = tl.where(condition, self, other)
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def where_tensor_scalar_kernel(condititon_ptr, self_ptr, other, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    condition = tl.load(condititon_ptr + offset, mask=mask)
    self = tl.load(self_ptr + offset, mask=mask)
    out = tl.where(condition, self, other)
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def where_scalar_tensor_kernel(condititon_ptr, it, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    condition = tl.load(condititon_ptr + offset, mask=mask)
    other = tl.load(other_ptr + offset, mask=mask)
    out = tl.where(condition, it, other)
    tl.store(out_ptr + offset, out, mask=mask)

@triton.jit
def where_scalar_scalar_kernel(condititon_ptr, it, other, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):    
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    condition = tl.load(condititon_ptr + offset, mask=mask)
    out = tl.where(condition, it, other)
    tl.store(out_ptr + offset, out, mask=mask)

def where(condition, it, other):
    out = torch.empty_like(condition).to(torch.float)
    n_elements = condition.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    if isinstance(it, torch.Tensor) and isinstance(other, torch.Tensor):
        where_tensor_tensor_kernel[(grid_size, 1, 1)](condition, it, other, out, n_elements, block_size)
        return out
    elif isinstance(it, torch.Tensor):
        where_tensor_scalar_kernel[(grid_size, 1, 1)](condition, it, other, out, n_elements, block_size)
        return out
    elif isinstance(other, torch.Tensor):
        where_scalar_tensor_kernel[(grid_size, 1, 1)](condition, it, other, out, n_elements, block_size)
        return out
    else:
        where_scalar_scalar_kernel[(grid_size, 1, 1)](condition, it, other, out, n_elements, block_size)
        return out
        
if __name__ == "__main__":  
    a = torch.randn([4, 4], device = "cuda")
    b = torch.randn([4, 4], device = "cuda")
    c = torch.randn([4, 4], device = "cuda")
    print(torch.where(a > 0, b, c))
    print(where(a > 0, b, c))
    print(torch.where(a > 0, 1, b))
    print(where(a > 0, 1, b))
    print(torch.where(a > 0, b, 1))
    print(where(a > 0, b, 1))
    print(torch.where(a > 0, 1, -1.1))
    print(where(a > 0, 1, -1.1))
    