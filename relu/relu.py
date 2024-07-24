import logging

import torch
import triton
import triton.language as tl
import math

@triton.jit
def relu_forward(x_ptr, out_ptr , n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_value = tl.load(x_ptr + offset, mask=mask)
    out = tl.where(x_value > 0, x_value, 0)
    tl.store(out_ptr + offset, out, mask=mask)    

@triton.jit
def relu_backward(x_ptr, dy, x_grad_ptr,  n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_value = tl.load(x_ptr + offset, mask=mask)
    out = tl.where(x_value > 0, dy, 0)
    tl.store(x_grad_ptr + offset, out, mask=mask)  

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        out = torch.empty_like(A)
        n_elements = A.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        relu_forward[(grid_size, 1, 1)](A, out, n_elements, block_size)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        (inp,) = ctx.saved_tensors
        in_grad = torch.empty_like(inp)
        n_elements = in_grad.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        relu_backward[(grid_size, 1, 1)](inp, out_grad, in_grad, n_elements, block_size)
        return in_grad

def relu(A):
    return Relu.apply(A)

if __name__ == "__main__":
    a = torch.randn([4, 4], device="cuda")
    print(torch.relu(a))
    print(relu(a))
