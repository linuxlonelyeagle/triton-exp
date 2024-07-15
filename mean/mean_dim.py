import torch
import triton
import triton.language as tl
import mean

# (2, 3, 4, 5) [1, 2]
def dim_compress(inp: torch.Tensor, dims):
    print(inp.shape)
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    # (60, 20, 5, 1)
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    # stride[1] = 20
    # strode[2] = 5
    # [1, 2]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    # [0, 3, 1, 2]
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4)
#         for m in [1, 2, 4, 8]
#     ],
#     key=["M", "N"],
# )

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] 
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)

def mean_dim(x, dim, keepdim=False, *, dtype=None):
  if dtype is None:
    dtype = x.dtype
  if dim is None:
    out = mean.mean(x, dtype=dtype)
    if not keepdim:
      out = out.reshape([1]*x.ndim)
    return out
  
  shape = list(x.shape)
  if isinstance(dim, int):
     dim = [dim]
  dim = [d % x.ndim for d in dim]
  x = dim_compress(x, dim)
  N = 1
  for i in dim:
    N *= shape[i]
    shape[i] = 1
  M = x.numel() // N
  out = torch.empty(shape, dtype=dtype, device=x.device)
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

  with torch.cuda.device(x.device):
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)
  if not keepdim:
    out = out.squeeze(dim)
  return out

a = torch.randn(4, 4, dtype= torch.float, device="cuda")

'''
print(a)
print(mean_dim(a, dim=None))
print(torch.mean(a, dim=0))
print(mean_dim(a, dim=0))
print(torch.mean(a, dim=1))
print(mean_dim(a, dim=1))
'''

b = torch.randn(2, 3, 4, 5, device="cuda")
# print(dim_compress(b, [1, 2]).shape)
print(mean_dim(b, [1, 2]))
