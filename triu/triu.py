import triton
import triton.language as tl
import torch

@triton.jit
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(row + diagonal <= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask and mn_mask
    x = tl.load(X + cols, mask, other=0.0)
    m = cols // N
    n = cols % N
    y = tl.where(m + diagonal <= n, x, 0.0)
    tl.store(Y + cols, y, mask=mask)


def triu(A, diagonal=0):
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    with torch.cuda.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal, M_BLOCK_SIZE=32, N_BLOCK_SIZE=8)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](B, out, batch, M * N, N, diagonal, BATCH_BLOCK_SIZE=32, MN_BLOCK_SIZE=8)
            out = out.view(A.shape)
    return out

if __name__ == "__main__":
    a = torch.randn(3, 3, device="cuda")
    print(torch.triu(a))
    print(triu(a))
    b = torch.randn(3, 3, 3, device="cuda")
    print(torch.triu(b))
    print(triu(b))
