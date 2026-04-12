"""Optional Triton kernels; falls back to torch.matmul on CPU / Windows / missing Triton."""

from __future__ import annotations

import os
import torch

_USE_TRITON = os.environ.get("NVFP4_USE_TRITON", "0") == "1"

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def matmul_fused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(M,K) @ (K,N) -> (M,N). Uses Triton only when explicitly enabled and available."""
    if _USE_TRITON and triton is not None and a.is_cuda and b.is_cuda:
        return _triton_matmul(a, b)
    return a @ b


def _triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    @triton.jit
    def _kernel(
        pa,
        pb,
        pc,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
        offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
        offs_k = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for kk in range(0, K, BLOCK):
            a_ptrs = pa + (offs_m[:, None] * stride_am + (offs_k[None, :] + kk) * stride_ak)
            b_ptrs = pb + ((offs_k[:, None] + kk) * stride_bk + offs_n[None, :] * stride_bn)
            mask_a = (offs_m[:, None] < M) & ((offs_k[None, :] + kk) < K)
            mask_b = ((offs_k[:, None] + kk) < K) & (offs_n[None, :] < N)
            a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc += tl.dot(a_block, b_block)
        c_ptrs = pc + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(a.dtype.element_ty), mask=mask_c)

    BLOCK = 32
    grid = (triton.cdiv(m, BLOCK), triton.cdiv(n, BLOCK))
    _kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK=BLOCK,
    )
    return c
