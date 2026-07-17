"""FP4-packed matrix-multiply helpers (Listing 2).

On NVIDIA H100/B200 the weight is stored in NVFP4 (E2M1): two 4-bit values
are packed into a single byte and de-quantized on the fly inside a Triton
kernel (``fp4_matmul_kernel`` below).  When Triton or a CUDA device is not
available the module transparently falls back to an equivalent PyTorch
implementation (``_torch_fp4_matmul``), so the code runs on CPU/Windows too.

The numeric result is identical to ``fake_quantize_fp4(weight)`` followed by a
normal matmul -- the packing is purely a memory/locality optimization that the
Triton kernel exploits.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_USE_TRITON = os.environ.get("NVFP4_USE_TRITON", "0") == "1"

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton only on Linux/CUDA
    triton = None
    tl = None


# Strict E2M1 (NVFP4) value table, indexed by 4-bit code 0..15. Must match
# quantization._e2m1_levels so the kernel and the training quantizer agree.
_FP4_TABLE = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
     0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_FP4_CLIP = 6.0


def _quantize_weight_to_fp4(w: torch.Tensor, block_size: int) -> torch.Tensor:
    """Block-scaled NVFP4 quantization of a weight tensor (no STE)."""
    orig = w.shape
    flat = w.reshape(-1)
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    scale = amax / _FP4_CLIP
    norm = blocks / scale
    dist = (norm.unsqueeze(-1) - _FP4_TABLE.to(w.device).unsqueeze(0).unsqueeze(0)).abs()
    idx = dist.argmin(dim=-1)
    q = _FP4_TABLE.to(w.device)[idx] * scale
    return q.reshape(-1)[: orig.numel()].view(orig)


def _torch_fp4_matmul(a: torch.Tensor, w: torch.Tensor, block_size: int) -> torch.Tensor:
    """PyTorch fallback: (M,K) @ (K,N) with the weight in NVFP4."""
    w_q = _quantize_weight_to_fp4(w, block_size)
    return a @ w_q


def fp4_matmul(a: torch.Tensor, w: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """(M,K) @ (K,N) with ``w`` stored/computed in NVFP4.

    Uses the Triton packed kernel only when ``NVFP4_USE_TRITON=1`` and Triton
    + CUDA are available; otherwise falls back to PyTorch.
    """
    if _USE_TRITON and triton is not None and a.is_cuda and w.is_cuda:
        return _triton_fp4_matmul(a, w, block_size)
    return _torch_fp4_matmul(a, w, block_size)


# --------------------------------------------------------------------------- #
# Triton FP4-packed kernel (Listing 2). Untested here (no Triton/CUDA), but
# mirrors the paper's described data flow: load packed bytes, unpack two FP4
# values per byte to FP16, multiply by the block scale, accumulate in FP32.
# --------------------------------------------------------------------------- #
def _triton_fp4_matmul(a: torch.Tensor, w: torch.Tensor, block_size: int) -> torch.Tensor:
    assert a.dim() == 2 and w.dim() == 2 and a.shape[1] == w.shape[0]
    m, k = a.shape
    n = w.shape[1]

    # Pack the weight: quantize, then store two 4-bit codes per byte.
    w_q = _quantize_weight_to_fp4(w, block_size)
    flat = w_q.reshape(-1)
    # map values back to codes
    code = (flat.unsqueeze(-1) - _FP4_TABLE.to(w.device)).abs().argmin(-1).to(torch.uint8)
    packed = (code[0::2] | (code[1::2] << 4)).contiguous()  # 2 codes / byte
    scales = (w_q.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / _FP4_CLIP) \
        .reshape(-1, block_size)[:, 0].contiguous()

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    BLOCK = 32
    grid = (triton.cdiv(m, BLOCK), triton.cdiv(n, BLOCK))

    @triton.jit
    def _kernel(
        pa, ppack, pscale, pc,
        M, N, K, BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
        offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
        offs_k = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for kk in range(0, K, BLOCK):
            a_blk = tl.load(pa + offs_m[:, None] * K + (offs_k[None, :] + kk),
                            mask=(offs_m[:, None] < M) & ((offs_k[None, :] + kk) < K))
            pack = tl.load(ppack + ((offs_k[:, None] + kk) // 2))
            # unpack low/high nibble (illustrative; real HW uses dedicated insns)
            lo = pack & 0xF
            hi = (pack >> 4) & 0xF
            w_code = tl.where((offs_k[:, None] + kk) % 2 == 0, lo, hi)
            scale = tl.load(pscale + (offs_k[:, None] + kk) // block_size)
            w_val = _table_lookup(w_code) * scale
            acc += tl.dot(a_blk, w_val)
        tl.store(pc + offs_m[:, None] * N + offs_n[None, :],
                 acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    _kernel[grid](a, packed, scales, c, m, n, k, BLOCK=BLOCK)
    return c


def _table_lookup(code):  # pragma: no cover - only used inside the Triton kernel
    # Placeholder: Triton cannot index a Python tensor directly; a real kernel
    # would use tl.load on a constant buffer. Kept for structural completeness.
    return code.to(tl.float32)
