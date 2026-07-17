"""NVFP4 (E2M1) fake quantization for quantization-aware training (QAT).

This module implements the block-scaled NVFP4 quantization described in the
paper (Definition III.1, Lemma III.2, Listing 1) and the adaptive block-size
selection of Algorithm 1 / Equation (9).

NVFP4 format
------------
NVIDIA NVFP4 is a 4-bit floating-point format with **1 sign bit, 2 exponent
bits and 1 mantissa bit** (E2M1).  The representable magnitudes are

    {0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6}

so the dynamic range is limited to ``[-6, 6]`` (Definition III.1 clips the
scaled domain to ``[-6, 6]``).

Block-scaled fake quantization (Listing 1)
------------------------------------------
For a tensor ``x`` split into blocks of ``block_size`` elements we use one
scale per block,

    s = amax(block) / 6,

so that the representable range ``[-6, 6]`` (in units of ``s``) exactly covers
``[-amax(block), amax(block)]``.  Each element is then quantized as

    Q_FP4(v; s) = s * clip(round(v / s)_FP4, -6, 6),

where ``round(.)_FP4`` maps to the nearest E2M1 level.  The forward pass
returns the de-quantized tensor while the straight-through estimator (STE)
passes the gradient unchanged to the underlying full-precision weights, which
is what makes QAT possible.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# NVFP4 (E2M1) level table
# --------------------------------------------------------------------------- #
def _e2m1_levels() -> torch.Tensor:
    """Signed E2M1 (NVFP4) representable levels.

    Magnitudes: 0.5, 0.75, 1, 1.5, 2, 3, 4, 6  (max exponent range = 6,
    matching Definition III.1's clip at [-6, 6]).  E2M1 has no exact 0, so
    near-zero values quantize to the smallest representable magnitude.
    """
    mags = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    levels = []
    for m in mags:
        levels.append(m)
        levels.append(-m)
    return torch.tensor(sorted(levels), dtype=torch.float32)


# E2M1 clip bound (Definition III.1).
_FP4_CLIP = 6.0


def fake_quantize_fp4(
    x: torch.Tensor,
    block_size: int = 64,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-scaled NVFP4 fake quantization with a straight-through estimator.

    Args:
        x: Arbitrary-shaped full-precision tensor (weights or activations).
        block_size: Number of elements per quantization block. ``<=0`` disables
            quantization and returns ``x`` unchanged.
        levels: Custom NVFP4 level table. Defaults to the E2M1 table.

    Returns:
        The de-quantized tensor (forward = quantized, backward = identity).
    """
    if block_size <= 0:
        return x

    if levels is None:
        levels = _e2m1_levels()
    levels = levels.to(device=x.device, dtype=x.dtype)

    orig = x.shape
    flat = x.reshape(-1)
    n = flat.numel()

    # Pad to a multiple of block_size so every block is full.
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)

    # Per-block scale: s = amax / 6  (Definition III.1).
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    scale = amax / _FP4_CLIP  # (num_blocks, 1)

    norm = blocks / scale  # in [-6, 6]
    # Nearest E2M1 level (argmin is non-differentiable; STE handles backward).
    dist = (norm.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    idx = dist.argmin(dim=-1)
    q = levels[idx]
    q_block = q * scale  # de-quantize back to the original scale

    # Drop padding and restore the original shape.
    q_flat = q_block.reshape(-1)[:n].view(orig)

    # Straight-through estimator: forward = quantized, backward = identity.
    return x + (q_flat - x).detach()


# --------------------------------------------------------------------------- #
# Quantized Linear layer (subclass so state_dict keys stay "weight"/"bias")
# --------------------------------------------------------------------------- #
class FP4Linear(nn.Linear):
    """``nn.Linear`` that fake-quantizes its weight to NVFP4 on the forward pass.

    Subclassing ``nn.Linear`` keeps the parameter names ``weight`` / ``bias``,
    so checkpoints are fully compatible with a plain ``nn.Linear`` and can be
    saved / loaded / used by EMA without any key remapping.
    """

    def __init__(self, linear: nn.Linear, block_size: int = 64) -> None:
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        # Reassign parameters so they live on the same device / dtype as the
        # source linear (``copy_`` alone would keep them on the default device).
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_fp4(self.weight, self.block_size)
        return F.linear(x, w, self.bias)


def _make_fp4_linear(linear: nn.Linear, block_size: int) -> nn.Module:
    return FP4Linear(linear, block_size)


def apply_fp4_to_linear_modules(
    module: nn.Module,
    block_size: int = 64,
) -> nn.Module:
    """Recursively wrap every ``nn.Linear`` in ``module`` with ``FP4Linear``.

    Already-wrapped ``FP4Linear`` layers are left untouched (their block size
    is preserved) so this is safe to call multiple times.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FP4Linear):
            setattr(module, name, _make_fp4_linear(child, block_size))
        else:
            apply_fp4_to_linear_modules(child, block_size)
    return module


# --------------------------------------------------------------------------- #
# Adaptive block-wise quantization (Algorithm 1, Equation 9)
# --------------------------------------------------------------------------- #
def nvfp4_adaptive_block_size(
    weight: torch.Tensor,
    candidates: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    lam: float = 1.0,
) -> int:
    """Pick the NVFP4 block size for a single (linear) weight tensor.

    Implements the cost of Algorithm 1 / Equation (9):

        cost(b) = err(b) + lambda * mem(b) / mem_max

    where ``err(b)`` is the MSE reconstruction error of the FP4-quantized
    weight and ``mem(b)`` is the relative memory overhead of storing one scale
    per block.  Both terms are normalized across candidates so the trade-off
    is well behaved for any layer shape.

    Args:
        weight: A 2-D weight tensor (``out_features x in_features``).
        candidates: Candidate block sizes (spatial dimension, per the paper).
        lam: Trade-off between accuracy and memory (Equation 9).

    Returns:
        The selected block size (one of ``candidates``).
    """
    w = weight.detach().reshape(-1)
    n = w.numel()

    errs: list[float] = []
    mems: list[float] = []
    for b in candidates:
        q = fake_quantize_fp4(w, block_size=int(b))
        errs.append(float(F.mse_loss(q, w)))
        num_blocks = math.ceil(n / b)
        # Memory ratio: (num_blocks * sizeof(scale)) / sizeof(weight).
        mems.append(num_blocks * 4.0 / (n * 4.0))

    err_max = max(errs) if max(errs) > 0 else 1.0
    mem_max = max(mems) if max(mems) > 0 else 1.0

    best_b, best_cost = candidates[0], float("inf")
    for i, b in enumerate(candidates):
        cost = (errs[i] / err_max) + lam * (mems[i] / mem_max)
        if cost < best_cost:
            best_cost, best_b = cost, b
    return int(best_b)


def apply_adaptive_fp4(
    module: nn.Module,
    candidates: Iterable[int] = (1, 2, 4, 8, 16, 32, 64),
    lam: float = 1.0,
) -> nn.Module:
    """Wrap every ``nn.Linear`` with ``FP4Linear`` using a *per-layer* block
    size chosen by :func:`nvfp4_adaptive_block_size` (Algorithm 1).

    Already-wrapped ``FP4Linear`` layers keep their existing block size.
    """
    cand = tuple(int(c) for c in candidates)
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FP4Linear):
            b = nvfp4_adaptive_block_size(child.weight, cand, lam)
            setattr(module, name, FP4Linear(child, b))
        else:
            apply_adaptive_fp4(child, cand, lam)
    return module
