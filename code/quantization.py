"""FP4-style fake quantization for QAT (simulates NVFP4-style range, not bit-exact hardware)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _fp4_levels() -> torch.Tensor:
    """Very small set of normalized levels in [0, 1] for 1 mantissa bit (illustrative)."""
    return torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)


def fake_quantize_fp4(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Block-scaled fake FP4: per-block scale + STE rounding to a few discrete levels."""
    if block_size <= 0:
        return x
    orig = x.shape
    flat = x.reshape(-1)
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    norm = blocks / amax
    levels = _fp4_levels().to(device=x.device, dtype=x.dtype)
    # map [0,1] to nearest level
    dist = (norm.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    idx = dist.argmin(dim=-1)
    q = levels[idx]
    out = (q * amax).view(-1)[: x.numel()].view(orig)
    return _round_ste(out)


class FP4LinearWeight(nn.Module):
    """Wraps nn.Linear to apply fake FP4 to weights during forward (QAT)."""

    def __init__(self, linear: nn.Linear, block_size: int = 64) -> None:
        super().__init__()
        self.linear = linear
        self.block_size = block_size

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quantize_fp4(self.linear.weight, self.block_size)
        return F.linear(x, w, self.linear.bias)


def apply_fp4_to_linear_modules(module: nn.Module, block_size: int = 64) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FP4LinearWeight(child, block_size))
        else:
            apply_fp4_to_linear_modules(child, block_size)
    return module
