"""Compact spatiotemporal DiT with optional audio cross-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class DiTConfig:
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_frames: int = 8
    frame_height: int = 32
    frame_width: int = 32
    audio_dim: int = 128


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (supports float or integer timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTAttention(nn.Module):
    """Self-attention with optional learnable per-head scale (Theorem III.6).

    Adding ``log(s_head)`` to the scaled query-key logits lets the network
    learn per-head dynamic range under low precision, which the paper shows is
    a sufficient condition for temporal-coherence preservation (Eq. 7 / 8).
    """

    def __init__(self, dim: int, heads: int, learnable_scale: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        if learnable_scale:
            self.head_scale = nn.Parameter(torch.ones(heads))
        else:
            self.register_parameter("head_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        qkv = self.to_qkv(x).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, h, n, dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, h, n, n)
        if self.head_scale is not None:
            attn = attn + self.head_scale.view(1, h, 1, 1).log()
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class CrossModalAttention(nn.Module):
    """Audio-conditioned cross-attention with learnable per-head scales (Eq. 10).

        Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + log(s_head) ) * V

    ``s_head in R^H`` is initialized to 1 (so ``log(1) = 0`` and the module
    starts as a standard cross-attention), then optimized during QAT to allocate
    precision for audio-visual fusion under FP4 quantization.
    """

    def __init__(self, dim: int, heads: int, audio_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(audio_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.head_scale = nn.Parameter(torch.ones(heads))

    def forward(self, x: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        q = self.to_q(x).reshape(b, n, h, d // h).permute(0, 2, 1, 3)  # (b, h, n, dh)
        kv = self.to_kv(audio).reshape(b, -1, 2, h, d // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (b, h, A, dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, h, n, A)
        # Learnable per-head bias (Eq. 10): add log(s_head) to every logit.
        attn = attn + self.head_scale.view(1, h, 1, 1).log()
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, audio_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DiTAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.norm_audio = nn.LayerNorm(dim)
        self.audio_attn = CrossModalAttention(dim, heads, audio_dim)

    def forward(self, x: torch.Tensor, audio: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # adaLN-style shift/scale from timestep (simplified: add t_emb per token).
        t = t_emb.unsqueeze(1)
        x = x + self.attn(self.norm1(x) + t)
        x = x + self.mlp(self.norm2(x) + t)
        xa = self.audio_attn(self.norm_audio(x), audio)
        return x + xa


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        fh, fw = cfg.frame_height // p, cfg.frame_width // p
        self.num_patches = cfg.num_frames * fh * fw
        patch_dim = cfg.in_channels * p * p
        self.patch_emb = nn.Linear(patch_dim, cfg.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, cfg.hidden_size))
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.audio_dim)
                for _ in range(cfg.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, patch_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) latents
        t: (B,) diffusion step indices (int) or (B,) float continuous steps
        audio: (B, A, D) preprocessed audio features (A = sequence length)
        """
        cfg = self.cfg
        p = cfg.patch_size
        x = rearrange(
            x,
            "b c t (h p1) (w p2) -> b (t h w) (c p1 p2)",
            p1=p,
            p2=p,
        )
        x = self.patch_emb(x)
        x = x + self.pos_emb
        t_emb = self.t_mlp(_timestep_embedding(t, cfg.hidden_size))
        for blk in self.blocks:
            x = blk(x, audio, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b (t h w) (c p1 p2) -> b c t (h p1) (w p2)",
            t=cfg.num_frames,
            h=cfg.frame_height // p,
            w=cfg.frame_width // p,
            c=cfg.in_channels,
            p1=p,
            p2=p,
        )
        return x
