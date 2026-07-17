"""Optional SyncNet interface for the audio-visual synchronization loss.

The paper's combined objective (Equation 11) includes a synchronization term

    L_sync = SyncNet(x_frames, c_audio)

where ``SyncNet`` [40] (Chung & Zisserman, ICCV 2019) scores how well the
generated frames are lip/audio synchronized.

A *pretrained* SyncNet checkpoint is required to produce a meaningful signal;
such a model is not bundled with this repository.  This module therefore
provides:

* a thin :class:`SyncNet` wrapper that loads a checkpoint if a path is given,
* a :class:`SyncLoss` that returns ``0`` (and warns once) when no SyncNet is
  available, so training stays runnable end-to-end without external weights.

Replace :meth:`SyncNet.forward` with a real pretrained implementation and pass
``--syncnet-path`` to ``train.py`` to activate the synchronization term.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn


class SyncNet(nn.Module):
    """Placeholder SyncNet. Loads a checkpoint if ``path`` is provided."""

    def __init__(self, embed_dim: int = 512, path: str | None = None) -> None:
        super().__init__()
        # Lazy projections so the placeholder adapts to any (frames, audio)
        # layout; a real SyncNet would contain the audiovisual encoder from
        # [40]. The sync score is produced by `forward`.
        self.audio_enc = nn.LazyLinear(embed_dim)
        self.visual_enc = nn.LazyLinear(embed_dim)
        self._warned = False
        if path is not None:
            self.load_state_dict(torch.load(Path(path), map_location="cpu"), strict=False)

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Return a (lower-is-better) synchronization error in [0, 1].

        With the placeholder encoder this is undefined; callers should only
        use this method with a real pretrained SyncNet.
        """
        if not self._warned:
            warnings.warn(
                "SyncNet placeholder is being evaluated; supply a real "
                "pretrained SyncNet via --syncnet-path for a meaningful loss.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True
        # frames: (B, T, D_vis), audio: (B, A, D_aud) -> pseudo distance
        vf = self.visual_enc(frames.mean(1))
        af = self.audio_enc(audio.mean(1))
        cos = torch.cosine_similarity(vf, af, dim=-1)
        return (1.0 - cos).mean()


class SyncLoss(nn.Module):
    """Synchronization loss term for Equation (11).

    When ``syncnet`` is ``None`` the loss is a no-op (returns ``0``) so the
    pipeline runs without external weights; the synchronization term is simply
    skipped in the total loss.
    """

    def __init__(self, syncnet: SyncNet | None = None) -> None:
        super().__init__()
        self.syncnet = syncnet

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if self.syncnet is None:
            return torch.zeros((), device=frames.device)
        return self.syncnet(frames, audio)
