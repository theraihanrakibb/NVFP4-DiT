"""Train DiT with MSE noise prediction and NVFP4 QAT (Equation 11)."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import DiT, DiTConfig
from quantization import apply_adaptive_fp4, apply_fp4_to_linear_modules
from syncnet import SyncLoss, SyncNet


class SyntheticLatentDataset(Dataset):
    def __init__(self, n: int, c: int, t: int, h: int, w: int, audio_dim: int, audio_len: int = 16) -> None:
        self.n = n
        self.shape = (c, t, h, w)
        self.audio_dim = audio_dim
        self.audio_len = audio_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = torch.randn(self.shape)
        audio = torch.randn(self.audio_len, self.audio_dim)
        return x, audio


def linear_beta_schedule(steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps)


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal style), used in the paper (Table XV)."""
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            v.copy_(v * d + msd[k] * (1.0 - d))


def temporal_coherence_loss(x0_hat: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame smoothness regularizer (Theorem III.6 / Equation 11, temp term).

    Penalizes large frame-to-frame differences of the predicted clean sample,
    encouraging temporal coherence under low-precision quantization.
    """
    if x0_hat.shape[2] < 2:
        return torch.zeros((), device=x0_hat.device)
    diff = x0_hat[:, :, 1:] - x0_hat[:, :, :-1]
    return diff.to(torch.float32).pow(2).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dit_s.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs in config")
    ap.add_argument(
        "--dataset-samples",
        type=int,
        default=None,
        help="Override synthetic dataset size (default: 256)",
    )
    ap.add_argument(
        "--syncnet-path",
        type=str,
        default=None,
        help="Path to a pretrained SyncNet checkpoint for the sync loss (Eq. 11)",
    )
    args = ap.parse_args()

    cfg_path = Path(__file__).resolve().parent / args.config
    with open(cfg_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)

    m = full["model"]
    tcfg = dict(full.get("train", {}))
    dcfg = dict(full.get("diffusion", {}))
    lcfg = dict(tcfg.get("loss", {}))
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs

    dit_cfg = DiTConfig(
        patch_size=m["patch_size"],
        in_channels=m["in_channels"],
        hidden_size=m["hidden_size"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        num_frames=m["num_frames"],
        frame_height=m["frame_height"],
        frame_width=m["frame_width"],
        audio_dim=m["audio_dim"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT(dit_cfg).to(device)

    # ---- NVFP4 QAT: wrap linears (optionally with per-layer block size) ---- #
    if tcfg.get("use_fp4_qat", False):
        if tcfg.get("adaptive_block_size", False):
            apply_adaptive_fp4(
                model,
                candidates=(1, 2, 4, 8, 16, 32, 64),
                lam=float(tcfg.get("adaptive_lam", 1.0)),
            )
        else:
            apply_fp4_to_linear_modules(model, block_size=int(tcfg.get("fp4_block_size", 64)))

    # ---- diffusion schedule ---- #
    timesteps = int(dcfg.get("timesteps", 1000))
    schedule = dcfg.get("schedule", "cosine")
    if schedule == "cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        betas = linear_beta_schedule(timesteps, dcfg.get("beta_start", 1e-4), dcfg.get("beta_end", 0.02))
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # ---- optimizer / EMA ---- #
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
    )
    ema = EMA(model, float(tcfg.get("ema_decay", 0.9999)))

    # ---- loss terms (Equation 11) ---- #
    lambda_sync = float(lcfg.get("lambda_sync", 0.1))
    lambda_temp = float(lcfg.get("lambda_temp", 0.01))
    use_sync = bool(lcfg.get("use_sync", False)) and args.syncnet_path is not None
    use_temp = bool(lcfg.get("use_temp", True))
    sync_loss = SyncLoss(SyncNet(path=args.syncnet_path) if use_sync else None).to(device)

    # ---- data ---- #
    n_samples = args.dataset_samples if args.dataset_samples is not None else 256
    ds = SyntheticLatentDataset(
        n=n_samples,
        c=m["in_channels"],
        t=m["num_frames"],
        h=m["frame_height"],
        w=m["frame_width"],
        audio_dim=m["audio_dim"],
    )
    loader = DataLoader(ds, batch_size=int(tcfg.get("batch_size", 2)), shuffle=True, drop_last=True)

    ckpt_dir = Path(__file__).resolve().parent / tcfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(__file__).resolve().parents[1] / "results" / "logs" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    num_epochs = int(tcfg.get("epochs", 5))
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{num_epochs}")
        for x0, audio in pbar:
            x0 = x0.to(device)
            audio = audio.to(device)
            bsz = x0.shape[0]
            t = torch.randint(0, timesteps, (bsz,), device=device)
            noise = torch.randn_like(x0)
            a = alphas_cumprod[t].view(bsz, 1, 1, 1, 1)
            a = a.clamp(min=1e-5)
            sqrt_a = torch.sqrt(a)
            xt = sqrt_a * x0 + torch.sqrt(1.0 - a) * noise

            pred = model(xt, t, audio)
            loss_diff = nn.functional.mse_loss(pred, noise)

            # Total loss (Equation 11): diffusion + sync + temporal smoothness.
            loss = loss_diff
            if use_temp:
                # Reconstruct the predicted clean sample and regularize its
                # frame-to-frame coherence.
                x0_hat = (xt - torch.sqrt(1.0 - a) * pred) / sqrt_a
                loss = loss + lambda_temp * temporal_coherence_loss(x0_hat)
            if use_sync:
                # Proxy "frames" from the latent (a real run decodes x0_hat first).
                frames = x0_hat.permute(0, 2, 1, 3, 4).reshape(bsz, m["num_frames"], -1)
                loss = loss + lambda_sync * sync_loss(frames, audio)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            ema.update(model)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))
            if global_step % int(tcfg.get("log_every", 10)) == 0:
                line = f"step={global_step} epoch={epoch+1} loss={loss.item():.6f}\n"
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(line)

        torch.save(
            {"model": model.state_dict(), "ema": ema.shadow.state_dict(), "cfg": full},
            ckpt_dir / "last.pt",
        )


if __name__ == "__main__":
    main()
