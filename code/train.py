"""Train DiT with MSE noise prediction and optional FP4 QAT."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from model import DiT, DiTConfig
from quantization import apply_fp4_to_linear_modules


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
    args = ap.parse_args()
    cfg_path = Path(__file__).resolve().parent / args.config
    with open(cfg_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)

    m = full["model"]
    tcfg = dict(full["train"])
    dcfg = full["diffusion"]
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
    if tcfg.get("use_fp4_qat", False):
        apply_fp4_to_linear_modules(model, block_size=int(tcfg.get("fp4_block_size", 64)))

    betas = linear_beta_schedule(dcfg["timesteps"], dcfg["beta_start"], dcfg["beta_end"])
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    ema = EMA(model, tcfg.get("ema_decay", 0.999))

    n_samples = args.dataset_samples if args.dataset_samples is not None else 256
    ds = SyntheticLatentDataset(
        n=n_samples,
        c=m["in_channels"],
        t=m["num_frames"],
        h=m["frame_height"],
        w=m["frame_width"],
        audio_dim=m["audio_dim"],
    )
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True)

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(__file__).resolve().parents[1] / "results" / "logs" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    num_epochs = int(tcfg["epochs"])
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{num_epochs}")
        for x0, audio in pbar:
            x0 = x0.to(device)
            audio = audio.to(device)
            bsz = x0.shape[0]
            t = torch.randint(0, dcfg["timesteps"], (bsz,), device=device)
            noise = torch.randn_like(x0)
            a = alphas_cumprod[t].view(bsz, 1, 1, 1, 1)
            xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
            pred = model(xt, t, audio)
            loss = nn.functional.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            ema.update(model)
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))
            if global_step % int(tcfg.get("log_every", 10)) == 0:
                line = f"step={global_step} epoch={epoch+1} loss={loss.item():.6f}\n"
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(line)

        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "cfg": full}, ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
