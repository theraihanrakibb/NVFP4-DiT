"""DDPM-style sampling from a trained NVFP4-DiT checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from model import DiT, DiTConfig
from quantization import apply_adaptive_fp4, apply_fp4_to_linear_modules


def _build_schedule(dcfg: dict, timesteps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    schedule = dcfg.get("schedule", "cosine")
    if schedule == "cosine":
        betas = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((betas / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)
    else:
        betas = torch.linspace(
            dcfg.get("beta_start", 1e-4), dcfg.get("beta_end", 0.02), timesteps
        )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dit_s.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--out", type=str, default="sample.pt")
    ap.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Override number of reverse diffusion steps (default: full schedule)",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    with open(base / args.config, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    m = full["model"]
    tcfg = full.get("train", {})
    dcfg = full.get("diffusion", {})

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
        if tcfg.get("adaptive_block_size", False):
            apply_adaptive_fp4(model, candidates=(1, 2, 4, 8, 16, 32, 64))
        else:
            apply_fp4_to_linear_modules(model, block_size=int(tcfg.get("fp4_block_size", 64)))

    ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    state = ckpt["ema"] if args.use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    T_train = int(dcfg.get("timesteps", 1000))
    T = T_train if args.sample_steps is None else max(2, min(T_train, args.sample_steps))
    betas, alphas, alphas_cumprod = _build_schedule(dcfg, T)
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)

    bsz = 1
    x = torch.randn(bsz, m["in_channels"], m["num_frames"], m["frame_height"], m["frame_width"], device=device)
    audio = torch.randn(16, m["audio_dim"], device=device).unsqueeze(0)

    with torch.no_grad():
        for step in reversed(range(T)):
            t = torch.full((bsz,), step, device=device, dtype=torch.long)
            beta_t = betas[step]
            alpha_t = alphas[step]
            alpha_bar_t = alphas_cumprod[step].clamp(min=1e-5)
            eps = model(x, t, audio)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps)
            if step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": x.cpu(), "config": full}, out_path)


if __name__ == "__main__":
    main()
