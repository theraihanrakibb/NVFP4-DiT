# NVFP4-DiT — training and inference code

This folder contains a minimal reference implementation for **4-bit (NVFP4-style) quantization-aware training** of a compact **Diffusion Transformer (DiT)** for video latents, plus optional Triton-style kernel hooks.

## Setup

```bash
cd code
pip install -r requirements.txt
```

On Windows, `triton` is omitted from the default install path; kernels fall back to PyTorch matmul.

## Layout

| File | Role |
|------|------|
| `train.py` | Training loop, EMA, checkpointing |
| `model.py` | DiT blocks, timestep + audio conditioning |
| `quantization.py` | FP4-style fake quantization, block scales |
| `kernels.py` | Packed matmul / attention helpers (Triton when available) |
| `inference.py` | Sampling loop from a checkpoint |
| `configs/*.yaml` | Model width, depth, and training hyperparameters |

## Quick start

```bash
# Train (small config, CPU-friendly latent shape for smoke test)
python train.py --config configs/dit_s.yaml

# Sample
python inference.py --config configs/dit_s.yaml --checkpoint checkpoints/last.pt --out ../results/sample_videos/gen.pt
```

Replace latent pipelines, datasets, and loss targets with your own data loader and VAE as you scale up.
