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

### Training flags

| Flag | Effect |
|------|--------|
| `--epochs N` | Override `train.epochs`. |
| `--dataset-samples N` | Override the synthetic dataset size (default 256). |
| `--syncnet-path PATH` | Load a pretrained SyncNet and enable the synchronization loss (Eq. 11). |

### Config knobs (per `configs/*.yaml`)

- `train.use_fp4_qat` — wrap every `nn.Linear` in `FP4Linear` (NVFP4 E2M1 QAT).
- `train.fp4_block_size` — fixed block size when `adaptive_block_size` is false.
- `train.adaptive_block_size` — select a per-layer block size via Algorithm 1 / Eq. (9).
- `train.loss.lambda_sync` / `lambda_temp` — weights for the sync and temporal-smoothness terms of Eq. (11) (Table XV: 0.1 / 0.01).
- `train.loss.use_sync` — enable the sync term (also needs `--syncnet-path`).
- `train.loss.use_temp` — enable the temporal-coherence regularizer (Theorem III.6).
- `diffusion.schedule` — `cosine` (paper default, Table XV) or `linear`.

### What maps to the paper

- `quantization.py` — NVFP4 (E2M1) block-scaled fake quantization, STE, and adaptive block-size selection (Definition III.1, Listing 1, Algorithm 1).
- `model.py` — `CrossModalAttention` adds learnable per-head scales `softmax(QKᵀ/√d_k + log s_head) V` (Eq. 10); `DiTAttention` adds per-head scales for temporal coherence (Theorem III.6).
- `kernels.py` — FP4-packed matmul with on-the-fly dequant (Listing 2); uses Triton when `NVFP4_USE_TRITON=1` on CUDA, else a PyTorch fallback.
- `syncnet.py` — pluggable `SyncNet` / `SyncLoss` for the synchronization term of Eq. (11).

Replace latent pipelines, datasets, and loss targets with your own data loader and VAE as you scale up.
