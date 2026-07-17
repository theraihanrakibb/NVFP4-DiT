"""Generate all paper figures into the repo-root ``images/`` folder.

Every figure is produced directly from the tables in the paper (main text +
appendix), so the plots stay in sync with the reported numbers. Run:

    python code/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent palette.
C_BASE = "#4c72b0"   # BF16 / baselines
C_OURS = "#dd8452"   # NVFP4-DiT (ours)
C_GRID = "#dddddd"
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "figure.autolayout": False,
})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# 1. Architecture / pipeline diagram
# --------------------------------------------------------------------------- #
def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#4c72b0", fs=9, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.3, color="#555"))

    # Inputs
    box(0.2, 5.3, 2.0, 0.9, "Text prompt\n(text encoder)", fc="#f7f0e6", ec="#dd8452")
    box(0.2, 3.9, 2.0, 0.9, "Audio\nmel-spectrogram", fc="#f7f0e6", ec="#dd8452")
    box(0.2, 2.5, 2.0, 0.9, "Noisy video\nlatents  x_t", fc="#f7f0e6", ec="#dd8452")

    # DiT backbone
    box(2.8, 1.4, 6.0, 5.0, "", fc="#fbfbfb", ec="#888", lw=1.0)
    ax.text(5.8, 6.15, "Audio-Guided Video DiT  (NVFP4 QAT)", ha="center", fontsize=10, weight="bold")
    box(3.1, 5.0, 5.4, 0.8, "Patch embed + positional + timestep (sinusoidal) embedding")
    box(3.1, 3.9, 5.4, 0.8, "DiT Block x L:  Self-Attn (per-head s_head) -> Cross-Modal Attn (Eq. 10) -> FFN",
        fc="#eef3fb")
    box(3.1, 2.8, 5.4, 0.8, "AdaLN modulation from timestep  +  LayerNorm")
    box(3.1, 1.7, 5.4, 0.8, "Final norm + linear head -> predicted noise eps_theta")

    # Output
    box(9.3, 3.6, 2.4, 1.0, "Denoised latents\n-> VAE decoder\n-> Video", fc="#e9f5ec", ec="#59a14f")

    # Kernels / serving band
    box(2.8, 0.25, 6.0, 0.9, "FP4-packed Triton kernels (Listing 2)  +  vLLM continuous batching (Alg. 3)",
        fc="#fdecea", ec="#c0392b")

    # arrows
    for y in (5.75, 4.35, 2.95):
        arrow(2.2, y, 2.8, 4.0)
    arrow(8.5, 4.0, 9.3, 4.1)
    arrow(5.8, 1.4, 5.8, 1.15)

    ax.set_title("NVFP4-DiT: 4-bit audio-guided video diffusion transformer", fontsize=12, weight="bold")
    _save(fig, "architecture.png")


# --------------------------------------------------------------------------- #
# 2. FP4 (E2M1) quantization concept
# --------------------------------------------------------------------------- #
def fp4_levels() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    mags = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    levels = sorted([-m for m in mags] + [0.0] + mags)
    ax1.stem(levels, np.ones(len(levels)), basefmt=" ")
    ax1.set_title("NVFP4 (E2M1) representable levels")
    ax1.set_xlabel("value (in units of block scale s)")
    ax1.set_yticks([])
    ax1.axvline(-6, color="r", ls="--", lw=0.8); ax1.axvline(6, color="r", ls="--", lw=0.8)
    ax1.text(-6, 1.02, "clip -6", color="r", fontsize=8, ha="center")
    ax1.text(6, 1.02, "clip +6", color="r", fontsize=8, ha="center")

    # block scaling illustration
    x = np.linspace(-1, 1, 200)
    y = np.tanh(3 * x)
    ax2.plot(x, y, color="#888", lw=1.2, label="original block")
    amax = 0.9
    s = amax / 6.0
    q = np.clip(np.round(y / s / 0.5) * 0.5 * s, -6 * s, 6 * s)
    ax2.step(x, q, color=C_OURS, lw=1.6, where="mid", label="FP4-quantized (s = amax/6)")
    ax2.set_title("Block-scaled FP4 fake quantization (STE)")
    ax2.set_xlabel("element index (block of 64)")
    ax2.legend(fontsize=8)
    _save(fig, "fp4_quantization.png")


# --------------------------------------------------------------------------- #
# 3. Main results (Table II)
# --------------------------------------------------------------------------- #
def main_results() -> None:
    methods = ["DiT-B\n(BF16)", "DiT-B\n+FP8", "DiT-B\n+INT8", "DiT-B\n+static FP4", "NVFP4-DiT\n(ours)"]
    fvd = [98.3, 100.1, 105.7, 187.3, 99.2]
    sync = [0.892, 0.885, 0.871, 0.723, 0.886]
    clip = [0.312, 0.308, 0.301, 0.241, 0.309]
    lpips = [0.087, 0.089, 0.094, 0.156, 0.088]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    titles = ["FVD (lower better)", "SyncNet (higher better)", "CLIP-T (higher better)", "LPIPS (lower better)"]
    data = [fvd, sync, clip, lpips]
    for ax, t, d in zip(axes, titles, data):
        colors = [C_BASE] * 4 + [C_OURS]
        ax.bar(methods, d, color=colors)
        ax.set_title(t, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Table II - Quality comparison on WebVid-10M", fontsize=11, weight="bold")
    _save(fig, "main_results.png")


# --------------------------------------------------------------------------- #
# 4. Quality across architectures (Table III)
# --------------------------------------------------------------------------- #
def architectures() -> None:
    models = ["DiT-S", "DiT-B", "DiT-L", "SVD"]
    fvd_b = [112.4, 98.3, 87.6, 82.1]; fvd_f = [113.8, 99.2, 88.9, 83.4]
    mem_b = [14.2, 28.5, 56.3, 42.1]; mem_f = [3.2, 6.8, 13.5, 10.1]
    x = np.arange(len(models)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x - w/2, fvd_b, w, label="BF16", color=C_BASE)
    ax1.bar(x + w/2, fvd_f, w, label="NVFP4-DiT", color=C_OURS)
    ax1.set_xticks(x); ax1.set_xticklabels(models); ax1.set_title("FVD (lower better)"); ax1.legend()
    ax2.bar(x - w/2, mem_b, w, label="BF16", color=C_BASE)
    ax2.bar(x + w/2, mem_f, w, label="NVFP4-DiT", color=C_OURS)
    ax2.set_xticks(x); ax2.set_xticklabels(models); ax2.set_title("Memory (GB) - 76.3% avg reduction"); ax2.legend()
    fig.suptitle("Table III - Quality & memory across architectures", fontsize=11, weight="bold")
    _save(fig, "quality_architectures.png")


# --------------------------------------------------------------------------- #
# 5. Inference performance (Table IV)
# --------------------------------------------------------------------------- #
def inference_perf() -> None:
    methods = ["BF16", "FP8", "INT8", "NVFP4-DiT"]
    throughput = [0.31, 0.67, 0.72, 1.05]
    latency = [101.2, 46.8, 43.6, 29.8]
    memory = [28.5, 14.3, 12.1, 6.8]
    energy = [892, 412, 384, 285]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    items = [("Throughput (videos/s, higher better)", throughput),
             ("Latency (ms/frame, lower better)", latency),
             ("Memory (GB, lower better)", memory),
             ("Energy (J/video, lower better)", energy)]
    for ax, (t, d) in zip(axes.flat, items):
        colors = [C_BASE]*3 + [C_OURS]
        ax.bar(methods, d, color=colors); ax.set_title(t, fontsize=9)
    fig.suptitle("Table IV - Inference performance (batch=4, 32 frames)", fontsize=11, weight="bold")
    _save(fig, "inference_performance.png")


# --------------------------------------------------------------------------- #
# 6. SyncNet noise robustness (Table V)
# --------------------------------------------------------------------------- #
def syncnet_robust() -> None:
    cond = ["Clean", "+10dB", "+20dB", "Cross-modal"]
    b = [0.892, 0.845, 0.781, 0.023]; f = [0.886, 0.839, 0.774, 0.021]
    x = np.arange(len(cond)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, b, w, label="BF16", color=C_BASE)
    ax.bar(x + w/2, f, w, label="NVFP4-DiT", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(cond); ax.set_ylabel("SyncNet accuracy")
    ax.set_title("Table V - Audio-visual sync under noise (<0.7% drop)", weight="bold"); ax.legend()
    _save(fig, "syncnet_robustness.png")


# --------------------------------------------------------------------------- #
# 7. Training cost (Table VI)
# --------------------------------------------------------------------------- #
def training_cost() -> None:
    prec = ["BF16", "FP8", "NVFP4-DiT"]
    gpu = [4800, 2400, 1600]; energy = [28.8, 14.4, 9.6]; cost = [9600, 4800, 3200]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, t, d in zip(axes, ["GPU hours", "Energy (MWh)", "Cost ($)"], [gpu, energy, cost]):
        ax.bar(prec, d, color=[C_BASE, "#8fa8c8", C_OURS]); ax.set_title(t, fontsize=9)
    fig.suptitle("Table VI - Training cost (66.7% reduction)", fontsize=11, weight="bold")
    _save(fig, "training_cost.png")


# --------------------------------------------------------------------------- #
# 8. Block-size ablation (Table VII)
# --------------------------------------------------------------------------- #
def block_size() -> None:
    blk = ["1x1", "4x4", "8x8", "16x16", "32x32", "Adaptive"]
    fvd = [98.7, 98.9, 99.2, 100.1, 104.3, 98.9]
    sync = [0.888, 0.887, 0.886, 0.882, 0.871, 0.888]
    x = np.arange(len(blk))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    bars = ax1.bar(x, fvd, color=[C_BASE]*5 + [C_OURS], alpha=0.85)
    ax1.set_ylabel("FVD (lower better)"); ax1.set_xticks(x); ax1.set_xticklabels(blk)
    ax2.plot(x, sync, color="#c0392b", marker="o", lw=1.6, label="SyncNet")
    ax2.set_ylabel("SyncNet (higher better)"); ax2.grid(False)
    ax1.set_title("Table VII - Effect of FP4 block size (DiT-B)", weight="bold")
    ax2.legend(loc="upper left")
    _save(fig, "block_size_ablation.png")


# --------------------------------------------------------------------------- #
# 9. QAT ablation (Table VIII)
# --------------------------------------------------------------------------- #
def qat_ablation() -> None:
    cfg = ["No QAT\n(PTQ)", "+Standard\nQAT", "+Learnable\nscales", "+Cross-modal\nQAT", "+Temporal\nloss"]
    fvd = [187.3, 112.4, 102.1, 99.8, 99.2]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(cfg, fvd, color=[C_BASE]*4 + [C_OURS])
    ax.set_ylabel("FVD (lower better)")
    ax.set_title("Table VIII - Impact of QAT components", weight="bold")
    for i, v in enumerate(fvd):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=8)
    _save(fig, "qat_ablation.png")


# --------------------------------------------------------------------------- #
# 10. Kernel microbenchmarks (Table IX)
# --------------------------------------------------------------------------- #
def kernels() -> None:
    k = ["FP16\nbaseline", "Naive FP4", "FP4 Triton\n(no pack)", "FP4 Triton\n(packed)", "FP4 FlashAttn\n(ours)"]
    t = [245, 187, 132, 89, 67]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(k, t, color=[C_BASE]*4 + [C_OURS])
    ax.set_ylabel("Time (us, lower better)")
    ax.set_title("Table IX - Kernel microbenchmarks (DiT-B attention)", weight="bold")
    for i, v in enumerate(t):
        ax.text(i, v + 3, f"{v}", ha="center", fontsize=8)
    _save(fig, "kernel_performance.png")


# --------------------------------------------------------------------------- #
# 11. Cross-dataset generalization (Table X)
# --------------------------------------------------------------------------- #
def cross_dataset() -> None:
    pairs = ["WebVid->UCF", "WebVid->VGG", "UCF->WebVid", "VGG->UCF"]
    b = [125.3, 98.7, 112.4, 118.7]; f = [126.8, 99.9, 114.1, 120.3]
    x = np.arange(len(pairs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, b, w, label="BF16", color=C_BASE)
    ax.bar(x + w/2, f, w, label="NVFP4-DiT", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(pairs); ax.set_ylabel("FVD")
    ax.set_title("Table X - Cross-dataset generalization (1.2-1.5% drop)", weight="bold"); ax.legend()
    _save(fig, "cross_dataset.png")


# --------------------------------------------------------------------------- #
# 12. Compression robustness (Table XI)
# --------------------------------------------------------------------------- #
def compression() -> None:
    crf = ["CRF 18", "CRF 23", "CRF 28"]
    b = [98.3, 101.2, 108.7]; f = [99.2, 102.4, 110.1]
    x = np.arange(len(crf)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(crf, b, marker="o", color=C_BASE, label="BF16")
    ax.plot(crf, f, marker="s", color=C_OURS, label="NVFP4-DiT")
    ax.set_ylabel("FVD"); ax.set_title("Table XI - Robustness to H.264 compression", weight="bold"); ax.legend()
    _save(fig, "compression_robustness.png")


# --------------------------------------------------------------------------- #
# 13. Resolution scaling (Table XII)
# --------------------------------------------------------------------------- #
def resolution() -> None:
    res = ["336x336", "224x224", "168x168"]
    b = [98.3, 112.4, 135.2]; f = [99.2, 113.8, 137.1]
    x = np.arange(len(res)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, b, w, label="BF16", color=C_BASE)
    ax.bar(x + w/2, f, w, label="NVFP4-DiT", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(res); ax.set_ylabel("FVD")
    ax.set_title("Table XII - Robustness to resolution scaling", weight="bold"); ax.legend()
    _save(fig, "resolution_robustness.png")


# --------------------------------------------------------------------------- #
# 14. User study (Table XIII)
# --------------------------------------------------------------------------- #
def user_study() -> None:
    metrics = ["Visual Quality", "Temporal Coherence", "Audio-Visual Sync", "Overall"]
    bf16 = [48, 46, 51, 47]; fp4 = [52, 54, 49, 53]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, bf16, w, label="Prefer BF16", color=C_BASE)
    ax.bar(x + w/2, fp4, w, label="Prefer NVFP4-DiT", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8); ax.set_ylabel("% preference")
    ax.set_title("Table XIII - User preference study (n=50)", weight="bold"); ax.legend()
    _save(fig, "user_study.png")


# --------------------------------------------------------------------------- #
# 15. Latency breakdown (Table XIV)
# --------------------------------------------------------------------------- #
def latency_breakdown() -> None:
    comp = ["Attention", "FFN", "Cross-modal fusion", "Other"]
    b = [112, 78, 45, 10]; f = [38, 26, 15, 5]
    x = np.arange(len(comp)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, b, w, label="BF16", color=C_BASE)
    ax.bar(x + w/2, f, w, label="NVFP4-DiT", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(comp); ax.set_ylabel("ms / 32-frame video")
    ax.set_title("Table XIV - Inference latency breakdown (245 -> 84 ms)", weight="bold"); ax.legend()
    _save(fig, "latency_breakdown.png")


def main() -> None:
    architecture()
    fp4_levels()
    main_results()
    architectures()
    inference_perf()
    syncnet_robust()
    training_cost()
    block_size()
    qat_ablation()
    kernels()
    cross_dataset()
    compression()
    resolution()
    user_study()
    latency_breakdown()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
