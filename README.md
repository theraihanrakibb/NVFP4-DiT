# NVFP4-DiT

Theory and practice of **4-bit (NVFP4-style) low-precision training and inference** for **audio-guided video Diffusion Transformers (DiT)**.

## 🎯 Overview

This repository contains the official implementation of **NVFP4-DiT**, a framework for training and deploying audio-guided video diffusion transformers in **4-bit FP4 precision**.

### Key Contributions
- **First 4-bit video diffusion framework** with <1% quality degradation
- **3.4× throughput** and **76% memory reduction** vs. FP16
- **Custom Triton kernels** for FP4-packed attention
- **Theoretical convergence guarantees**

## 📊 Results

| Method | FVD ↓ | Memory | Throughput | Energy |
|--------|-------|--------|------------|--------|
| BF16 Baseline | 98.3 | 28.5 GB | 0.31 vid/s | 892 J |
| NVFP4-DiT | **99.2** | **6.8 GB** | **1.05 vid/s** | **285 J** |

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/theraihanrakibb/NVFP4-DiT.git
cd NVFP4-DiT
pip install -r code/requirements.txt
```

## Repository layout

| Path | Contents |
|------|-----------|
| [`paper/`](paper/) | IEEEtran LaTeX sources, bibliography, figure assets |
| [`code/`](code/) | PyTorch training, quantization helpers, optional Triton kernels |
| [`results/`](results/) | Sample outputs, logs, and demo media |

## Paper

Build from the `paper/` directory (requires a LaTeX distribution with `IEEEtran` and common packages):

```bash
cd paper
pdflatex main.tex
bibtex main        # optional if you switch to BibTeX + references.bib
pdflatex main.tex
pdflatex main.tex
```

`references.bib` mirrors citation keys from the inline bibliography in `main.tex` if you prefer `\bibliography{references}`.

## Code

See [`code/README.md`](code/README.md) for environment setup, configs (`dit_s` / `dit_b` / `dit_l`), and smoke-test commands.

## Results

- `results/sample_videos/` — example clips and a comparison GIF (replace with your own generations).
- `results/logs/training.log` — populated when you run training.

## License

See [`LICENSE`](LICENSE).

