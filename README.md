# NVFP4-DiT

<p align="center">
  <img src="https://img.shields.io/badge/IEEE%20TMM-Under%20Review-blue" alt="IEEE TMM Under Review"/>
  <img src="https://img.shields.io/github/license/theraihanrakibb/NVFP4-DiT" alt="License"/>
  <img src="https://img.shields.io/github/stars/theraihanrakibb/NVFP4-DiT" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/theraihanrakibb/NVFP4-DiT" alt="Forks"/>
  <img src="https://img.shields.io/github/last-commit/theraihanrakibb/NVFP4-DiT" alt="Last Commit"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
</p>


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


## Citation

If you use NVFP4-DiT in your research, please cite:

```bibtex
@article{raihan2026nvfp4dit,
  title   = {NVFP4-DiT: Efficient 4-Bit Audio-Guided Video Diffusion Transformers for Low-Cost Video Generation},
  author  = {Raihan, Md Rakibul Islam and others},
  journal = {IEEE Transactions on Multimedia (TMM)},
  year    = {2026},
  note    = {Under review}
}
```

> Replace `and others` with your co-authors before submission.
