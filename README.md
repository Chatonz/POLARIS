# POLARIS

**Projection-Orthogonal Least Squares for Robust and Adaptive Inversion**

## About

POLARIS addresses a critical issue in diffusion-based image editing and restoration: the **noise approximation error** that accumulates during the DDIM inversion process. 

Traditional inversion methods approximate noise at step $t$ using predictions from step $t-1$, causing severe error accumulation. POLARIS reformulates inversion as an **error-origin problem** rather than an error-compensation problem.

**Key Innovation:** Instead of optimizing embeddings or latent codes to offset drift, POLARIS treats the guidance scale $\omega$ as a **step-wise variable** and derives a mathematically grounded formula to minimize inversion error at each step. This improves latent quality with **minimal code changes** and negligible performance overhead.

---

## Features

- **Image Editing** (`editing.py`): Text-guided image manipulation with spatial attention control
- **Reconstruction** (`reconstruction.py`): Compare DDIM inversion methods (Standard, POLARIS, Dynamic Exact)
- **Restoration** (`restoration.py`): Image restoration tasks (deblurring, super-resolution, inpainting, colorization) using DS-DDRM

## Installation

```bash
pip install torch torchvision diffusers transformers pillow
pip install numpy scipy scikit-image matplotlib lpips tqdm
```

## Quick Start

### Image Editing
```bash
python editing.py \
  --image_path <path> \
  --source_prompt "original description" \
  --target_prompt "target description" \
  --edit_phrase "word_to_edit"
```

### Image Reconstruction
```bash
python reconstruction.py --image_path <path> --prompt "image description"
```

### Image Restoration
```bash
python restoration.py --image_path <path> --prompt "image description"
```

## Usage Examples

```bash
# Edit a cat image
python editing.py \
  --image_path cat.jpg \
  --source_prompt "A cat on ther ight of a tennis racket. " \
  --target_prompt "A dog on ther ight of a tennis racket." \
  --edit_phrase "dog"

# Compare inversion methods
python reconstruction.py --image_path cat.jpg

# Restore degraded image
python restoration.py --image_path blurred_cat.jpg
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_path` | - | Input image path (required) |
| `--prompt` | - | Image description |
| `--steps` | 50/100 | Inference steps |
| `--guidance_scale` | 7.5 | CFG guidance scale |
| `--output_dir` | ./output | Output directory |
| `--mask_pow` | 10.0 | Attention mask sharpness |

## Output

- **editing.py**: `result_fixed.png`, `result_dynamic.png`, `comparison.png`
- **reconstruction.py**: `methods_comparison.png`, `scales_evolution.png`
- **restoration.py**: Task-specific degraded/restored images with metrics

## Requirements

- CUDA-compatible GPU (recommended)
- Python 3.8+
- ~7GB VRAM for inference



