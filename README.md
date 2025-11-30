# POLARIS

**Projection-Orthogonal Least Squares for Robust and Adaptive Inversion**

## About

POLARIS addresses a critical issue in diffusion-based image editing and restoration: the **noise approximation error** that accumulates during the DDIM inversion process. 

Traditional inversion methods approximate noise at step $t$ using predictions from step $t-1$, causing severe error accumulation. POLARIS reformulates inversion as an **error-origin problem** rather than an error-compensation problem.

**Key Innovation:** Instead of optimizing embeddings or latent codes to offset drift, POLARIS treats the guidance scale $\omega$ as a **step-wise variable** and derives a mathematically grounded formula to minimize inversion error at each step. This improves latent quality with **minimal code changes** and negligible performance overhead.

---

## Features

### 1. Image Editing (`editing/`)
Text-guided image editing with support for:
- **SAGE (Self-Attention Guided Editing)**: Attention-based image editing
- **Mask-Prompt-to-Prompt Editing**: Spatially-aware editing with prompt control

**Files**: 
- `sage.py` - SAGE editing with POLARIS inversion support
- `mask_p2p.py` - Prompt-to-prompt editing with spatial masks

### 2. Image Reconstruction (`reconstruction/`)
Compare different inversion methods and evaluate latent quality:
- **Standard DDIM**: Baseline fixed-scale inversion
- **POLARIS Inversion**: Dynamic-weighted precise inversion (score-based)


**Files**:
- `ddim.py` - DDIM inversion methods comparison
- `score_flow.py` - Score-based vs Flow-based POLARIS comparison

### 3. Image Restoration (`restoration/`)
Multi-task restoration using DDRM (Diffusion-based Data-driven Restoration Model):
- **Deblurring**: Restore clarity to blurred images
- **Super-Resolution**: 4x resolution enhancement
- **Inpainting**: Fill missing regions (circular mask)
- **Colorization**: Convert grayscale to color

**Files**:
- `ddrm.py` - DDRM restoration with POLARIS inversion

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Chatonz/POLARIS.git
cd POLARIS

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- PyTorch 2.5.1
- torchvision 0.20.1
- diffusers 0.32.2
- transformers 4.57.1
- PIL, numpy, scipy, scikit-image
- LPIPS (perceptual metrics)

## Quick Start

### 1. SAGE Image Editing (Interactive)
```bash
cd editing
python sage.py
# Follow the interactive prompts to:
# - Load an image
# - Enter source and target prompts
# - Compare Standard SAGE vs POLARIS results
```

### 2. Prompt-to-Prompt Editing with Masks
```bash
cd editing
python mask_p2p.py \
  --image_path <image_path> \
  --source_prompt "A cat sitting on a bench" \
  --target_prompt "A dog sitting on a bench" \
  --edit_phrase "dog" \
  --output_dir ./output
```

### 3. Compare Inversion Methods
```bash
cd reconstruction
python ddim.py --image_path <image_path> --prompt "image description"
# Compares: Standard, POLARIS, Dynamic (Exact)
```

### 4. Score vs Flow Comparison
```bash
cd reconstruction
python score_flow.py \
  --image_path <image_path> \
  --prompt "image description" \
  --output_dir ./output_score_flow
```

### 5. Image Restoration
```bash
cd restoration
python ddrm.py \
  --image_path <image_path> \
  --prompt "image description" \
  --output_dir ./results
# Performs: deblurring, super-resolution, inpainting, colorization
```

## Key Parameters

### SAGE (`editing/sage.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 8888 | Random seed for reproducibility |
| `side` | 512 | Image resolution |
| `ddim_steps` | 50 | Number of DDIM steps |
| `model_id` | CompVis/stable-diffusion-v1-4 | Model identifier |
| `cfg_value` | 7.5 | Classifier-free guidance scale |
| `reconstruction_type` | "sage_polaris" | "sage" or "sage_polaris" |
| `use_polaris_cfg` | True | Enable POLARIS dynamic guidance |
| `apply_color_correction` | True | Enable color transfer |

### DDIM Inversion (`reconstruction/ddim.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_path` | - | Input image path (required) |
| `--prompt` | - | Image description |
| `--steps` | 100 | Number of DDIM steps |
| `--guidance_scale` | 7.5 | CFG guidance strength |
| `--res` | 512 | Image resolution |
| `--seed` | 42 | Random seed |

### DDRM Restoration (`restoration/ddrm.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_path` | - | Input image (required) |
| `--prompt` | - | Image description |
| `--steps` | 100 | Number of inference steps |
| `--res` | 512 | Image resolution |
| `--device` | cuda | "cuda" or "cpu" |

## Output Results

- **SAGE**: Results saved in `./sage_results/` with comparison images
- **Mask P2P**: Results in specified `--output_dir` (fixed + POLARIS comparison)
- **DDIM**: Comparison grid and scale evolution plots
- **DDRM**: Task-specific results with quality metrics (MSE, PSNR, SSIM, LPIPS)


## Project Structure

```
POLARIS/
├── editing/
│   ├── sage.py              # Main SAGE editing script
│   └── mask_p2p.py          # Spatially-aware prompt editing
├── reconstruction/
│   ├── ddim.py              # DDIM inversion methods comparison
│   └── score_flow.py        # Score vs Flow parameterization
├── restoration/
│   └── ddrm.py              # Multi-task restoration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```





### Missing Model Downloads
- Models are auto-downloaded from Hugging Face on first run
- Ensure stable internet connection
- Default model: `runwayml/stable-diffusion-v1-5` or `CompVis/stable-diffusion-v1-4`


## Citation

If you use POLARIS in your research, please cite:
```bibtex
@article{polaris2024,
  title={POLARIS: Projection-Orthogonal Least Squares for Robust and Adaptive Inversion},
  author={[Your Name]},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [lihaosen@stu.ouc.edu.cn].





