import os
import argparse
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import lpips
from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="DDIM Inversion Methods Comparison")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Path or HF ID of the model")
    parser.add_argument("--image_path", type=str, default="C:\Users\24022\Desktop\POLARIS\pictures\gnochi_mirror.png", help="Path to the input image") 
    parser.add_argument("--prompt", type=str, default="A cat sitting next to a mirror.", help="Source prompt for the image")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale for reconstruction")
    parser.add_argument("--res", type=int, default=512, help="Image resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_image(path, res):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
    return Image.open(path).convert("RGB").resize((res, res))

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def get_text_emb(pipe, prompt, device):
    tok = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids
    return pipe.text_encoder(tok.to(device))[0]

@torch.no_grad()
def img2latent(pipe, image, device):
    img = np.array(image, dtype=np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)
    return pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor

@torch.no_grad()
def latent2img(pipe, latents):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    img = pipe.vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return Image.fromarray((img[0] * 255).astype(np.uint8))

def calculate_metrics(im1, im2, lpips_model, device):
    im1_np = np.array(im1).astype(np.float32)
    im2_np = np.array(im2).astype(np.float32)
    
    # MSE & PSNR
    mse = np.mean((im1_np - im2_np) ** 2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM
    im1_gray = np.array(im1.convert('L'))
    im2_gray = np.array(im2.convert('L'))
    ssim_score = ssim(im1_gray, im2_gray, data_range=255)
    
    # LPIPS
    im1_torch = torch.from_numpy(im1_np / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    im2_torch = torch.from_numpy(im2_np / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_score = lpips_model(im1_torch, im2_torch).item()
        
    return {"mse": mse, "psnr": psnr, "ssim": ssim_score, "lpips": lpips_score}


@torch.no_grad()
def ddim_inversion_fixed(pipe, latents, text_emb, steps, cfg=1.0):
    """Method 1: Standard Fixed Inversion"""
    uncond = get_text_emb(pipe, "", latents.device)
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps)
    z = latents.clone()
    
    for t in tqdm(inv_sched.timesteps, desc="Standard Inversion", leave=False):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        n_u, n_c = noise_pred.chunk(2)
        z = inv_sched.step(n_u + cfg * (n_c - n_u), t, z).prev_sample
    return z, None

@torch.no_grad()
def ddim_inversion_polaris(pipe, latents, text_emb, steps):
    """Method 2: POLARIS (formerly Dynamic Approx)"""
    uncond = get_text_emb(pipe, "", latents.device)
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps)
    z = latents.clone()
    prev_ep, prev_ec = None, None
    scales = []
    
    for t in tqdm(inv_sched.timesteps, desc="POLARIS Inversion", leave=False):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        ep, ec = noise_pred.chunk(2)
        omega_t = 1.0
        
        if prev_ep is not None:
            d_ep = (ep - prev_ep).float()
            d_ec = (ec - prev_ec).float()
            num = (d_ep * d_ep).sum() - (d_ep * d_ec).sum()
            den = ((d_ep - d_ec) ** 2).sum()
            if den.abs() > 1e-6:
                omega_t = (num / den).clamp(0.0, 15.0).item()
                
        scales.append(omega_t)
        z = inv_sched.step(ep + omega_t * (ec - ep), t, z).prev_sample
        prev_ep, prev_ec = ep.clone(), ec.clone()
        
    return z, scales

@torch.no_grad()
def ddim_inversion_dynamic_exact(pipe, latents, text_emb, steps):
    """Method 3: Dynamic Inversion (Exact)"""
    uncond = get_text_emb(pipe, "", latents.device)
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps)
    z = latents.clone()
    omega_t = 1.0
    prev_ep, prev_ec = None, None
    scales = []
    
    timesteps = inv_sched.timesteps
    for i in tqdm(range(len(timesteps)), desc="Dynamic (Exact)", leave=False):
        t = timesteps[i]
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        ep_t, ec_t = noise_pred.chunk(2)
        
        if prev_ep is not None:
            d_ep = (ep_t - prev_ep).float()
            d_ec = (ec_t - prev_ec).float()
            a = (1 - omega_t) * d_ep + omega_t * d_ec
            b = prev_ec - prev_ep
            a_dot_b = (a * b).sum()
            b_norm_sq = (b * b).sum()
            epsilon = 1e-6
            delta_omega = -a_dot_b / (b_norm_sq + epsilon)
            omega_t_minus_1 = omega_t - delta_omega.item()
            omega_t = np.clip(omega_t_minus_1, 0.0, 15.0)

        scales.append(omega_t)
        z = inv_sched.step(ep_t + omega_t * (ec_t - ep_t), t, z).prev_sample
        prev_ep, prev_ec = ep_t.clone(), ec_t.clone()
        
    return z, scales

@torch.no_grad()
def ddim_reconstruct(pipe, latents, text_emb, steps, cfg, scales=None, method_name="Standard"):
    uncond = get_text_emb(pipe, "", latents.device)
    pipe.scheduler.set_timesteps(steps)
    z = latents.clone()
    
    guidance_history = scales if scales is not None else [cfg] * steps
    
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"{method_name} Recon", leave=False)):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        n_u, n_c = noise_pred.chunk(2)
        # Handle index carefully
        s_index = min(i, len(guidance_history) - 1)
        s = guidance_history[s_index]
        z = pipe.scheduler.step(n_u + s * (n_c - n_u), t, z).prev_sample
        
    return z

if __name__ == '__main__':
    args = parse_args()
    
    # 1. Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Device: {device} | Output: {args.output_dir} ---")

    # 2. Load Models
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, scheduler=scheduler,
                                                   torch_dtype=torch.float16).to(device)
    
    # Memory optimization
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()

    loss_fn = lpips.LPIPS(net='alex').to(device)

    # 3. Load Data
    try:
        original_img = load_image(args.image_path, args.res)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure '{args.image_path}' exists or specify it with --image_path")
        exit(1)
        
    src_emb = get_text_emb(pipe, args.prompt, device)
    init_z = img2latent(pipe, original_img, device)

    results = {}
    methods = [
        ("Standard", ddim_inversion_fixed, None),
        ("POLARIS", ddim_inversion_polaris, None), # Renamed here
        ("Dynamic (Exact)", ddim_inversion_dynamic_exact, None)
    ]
    
    scale_records = {}

    # 4. Run Methods
    print("\n--- Running Inversions ---")
    for name, func, extra_args in methods:
        # Inversion
        if name == "Standard":
            z_inv, _ = func(pipe, init_z.clone(), src_emb, args.steps, cfg=1.0)
            recon_scales = None
        else:
            z_inv, scales = func(pipe, init_z.clone(), src_emb, args.steps)
            recon_scales = scales
            scale_records[name] = scales
            
        # Reconstruction
        recon_z = ddim_reconstruct(pipe, z_inv, src_emb, args.steps, args.guidance_scale, 
                                   scales=recon_scales, method_name=name)
        img_recon = latent2img(pipe, recon_z)
        
        # Metrics
        met = calculate_metrics(original_img, img_recon, loss_fn, device)
        results[name] = {"image": img_recon, "metrics": met}

    # 5. Reporting
    print("\n==================== RESULTS ====================")
    for name, data in results.items():
        m = data["metrics"]
        print(f"[{name}]")
        print(f"  MSE: {m['mse']:.2f} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f} | LPIPS: {m['lpips']:.4f}")

    # 6. Visualization
    print("\n--- Saving Visualizations ---")
    
    # Grid Plot
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    for i, (name, data) in enumerate(results.items(), 1):
        m = data["metrics"]
        axes[i].imshow(data["image"])
        axes[i].set_title(f"{name}\nPSNR:{m['psnr']:.2f} / LPIPS:{m['lpips']:.4f}")
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "methods_comparison.png"), dpi=200)
    plt.close(fig)

    # Scale Evolution Plot
    if scale_records:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, scales in scale_records.items():
            # Update plot styling logic
            is_polaris = 'POLARIS' in name
            ax.plot(scales, 
                    marker='o' if is_polaris else '^', 
                    linestyle='-' if is_polaris else '--', 
                    label=name, alpha=0.8)
        
        ax.set_title("Evolution of Guidance Scale during Inversion")
        ax.set_xlabel("Inversion Step")
        ax.set_ylabel("Guidance Scale (ω)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "scales_evolution.png"), dpi=200)
        plt.close(fig)

    print(f"Done. Results saved to {args.output_dir}")