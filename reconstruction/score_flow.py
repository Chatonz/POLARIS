import torch
import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description="POLARIS Score vs Flow Comparison")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Image description")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./output_score_flow")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = args.device

print(f"Loading model from {args.model_path}...")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(args.model_path, scheduler=scheduler,
                                               torch_dtype=torch.float16).to(device)
pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

def get_sigmas(timesteps):
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    sigmas = []
    alphas = []
    for t in timesteps:
        idx = min(t.item(), 999)
        alpha_prod = alphas_cumprod[idx]
        alphas.append(alpha_prod)
        sigmas.append((1 - alpha_prod) ** 0.5)
    return torch.tensor(alphas).to(device), torch.tensor(sigmas).to(device)

@torch.no_grad()
def get_text_emb(prompt):
    tok = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids
    return pipe.text_encoder(tok.to(device))[0]

@torch.no_grad()
def img2latent(image):
    img = np.array(image, dtype=np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)
    return pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor

@torch.no_grad()
def latent2img(latents):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    img = pipe.vae.decode(latents).sample
    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return Image.fromarray((img[0] * 255).astype(np.uint8))

@torch.no_grad()
def ddim_inversion_score_polaris(latents, text_emb, steps):
    uncond = get_text_emb("")
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps)
    z = latents.clone()
    
    prev_s_u, prev_s_c = None, None
    scales = []
    alphas_list, sigmas_list = get_sigmas(inv_sched.timesteps)

    for i, t in enumerate(inv_sched.timesteps):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        eps_u, eps_c = noise_pred.chunk(2)
        
        sigma_t = sigmas_list[i]
        if sigma_t < 1e-3:
            s_u = -eps_u / 1e-3
            s_c = -eps_c / 1e-3
        else:
            s_u = -eps_u / sigma_t
            s_c = -eps_c / sigma_t
        
        omega_t = 1.0
        if prev_s_u is not None:
            d_su = (s_u - prev_s_u).float()
            d_sc = (s_c - prev_s_c).float()
            num = (d_su * d_su).sum() - (d_su * d_sc).sum()
            den = ((d_su - d_sc) ** 2).sum()
            if den.abs() > 1e-6:
                omega_t = (num / den).clamp(0.0, 15.0).item()
        
        scales.append(omega_t)
        z = inv_sched.step(eps_u + omega_t * (eps_c - eps_u), t, z).prev_sample
        prev_s_u, prev_s_c = s_u.clone(), s_c.clone()
        
    return z, scales

@torch.no_grad()
def ddim_inversion_flow_polaris(latents, text_emb, steps):
    uncond = get_text_emb("")
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps)
    z = latents.clone()
    
    prev_v_u, prev_v_c = None, None
    scales = []
    alphas_list, sigmas_list = get_sigmas(inv_sched.timesteps)

    for i, t in enumerate(inv_sched.timesteps):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        eps_u, eps_c = noise_pred.chunk(2)
        
        alpha_sqrt = alphas_list[i] ** 0.5
        sigma = sigmas_list[i]
        v_u = alpha_sqrt * eps_u - sigma * z
        v_c = alpha_sqrt * eps_c - sigma * z
        
        omega_t = 1.0
        if prev_v_u is not None:
            d_vu = (v_u - prev_v_u).float()
            d_vc = (v_c - prev_v_c).float()
            num = (d_vu * d_vu).sum() - (d_vu * d_vc).sum()
            den = ((d_vu - d_vc) ** 2).sum()
            if den.abs() > 1e-6:
                omega_t = (num / den).clamp(0.0, 15.0).item()
        
        scales.append(omega_t)
        z = inv_sched.step(eps_u + omega_t * (eps_c - eps_u), t, z).prev_sample
        prev_v_u, prev_v_c = v_u.clone(), v_c.clone()
        
    return z, scales

@torch.no_grad()
def ddim_reconstruct(latents, text_emb, steps, scales=None):
    uncond = get_text_emb("")
    pipe.scheduler.set_timesteps(steps)
    z = latents.clone()

    if scales is None:
        guidance_history = [args.guidance_scale] * steps
    else:
        guidance_history = scales

    for i, t in enumerate(pipe.scheduler.timesteps):
        noise_pred = pipe.unet(torch.cat([z, z]), t,
                               encoder_hidden_states=torch.cat([uncond, text_emb])).sample
        n_u, n_c = noise_pred.chunk(2)

        if scales is not None:
            s = guidance_history[len(guidance_history) - 1 - i]
        else:
            s = args.guidance_scale

        z = pipe.scheduler.step(n_u + s * (n_c - n_u), t, z).prev_sample
    return z

def calculate_metrics(im1, im2, lpips_model):
    im1_np = np.array(im1).astype(np.float32)
    im2_np = np.array(im2).astype(np.float32)
    
    mse = np.mean((im1_np - im2_np) ** 2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    
    im1_gray = np.array(im1.convert('L'))
    im2_gray = np.array(im2.convert('L'))
    ssim_score = ssim(im1_gray, im2_gray, data_range=255)
    
    im1_torch = torch.from_numpy(im1_np / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    im2_torch = torch.from_numpy(im2_np / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_score = lpips_model(im1_torch, im2_torch).item()
        
    return {"mse": mse, "psnr": psnr, "ssim": ssim_score, "lpips": lpips_score}

if __name__ == '__main__':
    print(f"Processing image: {args.image_path}")
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        exit(1)
    
    original_img = Image.open(args.image_path).convert("RGB").resize((args.resolution, args.resolution))
    
    src_emb = get_text_emb(args.prompt)
    init_z = img2latent(original_img)

    print("\n[1] Running POLARIS (Score-based)...")
    z_score, scales_score = ddim_inversion_score_polaris(init_z.clone(), src_emb, args.steps)
    img_score = latent2img(ddim_reconstruct(z_score, src_emb, args.steps, scales=scales_score))
    m_score = calculate_metrics(original_img, img_score, loss_fn_alex)
    
    print("[2] Running POLARIS (Flow-based)...")
    z_flow, scales_flow = ddim_inversion_flow_polaris(init_z.clone(), src_emb, args.steps)
    img_flow = latent2img(ddim_reconstruct(z_flow, src_emb, args.steps, scales=scales_flow))
    m_flow = calculate_metrics(original_img, img_flow, loss_fn_alex)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nPOLARIS (Score-based):")
    print(f"  MSE   (↓): {m_score['mse']:.4f}")
    print(f"  PSNR  (↑): {m_score['psnr']:.4f}")
    print(f"  SSIM  (↑): {m_score['ssim']:.4f}")
    print(f"  LPIPS (↓): {m_score['lpips']:.4f}")
    
    print(f"\nPOLARIS (Flow-based):")
    print(f"  MSE   (↓): {m_flow['mse']:.4f}")
    print(f"  PSNR  (↑): {m_flow['psnr']:.4f}")
    print(f"  SSIM  (↑): {m_flow['ssim']:.4f}")
    print(f"  LPIPS (↓): {m_flow['lpips']:.4f}")
    print("="*60)

    img_score.save(os.path.join(args.output_dir, "result_score_based.png"))
    img_flow.save(os.path.join(args.output_dir, "result_flow_based.png"))
    original_img.save(os.path.join(args.output_dir, "original.png"))
    
    print(f"\nResults saved to {args.output_dir}")
