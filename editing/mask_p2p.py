import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List, Union, Optional
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from diffusers.models.attention_processor import Attention

# ======================================================================================
# 1. Arguments & Configuration
# ======================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="POLARIS Spatially-Aware Inversion & Generation")
    
    # Paths & Model
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Prompts
    parser.add_argument("--source_prompt", type=str, required=True, help="Description of the original image")
    parser.add_argument("--target_prompt", type=str, required=True, help="Description of the target edit")
    parser.add_argument("--edit_phrase", type=str, required=True, help="Words in target prompt to edit (determines mask)")
    
    # Hyperparameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_text", type=float, default=7.5, help="Guidance scale for edited regions")
    parser.add_argument("--mask_pow", type=float, default=10.0, help="Exponent to sharpen the attention mask")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

# ======================================================================================
# 2. Attention Control (Prompt-to-Prompt Logic)
# ======================================================================================
class AttentionStore:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self._maps = {}
        self.reset()

    def reset(self):
        self._maps.clear()

    @property
    def attention_maps(self):
        return self._maps

    def __call__(self, attn_probs: torch.Tensor, place_in_unet: str):
        # Store cross-attention maps only
        head_dim = attn_probs.shape[0] // (2 * self.batch_size)
        cond_part = attn_probs[-self.batch_size * head_dim:]
        self._maps.setdefault(place_in_unet, []).append(cond_part)

class AttentionStoreProcessor:
    def __init__(self, store_cb, place_in_unet):
        self.store_cb = store_cb
        self.place = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        b, seqlen, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seqlen, b)
        q = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        probs = attn.get_attention_scores(q, k, attention_mask)
        
        # Callback to store attention maps
        if is_cross: 
            self.store_cb(probs, self.place)
            
        out = torch.bmm(probs, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

def register_attention_control(pipe, controller: AttentionStore):
    original_processors = pipe.unet.attn_processors
    new_processors = {}
    
    for key, proc in original_processors.items():
        if "attn2" in key: # Focus on Cross-Attention
            new_processors[key] = AttentionStoreProcessor(controller, key)
        else:
            new_processors[key] = proc
            
    pipe.unet.set_attn_processor(new_processors)

    def deregister():
        pipe.unet.set_attn_processor(original_processors)

    return deregister

# ======================================================================================
# 3. Core Logic: Inversion & Generation
# ======================================================================================
@torch.no_grad()
def ddim_inversion_polaris(pipe, latents, text_embeddings, steps):
    """
    Inverts the image using POLARIS CFG (solving for omega_t).
    Returns inverted latents and the list of calculated scales.
    """
    uncond = pipe.text_encoder(pipe.tokenizer("", return_tensors="pt").input_ids.to(pipe.device))[0]
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps, device=pipe.device)
    
    z = latents.clone()
    prev_ep, prev_ec = None, None
    scales = []
    
    for t in tqdm(inv_sched.timesteps, desc="POLARIS Inversion", leave=False):
        latent_in = torch.cat([z, z], dim=0)
        embeds = torch.cat([uncond, text_embeddings], dim=0)
        
        noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=embeds).sample
        ep, ec = noise_pred.chunk(2)
        
        # Calculate POLARIS scale (omega)
        omega_t = 1.0
        if prev_ep is not None:
            d_ep = (ep - prev_ep).float()
            d_ec = (ec - prev_ec).float()
            num = (d_ep * d_ep).sum() - (d_ep * d_ec).sum()
            den = ((d_ep - d_ec) ** 2).sum()
            if den.abs() > 1e-6:
                omega_t = (num / den).clamp(0.0, 15.0).item()
        
        scales.append(omega_t)
        
        # Inverse Step
        z = inv_sched.step(ep + omega_t * (ec - ep), t, z).prev_sample
        prev_ep, prev_ec = ep.clone(), ec.clone()
        
    return z, scales

@torch.no_grad()
def ddim_inversion_fixed(pipe, latents, text_embeddings, steps, scale=1.0):
    uncond = pipe.text_encoder(pipe.tokenizer("", return_tensors="pt").input_ids.to(pipe.device))[0]
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sched.set_timesteps(steps, device=pipe.device)
    
    z = latents.clone()
    for t in tqdm(inv_sched.timesteps, desc="Fixed Inversion", leave=False):
        latent_in = torch.cat([z, z], dim=0)
        embeds = torch.cat([uncond, text_embeddings], dim=0)
        noise = pipe.unet(latent_in, t, encoder_hidden_states=embeds).sample
        ep, ec = noise.chunk(2)
        z = inv_sched.step(ep + scale * (ec - ep), t, z).prev_sample
    return z

@torch.no_grad()
def generate_spatially_aware(
    pipe, latents, text_embeddings, controller, steps, 
    target_scale, recon_scales, edit_indices, preserve_indices, mask_pow
):
    """
    Generates image using spatially aware guidance.
    Uses target_scale for edit_indices and recon_scales for preserve_indices.
    """
    uncond = pipe.text_encoder(pipe.tokenizer("", return_tensors="pt").input_ids.to(pipe.device))[0]
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    
    is_polaris_mode = isinstance(recon_scales, list)
    desc = "Gen (POLARIS)" if is_polaris_mode else "Gen (Fixed)"
    
    z = latents.clone()
    
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=steps, desc=desc, leave=False):
        controller.reset()
        
        latent_in = torch.cat([z, z], dim=0)
        embeds = torch.cat([uncond, text_embeddings], dim=0)
        
        noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=embeds).sample
        nu, nc = noise_pred.chunk(2)
        
        # Retrieve attention maps
        attn_maps = [m for lst in controller.attention_maps.values() for m in lst]
        current_recon_scale = recon_scales[i] if is_polaris_mode else recon_scales

        if not attn_maps or not edit_indices:
            # Fallback if no attention info
            guided_noise = nu + current_recon_scale * (nc - nu)
        else:
            # Create Spatial Mask from Cross-Attention
            # 1. Aggregate maps to a common resolution (usually 16x16 or 32x32)
            min_sq = min(a.shape[1] for a in attn_maps)
            tgt_res = int(np.sqrt(min_sq))
            
            avg_map = torch.zeros(1, tgt_res*tgt_res, 77, device=pipe.device, dtype=pipe.unet.dtype)
            count = 0
            for item in attn_maps:
                res = int(np.sqrt(item.shape[1]))
                item = item.mean(0, keepdim=True) # Average over heads
                if res != tgt_res:
                    # Resize to target resolution
                    item = item.permute(0, 2, 1).reshape(1, 77, res, res)
                    item = F.interpolate(item, size=(tgt_res, tgt_res), mode='bilinear', align_corners=False)
                    item = item.reshape(1, 77, -1).permute(0, 2, 1)
                avg_map += item
                count += 1
            avg_map /= max(1, count)
            
            # 2. Extract specific token attention
            edit_attn = avg_map[:, :, edit_indices].sum(-1)
            keep_attn = avg_map[:, :, preserve_indices].sum(-1)
            
            # 3. Compute Soft Mask
            mask = (edit_attn / (edit_attn + keep_attn + 1e-6)).clamp(0, 1)
            mask = mask ** mask_pow # Sharpen mask
            
            # 4. Interpolate Mask to Latent Resolution
            H, W = z.shape[-2:]
            mask_2d = mask.reshape(1, 1, tgt_res, tgt_res)
            mask_upscaled = F.interpolate(mask_2d, size=(H, W), mode='bilinear', align_corners=False)
            
            # 5. Spatially Varying Guidance Scale
            # Mask=1 (Edit) -> Target Scale; Mask=0 (Keep) -> Reconstruction Scale
            spatial_scale = mask_upscaled * target_scale + (1 - mask_upscaled) * current_recon_scale
            
            guided_noise = nu + spatial_scale * (nc - nu)
            
        z = pipe.scheduler.step(guided_noise, t, z).prev_sample
        
    return z

# ======================================================================================
# 4. Helpers
# ======================================================================================
def load_image(path, res):
    img = Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)
    return img

def img2latent(pipe, image):
    x = np.array(image, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(pipe.device, dtype=pipe.unet.dtype)
    return pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor

def latent2img(pipe, z):
    z = z / pipe.vae.config.scaling_factor
    x = pipe.vae.decode(z).sample
    x = (x / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return Image.fromarray((x[0] * 255).astype(np.uint8))

def get_token_indices(pipe, prompt, phrase):
    tokens = pipe.tokenizer.tokenize(prompt)
    target_tokens = pipe.tokenizer.tokenize(phrase)
    if not target_tokens: return []
    idxs = []
    # Simple substring matching in token list
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i + len(target_tokens)] == target_tokens:
            # +1 because 0 is usually Start-of-Sentence
            idxs.extend(list(range(i + 1, i + 1 + len(target_tokens))))
    return idxs

# ======================================================================================
# 5. Main
# ======================================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Randomness
    torch.manual_seed(args.seed)
    
    # Load Pipeline
    print(f"Loading model: {args.model_id}...")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=scheduler, 
        torch_dtype=torch.float16 if args.device=="cuda" else torch.float32
    ).to(args.device)
    
    # Optimization (Optional)
    try:
        pipe.disable_xformers_memory_efficient_attention()
    except: pass
    pipe.unet.eval()
    pipe.vae.eval()
    
    # Prepare Data
    print(f"Processing image: {args.image_path}")
    raw_img = load_image(args.image_path, args.res)
    init_z = img2latent(pipe, raw_img)
    
    src_emb = pipe.text_encoder(pipe.tokenizer(args.source_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(args.device))[0]
    tgt_emb = pipe.text_encoder(pipe.tokenizer(args.target_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(args.device))[0]
    
    # Identify Tokens
    edit_indices = get_token_indices(pipe, args.target_prompt, args.edit_phrase)
    if not edit_indices:
        print(f"Warning: Edit phrase '{args.edit_phrase}' not found in target prompt tokens.")
    
    # All other tokens are preserve tokens
    all_indices = list(range(1, len(pipe.tokenizer.tokenize(args.target_prompt)) + 1))
    preserve_indices = [i for i in all_indices if i not in edit_indices]
    
    print(f"Edit Token Indices: {edit_indices}")
    print(f"Preserve Token Indices: {len(preserve_indices)} tokens")

    # --- Method A: Fixed Scale Baseline ---
    print("\n[A] Running Fixed Scale Inversion...")
    z_inv_fixed = ddim_inversion_fixed(pipe, init_z, src_emb, args.steps, scale=1.0)
    
    print("    Generating...")
    controller = AttentionStore()
    deregister = register_attention_control(pipe, controller)
    z_fixed_res = generate_spatially_aware(
        pipe, z_inv_fixed, tgt_emb, controller, args.steps,
        target_scale=args.cfg_text, recon_scales=1.0, # Fixed recon scale
        edit_indices=edit_indices, preserve_indices=preserve_indices, mask_pow=args.mask_pow
    )
    deregister()
    img_fixed = latent2img(pipe, z_fixed_res)
    img_fixed.save(os.path.join(args.output_dir, "result_fixed.png"))

    # --- Method B: POLARIS (Ours) ---
    print("\n[B] Running POLARIS Inversion...")
    z_inv_polaris, polaris_scales = ddim_inversion_polaris(pipe, init_z, src_emb, args.steps)
    
    print("    Generating...")
    controller = AttentionStore()
    deregister = register_attention_control(pipe, controller)
    z_polaris_res = generate_spatially_aware(
        pipe, z_inv_polaris, tgt_emb, controller, args.steps,
        target_scale=args.cfg_text, recon_scales=polaris_scales, # POLARIS recon scales
        edit_indices=edit_indices, preserve_indices=preserve_indices, mask_pow=args.mask_pow
    )
    deregister()
    img_polaris = latent2img(pipe, z_polaris_res)
    img_polaris.save(os.path.join(args.output_dir, "result_polaris.png"))

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(raw_img); axs[0].set_title("Original")
    axs[1].imshow(img_fixed); axs[1].set_title("Fixed Inversion")
    axs[2].imshow(img_polaris); axs[2].set_title("POLARIS Inversion")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison.png"))
    
    print(f"\nDone. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()