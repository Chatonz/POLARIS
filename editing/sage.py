import os
import torch
import torch.nn.functional as nnf
from torchvision.transforms.functional import pil_to_tensor
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
import numpy as np
from tqdm import tqdm  
import PIL.Image
import matplotlib.pyplot as plt
import json
import re

try:
    from guided_editing import EditableAttnProcessor, AttnStore, AttnCost, AttnReplace, AttnMask, AttnDummy, \
                               compose_editing, get_prompt_embedding, get_index
    from visualization import torch_to_pil
except ImportError:
    print("Error: Please ensure guided_editing.py and visualization.py are in the current directory.")
    exit()


# --- Global Configuration ---
is_cuda = torch.cuda.is_available()
if is_cuda:
    DTYPE = torch.float16
    DEVICE = torch.device("cuda:0")
    DEFAULT_LOSS_SCALE = 500
else:
    DTYPE = torch.float32
    DEVICE = 'cpu'
    DEFAULT_LOSS_SCALE = 300

loss_dict = dict(mae=nnf.l1_loss, mse=nnf.mse_loss)

pipe = None
last_model_id = None

# --- Model Loading and Helper Functions ---

def color_transfer(source_img: PIL.Image.Image, target_img: PIL.Image.Image) -> PIL.Image.Image:
    """Transfer color from source image to target image using YCbCr space."""
    if source_img.size != target_img.size:
        source_img = source_img.resize(target_img.size, PIL.Image.LANCZOS)
    source_ycbcr = source_img.convert('YCbCr')
    target_ycbcr = target_img.convert('YCbCr')
    source_y, source_cb, source_cr = source_ycbcr.split()
    target_y, _, _ = target_ycbcr.split()
    merged_ycbcr = PIL.Image.merge('YCbCr', [target_y, source_cb, source_cr])
    return merged_ycbcr.convert('RGB')

def load_model(model_id, use_ft_vae=True):
    """Load Stable Diffusion pipeline and optionally replace VAE."""
    global pipe, last_model_id
    current_config = (model_id, use_ft_vae)
    if current_config != last_model_id:
        print(f"Loading model: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        if use_ft_vae:
            print("Loading and replacing with fine-tuned VAE (ft-mse)...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", 
                torch_dtype=DTYPE
            ).to(DEVICE)
            pipe.vae = vae
        last_model_id = current_config

def load_all(model_id):
    """Load model with default settings."""
    load_model(model_id, use_ft_vae=True)

def clean_str(s):
    """Clean whitespace in string."""
    return " ".join(s.split())

# --- Core Execution Function ---
def run(
    # Input
    input_image: PIL.Image.Image,
    prompt_str,
    edited_prompt_str,
    
    # System parameters
    seed = 8888,
    side = 512,
    ddim_steps=50,
    model_id = "CompVis/stable-diffusion-v1-4",
    
    # I2I parameters
    self_latent_guidance_scale = 250.,
    cfg_value = 7.5,
    loss_scale = None,
    max_steps = 40,
    use_monotonical_scale = True,
    loss_type = "mae",
    
    self_layer=1,
    cross_layer=1,
    reconstruction_type="sage_polaris",
    
    # Cross-attention control parameters
    replace_words: tuple = None,
    blend_words: tuple = None,
    cross_replace_steps: float = 0.8,
    sag_mask_min: float = 0.1,
    
    # POLARIS-CFG Control
    use_polaris_cfg: bool = True,
    
    # ***** New Parameter *****
    use_standard_generation_guidance: bool = False,
    
    # Control flags
    low_memory: bool = False,
    disable_tqdm=False,
    
    # Color Correction
    apply_color_correction: bool = True
):
    # --- Parse and Set Parameters ---
    use_polaris_inversion = reconstruction_type == "sage_polaris"
    loss_scale = DEFAULT_LOSS_SCALE if loss_scale is None or loss_scale == 0 else loss_scale
    side, ddim_steps, seed = int(side), int(ddim_steps), int(seed)
    prompt_str, edited_prompt_str = clean_str(prompt_str), clean_str(edited_prompt_str)
    
    forward_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    forward_scheduler.set_timesteps(ddim_steps, device=DEVICE)
    inverse_scheduler.set_timesteps(ddim_steps, device=DEVICE)
    generation_timesteps = forward_scheduler.timesteps.tolist()
    inversion_timesteps = inverse_scheduler.timesteps.tolist()
        
    torch.manual_seed(seed)
    if is_cuda: torch.cuda.manual_seed_all(seed)
    
    original_image_for_color = input_image.convert("RGB").resize((side, side), PIL.Image.LANCZOS)
    img_tensor = pil_to_tensor(original_image_for_color).unsqueeze(0).to(DEVICE, DTYPE) / 127.5 - 1.0
    
    # --- Set Custom Attention Processors ---
    self_side = side // (8 * (2**self_layer))
    cross_side = side // (8 * (2**cross_layer))
    attn_store = AttnStore(DEVICE, DTYPE, self_side=self_side, cross_side=cross_side)
    
    if use_polaris_cfg:
        if replace_words:
            try:
                source_id = get_index(prompt_str, replace_words[0], pipe.tokenizer)
                edited_id = get_index(edited_prompt_str, replace_words[1], pipe.tokenizer)
                max_t_replace = generation_timesteps[int(len(generation_timesteps) * cross_replace_steps)]
                attn_replace = AttnReplace(DEVICE, DTYPE, source_id, edited_id, cross_side=cross_side, max_t=max_t_replace)
            except ValueError:
                print(f"Warning: Cannot find replace_words '{replace_words}' in prompt. Replacement disabled.")
                attn_replace = AttnDummy()
        else:
            attn_replace = AttnDummy()

        if blend_words:
            attn_mask = AttnMask(pipe.tokenizer, [prompt_str, edited_prompt_str], blend_words, DEVICE, DTYPE, cross_side=cross_side)
        else:
            attn_mask = AttnDummy()
    else:
        attn_replace = AttnDummy()
        attn_mask = AttnDummy()

    editing = compose_editing(attn_replace, attn_mask)
    guidance_processor = EditableAttnProcessor(AttnCost(store=attn_store, editing=editing))
    cond_processor = EditableAttnProcessor(AttnCost(store=AttnStore(DEVICE, DTYPE, self_side=self_side, cross_side=cross_side), editing=editing))
    default_processor = AttnProcessor()
        
    with torch.no_grad():
        latents_t0 = pipe.vae.encode(img_tensor).latent_dist.mean * pipe.vae.config.scaling_factor
        _, edited_prompt_embeds = get_prompt_embedding(pipe, prompt_str, edited_prompt_str)
        source_embeddings, edited_embeddings = edited_prompt_embeds.chunk(2)
        uncond_embeddings = get_prompt_embedding(pipe, "", "")[1].chunk(2)[0]
        
        # --- Inversion Phase ---
        latents = latents_t0.clone()
        polaris_weights = {}
        inversion_source_preds = {}
        
        inversion_loop = tqdm(inversion_timesteps, desc="Inversion", leave=False, disable=disable_tqdm)
        if use_polaris_inversion:
            eps_phi_prev, eps_c_prev = None, None
            for i, t in enumerate(inversion_loop):
                t_tensor = torch.tensor([t], device=DEVICE)
                pipe.unet.set_attn_processor(default_processor)
                eps_phi_t = pipe.unet(latents, t_tensor, encoder_hidden_states=uncond_embeddings).sample
                
                attn_store.set_t(t)
                attn_replace.set_t(t)
                pipe.unet.set_attn_processor(guidance_processor)
                eps_c_t = pipe.unet(latents, t_tensor, encoder_hidden_states=source_embeddings).sample
                
                if i == 0: wt = 1.0
                else:
                    delta_eps_phi, delta_eps_c = eps_phi_t - eps_phi_prev, eps_c_t - eps_c_prev
                    numerator = (delta_eps_phi * delta_eps_phi).sum() - (delta_eps_phi * delta_eps_c).sum()
                    denominator = ((delta_eps_phi - delta_eps_c)**2).sum()
                    wt = (numerator / (denominator + 1e-6)).clamp(0, 15).item()
                
                polaris_weights[t] = wt
                inversion_source_preds[t] = eps_c_t.detach().clone()
                guided_noise = (1 - wt) * eps_phi_t + wt * eps_c_t
                latents = inverse_scheduler.step(guided_noise, t, latents).prev_sample
                eps_phi_prev, eps_c_prev = eps_phi_t, eps_c_t
        else:
            pipe.unet.set_attn_processor(guidance_processor)
            for t in inversion_loop:
                attn_store.set_t(t)
                t_tensor = torch.tensor([t], device=DEVICE)
                noise_pred = pipe.unet(latents, t_tensor, encoder_hidden_states=source_embeddings).sample
                latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample

        init_latents = latents.detach().clone()
        attn_store.fix()
        attn_replace.fix()
        
    # --- Generation Phase (Guided Diffusion) ---
    attn_mask.reset()
    attn_mask.enable()
    attn_replace.enable()
    
    generation_loop = tqdm(generation_timesteps, desc="Generation", leave=False, disable=disable_tqdm)
    for i, t in enumerate(generation_loop):
        t_tensor = torch.tensor([t], device=DEVICE)
        attn_store.set_t(t)
        attn_replace.set_t(t)
        
        latents_grad = latents.detach().clone()
        latents_grad.requires_grad = True

        # Determine guidance baseline based on configuration
        if use_polaris_cfg and not use_standard_generation_guidance:
            # True POLARIS guidance: use empty prompt as baseline
            guidance_prompt_embeddings = uncond_embeddings
        else:
            # Standard SAGE or hybrid guidance: use original prompt as baseline
            guidance_prompt_embeddings = source_embeddings

        attn_replace.disable()
        pipe.unet.set_attn_processor(guidance_processor)
        attn_mask.set_prompt_idx(0)
        uncond_pred = pipe.unet(latents_grad, t_tensor, encoder_hidden_states=guidance_prompt_embeddings).sample
        
        grad = torch.zeros_like(latents)
        attn_pair = attn_store.get_t(t)
        if attn_pair and attn_pair.reference and attn_pair.current:
            loss = loss_dict[loss_type](torch.cat(attn_pair.current.self), torch.cat(attn_pair.reference.self)) * loss_scale
            grad = torch.autograd.grad(loss, latents_grad)[0]
        
        attn_replace.enable()
        pipe.unet.set_attn_processor(cond_processor)
        attn_mask.set_prompt_idx(1)
        with torch.no_grad():
            cond_pred = pipe.unet(latents, t_tensor, encoder_hidden_states=edited_embeddings).sample
        
        noise_pred = uncond_pred.detach() + cfg_value * (cond_pred - uncond_pred.detach())
        
        if low_memory:
            attn_store.clear_t(t)
            
        with torch.no_grad():
            monotonical_scale = 1 - (i / ddim_steps) if use_monotonical_scale and i < max_steps else 0
            latents = forward_scheduler.step(noise_pred, t, latents).prev_sample
            grad_guidance = self_latent_guidance_scale * grad * monotonical_scale
            if use_polaris_cfg and blend_words and sag_mask_min < 1.0:
                preservation_mask = (1 - attn_mask.mask_like(latents)).clamp(sag_mask_min, 1.0)
                grad_guidance *= preservation_mask
            latents -= grad_guidance
            
    pipe.unet.set_attn_processor(guidance_processor)

    with torch.no_grad():
        edited_image_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        edited_image_pil = torch_to_pil(edited_image_tensor)
    
    if apply_color_correction:
        edited_image_pil = color_transfer(original_image_for_color, edited_image_pil)
    
    return edited_image_pil

# --- Main Interactive Editing Loop ---
def main():
    """Interactive image editing with SAGE and POLARIS methods."""
    # --- Configuration ---
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    BASE_OUTPUT_DIR = "./sage_results"
    SEED = 42

    # --- Step 1: Setup Output Directory ---
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # --- Step 2: Load Model ---
    print("Loading model...")
    load_all(MODEL_ID)
    print("✓ Model loaded successfully!\n")

    # --- Step 3: Get Input ---
    print("="*50)
    print("SAGE Image Editing Tool")
    print("="*50)
    
    # Get image path
    image_path = input("\nEnter image path: ").strip()
    if not os.path.exists(image_path):
        print(f"Error: Image not found '{image_path}'.")
        return
    
    try:
        input_image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: Cannot open image. {e}")
        return
    
    # Get prompts
    source_prompt = input("Enter source prompt: ").strip()
    if not source_prompt:
        print("Error: Prompt cannot be empty.")
        return
    
    target_prompt = input("Enter target prompt: ").strip()
    if not target_prompt:
        print("Error: Prompt cannot be empty.")
        return
    
    # --- Step 4: Run Both Methods ---
    print(f"\n{'='*50}")
    print("Processing image...")
    print(f"{'='*50}")
    
    # Standard SAGE
    print("\n[1/2] Running Standard SAGE...")
    sage_result = run(
        input_image=input_image,
        prompt_str=source_prompt,
        edited_prompt_str=target_prompt,
        model_id=MODEL_ID,
        seed=SEED,
        reconstruction_type="sage",
        use_polaris_cfg=False,
        apply_color_correction=False,
        disable_tqdm=False
    )

    # POLARIS Inversion + Standard Guidance
    print("\n[2/2] Running POLARIS Inversion + Standard Guidance...")
    advanced_result = run(
        input_image=input_image,
        prompt_str=source_prompt,
        edited_prompt_str=target_prompt,
        model_id=MODEL_ID,
        seed=SEED,
        reconstruction_type="sage_polaris",
        use_polaris_cfg=True,
        use_standard_generation_guidance=True,
        apply_color_correction=False,
        disable_tqdm=False
    )

    # --- Step 5: Save Results ---
    timestamp = str(int(torch.seed() % 1000000))
    
    sage_path = os.path.join(BASE_OUTPUT_DIR, f"sage_result_{timestamp}.png")
    advanced_path = os.path.join(BASE_OUTPUT_DIR, f"polaris_result_{timestamp}.png")
    
    sage_result.save(sage_path)
    advanced_result.save(advanced_path)

    # --- Step 6: Visualize Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(input_image.resize((512, 512)))
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(sage_result)
    axes[1].set_title("Standard SAGE", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(advanced_result)
    axes[2].set_title("POLARIS Inversion + Standard Guidance", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(BASE_OUTPUT_DIR, f"comparison_{timestamp}.png")
    plt.savefig(comparison_path)
    plt.close(fig)

    print(f"\n✓ Processing Complete!")
    print(f"  - Standard SAGE result: {sage_path}")
    print(f"  - POLARIS result: {advanced_path}")
    print(f"  - Comparison image: {comparison_path}")
    print(f"\nAll results saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()