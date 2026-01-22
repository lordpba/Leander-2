#!/usr/bin/env python3
"""
Leander 2 - Asset Generator
Generates pixel art assets using Stable Diffusion locally (RTX 3060 12GB)
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import argparse
import os

# Modelli consigliati per pixel art (con safetensors per sicurezza)
MODELS = {
    "pixel-art": "nerijs/pixel-art-xl",                    # SDXL fine-tuned per pixel art
    "pixel-diffusion": "CompVis/stable-diffusion-v1-4",    # SD 1.4 (sempre pubblico)
    "retro": "dreamlike-art/dreamlike-diffusion-1.0",      # Ottimo stile artistico
}

# Preset per diversi tipi di asset
PRESETS = {
    "background": {
        "width": 320,
        "height": 256,
        "prompt_prefix": "pixel art, 16-bit Amiga game background, dark fantasy, Psygnosis style, ",
        "prompt_suffix": ", deep blue and purple shadows, moody atmosphere, medieval fantasy, detailed dithering, low saturation",
        "negative": "blurry, 3d render, photo, realistic, modern, text, watermark, bright colors, cartoon, anime, high saturation"
    },
    "sprite": {
        "width": 64,
        "height": 64, 
        "prompt_prefix": "pixel art sprite, 16-bit character, transparent background, ",
        "prompt_suffix": ", game asset, clean pixels, no antialiasing",
        "negative": "blurry, 3d, realistic, background, text"
    },
    "tile": {
        "width": 32,
        "height": 32,
        "prompt_prefix": "pixel art tile, seamless texture, game asset, ",
        "prompt_suffix": ", tileable, retro game style",
        "negative": "blurry, 3d, text, border"
    },
    "leander": {
        "width": 320,
        "height": 256,
        "prompt_prefix": "16-bit pixel art, dark fantasy platformer background, Amiga AGA graphics, ",
        "prompt_suffix": ", deep blue purple color palette, gothic atmosphere, moody lighting, detailed pixel dithering, stone textures, mystical, medieval castle dungeon style",
        "negative": "bright, colorful, cartoon, anime, 3d, realistic, photo, modern, text, watermark, blurry, high saturation, green grass"
    }
}

def load_pipeline(model_name="pixel-diffusion", use_cpu=False):
    """Carica il modello Stable Diffusion"""
    
    model_id = MODELS.get(model_name, model_name)
    print(f"📥 Caricamento modello: {model_id}")
    
    device = "cpu" if use_cpu else "cuda"
    dtype = torch.float32 if use_cpu else torch.float16
    
    # Per modelli SDXL
    if "xl" in model_id.lower():
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if not use_cpu else None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True
        )
    
    # Ottimizzazioni per VRAM
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()  # Se hai xformers
    
    print(f"✅ Modello caricato su {device}")
    return pipe

def load_img2img_pipeline(model_name="pixel-diffusion", use_cpu=False):
    """Carica il modello per img2img"""
    
    model_id = MODELS.get(model_name, model_name)
    print(f"📥 Caricamento modello img2img: {model_id}")
    
    device = "cpu" if use_cpu else "cuda"
    dtype = torch.float32 if use_cpu else torch.float16
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=True
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    print(f"✅ Modello img2img caricato su {device}")
    return pipe

def generate_variation(pipe, reference_image_path, prompt, preset_name="leander", 
                       output_path="output.png", seed=None, steps=30, 
                       guidance=7.5, strength=0.6):
    """Genera una variazione basata su un'immagine di riferimento"""
    
    preset = PRESETS.get(preset_name, PRESETS["leander"])
    
    # Carica immagine di riferimento
    ref_image = Image.open(reference_image_path).convert("RGB")
    ref_image = ref_image.resize((preset["width"] * 2, preset["height"] * 2))
    
    full_prompt = preset["prompt_prefix"] + prompt + preset["prompt_suffix"]
    
    print(f"🎨 Generazione variazione da: {reference_image_path}")
    print(f"   Prompt: {prompt}")
    print(f"   Strength: {strength} (più alto = più diverso dall'originale)")
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"   Seed: {seed}")
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=preset["negative"],
        image=ref_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    # Ridimensiona e quantizza
    image = image.resize(
        (preset["width"], preset["height"]), 
        Image.Resampling.NEAREST
    )
    image = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    image = image.convert("RGB")
    
    image.save(output_path)
    print(f"💾 Salvato: {output_path}")
    
    return image

def generate_asset(pipe, prompt, preset_name="background", output_path="output.png", 
                   seed=None, steps=25, guidance=7.5, upscale=True):
    """Genera un singolo asset"""
    
    preset = PRESETS.get(preset_name, PRESETS["background"])
    
    full_prompt = preset["prompt_prefix"] + prompt + preset["prompt_suffix"]
    
    print(f"🎨 Generazione: {prompt}")
    print(f"   Preset: {preset_name} ({preset['width']}x{preset['height']})")
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        print(f"   Seed: {seed}")
    
    # Genera a risoluzione più alta se upscale
    gen_width = preset["width"] * 2 if upscale else preset["width"]
    gen_height = preset["height"] * 2 if upscale else preset["height"]
    
    # Arrotonda a multipli di 8 (richiesto da SD)
    gen_width = (gen_width // 8) * 8
    gen_height = (gen_height // 8) * 8
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=preset["negative"],
        width=gen_width,
        height=gen_height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    # Ridimensiona alla risoluzione target con NEAREST per mantenere pixel nitidi
    if upscale:
        image = image.resize(
            (preset["width"], preset["height"]), 
            Image.Resampling.NEAREST
        )
    
    # Riduci a 32 colori (compatibile Amiga AGA)
    image = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    image = image.convert("RGB")
    
    image.save(output_path)
    print(f"💾 Salvato: {output_path}")
    
    return image

def batch_generate(pipe, prompts_file, output_dir, preset_name="background"):
    """Genera asset da un file di prompt"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output_dir, f"asset_{i:03d}.png")
        generate_asset(pipe, prompt, preset_name, output_path)

def interactive_mode(pipe):
    """Modalità interattiva per sperimentare"""
    
    print("\n🎮 MODALITÀ INTERATTIVA")
    print("=" * 50)
    print("Comandi: quit, preset <nome>, seed <numero>")
    print("Preset disponibili:", list(PRESETS.keys()))
    print("=" * 50)
    
    current_preset = "background"
    current_seed = None
    
    while True:
        try:
            user_input = input(f"\n[{current_preset}] Prompt: ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower().startswith("preset "):
            current_preset = user_input.split()[1]
            print(f"Preset cambiato: {current_preset}")
            continue
        if user_input.lower().startswith("seed "):
            current_seed = int(user_input.split()[1])
            print(f"Seed impostato: {current_seed}")
            continue
        
        output_path = f"assets/generated_{current_preset}.png"
        generate_asset(pipe, user_input, current_preset, output_path, seed=current_seed)

def main():
    parser = argparse.ArgumentParser(description="Genera asset pixel art per Leander 2")
    parser.add_argument("--model", default="pixel-diffusion", 
                        choices=list(MODELS.keys()) + ["custom"],
                        help="Modello da usare")
    parser.add_argument("--custom-model", help="Path/ID modello custom")
    parser.add_argument("--prompt", "-p", help="Prompt per generazione singola")
    parser.add_argument("--preset", default="background",
                        choices=list(PRESETS.keys()),
                        help="Tipo di asset")
    parser.add_argument("--output", "-o", default="assets/generated.png",
                        help="File di output")
    parser.add_argument("--batch", help="File con lista di prompt")
    parser.add_argument("--batch-dir", default="assets/generated",
                        help="Directory per output batch")
    parser.add_argument("--seed", type=int, help="Seed per riproducibilità")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--cpu", action="store_true", help="Usa CPU invece di GPU")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Modalità interattiva")
    # Img2Img arguments
    parser.add_argument("--reference", "-r", help="Immagine di riferimento per img2img")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Quanto modificare l'originale (0.0-1.0, default 0.5)")
    
    args = parser.parse_args()
    
    # Verifica CUDA
    if not args.cpu:
        if torch.cuda.is_available():
            print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA non disponibile, uso CPU")
            args.cpu = True
    
    model = args.custom_model if args.model == "custom" else args.model
    
    if args.interactive:
        pipe = load_pipeline(model, use_cpu=args.cpu)
        interactive_mode(pipe)
    elif args.reference and args.prompt:
        # Modalità img2img
        pipe = load_img2img_pipeline(model, use_cpu=args.cpu)
        generate_variation(pipe, args.reference, args.prompt, args.preset,
                          args.output, args.seed, args.steps, args.guidance, args.strength)
    elif args.batch:
        pipe = load_pipeline(model, use_cpu=args.cpu)
        batch_generate(pipe, args.batch, args.batch_dir, args.preset)
    elif args.prompt:
        pipe = load_pipeline(model, use_cpu=args.cpu)
        generate_asset(pipe, args.prompt, args.preset, args.output, 
                      args.seed, args.steps, args.guidance)
    else:
        # Demo
        print("\n📋 Esempio di utilizzo:")
        print("  python generate_assets.py -p 'dark fantasy forest' --preset leander")
        print("  python generate_assets.py -r assets/original/leander_03.png -p 'ice cave' --strength 0.6")
        print("  python generate_assets.py -i  # Modalità interattiva")
        print("\n🎮 Genero un asset demo...")
        pipe = load_pipeline(model, use_cpu=args.cpu)
        generate_asset(pipe, "dark medieval castle, moonlight, gothic", 
                      "leander", "assets/generated/demo_bg.png", seed=42)

if __name__ == "__main__":
    main()
