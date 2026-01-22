#!/usr/bin/env python3
"""
Leander 2 - Asset Generator v2
Genera asset pixel art di alta qualità usando SDXL + LoRA
Supporta generazione layer parallax (sky, far, near) e sprites
"""

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
import argparse
import os
import json

# === CONFIGURAZIONE ===

# Modello SDXL base + LoRA pixel art
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
PIXEL_ART_LORA = "nerijs/pixel-art-xl"
PIXEL_ART_LORA_FILE = "pixel-art-xl.safetensors"

# Dimensioni layer per Amiga (320x256 totale, ma layer possono essere più larghi per scroll)
LAYER_CONFIGS = {
    "sky": {
        "width": 320,      # Cielo fisso o scroll lento
        "height": 80,      # Parte superiore
        "description": "Sky layer - gradient, clouds, stars, moon"
    },
    "far": {
        "width": 640,      # 2x per scroll parallax
        "height": 128,
        "description": "Far background - distant mountains, buildings, horizon"
    },
    "near": {
        "width": 640,      # 2x per scroll
        "height": 256,     # Full height playfield
        "description": "Near background - platforms, walls, interactive elements"
    },
    "full": {
        "width": 320,
        "height": 256,
        "description": "Full single background (no parallax)"
    },
    "sprite": {
        "width": 128,
        "height": 128,
        "description": "Character/enemy sprites"
    },
    "tile": {
        "width": 64,
        "height": 64,
        "description": "Tileable textures"
    }
}

# Stili predefiniti per Leander 2
STYLES = {
    "gothic": {
        "prompt_prefix": "pixel art, 16-bit retro game art, dark gothic fantasy, ",
        "prompt_suffix": ", deep blue and purple shadows, stone textures, torchlight, moody atmosphere, detailed dithering, Amiga AGA style",
        "negative": "blurry, 3d render, photo, realistic, modern, text, watermark, bright colors, high saturation, cartoon, anime, smooth gradients"
    },
    "cave": {
        "prompt_prefix": "pixel art, 16-bit game background, dark underground cavern, ",
        "prompt_suffix": ", stalactites, rock formations, dim lighting, purple and blue tones, detailed pixel texture, Amiga retro style",
        "negative": "blurry, 3d, photo, realistic, bright, colorful, text, modern, anime"
    },
    "forest": {
        "prompt_prefix": "pixel art, 16-bit dark fantasy forest, ",
        "prompt_suffix": ", twisted dead trees, fog, moonlight through branches, dark green and purple palette, gothic atmosphere, retro game style",
        "negative": "blurry, 3d, photo, realistic, bright green, cartoon, anime, text, modern"
    },
    "castle": {
        "prompt_prefix": "pixel art, 16-bit medieval castle interior, ",
        "prompt_suffix": ", stone walls, gothic arches, wooden beams, torches, dark blue shadows, Psygnosis style, detailed pixel art",
        "negative": "blurry, 3d, photo, realistic, bright, modern, text, anime"
    },
    "sky_night": {
        "prompt_prefix": "pixel art, 16-bit night sky, ",
        "prompt_suffix": ", dark blue to purple gradient, stars, crescent moon, subtle clouds, retro game style, smooth dithering",
        "negative": "blurry, 3d, photo, realistic, bright, sun, daylight, text"
    },
    "sky_sunset": {
        "prompt_prefix": "pixel art, 16-bit sunset sky, ",
        "prompt_suffix": ", orange to purple gradient, dramatic clouds, retro game style, smooth color bands",
        "negative": "blurry, 3d, photo, realistic, text, modern"
    },
    "mountains": {
        "prompt_prefix": "pixel art, 16-bit distant mountain silhouettes, ",
        "prompt_suffix": ", layered parallax mountains, dark purple and blue, misty atmosphere, retro game background",
        "negative": "blurry, 3d, photo, realistic, detailed, close-up, text"
    }
}


class PixelArtGenerator:
    def __init__(self, use_cpu=False, low_vram=False):
        self.device = "cpu" if use_cpu else "cuda"
        self.dtype = torch.float32 if use_cpu else torch.float16
        self.pipe = None
        self.img2img_pipe = None
        self.low_vram = low_vram
        
    def load_model(self):
        """Carica il modello SDXL base + LoRA pixel art"""
        if self.pipe is not None:
            return
            
        print(f"📥 Caricamento SDXL base: {SDXL_BASE}")
        print(f"   + LoRA: {PIXEL_ART_LORA}")
        print(f"   Device: {self.device}, dtype: {self.dtype}")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        
        # Carica LoRA pixel art
        print("   Caricamento LoRA pixel art...")
        self.pipe.load_lora_weights(PIXEL_ART_LORA, weight_name=PIXEL_ART_LORA_FILE)
        
        # Scheduler veloce
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Ottimizzazioni VRAM
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            if self.low_vram:
                self.pipe.enable_model_cpu_offload()
                print("   ⚠️ Low VRAM mode: CPU offload attivo")
        
        print(f"✅ Modello caricato")
        
    def load_img2img_model(self):
        """Carica modello per img2img"""
        if self.img2img_pipe is not None:
            return
            
        print(f"📥 Caricamento SDXL img2img + LoRA")
        
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SDXL_BASE,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        
        # Carica LoRA
        self.img2img_pipe.load_lora_weights(PIXEL_ART_LORA, weight_name=PIXEL_ART_LORA_FILE)
        
        self.img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.img2img_pipe.scheduler.config
        )
        
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        
        if self.device == "cuda":
            self.img2img_pipe.enable_attention_slicing()
            if self.low_vram:
                self.img2img_pipe.enable_model_cpu_offload()
        
        print(f"✅ Modello img2img caricato")

    def generate(self, prompt, layer="full", style="gothic", output_path=None,
                 seed=None, steps=30, guidance=7.0):
        """Genera un singolo asset"""
        
        self.load_model()
        
        config = LAYER_CONFIGS.get(layer, LAYER_CONFIGS["full"])
        style_config = STYLES.get(style, STYLES["gothic"])
        
        # Costruisci prompt completo
        full_prompt = style_config["prompt_prefix"] + prompt + style_config["prompt_suffix"]
        
        print(f"\n🎨 Generazione: {prompt}")
        print(f"   Layer: {layer} ({config['width']}x{config['height']})")
        print(f"   Stile: {style}")
        
        # Generator per seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"   Seed: {seed}")
        
        # SDXL lavora meglio a risoluzioni più alte, poi ridimensioniamo
        # Minimo 512x512 per SDXL, usiamo multipli
        gen_width = max(config["width"], 512)
        gen_height = max(config["height"], 512)
        
        # Arrotonda a multipli di 8
        gen_width = ((gen_width + 7) // 8) * 8
        gen_height = ((gen_height + 7) // 8) * 8
        
        print(f"   Generazione a: {gen_width}x{gen_height}")
        
        # Genera
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=style_config["negative"],
            width=gen_width,
            height=gen_height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        # Ridimensiona alla dimensione target con NEAREST per pixel art croccante
        if gen_width != config["width"] or gen_height != config["height"]:
            image = image.resize(
                (config["width"], config["height"]),
                Image.Resampling.NEAREST
            )
        
        # Quantizza a 32 colori (Amiga AGA)
        image = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
        image = image.convert("RGB")
        
        # Salva
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)
            print(f"💾 Salvato: {output_path}")
        
        return image

    def generate_variation(self, reference_path, prompt, layer="full", style="gothic",
                          output_path=None, seed=None, steps=30, guidance=7.0, strength=0.5):
        """Genera variazione da immagine di riferimento"""
        
        self.load_img2img_model()
        
        config = LAYER_CONFIGS.get(layer, LAYER_CONFIGS["full"])
        style_config = STYLES.get(style, STYLES["gothic"])
        
        # Carica riferimento
        ref_image = Image.open(reference_path).convert("RGB")
        
        # Ridimensiona riferimento per SDXL
        gen_width = max(config["width"], 512)
        gen_height = max(config["height"], 512)
        gen_width = ((gen_width + 7) // 8) * 8
        gen_height = ((gen_height + 7) // 8) * 8
        
        ref_image = ref_image.resize((gen_width, gen_height), Image.Resampling.LANCZOS)
        
        full_prompt = style_config["prompt_prefix"] + prompt + style_config["prompt_suffix"]
        
        print(f"\n🎨 Variazione da: {reference_path}")
        print(f"   Prompt: {prompt}")
        print(f"   Strength: {strength}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"   Seed: {seed}")
        
        image = self.img2img_pipe(
            prompt=full_prompt,
            negative_prompt=style_config["negative"],
            image=ref_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        # Ridimensiona e quantizza
        if gen_width != config["width"] or gen_height != config["height"]:
            image = image.resize(
                (config["width"], config["height"]),
                Image.Resampling.NEAREST
            )
        
        image = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
        image = image.convert("RGB")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)
            print(f"💾 Salvato: {output_path}")
        
        return image

    def generate_level_pack(self, level_name, theme_prompt, style="gothic", 
                           output_dir="assets/generated", seed=None):
        """Genera un set completo di layer per un livello"""
        
        print(f"\n{'='*50}")
        print(f"🏰 Generazione Level Pack: {level_name}")
        print(f"   Tema: {theme_prompt}")
        print(f"   Stile: {style}")
        print(f"{'='*50}")
        
        level_dir = os.path.join(output_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
        
        base_seed = seed or 42
        
        # 1. Sky layer
        print("\n[1/3] Generazione SKY...")
        self.generate(
            f"night sky for {theme_prompt}",
            layer="sky",
            style="sky_night",
            output_path=os.path.join(level_dir, "sky.png"),
            seed=base_seed
        )
        
        # 2. Far background
        print("\n[2/3] Generazione FAR BACKGROUND...")
        self.generate(
            f"distant view of {theme_prompt}, silhouettes",
            layer="far",
            style=style,
            output_path=os.path.join(level_dir, "far_bg.png"),
            seed=base_seed + 1
        )
        
        # 3. Near/playfield
        print("\n[3/3] Generazione NEAR BACKGROUND (playfield)...")
        self.generate(
            f"{theme_prompt} with platforms and obstacles",
            layer="near",
            style=style,
            output_path=os.path.join(level_dir, "near_bg.png"),
            seed=base_seed + 2
        )
        
        # Salva metadata
        metadata = {
            "level_name": level_name,
            "theme": theme_prompt,
            "style": style,
            "seed": base_seed,
            "layers": ["sky.png", "far_bg.png", "near_bg.png"]
        }
        
        with open(os.path.join(level_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Level pack completato: {level_dir}/")
        return level_dir


def main():
    parser = argparse.ArgumentParser(
        description="Genera asset pixel art per Leander 2 (SDXL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Genera singolo background
  %(prog)s -p "gothic castle dungeon" --layer full --style gothic
  
  # Genera layer sky
  %(prog)s -p "stormy clouds" --layer sky --style sky_night
  
  # Genera level pack completo (sky + far + near)
  %(prog)s --level-pack "level1_castle" --theme "abandoned stone castle"
  
  # Variazione da riferimento
  %(prog)s -r assets/original/leander_03.png -p "ice version" --strength 0.5

Layer disponibili: sky, far, near, full, sprite, tile
Stili disponibili: gothic, cave, forest, castle, sky_night, sky_sunset, mountains
        """
    )
    
    # Generazione singola
    parser.add_argument("-p", "--prompt", help="Prompt per generazione")
    parser.add_argument("--layer", default="full", 
                        choices=list(LAYER_CONFIGS.keys()),
                        help="Tipo di layer")
    parser.add_argument("--style", default="gothic",
                        choices=list(STYLES.keys()),
                        help="Stile grafico")
    parser.add_argument("-o", "--output", default="assets/generated/output.png",
                        help="File output")
    
    # Level pack
    parser.add_argument("--level-pack", metavar="NAME",
                        help="Genera set completo layer per un livello")
    parser.add_argument("--theme", help="Tema per level pack")
    
    # Img2img
    parser.add_argument("-r", "--reference", help="Immagine riferimento per variazione")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Forza modifica (0.0-1.0)")
    
    # Parametri generazione
    parser.add_argument("--seed", type=int, help="Seed per riproducibilità")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    
    # Hardware
    parser.add_argument("--cpu", action="store_true", help="Usa CPU")
    parser.add_argument("--low-vram", action="store_true", 
                        help="Modalità low VRAM (più lento)")
    
    args = parser.parse_args()
    
    # Info GPU
    if not args.cpu and torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {vram:.1f} GB")
        if vram < 10:
            print("   ⚠️ Consigliato --low-vram per GPU con meno di 10GB")
    
    # Crea generatore
    gen = PixelArtGenerator(use_cpu=args.cpu, low_vram=args.low_vram)
    
    # Esegui
    if args.level_pack:
        if not args.theme:
            print("❌ Errore: --theme richiesto con --level-pack")
            return
        gen.generate_level_pack(
            args.level_pack, args.theme, args.style,
            seed=args.seed
        )
    elif args.reference and args.prompt:
        gen.generate_variation(
            args.reference, args.prompt, args.layer, args.style,
            args.output, args.seed, args.steps, args.guidance, args.strength
        )
    elif args.prompt:
        gen.generate(
            args.prompt, args.layer, args.style,
            args.output, args.seed, args.steps, args.guidance
        )
    else:
        parser.print_help()
        print("\n" + "="*50)
        print("🎮 Demo: genero un level pack di esempio...")
        print("="*50)
        gen.generate_level_pack(
            "demo_gothic", 
            "dark gothic castle with stone walls",
            style="gothic",
            seed=42
        )


if __name__ == "__main__":
    main()
