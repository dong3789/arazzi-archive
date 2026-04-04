#!/usr/bin/env python3
"""AraZZi SDXL LoRA - generate 5 sample images"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os

PROJECT_DIR = "/Users/yoon/Projects/arazzi-archive"
BASE_MODEL = "/Users/yoon/Documents/ComfyUI/models/checkpoints/novaAnimeXL_ilV160.safetensors"
LORA_PATH = f"{PROJECT_DIR}/training/output/arazzi_lora_sdxl_v1.safetensors"
OUTPUT_DIR = f"{PROJECT_DIR}/training/output/samples_sdxl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading SDXL base model...")
pipe = StableDiffusionXLPipeline.from_single_file(
    BASE_MODEL,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("Loading LoRA weights...")
pipe.load_lora_weights(LORA_PATH)

print("Moving to MPS...")
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompts = [
    {
        "prompt": "arazzi, a cute beige hamster character, standing front view, happy smiling expression, waving hand, white background, full body, 2d illustration, soft pastel colors, brown outline",
        "seed": 42,
        "name": "arazzi_sdxl_hello.png",
    },
    {
        "prompt": "arazzi, a cute beige hamster character, sitting, eating a large rice ball, eyes closed, content expression, white background, 2d illustration, soft pastel colors, brown outline",
        "seed": 77,
        "name": "arazzi_sdxl_eating.png",
    },
    {
        "prompt": "arazzi, a cute beige hamster character, sleeping curled up, peaceful expression, soft blanket, warm cozy room, 2d illustration, soft pastel colors, brown outline",
        "seed": 123,
        "name": "arazzi_sdxl_sleeping.png",
    },
    {
        "prompt": "arazzi, a cute beige hamster character, sitting under cherry blossom tree, petals falling, spring, happy expression, pink blush on cheeks, 2d illustration, soft pastel colors, brown outline",
        "seed": 256,
        "name": "arazzi_sdxl_cherry_blossom.png",
    },
    {
        "prompt": "arazzi, a cute beige hamster character, holding a small sunflower, standing in a garden, summer day, cheerful expression, 2d illustration, soft pastel colors, brown outline",
        "seed": 512,
        "name": "arazzi_sdxl_sunflower.png",
    },
]

negative_prompt = "realistic, 3d, photo, low quality, blurry, deformed, ugly, text, watermark, multiple characters, nsfw"

print(f"Generating {len(prompts)} SDXL images...")
for i, p in enumerate(prompts):
    print(f"  [{i+1}/{len(prompts)}] {p['name']}...")
    generator = torch.Generator("mps").manual_seed(p["seed"])
    image = pipe(
        prompt=p["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.0,
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]
    out_path = os.path.join(OUTPUT_DIR, p["name"])
    image.save(out_path)
    print(f"  Saved: {out_path}")

print("\nDone! All SDXL samples generated.")
