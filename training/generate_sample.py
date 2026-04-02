#!/usr/bin/env python3
"""AraZZi LoRA sample image generator"""

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from safetensors.torch import load_file
import os

PROJECT_DIR = "/Users/yoon/Projects/arazzi-archive"
MODEL_PATH = f"{PROJECT_DIR}/models/dreamshaper-8"
LORA_PATH = f"{PROJECT_DIR}/training/output/arazzi_lora_v1.safetensors"
OUTPUT_DIR = f"{PROJECT_DIR}/training/output/samples"

print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)

# Use Euler Ancestral scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("Loading LoRA weights...")
pipe.load_lora_weights(LORA_PATH)

print("Moving to MPS...")
pipe = pipe.to("mps")

# Enable attention slicing for memory efficiency
pipe.enable_attention_slicing()

prompts = [
    "arazzi, a cute beige hamster character, standing front view, happy smiling expression, white background, 2d illustration, soft pastel colors, brown outline, simple clean style",
    "arazzi, a cute beige hamster character, sitting, eating a snack, eyes closed, content expression, white background, 2d illustration, soft pastel colors, brown outline",
    "arazzi, a cute beige hamster character, waving hand, cheerful expression, blushing pink cheeks, white background, full body, 2d illustration, soft pastel colors, brown outline",
]

negative_prompt = "realistic, 3d, photo, low quality, blurry, deformed, ugly, text, watermark, multiple characters"

print(f"Generating {len(prompts)} images...")
for i, prompt in enumerate(prompts):
    print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")

    generator = torch.Generator("mps").manual_seed(42 + i)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"arazzi_sample_{i+1}.png")
    image.save(out_path)
    print(f"  Saved: {out_path}")

print("\nDone! All samples generated.")
