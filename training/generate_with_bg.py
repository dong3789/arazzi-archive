#!/usr/bin/env python3
"""AraZZi LoRA - generate images with backgrounds"""

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import os

PROJECT_DIR = "/Users/yoon/Projects/arazzi-archive"
MODEL_PATH = f"{PROJECT_DIR}/models/dreamshaper-8"
LORA_PATH = f"{PROJECT_DIR}/training/output/arazzi_lora_v1.safetensors"
OUTPUT_DIR = f"{PROJECT_DIR}/training/output/samples"

print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LORA_PATH)
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompts = [
    {
        "prompt": "arazzi, a cute beige hamster character, sitting on a grassy hill under cherry blossom trees, petals falling, spring afternoon, warm sunlight, soft pastel sky, 2d illustration, soft pastel colors, brown outline, cozy peaceful scene",
        "seed": 77,
        "name": "arazzi_cherry_blossom.png",
    },
    {
        "prompt": "arazzi, a cute beige hamster character, sitting at a tiny cafe table, holding a cup of hot cocoa, steam rising, rainy window background, cozy indoor scene, warm lighting, 2d illustration, soft pastel colors, brown outline, cute storybook style",
        "seed": 123,
        "name": "arazzi_rainy_cafe.png",
    },
]

negative_prompt = "realistic, 3d, photo, low quality, blurry, deformed, ugly, text, watermark, multiple characters, nsfw"

for i, p in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] {p['name']}...")
    generator = torch.Generator("mps").manual_seed(p["seed"])
    image = pipe(
        prompt=p["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    out_path = os.path.join(OUTPUT_DIR, p["name"])
    image.save(out_path)
    print(f"  Saved: {out_path}")

print("\nDone!")
