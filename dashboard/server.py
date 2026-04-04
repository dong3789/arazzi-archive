#!/usr/bin/env python3
"""AraZZi Gallery — generation server"""

import os, json, uuid, time, threading
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

PROJECT = Path("/Users/yoon/Projects/arazzi-archive")
DASHBOARD = PROJECT / "dashboard"
IMAGES_DIR = DASHBOARD / "images"
GALLERY_JSON = DASHBOARD / "gallery.json"
LORA_PATH = PROJECT / "training/output/arazzi_lora_sdxl_v1.safetensors"
MODEL_PATH = "/Users/yoon/Documents/ComfyUI/models/checkpoints/novaAnimeXL_ilV160.safetensors"

# --- prompt template ---
PROMPT_TEMPLATE = (
    "arazzi, a cute beige hamster character, {scene}, "
    "simple body, short round arms, no fingers, stubby limbs, "
    "2d illustration, soft pastel colors, brown outline, cute storybook style, "
    "clean lineart, flat color, chibi proportions"
)
NEGATIVE = (
    "realistic, 3d, photo, low quality, blurry, deformed, ugly, "
    "text, watermark, multiple characters, nsfw, "
    "extra fingers, mutated hands, bad hands, extra hands, missing fingers, "
    "fused fingers, too many fingers, extra arms, extra legs, bad anatomy, "
    "malformed limbs, deformed body, disfigured, poorly drawn hands, "
    "poorly drawn face, mutation, extra limbs, gross proportions"
)

# Korean → English basic scene keywords
KO_EN = {
    "벚꽃": "cherry blossom",
    "카페": "cafe",
    "비": "rain",
    "눈": "snow",
    "바다": "sea, ocean, beach",
    "숲": "forest",
    "밤": "night sky, stars",
    "봄": "spring",
    "여름": "summer",
    "가을": "autumn, falling leaves",
    "겨울": "winter, snow",
    "먹": "eating",
    "자": "sleeping",
    "웃": "smiling, happy",
    "울": "crying, sad",
    "춤": "dancing",
    "노래": "singing",
    "요리": "cooking",
    "산책": "walking",
    "운동": "exercising",
    "공부": "studying",
    "게임": "playing games",
    "음악": "music, headphones",
    "커피": "coffee, hot drink",
    "케이크": "cake",
    "피자": "pizza",
    "라면": "ramen, noodles",
    "아이스크림": "ice cream",
    "우산": "umbrella",
    "모자": "hat",
    "안경": "glasses",
    "꽃": "flowers",
    "별": "stars, sparkles",
    "달": "moon, moonlight",
    "해": "sun, sunshine",
    "무지개": "rainbow",
    "풍선": "balloons",
    "선물": "gift box",
    "하트": "hearts",
    "고양이": "with a cat",
    "강아지": "with a puppy",
    "크리스마스": "christmas, festive",
    "생일": "birthday, party",
    "소풍": "picnic, outdoor",
    "캠핑": "camping, tent",
    "수영": "swimming, pool",
    "낚시": "fishing",
    "그림": "painting, drawing",
    "책": "reading a book",
    "편지": "writing a letter",
    "전화": "talking on phone",
    "사진": "taking a photo",
    "여행": "traveling, suitcase",
    "비행기": "airplane",
    "기차": "train",
    "자전거": "riding a bicycle",
    "잠": "sleeping, zzz",
    "화남": "angry, frustrated",
    "놀람": "surprised, shocked",
    "사랑": "in love, hearts floating",
    "행복": "very happy, joyful",
    "슬픔": "sad, tears",
    "피곤": "tired, exhausted",
    "배고픔": "hungry",
    "정원": "garden",
    "방": "cozy room, indoors",
    "학교": "school, classroom",
    "공원": "park, bench",
    "도서관": "library",
    "놀이공원": "amusement park",
    "마트": "supermarket, shopping",
    "병원": "hospital, nurse outfit",
}


def translate_prompt(korean_input: str) -> str:
    """Translate Korean keywords to English scene description."""
    parts = []
    remaining = korean_input

    for ko, en in sorted(KO_EN.items(), key=lambda x: -len(x[0])):
        if ko in remaining:
            parts.append(en)
            remaining = remaining.replace(ko, "")

    # If we translated something, use it; otherwise pass through as-is
    if parts:
        scene = ", ".join(parts)
    else:
        scene = korean_input

    return PROMPT_TEMPLATE.format(scene=scene)


def load_gallery() -> list:
    if GALLERY_JSON.exists():
        return json.loads(GALLERY_JSON.read_text())
    return []


def save_gallery(gallery: list):
    GALLERY_JSON.write_text(json.dumps(gallery, ensure_ascii=False, indent=2))


# --- Load pipeline at startup ---
print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float32,
    safety_checker=None,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(str(LORA_PATH))
pipe = pipe.to("mps")
pipe.enable_attention_slicing()
print("SDXL Pipeline ready!")

# --- FastAPI ---
app = FastAPI()


# --- Job queue ---
jobs: dict = {}  # job_id -> {status, entry, error, position}
job_queue: list = []  # ordered list of pending job_ids
gen_lock = threading.Lock()
queue_lock = threading.Lock()
worker_thread = None


def queue_worker():
    """Process jobs one at a time from the queue."""
    while True:
        with queue_lock:
            if not job_queue:
                return
            job_id = job_queue[0]

        job = jobs.get(job_id)
        if not job:
            with queue_lock:
                job_queue.pop(0)
            continue

        jobs[job_id]["status"] = "generating"
        # Update positions for remaining jobs
        with queue_lock:
            for i, jid in enumerate(job_queue):
                if jid != job_id and jid in jobs:
                    jobs[jid]["position"] = i

        try:
            prompt = translate_prompt(job["prompt_ko"])
            seed = int(time.time()) % 100000
            generator = torch.Generator("mps").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                num_inference_steps=30,
                guidance_scale=7.0,
                width=1024,
                height=1024,
                generator=generator,
            ).images[0]

            filename = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
            filepath = IMAGES_DIR / filename
            image.save(str(filepath))

            entry = {
                "filename": filename,
                "prompt_ko": job["prompt_ko"],
                "prompt_en": prompt,
                "seed": seed,
                "created": datetime.now().isoformat(),
            }

            gallery = load_gallery()
            gallery.insert(0, entry)
            save_gallery(gallery)

            jobs[job_id] = {"status": "done", "entry": entry}
        except Exception as e:
            jobs[job_id] = {"status": "error", "error": str(e)}

        with queue_lock:
            if job_queue and job_queue[0] == job_id:
                job_queue.pop(0)


def ensure_worker():
    """Start the worker thread if not running."""
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=queue_worker, daemon=True)
        worker_thread.start()


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    user_prompt = body.get("prompt", "").strip()
    if not user_prompt:
        return JSONResponse({"error": "prompt is required"}, 400)

    job_id = uuid.uuid4().hex[:8]

    with queue_lock:
        position = len(job_queue)
        job_queue.append(job_id)

    jobs[job_id] = {
        "status": "queued",
        "prompt_ko": user_prompt,
        "position": position,
    }

    ensure_worker()

    return {"ok": True, "job_id": job_id, "position": position}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, 404)
    return job


@app.get("/api/queue")
async def get_queue():
    with queue_lock:
        return {"length": len(job_queue), "jobs": job_queue}


@app.get("/api/gallery")
async def get_gallery():
    return load_gallery()


@app.get("/api/stats")
async def get_stats():
    gallery = load_gallery()
    prompts = [g["prompt_ko"] for g in gallery]
    return {
        "total": len(gallery),
        "prompts": prompts,
    }


# Serve static files (index.html, images/)
app.mount("/", StaticFiles(directory=str(DASHBOARD), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
