#!/bin/bash
# AraZZi LoRA Training Script (SDXL)
# Mac mini M4 24GB — fp16 with patched accelerate

# MPS fallback 활성화
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 프로젝트 경로
PROJECT_DIR="/Users/yoon/Projects/arazzi-archive"
VENV_DIR="${PROJECT_DIR}/.venv"

# venv 활성화
source "${VENV_DIR}/bin/activate"

# sd-scripts 디렉토리로 이동 (import 경로 해결)
SD_SCRIPTS_DIR="${PROJECT_DIR}/tools/sd-scripts"
cd "${SD_SCRIPTS_DIR}"

# 다른 앱 닫아서 메모리 확보 권장
echo "=== SDXL LoRA Training (fp16) ==="
echo "RAM을 최대한 확보하세요 (다른 앱 종료 권장)"
echo ""

accelerate launch --num_cpu_threads_per_process=4 \
  "${SD_SCRIPTS_DIR}/sdxl_train_network.py" \
  --pretrained_model_name_or_path="/Users/yoon/Documents/ComfyUI/models/checkpoints/novaAnimeXL_ilV160.safetensors" \
  --dataset_config="${PROJECT_DIR}/training/dataset_config_sdxl.toml" \
  --output_dir="${PROJECT_DIR}/training/output" \
  --output_name="arazzi_lora_sdxl_v1" \
  --save_model_as=safetensors \
  --save_every_n_epochs=5 \
  --max_train_epochs=20 \
  --network_module=networks.lora \
  --network_dim=16 \
  --network_alpha=8 \
  --learning_rate=1e-4 \
  --network_train_unet_only \
  --lr_scheduler=cosine_with_restarts \
  --lr_warmup_steps=50 \
  --optimizer_type=AdamW \
  --mixed_precision=fp16 \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --noise_offset=0.05 \
  --seed=42 \
  --sdpa \
  --caption_extension=".txt" \
  --max_token_length=150 \
  --logging_dir="${PROJECT_DIR}/training/logs" \
  --log_with=tensorboard

echo ""
echo "=== 학습 완료 ==="
echo "결과물: ${PROJECT_DIR}/training/output/"
