#!/bin/bash
# AraZZi LoRA Training Script (SD 1.5)
# Mac mini M4 24GB 최적화 설정

# MPS fallback 활성화 (필수)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 프로젝트 경로
PROJECT_DIR="/Users/yoon/Projects/arazzi-archive"
SD_SCRIPTS_DIR="${PROJECT_DIR}/tools/sd-scripts"
VENV_DIR="${PROJECT_DIR}/.venv"

# venv 활성화
source "${VENV_DIR}/bin/activate"

# ===== 베이스 모델 경로 =====
# SD 1.5 기반 모델 (SDXL은 24GB에서 float32 메모리 초과)
MODEL_PATH="${PROJECT_DIR}/models/dreamshaper-8"

# ===== 학습 실행 =====
accelerate launch --num_cpu_threads_per_process=4 \
  "${SD_SCRIPTS_DIR}/train_network.py" \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --dataset_config="${PROJECT_DIR}/training/dataset_config.toml" \
  --output_dir="${PROJECT_DIR}/training/output" \
  --output_name="arazzi_lora_v1" \
  --save_model_as=safetensors \
  --save_every_n_epochs=5 \
  --max_train_epochs=25 \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=1e-4 \
  --network_train_unet_only \
  --lr_scheduler=cosine_with_restarts \
  --lr_warmup_steps=100 \
  --optimizer_type=AdamW \
  --mixed_precision=no \
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
