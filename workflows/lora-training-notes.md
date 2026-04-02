# LoRA Training Notes

## 환경
- Mac mini M4, 24GB unified memory
- Python 3.12 (ARM64)
- PyTorch 2.11.0 (MPS backend)
- kohya-ss/sd-scripts
- venv: `.venv/`

## 베이스 모델 선택
- **SDXL 계열 권장** (SD 1.5보다 2D 캐릭터 품질이 좋음)
- 추천 모델 (anime/cartoon 특화):
  - Illustrious XL v1.0
  - AnimagineXL
  - PonyDiffusion XL
- 모델 저장 경로: `models/`

## 데이터셋 구성
- 25장 train / 3장 eval
- 3 repeats × 25 = 75 steps/epoch
- 25 epochs → 총 ~1,875 steps
- 캡션: 트리거 워드 `arazzi` + 영문 설명
- kohya 형식: `training/image/3_arazzi/` (이미지+캡션 쌍)

## 학습 파라미터 (v1)
```
network_dim: 32
network_alpha: 16
learning_rate: 1e-4
text_encoder_lr: 5e-5
lr_scheduler: cosine_with_restarts
optimizer: AdamW
mixed_precision: fp16
resolution: 1024 (bucketing 768~1536)
batch_size: 1
epochs: 25
noise_offset: 0.05
gradient_checkpointing: true
cache_latents: true
cache_text_encoder_outputs: true
attention: sdpa (xformers 불가)
```

## Apple Silicon (MPS) 주의사항
1. `PYTORCH_ENABLE_MPS_FALLBACK=1` 환경변수 필수
2. xformers 사용 불가 → `--sdpa` 사용
3. bf16 대신 **fp16** 사용 (bf16은 MPS에서 NaN 발생 가능)
4. `torch.compile` 미지원 → eager mode
5. `blocks_to_swap` 미지원 (CUDA 전용)
6. 학습 중 다른 앱 닫아 메모리 확보 (unified memory 공유)
7. bitsandbytes 8bit 옵티마이저 호환 불확실 → 표준 AdamW 사용

## 실행 방법
```bash
# 1. 베이스 모델 다운로드 후 models/ 에 배치
# 2. train_arazzi.sh 내 MODEL_PATH 수정
# 3. 실행
cd /Users/yoon/Projects/arazzi-archive
./training/train_arazzi.sh
```

## 예상 소요 시간
- SDXL LoRA, 25 epochs: 약 3~6시간

## 평가 계획
- eval 이미지 3장으로 캐릭터 재현성 비교
- 프롬프트별 일관성 체크
- 실패 패턴 기록 후 파라미터 조정

## TODO
- [ ] 베이스 모델 다운로드 (Illustrious XL 권장)
- [ ] 1차 학습 실행
- [ ] 결과 평가
- [ ] 필요시 파라미터 조정 후 재학습
