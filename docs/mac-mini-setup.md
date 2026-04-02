# Mac mini Setup (M4 / 24GB)

## 현재 상태: 설치 완료

### 하드웨어
- Mac mini M4, 24GB unified memory
- macOS Darwin 25.3.0

### 소프트웨어 환경
- Python 3.12.x (ARM64, Homebrew)
- venv: `arazzi-archive/.venv/`
- PyTorch 2.11.0 (MPS backend 활성)
- kohya-ss/sd-scripts: `tools/sd-scripts/`

### 설치된 주요 패키지
- torch 2.11.0
- diffusers 0.32.1
- transformers 4.54.1
- accelerate 1.6.0
- safetensors 0.4.5

## 남은 작업
1. **베이스 모델 다운로드** → `models/` 디렉토리에 배치
   - 추천: Illustrious XL v1.0 (Civitai 또는 HuggingFace)
   - 파일 크기: ~6.5GB
2. `training/train_arazzi.sh` 내 `MODEL_PATH` 경로 수정
3. 학습 실행

## 활성화 방법
```bash
cd /Users/yoon/Projects/arazzi-archive
source .venv/bin/activate
```

## MPS 확인
```python
import torch
print(torch.backends.mps.is_available())  # True
```
