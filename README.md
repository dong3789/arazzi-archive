# AraZZi Revival

아라찌 캐릭터를 로컬 환경에서 지속적으로 재현/확장 생성하기 위한 프로젝트.

목표:
- 참조 이미지 기반으로 아라찌 캐릭터 일관성 확보
- ComfyUI에서 빠른 생성 워크플로우 구성
- Mac mini (M4, 24GB RAM) 기준으로 로컬 작업 가능하게 세팅
- 최종적으로 아라찌 전용 LoRA 학습 및 운영

## 프로젝트 단계

### Phase 1 — 빠른 재현 테스트
- 대표 참조 이미지 3~5장 선별
- ComfyUI + IPAdapter로 유사도 테스트
- 캐릭터 특징 언어화 및 프롬프트 템플릿 정리

### Phase 2 — 데이터셋 준비
- 학습용 이미지 20~40장 수집
- 단독샷/표정/포즈/소품별 정리
- 캡션/태그 정리
- 학습 제외 이미지 분리

### Phase 3 — LoRA 학습
- 아라찌 전용 캐릭터 LoRA 1차 학습
- 재현성 평가
- 실패 케이스 분석
- 2차 데이터 보정 후 재학습

### Phase 4 — 운영
- LoRA + IPAdapter 조합 워크플로우 고정
- 자주 쓰는 프롬프트 세트 정리
- 결과물 관리 규칙 수립

## 현재 권장 전략

1. **IPAdapter 먼저**
   - 아라찌와 얼마나 비슷하게 나오는지 빠르게 확인
2. **LoRA로 캐릭터 고정**
   - 장기적으로는 필수
3. **LoRA + IPAdapter 병행**
   - 정체성 유지 + 장면 제어 둘 다 확보

## Mac mini (M4, 24GB) 판단

- ComfyUI 추론: 충분히 가능
- IPAdapter 테스트: 가능
- 소규모 캐릭터 LoRA 학습: 가능성 높음
- 무거운 고해상도 학습: 세팅 최적화 필요

즉, 이 프로젝트는 Mac mini 로컬 시작이 현실적임.

## 권장 학습 자료

### 먼저 볼 자료
1. ComfyUI 캐릭터 일관성 / IPAdapter 튜토리얼
2. Kohya-SS LoRA 학습 기본 튜토리얼
3. Apple Silicon에서 Stable Diffusion/LoRA 학습 세팅 자료

### 참고 링크
- ComfyUI character consistency (IPAdapter / ControlNet)
  - https://learn.runcomfy.com/create-consistent-characters-with-controlnet-ipadapter
- ComfyUI LoRA training guide
  - https://www.apatero.com/blog/comfyui-lora-training-character-consistency-guide-2026
- Kohya-SS basics to testing
  - https://lilys.ai/en/notes/training-lora-20260208/kohya-ss-lora-training-guide
- Kohya-SS YouTube tutorial
  - https://www.youtube.com/watch?v=wTVI0SONkpc
- Apple Silicon LoRA training reference
  - https://www.reallyar.com/training-stable-diffusion-lora-on-apple-silicon-m2-mac-gpus-metal/
- Stable Diffusion on Apple Silicon overview
  - https://stable-diffusion-art.com/install-mac/

## 추천 폴더 구조

```text
arazzi-revival/
├── README.md
├── docs/
│   ├── roadmap.md
│   ├── dataset-guide.md
│   ├── learning-resources.md
│   ├── mac-mini-setup.md
│   └── workflow-plan.md
├── prompts/
│   ├── character-core.md
│   ├── prompt-templates.md
│   └── negative-prompts.md
├── datasets/
│   ├── raw/
│   ├── selected/
│   ├── train/
│   ├── eval/
│   └── captions/
└── workflows/
    ├── comfyui-ipadapter-workflow-notes.md
    └── lora-training-notes.md
```

## 즉시 할 일

1. 아라찌 원본 이미지 최대한 모으기
2. 대표 이미지 3~5장 고르기
3. ComfyUI 설치 상태 확인
4. IPAdapter 테스트 시작
5. LoRA 학습용 이미지 선별 시작

## 주의

- 회사 IP라면 상업적 사용 전 권리 확인 필요
- 개인 보존/연구 목적과 상업 활용은 구분할 것
