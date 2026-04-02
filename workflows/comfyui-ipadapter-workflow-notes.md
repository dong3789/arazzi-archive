# ComfyUI IPAdapter Workflow Notes

기본 구성:
- Load Checkpoint
- CLIP Text Encode (positive)
- CLIP Text Encode (negative)
- Load Image (reference)
- IPAdapter
- KSampler
- VAE Decode
- Save Image

## 목적
- 아라찌 유사도 빠른 확인
- 대표 이미지별 반응 차이 테스트

## TODO
- 실제 사용하는 모델명 기록
- 가장 잘 나온 설정값 기록
