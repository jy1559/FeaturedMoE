# FeaturedMoE_HiR2 Quick Guide

`FeaturedMoE_HiR2`는 기존 HiR와 분리된 Stage-first 2단 게이팅 모델입니다.

- 1단: session-level stage allocator (`macro/mid/micro` 비율)
- 2단: stage 내부 token-level expert router
- 결합 모드:
  - `serial_weighted`
  - `parallel_weighted`

기본 모델 설정:
- `experiments/configs/model/featured_moe_hir2.yaml`
- 튜닝 설정:
  `experiments/configs/model/featured_moe_hir2_tune.yaml`
