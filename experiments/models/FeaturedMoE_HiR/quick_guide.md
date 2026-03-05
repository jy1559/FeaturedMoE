# FeaturedMoE_HiR Quick Guide

## 1) 한 줄 요약
`FeaturedMoE_HiR`는 FeaturedMoE의 stage 구조를 유지하면서, 각 stage 내부를 **2단 라우팅(bundle -> intra-bundle expert)**으로 바꾼 모델입니다.

## 2) FeaturedMoE와의 핵심 차이
- 공통: layout 기반 stage 배치, Transformer pre/post scaffold
- 차이:
  - Stage 내부 라우팅: 1단 -> 2단(Hierarchical-in-Stage)
  - expert: hidden-only FFN (`expert_use_feature`는 강제로 비활성 처리)
  - stage 병합: `stage_merge_mode=serial|parallel`
  - HiR 전용 aux loss: bundle-level balance 항 추가 가능

핵심 구현 파일:
- `featured_moe_hir.py`: 모델 본체/스케줄/손실
- `hir_moe_stages.py`: `HierarchicalStageMoE`, `HierarchicalMoEHiR`
- 공용 모듈 재사용: `FeaturedMoE/routers.py`, `FeaturedMoE/transformer.py`

## 3) 영향 큰 하이퍼파라미터
- 공통(FeaturedMoE와 동일하게 중요)
  - `arch_layout_catalog`, `arch_layout_id`
  - `moe_top_k*`, `moe_top_k_policy`, `moe_top_k_ratio`
  - `expert_scale`, `d_expert_hidden`, `d_router_hidden`
  - `mid/micro_router_temperature`, `*_feature_dropout`, `balance_loss_lambda`
- HiR 전용(특히 중요)
  - `bundle_top_k`
    - bundle 선택 sparsity 강도.
  - `stage_merge_mode`
    - `serial`: Macro->Mid->Micro 순차
    - `parallel`: stage delta를 stage gate로 가중합
  - `parallel_stage_gate_top_k`
    - parallel mode에서 stage gate sparsity
  - `hir_use_bundle_aux_loss`, `hir_bundle_aux_lambda_scale`
    - expert balance 외 bundle balance 항 강도

## 4) 권장 튜닝 순서
1. `stage_merge_mode`(serial/parallel)와 layout 축 먼저 확정
2. `bundle_top_k`, `moe_top_k*`로 sparsity 축 조정
3. `parallel_stage_gate_top_k`(parallel일 때) 미세조정
4. `hir_*bundle_aux*`와 `balance_loss_lambda` 균형 조정

## 5) 운영 참고
- 활성 run track: `experiments/run/fmoe_hir/*`
- 기본 산출물: `experiments/run/artifacts/logs/fmoe_hir`, `experiments/run/artifacts/results/fmoe_hir`
