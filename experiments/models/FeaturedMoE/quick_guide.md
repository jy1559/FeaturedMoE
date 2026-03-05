# FeaturedMoE Quick Guide

## 1) 한 줄 요약
`FeaturedMoE`는 **Transformer + 3-stage(Macro/Mid/Micro) Stage-MoE**를 교차 배치하고, `arch_layout_id`로 attention/MoE 배치를 제어하는 순차 추천 모델입니다.

## 2) 구조 개요
- 입력: `item_id_list` + feature list(`mac_*`, `mid_*`, `mic_*`)
- 백본: `pre_transformer` -> stage별 `pre-attn + MoE` -> `post_transformer`
- Stage-MoE: `Router`가 expert 가중치 생성, `ExpertGroup` 출력의 가중합을 residual로 더함
- 출력: 마지막 valid position hidden으로 next-item logits 계산

핵심 구현 파일:
- `featured_moe.py`: 모델 본체, layout/schedule/loss
- `moe_stages.py`: `MoEStage`, `HierarchicalMoE`
- `experts.py`: expert MLP/그룹
- `routers.py`: gating + load-balance loss
- `transformer.py`: Transformer(+optional FFN-MoE)

## 3) 영향 큰 하이퍼파라미터
- `arch_layout_catalog`, `arch_layout_id`
  - stage별 depth/bypass를 결정하는 최상위 구조 파라미터.
- `moe_top_k`, `moe_top_k_policy`, `moe_top_k_ratio`, `moe_top_k_min`
  - dense/sparse gating 강도와 expert 선택 폭을 결정.
- `expert_scale`, `d_expert_hidden`, `d_router_hidden`
  - 전문가 표현력/라우터 용량/연산량에 직접 영향.
- `macro_routing_scope`, `macro_session_pooling`
  - macro stage가 session-level로 묶어 볼지(token/session) 결정.
- `mid_router_temperature`, `micro_router_temperature`
  - gating 분산/집중도에 큰 영향.
- `mid_router_feature_dropout`, `micro_router_feature_dropout`, `use_valid_ratio_gating`
  - feature 신뢰도 반영 및 과적합/노이즈 제어.
- `stage_moe_repeat_after_pre_layer`
  - depth>0 stage에서 `pre-attn -> MoE` 반복 여부.
- `balance_loss_lambda`
  - expert collapse 방지 강도.
- `fmoe_schedule_enable` + warmup 계열
  - 초기 학습 안정화(temperature/top-k/alpha warmup).

## 4) 권장 튜닝 순서
1. `arch_layout_id`로 구조 먼저 고정
2. `learning_rate`, `weight_decay`, `hidden_dropout_prob`, `balance_loss_lambda` 조정
3. routing(`moe_top_k*`, `temperature*`, `*_feature_dropout`) 미세조정
4. 필요 시 `fmoe_schedule_enable` ablation

## 5) 운영 참고
- 활성 run track: `experiments/run/fmoe/*`
- 기본 산출물: `experiments/run/artifacts/logs/fmoe`, `experiments/run/artifacts/results/fmoe`
