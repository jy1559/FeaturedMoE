# FeaturedMoE_v2 Quick Guide

## 한 줄 요약
`FeaturedMoE_v2`는 stage 경계를 명시하는 layout 기반 Stage-MoE 모델이며, `serial`/`parallel`과 repeat를 통합 지원합니다.

## 구조 개요
- 입력: `item_id_list` + engineered feature (`mac_*`, `mid_*`, `mic_*`)
- 전역 흐름: `global_pre_layers -> stage executor -> global_post_layers`
- stage 경계 제어:
  - `pass_layers`: MoE 비적용 통과 레이어
  - `moe_blocks`: `[1-layer attention + 1 MoE]` 반복 횟수
- `parallel` 모드에서는 stage delta를 merge router로 가중합

## 영향 큰 하이퍼파라미터
- `fmoe_v2_layout_catalog`, `fmoe_v2_layout_id`
  - MoE 적용 시작점/깊이/반복 강도에 직접 영향
- `fmoe_stage_execution_mode`
  - `serial` vs `parallel` 계산 방식 차이
- `moe_top_k`, `moe_top_k_policy`, `moe_top_k_ratio`
  - expert 라우팅 sparsity 제어
- `router_impl`, `router_impl_by_stage`, `rule_router.*`
  - learned vs rule_soft 및 stage별 mixed routing 제어
- `fmoe_v2_parallel_stage_gate_top_k`, `fmoe_v2_parallel_stage_gate_temperature`
  - parallel merge 안정성/표현력 제어
- `balance_loss_lambda`, `fmoe_v2_stage_merge_aux_enable`
  - collapse 억제 및 merge regularization 강도

## 추천 튜닝 순서
1. layout/execution 고정 (`fmoe_v2_layout_id`, `fmoe_stage_execution_mode`)
2. optimizer 튜닝 (`learning_rate`, `weight_decay`, `hidden_dropout_prob`)
3. routing 튜닝 (`moe_top_k*`, 온도, feature dropout)
4. parallel일 때 merge gate/merge aux 튜닝

## 운영 경로
- 모델: `experiments/models/FeaturedMoE_v2`
- 런 트랙: `experiments/run/fmoe_v2`
- rule ablation 트랙: `experiments/run/fmoe_rule`
- 산출물: `experiments/run/artifacts/{logs,results}/fmoe_v2`
