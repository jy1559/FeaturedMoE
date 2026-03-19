# Phase 7 Plan: Router Input Decision + Aux/Reg Direction Test

작성일: 2026-03-18  
대상: FeaturedMoE_N3 on KuaiRecLargeStrictPosV2_0.2 (우선), 필요 시 lastfm0.03 확장

## 0) 실험 목적
Phase7은 나머지 세팅을 고정하고 아래 2가지만 분리 검증한다.

1. Router 입력 방식 8종 비교
- standard
- factored
- factored 비중 강화(factored-heavy)
- factored-only(no stage-whole feature routing)
- hierarchical-like multiplicative(HIR)
- group importance bias(FAC_GROUP)
- fac-only + group router both(FAC_ONLY_BOTH)
- factored-heavy + group router feature(FAC_HEAVY_FEAT)

2. Aux/Reg 방향성 8종 비교
- baseline router 2종(standard, factored-heavy)을 고정
- balance 유도 2종 vs specialization 유도 2종
- 총 8설정

핵심 관찰 포인트:
- 성능: valid/test MRR@20
- special: cold item, short session(1-2, 3-5)
- diag: expert usage 분포, top1 편향, n_eff, macro/micro jitter

## 1) 왜 이 설계인가 (Phase5/6 근거)
- Phase6에서 standard/factored 모두 경쟁력이 있었고, context/입력 경로에 따라 cold slice 차이가 나타남.
- Phase5/6에서 balance 계열은 강하게 넣을수록 성능 이득이 제한적이거나 하락하는 경우가 관찰됨.
- 반대로 specialization 계열은 전체 성능 고점/콜드 슬라이스 이득이 서로 다른 조합에서 나옴.
- 따라서 Phase7은 "라우터 입력"과 "aux/reg 철학(balance vs specialization)"을 분리해 인과 확인한다.

## 2) 고정 공통 세팅 (16설정 전체 공통)
아래는 phase7에서 고정한다.

- dataset: KuaiRecLargeStrictPosV2_0.2
- model: featured_moe_n3_tune
- layer_layout: [macro, mid, micro]
- max_item_list_length: 20
- batch_size(train/eval): 4096
- embedding_size: 128
- d_feat_emb: 16
- d_router_hidden: 64
- d_expert_hidden: 128
- expert_scale: 3
- num_heads: 4
- d_ff: 256
- attn_dropout_prob: 0.1
- lr scheduler: warmup_cosine
- learning_rate search: [2e-4, 8e-3] (loguniform)
- weight_decay: 1e-6 (고정)
- hidden_dropout_prob: 0.15 (고정)
- max-evals: 10
- tune_epochs: 100
- tune_patience: 10

기본 residual/진단 로깅:
- stage_residual_mode: base(고정)
- fmoe_diag_logging: true
- fmoe_special_logging: true
- fmoe_feature_ablation_logging: false
- logging root: run/artifacts/logging

## 3) 실험군 A: Router 입력 방식 8설정
Aux/Reg는 동일 baseline으로 고정한다.

Baseline aux/reg (router 비교 공통):
- balance_loss_lambda: 0.002
- z_loss_lambda: 1e-4
- route_smoothness_lambda: 0.01
- route_consistency_lambda: 0.0
- route_sharpness_lambda: 0.0
- route_monopoly_lambda: 0.0
- route_prior_lambda: 0.0
- factored 계열일 때만: group_prior_align_lambda=5e-4, factored_group_balance_lambda=1e-3
- standard일 때: group_prior_align_lambda=0.0, factored_group_balance_lambda=0.0

### A-1) R0_STD (standard)
- stage_router_type: standard
- stage_router_source: both
- stage_feature_injection: gated_bias
- topk_scope_mode: global_flat
- moe_top_k: 0

### A-2) R1_FAC (factored)
- stage_router_type: factored
- stage_router_source: both
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: feature
- stage_factored_group_logit_scale: 1.0
- stage_factored_intra_logit_scale: 1.0
- topk_scope_mode: group_dense
- moe_top_k: 0

### A-3) R2_FAC_HEAVY (factored 비중 강화)
- stage_router_type: factored
- stage_router_source: feature
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: both
- stage_factored_group_logit_scale: 1.6
- stage_factored_intra_logit_scale: 1.0
- topk_scope_mode: group_dense
- moe_top_k: 0

### A-4) R3_FAC_ONLY (factored-only, no stage-whole feature routing)
- stage_router_type: factored
- stage_router_source: hidden
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: feature
- stage_factored_group_logit_scale: 1.0
- stage_factored_intra_logit_scale: 0.0 (group logit만 사용)
- topk_scope_mode: group_dense
- moe_top_k: 0

### A-5) R4_HIR (hierarchical-like multiplicative)
- stage_router_type: factored
- stage_router_source: both
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: both
- stage_factored_combine_mode: hir
- 결합: group 분포 x group-intra 분포(곱)

### A-6) R5_FAC_GROUP (group importance bias)
- stage_router_type: factored
- stage_router_source: both
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: feature
- stage_factored_combine_mode: fac_group
- 결합: dense expert logits + group 중요도 bias(add)

### A-7) R6_FAC_ONLY_BOTH
- stage_router_type: factored
- stage_router_source: hidden
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: both
- stage_factored_intra_logit_scale: 0.0

### A-8) R7_FAC_HEAVY_FEAT
- stage_router_type: factored
- stage_router_source: feature
- stage_feature_injection: group_gated_bias
- stage_factored_group_router_source: feature
- stage_factored_group_logit_scale: 1.6

## 4) 실험군 B: Aux/Reg 8설정
Router anchor 2개를 고정하고, aux/reg만 바꾼다.

Router anchor:
- standard anchor: R0_STD
- factored-heavy anchor: R2_FAC_HEAVY

### balance 계열 2종
B1) BAL_A (light)
- balance_loss_lambda: 0.002
- z_loss_lambda: 1e-4
- route_smoothness/consistency/sharp/monopoly/prior: 0

B2) BAL_B (strong)
- balance_loss_lambda: 0.006
- z_loss_lambda: 3e-4
- route_smoothness/consistency/sharp/monopoly/prior: 0

### specialization 계열 2종
S1) SPEC_A (smoothness)
- balance_loss_lambda: 0.002
- z_loss_lambda: 1e-4
- route_smoothness_lambda: 0.04
- route_consistency_lambda: 0
- route_sharpness_lambda: 0
- route_monopoly_lambda: 0
- route_prior_lambda: 0

S2) SPEC_B (high sharp + monopoly)
- balance_loss_lambda: 0.002
- z_loss_lambda: 1e-4
- route_sharpness_lambda: 0.01
- route_monopoly_lambda: 0.04
- route_monopoly_tau: 0.25
- route_smoothness_lambda: 0
- route_consistency_lambda: 0
- route_prior_lambda: 0

총 8설정 구성:
- standard x {BAL_A, BAL_B, SPEC_A, SPEC_B}
- factored-heavy x {BAL_A, BAL_B, SPEC_A, SPEC_B}

## 5) 총 실험 수 / 시드 / GPU 배분
- setting 수: 16
- seed 수: 4 (1,2,3,4)
- 총 run 수: 16 x 4 = 64

GPU 8개 사용 시:
- round-robin 할당
- 각 GPU 6 run
- 각 GPU는 독립 FIFO 큐
- 다른 GPU와 무관하게 현재 run 종료 즉시 다음 run 실행

## 6) 실행 스크립트
### 파일
- launcher: experiments/run/fmoe_n3/run_phase7_router_aux.py
- shell wrapper: experiments/run/fmoe_n3/phase_7_router_aux.sh

### 전체 64run dry-run 확인
```bash
bash experiments/run/fmoe_n3/phase_7_router_aux.sh \
  --datasets KuaiRecLargeStrictPosV2_0.2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --seeds 1,2,3,4 \
  --group all \
  --dry-run
```

### 전체 실행
```bash
bash experiments/run/fmoe_n3/phase_7_router_aux.sh \
  --datasets KuaiRecLargeStrictPosV2_0.2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --seeds 1,2,3,4 \
  --group all
```

### 실험군 분리 실행
- Router 32run만:
```bash
bash experiments/run/fmoe_n3/phase_7_router_aux.sh --group router
```

- Aux 32run만:
```bash
bash experiments/run/fmoe_n3/phase_7_router_aux.sh --group aux
```

## 7) 산출물 경로와 확인 체크
텍스트 실행 로그:
- experiments/run/artifacts/logs/fmoe_n3/phase7_router_aux_v1/P7/{dataset}/FMoEN3/*.log

run 결과 json:
- experiments/run/artifacts/results/fmoe_n3/*.json

추가 logging bundle:
- experiments/run/artifacts/logging/fmoe_n3/{dataset}/P7/{run_id}/

최소 확인 항목(각 run):
1. run_summary.json
2. special_metrics.json
3. diag json/csv (router 진단)

## 8) 해석 계획 (MRR@20 중심)
1차 비교:
- Router 8설정의 valid/test MRR@20 평균/표준편차/최대

2차 비교:
- cold item(<=5), short session(1-2, 3-5)에서 standard vs factored 계열 차이

3차 비교:
- diag에서 macro/mid/micro별
  - n_eff
  - cv_usage
  - top1_over_scope_uniform
  - route_jitter_adjacent

판단 기준:
- 성능이 비슷하면 cold/special + diag 안정성(낮은 jitter, 과도한 monopoly 회피) 우선

## 9) 다음 후보 (Phase7 종료 후 1~3개)
1. Router winner 고정 후 aux 4종만 재시드(추가 4 seeds)로 분산 검증
2. SPEC_B winner 시 monopoly tau (0.22/0.25/0.30)만 미세 탐색
3. Factored winner 시 stage_router_source(feature vs hidden)만 2x 재검증
