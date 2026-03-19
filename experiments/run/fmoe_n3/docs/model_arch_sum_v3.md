# FeaturedMoE_N3 모델 구조 정리 v3 (코드 기준)

작성일: 2026-03-17  
기준 코드: `experiments/models/FeaturedMoE_N3`

## 1. 문서 목적
- 현재 코드에 반영된 FeaturedMoE_N3의 실제 아키텍처를 발표용으로 정리한다.
- 바꿀 수 있는 실험 축(axis)을 구조/라우팅/정규화/학습 관점에서 체계화한다.
- 지금까지 실험에서 충분히 본 축과 부족한 축을 구분해, 다음 baseline 선정 기준을 만든다.

## 2. 코드 구조 개요

### 2.1 핵심 파일 역할
- `featured_moe_n3.py`
  - 모델 초기화, stage별 설정 파싱, feature 수집/ablation, loss 합산, 진단 수집.
- `stage_executor.py`
  - `layer_layout` 토큰(`layer`, `macro`, `mid`, `micro`, `*_ffn`)을 실제 연산 그래프로 컴파일.
- `stage_modules.py`
  - `N3StageBlock` 구현(라우터, expert 연산, feature injection, residual merge, route regularization).

### 2.2 실행 흐름(요약)
1. 입력 시퀀스 embedding + positional embedding.
2. `layer_layout` 순서대로 attention/stage block 실행.
3. stage block 내부에서
   - feature encoder
   - router logits/weights
   - expert mixture
   - residual merge (`base` 또는 `shared+moe` 계열)
   수행.
4. 최종 hidden으로 CE loss 계산.
5. aux/reg(loss) 항목을 조건부로 합산.

## 3. 변경 가능한 축(코드 기준 최신)

## 3.1 구조 축
- `layer_layout`
  - 예: `[macro, mid, micro]`, `[layer, layer, macro, mid, micro]`, `[layer, macro_ffn, mid_ffn, micro_ffn]`
- `MAX_ITEM_LIST_LENGTH`
  - 주로 20/30/50.
- `embedding_size`, `d_ff`, `d_expert_hidden`, `d_router_hidden`, `expert_scale`
  - 용량/표현력 축.

## 3.2 Stage 계산 방식
- `stage_compute_mode`: `none`, `dense_plain`, `moe`
  - 현재 메인 트랙은 `moe`.

## 3.3 라우터 입력/형태 축
- `stage_router_source`: `hidden`, `feature`, `both`
- `stage_router_type`: `standard`, `factored`
- `stage_router_mode`: `none`, `learned`, `rule_soft`
- `stage_router_granularity`: `session`, `token` (micro는 token 강제)
- `moe_top_k`: dense(`0`) vs sparse(`k>0`)
- `topk_scope_mode` (런처 오버라이드): global/group 단위 top-k 동작 방식.

## 3.4 Feature 조건화 축
- `stage_feature_encoder_mode`: `linear`, `complex`
- `stage_feature_injection`: `none`, `film`, `gated_bias`, `group_gated_bias`
- `stage_feature_family_mask`
  - Tempo/Memory/Focus/Exposure 그룹 선택.
- `macro_history_window`

## 3.5 Residual 축 (v3에서 중요)
`stage_residual_mode`:
- `base`
- `shared_only`
- `shared_moe_fixed`
- `shared_moe_learned`
- `shared_moe_global`
- `shared_moe_learned_warmup`

추가 파라미터:
- `residual_alpha_fixed`
- `residual_alpha_init`
- `residual_shared_ffn_scale`
- schedule 연동 시 `alpha_warmup_*`

즉, 현재 코드는 `h + MoE(...)`만 있는 것이 아니라, shared branch와 alpha 스케줄까지 포함한 구조 실험이 가능하다.

## 3.6 Aux/Reg 축
기존:
- `balance_loss_lambda`
- `z_loss_lambda`
- `gate_entropy_lambda`
- `rule_agreement_lambda`
- `group_prior_align_lambda`
- `factored_group_balance_lambda`

확장(phase5 포함):
- `route_smoothness_lambda`
- `route_consistency_lambda`
- `route_sharpness_lambda`
- `route_monopoly_lambda`, `route_monopoly_tau`
- `route_prior_lambda`, `route_prior_bias_scale`
- `route_smoothness_stage_weight`, `route_consistency_pairs`

요약하면, 단순 balance 정규화에서 벗어나 "specialization 유지 + jitter 억제" 방향의 정규화까지 코드에 들어와 있다.

## 3.7 진단/로깅 축
- `fmoe_special_logging`
- `fmoe_diag_logging`
- `fmoe_feature_ablation_logging`
- 주요 진단치: `n_eff`, `cv_usage`, `top1_max_frac`, `route_jitter_*`, `alpha_value`, branch norm.

## 4. 축별 실험 커버리지 평가

### 4.1 충분히 검증된 축
- `stage_compute_mode` (dense_plain vs moe)
- `router_source`에서 `both` 우위, `feature-only` 리스크 확인
- `standard vs factored` (phase2~4, phase3 S2 포함)
- `feature_injection`에서 gated_bias 계열 유효성
- residual 대분류(`base/shared_only/fixed/global/stage/warmup`) phase4에서 확인

### 4.2 부분 검증 축
- sparse top-k (global/group) 성능/안정성
- macro window(5 vs 10 이상)
- family mask 조합별 일반화
- route regularizer(method M0~M4)는 phase5 문서상 신호 존재하나 재현 검증 필요

### 4.3 부족/보완 필요 축
- phase5 산출물의 공식 result 집계 일관성(문서와 root 결과 파일 불일치)
- 동일 예산/동일 seed의 공정한 standard vs factored 재대결(최종 후보군 한정)
- feature ablation 파일 누락 케이스 보완

## 5. 현재 시점 기준 권장 아키텍처 후보 3개

### 후보 A: 안정형 기준선 (A05/PA05 계열)
- `layer_layout=[macro,mid,micro]`
- `router_type=standard`
- `router_source=both`
- `feature_injection=gated_bias`
- `residual_mode=base`
- `moe_top_k=0`

장점: 재현성과 비교 가능성이 가장 높음.

### 후보 B: specialization 강화형 (P3S2 + K/group_dense 계열)
- `router_type=factored`
- `feature_injection=group_gated_bias`
- `topk_scope=group_dense` 또는 `group_top2`
- `residual_mode=base`
- `group_prior_align_lambda` 약하게 유지

장점: K/F/diag에서 특화 라우팅 신호가 좋고, 최고 test 구간을 자주 만듦.

### 후보 C: 안정-특화 절충형 (warmup residual)
- 후보 B 기반 + `residual_mode=shared_moe_learned_warmup`
- `residual_alpha_init<0`, `alpha_warmup_until` 짧게
- route jitter 억제형 regularizer를 약하게 병행

장점: 초반 안정성과 후반 specialization을 동시에 노리는 구조.

## 6. 발표용 핵심 메시지
- FeaturedMoE_N3는 이제 "MoE 추가 모델"이 아니라,
  - stage별 router/feature/residual을 독립 제어하고,
  - specialization 강도와 안정성을 정규화로 튜닝하는
  실험 플랫폼으로 확장됨.
- 다음 baseline은 A/B/C 3개로 고정하고, 동일 예산 비교로 최종 아키텍처를 확정하는 것이 가장 효율적이다.
